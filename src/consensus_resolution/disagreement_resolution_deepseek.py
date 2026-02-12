import json
import mlflow
import os
import time
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict
# import pprint
import ast

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class ResolutionDetails(BaseModel):
    accepted_critique_points: Dict[str, List[str]]  # Accepted points per category
    rejected_critique_points: Dict[str, List[str]]  # Rejected points per category
    final_resolution_summary: str  # Summary of the resolved stance

class DisagreementResolutionResult(BaseModel):
    paper_id: str
    review_pair: List[str]
    resolution_details: ResolutionDetails

def load_completed_papers(output_file):
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        return set(df["paper_id"].astype(str))
    return set()

output_file = "data/processed/consensus_resolution/disagreement_resolution_batch_2.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
completed_papers = load_completed_papers(output_file)

papers = pd.read_csv("data/processed/search/papers_with_search_2.csv")
disagreements = pd.read_csv("data/processed/disagreement_detection/disagreement_aggregated_2.csv")
merged_data = pd.merge(papers, disagreements, on="paper_id", how="inner")

def construct_prompt(row):
    """
    Constructs a structured prompt for DeepSeek-R1 to resolve disagreements using reasoning and evidence verification.
    """
    paper_title = row['paper_title']
    paper_abstract = row['paper_abstract']
    disagreement_score = row['disagreement_score']
    disagreement_details = eval(row['disagreement_details'])
    critique_points = eval(row['Combined Critiques'])
    sota_results = row['SoTA Results']
    retrieved_evidence = row['Retrieved Evidence']

    system_prompt = """
    You are an AI specialized in resolving academic peer review disagreements. 
    Your task is to analyze critiques, verify evidence, and provide a structured resolution.
    
    Respond in the following JSON format:
    {
      "accepted_critique_points": {{"category": ["critique_1", "critique_2"]}},
      "rejected_critique_points": {{"category": ["critique_3"]}},
      "final_resolution_summary": "After analyzing critiques and evidence, we conclude that..."
    }
    """

    user_prompt = f"""
    ### **Paper Details**
    **Title:** {paper_title}  
    **Abstract:** {paper_abstract}  

    ### **Reviewer Disagreement (Score: {disagreement_score})**
    - **Methodology:** {disagreement_details.get('Methodology', 'N/A')}
    - **Experiments:** {disagreement_details.get('Experiments', 'N/A')}
    - **Clarity:** {disagreement_details.get('Clarity', 'N/A')}
    - **Significance:** {disagreement_details.get('Significance', 'N/A')}
    - **Novelty:** {disagreement_details.get('Novelty', 'N/A')}

    ### **Supporting Information**
    **Critique Points from Reviews:**  
    {json.dumps(critique_points, indent=2)}

    **State-of-the-Art (SoTA) Findings:**  
    {sota_results}

    **Retrieved Evidence:**  
    {retrieved_evidence}

    ### **Resolution Task**
    1. Validate critique points and categorize them into accepted or rejected.
    2. Compare with SoTA research and retrieved evidence.
    3. Provide a final resolution summary explaining whether the disagreement is justified.
    """

    return system_prompt, user_prompt

mlflow.set_experiment("Disagreement Resolution Analysis")

with mlflow.start_run():

    for _, row in tqdm(merged_data.iterrows(), desc="Resolving Disagreements", total=len(merged_data)):
        paper_id = str(row["paper_id"])
        if paper_id in completed_papers:
            print(f"{paper_id} paper already computed!")
            continue
        
        system_prompt, user_prompt = construct_prompt(row)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        retries, max_retries, base_wait = 0, 5, 5
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-r1-zero:free",
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                if not response.choices or not response.choices[0].message.content.strip():
                    # raise ValueError("Empty response from DeepSeek-R1")
                    wait_time = base_wait * (2 ** retries)
                    print(f"\nEmpty response from DeepSeek-R1. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    continue
                llm_output = json.loads(response.choices[0].message.content[6:])
                if set(llm_output.keys()) != {"accepted_critique_points", "rejected_critique_points", "final_resolution_summary"}:
                    wait_time = base_wait * (2 ** retries)
                    print(f"\nNot all columns present. Columns present right now: {llm_output.keys()}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    continue

            except Exception as e:
                    wait_time = base_wait * (2 ** retries)
                    print(f"\n{str(e).lower()}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    continue

            try:
                resolved_disagreement = DisagreementResolutionResult(
                    paper_id=row["paper_id"],
                    review_pair=ast.literal_eval(row["review_pair"]),
                    resolution_details=ResolutionDetails(
                        accepted_critique_points=llm_output["accepted_critique_points"],
                        rejected_critique_points=llm_output["rejected_critique_points"],
                        final_resolution_summary=llm_output["final_resolution_summary"],
                    ),
                )
                break
            except Exception as e:
                    wait_time = base_wait * (2 ** retries)
                    print(f"\n{str(e).lower()}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    continue

        else:
            print("\nMax retries reached. Skipping.")
            llm_output = {"error": "LLM failed after retries"}

        # if "error" in llm_output:
        #     continue

        results_df = pd.DataFrame([{  
            "paper_id": resolved_disagreement.paper_id,
            "review_pair": "; ".join(resolved_disagreement.review_pair),
            "accepted_critique_points": json.dumps(resolved_disagreement.resolution_details.accepted_critique_points),
            "rejected_critique_points": json.dumps(resolved_disagreement.resolution_details.rejected_critique_points),
            "final_resolution_summary": resolved_disagreement.resolution_details.final_resolution_summary,
        }])

        results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False, encoding="utf-8")
        mlflow.log_metric("resolved_disagreement_score", row["disagreement_score"])

#     output_file = "data/processed/disagreement_resolution.csv"
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)

#     results_df = pd.DataFrame([
#         {
#             "paper_id": res["paper_id"],
#             "review_pair": "; ".join(res["review_pair"]),
#             "accepted_critique_points": json.dumps(res["resolution_details"]["accepted_critique_points"]),
#             "rejected_critique_points": json.dumps(res["resolution_details"]["rejected_critique_points"]),
#             "final_resolution_summary": res["resolution_details"]["final_resolution_summary"],
#         }
#         for res in results
#     ])

#     results_df.to_csv(output_file, index=False, encoding="utf-8")
#     mlflow.log_artifact(output_file)

# print("Disagreement resolution complete. Results saved as CSV.")
