import json
import mlflow
import os
import time
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class MetaReviewResult(BaseModel):
    paper_id: str
    meta_review: str 

disagreement_resolution_df = pd.read_csv("data/processed/consensus_resolution/disaagreement_resolution_final_batch_2.csv")
papers = pd.read_csv("data/processed/search/papers_with_search_2.csv")
merged_data = pd.merge(papers, disagreement_resolution_df, on="paper_id", how="inner")

output_file = "data/processed/consensus_resolution/meta_reviews_batch_2.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if os.path.exists(output_file):
    existing_results = pd.read_csv(output_file)
    completed_papers = set(existing_results["paper_id"].astype(str))
else:
    existing_results = pd.DataFrame(columns=["paper_id", "meta_review"])
    completed_papers = set()

def construct_prompt(row):
    paper_title = row['paper_title']
    paper_abstract = row['paper_abstract']
    sota_results = row['SoTA Results']
    retrieved_evidence = row['Retrieved Evidence']
    accepted_critique_points = eval(row['accepted_critique_points'])
    rejected_critique_points = eval(row['rejected_critique_points'])
    final_resolution_summary = row['final_resolution_summary']

    system_prompt = """
    You are an expert meta-reviewer. Your task is to generate a structured meta-review based on reviewer critiques, disagreements, and resolutions.
    Your review should be clear, concise, and well-structured.
    Respond with only the meta-review text.
    """

    user_prompt = f"""
    ### **Paper Details**
    **Title:** {paper_title}  
    **Abstract:** {paper_abstract}  

    ### **Final Disagreement Resolution Summary**
    {final_resolution_summary}

    ### **Accepted Critique Points (Valid Feedback)**
    {json.dumps(accepted_critique_points, indent=2)}

    ### **Rejected Critique Points (Unjustified Criticism)**
    {json.dumps(rejected_critique_points, indent=2)}

    ### **State-of-the-Art (SoTA) Findings**
    {sota_results}

    ### **Retrieved Evidence for Validation**
    {retrieved_evidence}

    ### **Meta-Review Task**
    1. Summarize the strengths and weaknesses of the paper.
    2. Discuss key disagreements among reviewers and how they were resolved.
    3. Compare the paper’s claims with state-of-the-art research and evidence.
    4. Provide a final verdict on the paper’s quality, clarity, and contribution.
    """
    
    return system_prompt, user_prompt

mlflow.set_experiment("Meta-Review Generation")

with mlflow.start_run():
    for _, row in tqdm(merged_data.iterrows(), desc="Generating Meta-Reviews", total=len(merged_data)):
        if row["paper_id"] in completed_papers:
            print(f"Skipping over {row['paper_id']} paper!")
            continue  # Skip already processed papers
        
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
                )
                
                meta_review_text = response.choices[0].message.content.strip()
                if not meta_review_text:
                    raise ValueError("Empty response from LLM")
                
                break  
            except Exception as e:
                wait_time = base_wait * (2 ** retries)
                print(f"\nError: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
        else:
            print("\nMax retries reached. Skipping.")
            meta_review_text = "Error generating meta-review"

        result_df = pd.DataFrame([{ "paper_id": row["paper_id"], "meta_review": meta_review_text }])
        result_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False, encoding="utf-8")
        
        mlflow.log_text(meta_review_text, f"meta_reviews/{row['paper_id']}.txt")

print("Meta-review generation complete. Results saved as CSV.")
