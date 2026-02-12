import json
import mlflow
from itertools import combinations
import google.genai as genai
from pydantic import BaseModel, Field, conlist
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class DisagreementDetails(BaseModel):
    Methodology: list[str] = Field(default_factory=list)
    Experiments: list[str] = Field(default_factory=list)
    Clarity: list[str] = Field(default_factory=list)
    Significance: list[str] = Field(default_factory=list)
    Novelty: list[str] = Field(default_factory=list)

class DisagreementResult(BaseModel):
    paper_id: str
    review_pair: list[str]
    disagreement_score: float = Field(..., ge=0.0, le=1.0)
    disagreement_details: DisagreementDetails ## More detailed

def list_to_string(lst):
    return "\n".join(lst) if lst else ""

with open("data/processed/critique_points/critique_points_llm_2.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

mlflow.set_experiment("Review Disagreement Analysis")
with mlflow.start_run():
    results = []
    
    for paper in tqdm(papers, desc="Processing Papers"): ## Add TQDM 
        paper_id = paper["paper_id"]
        critique_points = paper["critique_points"]
        
        review_pairs = list(combinations(range(len(critique_points)), 2))
        
        for r1, r2 in review_pairs:
            review1, review2 = critique_points[r1], critique_points[r2]
            
            # Change prompt to handle empty comments
            prompt = f"""
            Compare the following critiques for the paper '{paper_id}'.
            Assess disagreement across Methodology, Experiments, Clarity, Significance, and Novelty.
            Return a disagreement score (0-1) and points of disagreement.
            
            Review 1:
            Methodology: {list_to_string(review1.get('Methodology', []))}
            Experiments: {list_to_string(review1.get('Experiments', []))}
            Clarity: {list_to_string(review1.get('Clarity', []))}
            Significance: {list_to_string(review1.get('Significance', []))}
            Novelty: {list_to_string(review1.get('Novelty', []))}

            Review 2:
            Methodology: {list_to_string(review2.get('Methodology', []))}
            Experiments: {list_to_string(review2.get('Experiments', []))}
            Clarity: {list_to_string(review2.get('Clarity', []))}
            Significance: {list_to_string(review2.get('Significance', []))}
            Novelty: {list_to_string(review2.get('Novelty', []))}
            
            Output JSON with keys: disagreement_score (float), disagreement_details (dict)
            """

            config = {
                "response_mime_type": "application/json",
                "response_schema": DisagreementResult,
            }

            retries, max_retries, base_wait = 0, 5, 5
            while retries < max_retries:
                try:
                    response = client.models.generate_content(
                        contents=prompt, model="gemini-2.0-flash-lite", config=config
                    )
                    llm_output = json.loads(response.text) 
                    break  

                except genai.errors.ClientError as e:
                    if "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = base_wait * (2 ** retries)
                        print(f"Quota exceeded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        print(f"API error: {e}")
                        llm_output = {"error": str(e)}
                        break  

                except json.JSONDecodeError:
                    llm_output = {"error": "Failed to parse LLM response"}
                    break  

            else: 
                print("Max retries reached. Skipping batch.")
                llm_output = {"error": "LLM failed after retries"}

            # If API failed, continue to next pair
            if "error" in llm_output:
                continue

            disagreement_result = DisagreementResult(
                paper_id=paper_id,
                review_pair=[str(r1), str(r2)],  # Ensure it's a list of strings
                disagreement_score=llm_output["disagreement_score"],
                disagreement_details=llm_output["disagreement_details"]
            )
            
            results.append(disagreement_result.model_dump_json())

            with open('arya_disagg_llm2.json', 'a', encoding='utf-8') as file:
                json.dump(disagreement_result.model_dump_json(), file, indent=4)

            mlflow.log_metric("disagreement_score", disagreement_result.disagreement_score)
    
    output_file = "data/processed/disagreement_detection/disagreement_scores_llm_2.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    mlflow.log_artifact("data/processed/disagreement_detection/disagreement_scores_llm_2.json")

print("Disagreement detection complete. Results saved.")