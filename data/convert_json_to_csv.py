import json
import pandas as pd
import ast

input_path = "data\processed\disagreement_detection\disagreement_scores_llm_2.json"

# with open(input_path, "r") as f:
#     parsed_data = [json.loads(line) for line in f]

with open(input_path, "r", encoding="utf-8") as file_name:
    parsed_data = json.load(file_name)

rows = []
for entry in parsed_data:
    entry = ast.literal_eval(entry)
    paper_id = entry["paper_id"]
    review_pair = entry["review_pair"]
    disagreement_score = entry["disagreement_score"]
    
    rows.append({
        "paper_id": paper_id,
        "review_pair": review_pair,
        "disagreement_score": disagreement_score,
        "disagreement_details": entry["disagreement_details"]
    })

df = pd.DataFrame(rows)
df.to_csv("data\processed\disagreement_detection\disagreement_scores_llm_2.csv", index=False)
