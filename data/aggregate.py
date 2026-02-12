import pandas as pd
import json
import ast

def merge_disagreement_details(series):
    merged_details = {}
    for detail_json in series:
        if isinstance(detail_json, str): 
            try:
                details_dict = ast.literal_eval(detail_json)
                for category, comments in details_dict.items():
                    if category not in merged_details:
                        merged_details[category] = []
                    if isinstance(comments, list):
                        merged_details[category].extend(comments)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON: {detail_json}")
    return json.dumps(merged_details)

def merge_review_pairs(series):
    review_pairs = []
    for pair_list in series:
        try:
            # pair_list = json.loads(pair_str)
            review_pairs.append(pair_list)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON for review_pair: {pair_list}")
    return json.dumps(review_pairs) 

df = pd.read_csv("data/processed/disagreement_detection/disagreement_scores_llm_2.csv") 

merged_df = df.groupby('paper_id').agg({
    'review_pair': merge_review_pairs,
    'disagreement_score': 'mean',
    'disagreement_details': merge_disagreement_details
}).reset_index()

merged_df.columns = ['paper_id', 'review_pair', 'disagreement_score', 'disagreement_details']
merged_df.to_csv("data/processed/disagreement_detection/disagreement_aggregated_2.csv", index=False)
