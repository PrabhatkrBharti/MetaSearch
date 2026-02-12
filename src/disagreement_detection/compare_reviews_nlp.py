import json
import numpy as np
import mlflow
import mlflow.sklearn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm

mlflow.set_experiment("XAI Paper Disagreement Detection")

embedder = SentenceTransformer("allenai/specter")

def load_critique_points(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_disagreement_score(critique1, critique2):
    if not critique1 or not critique2:
        return 0.0  

    # Ensure non-empty lists before encoding
    critique1 = [c for c in critique1 if c.strip()]
    critique2 = [c for c in critique2 if c.strip()]
    
    if not critique1 or not critique2:
        return 0.0  

    embeddings1 = embedder.encode(critique1)
    embeddings2 = embedder.encode(critique2)

    pairwise_similarities = cosine_similarity(embeddings1, embeddings2)
    avg_sim = np.mean(pairwise_similarities)

    disagreement_score = 1 - avg_sim
    return float(disagreement_score if disagreement_score > 0.1 else 0.1) # Higher means more disagreement

def detect_disagreements(reviews):
    disagreement_scores = []
    
    with mlflow.start_run():
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        review_count = 0
        for review in tqdm(reviews, desc="Processing Reviews....", position=0, leave=True):
            review_count += 1
            paper_id = review.get("paper_id", "unknown")
            disagreements = []
            critique_lists = review.get("critique_points", [])
            
            for i in range(len(critique_lists)):
                for j in range(i + 1, len(critique_lists)):
                    score = compute_disagreement_score(
                        critique_lists[i], critique_lists[j]
                    )
                    mlflow.log_metric(f"disagreement_score_{paper_id}", score)
                    disagreements.append(score)
                    
            avg_disagreement = np.mean(disagreements) if disagreements else 0
            mlflow.log_metric(f"disagreement_score_{paper_id}", avg_disagreement)
            disagreement_scores.append({"paper_id": paper_id, "score": avg_disagreement})
            # print(f"Review {review_count} disagreement detection completed!")

        # mlflow.log_artifact("data/processed/critique_points_llm_2.json")
    
    return disagreement_scores

input_file = "data/processed/critique_points/critique_points_llm_2.json"
reviews = load_critique_points(input_file)
disagreement_results = detect_disagreements(reviews)

output_file = "data/processed/disagreement_detection/disagreement_scores_nlp_2.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(disagreement_results, f, indent=4)

print(f"Disagreement detection completed. Results saved to {output_file}")
