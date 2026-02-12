import json
import numpy as np
import nltk
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # Import tqdm
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")
nltk.download("punkt_tab")

class CritiquePoint(BaseModel):
    Methodology: list[str] = []
    Experiments: list[str] = []
    Clarity: list[str] = []
    Significance: list[str] = []
    Novelty: list[str] = []

CATEGORY_EXEMPLARS = {
    "Methodology": [
        "The approach is not well-justified.",
        "There is a lack of theoretical grounding.",
        "The method lacks proper motivation.",
    ],
    "Experiments": [
        "The experiments are insufficient.",
        "More baselines should be compared.",
        "Results are not statistically significant.",
    ],
    "Clarity": [
        "The paper is not well-written.",
        "Some sections are unclear.",
        "Key assumptions are not explained.",
    ],
    "Significance": [
        "The contributions are not impactful.",
        "The problem addressed is too narrow.",
        "This work lacks real-world applicability.",
    ],
    "Novelty": [
        "The paper lacks novelty.",
        "It is too similar to prior work.",
        "There is little innovation in this work.",
    ],
}

category_embeddings = {
    category: np.mean(embedder.encode(exemplars), axis=0)
    for category, exemplars in CATEGORY_EXEMPLARS.items()
}

# def load_reviews(path):
#     with open(path, "r") as f:
#         return [json.loads(line) for line in f]

def classify_sentence(sentence_embedding):
    similarities = {
        category: cosine_similarity(
            [sentence_embedding], [category_embeddings[category]]
        )[0][0]
        for category in CATEGORY_EXEMPLARS.keys()
    }
    return max(similarities, key=similarities.get)  

def extract_critique_points(review_text):
    critique = CritiquePoint()
    sentences = nltk.sent_tokenize(review_text)
    sentence_embeddings = embedder.encode(sentences)

    for sent, sent_emb in zip(sentences, sentence_embeddings):
        category = classify_sentence(sent_emb)
        getattr(critique, category).append(sent)

    return critique.model_dump_json()

def process_reviews(reviews):
    for review in tqdm(reviews, desc="Processing Reviews"):  # Add tqdm progress bar
        extracted_reviews = []

        for writer, content in zip(review["review_writers"], review["review_contents"]):
            if writer == "official_reviewer" and len(content) > 100:
                extracted_reviews.append(extract_critique_points(content))

        review["critique_points"] = extracted_reviews

        if "meta_review" in review and len(review["meta_review"]) > 100:
            review["meta_review_critique"] = extract_critique_points(review["meta_review"])

    return reviews

input_file = "data/split/sample_subset_train_2.json"
reviews = json.load(open(input_file, "r"))
critique_data = process_reviews(reviews)

output_file = "data/processed/critique_points/critique_points_nlp_2.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(critique_data, f, indent=4)

print(f"Extracted critique points saved to {output_file}")