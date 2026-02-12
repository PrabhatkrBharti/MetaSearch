import json
import os
import time
import google.genai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class CritiquePoint(BaseModel):
    Methodology: list[str]
    Experiments: list[str]
    Clarity: list[str]
    Significance: list[str]
    Novelty: list[str]

def extract_critique_points_llm(review_text):
    prompt = f"""
    Extract key critique points from the following research paper review.
    Categorize them into aspects: Methodology, Experiments, Clarity, Significance, Novelty.
    Return a structured JSON. 

    Review:
    {json.dumps(review_text, indent=4)} 
    """
    config = {
        "response_mime_type": "application/json",
        "response_schema": CritiquePoint,
    }

    retries, max_retries, base_wait = 0, 5, 5
    while retries < max_retries:
        try:
            response = client.models.generate_content(
                contents=prompt, model="gemini-2.0-flash-lite", config=config
            )
            return json.loads(response.text) 

        except genai.errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                wait_time = base_wait * (2 ** retries)
                print(f"Quota exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"API error: {e}")
                return {"error": str(e)}

        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response"}

    print("Max retries reached. Skipping batch.")
    return {"error": "LLM failed after retries"}

def process_reviews_llm(reviews):
    for i, review in enumerate(tqdm(reviews, desc="Processing Reviews", leave=False)):
        extracted_reviews = []

        for writer, content in zip(review["review_writers"], review["review_contents"]):
            if writer == "official_reviewer" and len(content) > 100:
                extracted_reviews.append(extract_critique_points_llm(content))

        review["critique_points"] = extracted_reviews

        if "meta_review" in review and len(review["meta_review"]) > 100:
            review["meta_review_critique"] = extract_critique_points_llm(review["meta_review"])

        print(f"Critique Points for Review {i+1} generated")

    return reviews

input_file = "data/split/sample_subset_train_2.json"
reviews = json.load(open(input_file, "r"))
critique_data = process_reviews_llm(reviews)

output_file_1 = "data/processed/critique_points/critique_points_llm_2.json"
os.makedirs(os.path.dirname(output_file_1), exist_ok=True)
with open(output_file_1, "w", encoding="utf-8") as f:
    json.dump(critique_data, f, indent=4)


print(f"Extracted critique points (LLM only) saved to {output_file_1}.")