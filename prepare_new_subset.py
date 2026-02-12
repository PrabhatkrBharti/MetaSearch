import json
import random

def create_sample_dataset(input_path, output_path, exclude_path, sample_size=100):
    with open(input_path, "r") as f:
        papers = [json.loads(line) for line in f]

    try:
        with open(exclude_path, "r") as f:
            excluded_papers = {paper["paper_id"] for paper in json.load(f)}
    except FileNotFoundError:
        excluded_papers = set()

    remaining_papers = [paper for paper in papers if paper["paper_id"] not in excluded_papers]

    sampled_papers = random.sample(remaining_papers, min(sample_size, len(remaining_papers)))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_papers, f, indent=4)

    print(f"Sample dataset of {len(sampled_papers)} reviews saved to {output_path}")

input_file = "data/split/train.json"
first_subset_file = "data/split/sample_subset_train.json"  # Already created first subset
second_subset_file = "data/split/sample_subset_train_2.json"

create_sample_dataset(input_file, second_subset_file, first_subset_file, sample_size=100)
