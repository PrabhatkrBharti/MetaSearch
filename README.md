# MetaSearch: Search-Augmented LLM with Reasoning for Consensus Resolution in Peer Review

This repository contains the implementation and data processing pipeline for the paper "MetaSearch: Search-Augmented LLM with Reasoning for Consensus Resolution in Peer Review".

## Repository Structure

```
MetaSearch/
├── .gitignore                         # Files to ignore
├── LICENSE                            # License info
├── README.md                          # Overview, setup, and usage
├── requirements.txt                   # Dependencies
├── Supplementary Materials.pdf        # PDF of Supplementary Materials
├── workflow_drawio.png                # Visual workflow diagram
├── data                               # Data processing and datasets
│   ├── aggregate.py                   # Aggregates extracted data
│   ├── convert_json_to_csv.py         # Converts JSON files to CSV
│   ├── merge_final.py                 # Merges finalized datasets
│   ├── merge_search_critique.py       # Merges search results with critiques
│   ├── prepare_new_subset.py          # Prepares a new dataset subset
│   ├── prepare_sample_subset.py       # Prepares a sample dataset subset
│   ├── processed                      # Processed data outputs
│   │   ├── consensus_resolution       # Consensus resolution outputs
│   │   │   ├── aggregated_dr.csv              # Aggregated disagreement resolution data
│   │   │   ├── aggregated_mr.csv              # Aggregated meta-review data
│   │   │   ├── combined_final.csv             # Combined final dataset
│   │   │   ├── disaagreement_resolution_final.csv  # Final resolution with discrepancies
│   │   │   ├── disagreement_resolution.csv     # Disagreement resolution data
│   │   │   ├── disagreement_resolution_final.csv# Final disagreement resolution data
│   │   │   ├── meta_reviews.csv                # Generated meta reviews
│   │   │   └── meta_reviews_final.csv          # Final meta reviews
│   │   ├── critique_points            # Extracted critique points
│   │   │   ├── critique_points_llm.json        # LLM-extracted (v1)
│   │   │   ├── critique_points_llm_2.json      # LLM-extracted (v2)
│   │   │   ├── critique_points_nlp.json        # NLP-extracted (v1)
│   │   │   └── critique_points_nlp_2.json      # NLP-extracted (v2)
│   │   ├── disagreement_detection     # Disagreement detection outputs
│   │   │   ├── aggregated_ds_llm.csv            # LLM-based aggregated disagreement scores
│   │   │   ├── disagreement_aggregated.csv      # Aggregated disagreement data (v1)
│   │   │   ├── disagreement_aggregated_2.csv    # Aggregated disagreement data (v2)
│   │   │   ├── disagreement_scores_llm.csv      # LLM disagreement scores (CSV)
│   │   │   ├── disagreement_scores_llm.json     # LLM disagreement scores (JSON, v1)
│   │   │   ├── disagreement_scores_llm_2.csv      # LLM disagreement scores (CSV, v2)
│   │   │   ├── disagreement_scores_llm_2.json     # LLM disagreement scores (JSON, v2)
│   │   │   ├── disagreement_scores_llm_2_arya.json# Alternative LLM disagreement scores
│   │   │   ├── disagreement_scores_nlp.json       # NLP disagreement scores
│   │   │   └── metrics.csv                        # Evaluation metrics
│   │   └── search                     # Search-retrieved evidence and logs
│   │       ├── aggregated_cp.json               # Aggregated critique points from search
│   │       ├── aggregated_papers_with_search.csv # Papers with search annotations
│   │       ├── critique_points_llm_2.json         # Additional LLM critique points (v2)
│   │       ├── drive-download-20250329T065307Z-001# Drive download (timestamped)
│   │       ├── logs.json                         # Search process logs
│   │       ├── papers_with_search_2.csv           # Papers with search results (CSV, v2)
│   │       ├── papers_with_search_2.json          # Papers with search results (JSON, v2)
│   │       ├── papers_with_search_final.csv       # Final papers with search results
│   │       ├── papers_with_search_initial.json    # I nitial papers with search results
│   │       ├── papers_with_search_progress.csv    # Search progress tracking
│   │       ├── search_batch_2.json                # Search results batch (v2)
│   │       ├── search_final_2.json                # Final search results (v2)
│   │       ├── search_intermediate.json           # Intermediate search results (v1)
│   │       └── search_intermediate_2.json         # Intermediate search results (v2)
│   ├── raw                           # Original, unprocessed data
│   │   ├── cache-2ff1e739000f2062.arrow         # Cached data (v1)
│   │   ├── cache-3465735dd11c43b2.arrow         # Cached data (v2)
│   │   ├── cache-48201879335d942f.arrow         # Cached data (v3)
│   │   ├── data-00000-of-00001.arrow            # Primary raw data file
│   │   ├── dataset_info.json                    # Dataset metadata
│   │   └── state.json                           # Data processing state
│   ├── split                         # Dataset splits (train/test/val)
│   │   ├── sample_subset_train.json             # Sample training subset (v1)
│   │   ├── sample_subset_train_2.json           # Sample training subset (v2)
│   │   ├── test                      # Test split
│   │   │   ├── data-00000-of-00001.arrow        # Test data
│   │   │   ├── dataset_info.json                # Test info
│   │   │   └── state.json                       # Test state
│   │   ├── test.json                            # Test metadata
│   │   ├── train                     # Training split
│   │   │   ├── data-00000-of-00001.arrow        # Train data
│   │   │   ├── dataset_info.json                # Train info
│   │   │   └── state.json                       # Train state
│   │   ├── train.json                           # Training metadata
│   │   ├── val                       # Validation split
│   │   │   ├── data-00000-of-00001.arrow        # Validation data
│   │   │   ├── dataset_info.json                # Validation info
│   │   │   └── state.json                       # Validation state
│   │   └── val.json                             # Validation metadata
│   └── split_dataset.py                # Generates data splits
├── figures                          # Visualizations and figures (mentioned in detail in Supplementary Materials PDF document)
│   ├── Acceptance vs. Review Ratings.png
│   ├── Average Disagreement Score Trend (NeurIPS and ICLR).png
│   ├── Cosine similarity - Generated Meta Review vs Original Meta Review.png
│   ├── Distribution of Disagreement Scores.png
│   ├── Distribution of Review Lengths.png
│   ├── Distribution of Reviewer Disagreements.png
│   ├── Jaccard Similarity Distributions.png
│   ├── Review Length Distribution by Confidence Score.png
│   ├── Review Rating Distribution for Accepted vs Rejected Papers.png
│   ├── SOTA Results and Paper Title Similarity Score.png
│   ├── Sentiment Score Distribution for Clarity.png
│   ├── Sentiment Subjectivity Histogram.png
│   └── figures_in_paper
│       ├── Review Rating Distribution by Paper Acceptance.png
│       ├── cosine.png
│       └── meta_review_bert.png
├── notebooks                        # Jupyter notebooks for analysis
│   ├── 01_Exploratory_Analysis.ipynb
│   ├── 02_EDA_CritiqueLLM.ipynb
│   ├── 03_Disagreement_Scores_LLM.ipynb
│   ├── 04_Search_Result_Analysis.ipynb
│   ├── 05_Retrieved_Evidences_Analysis.ipynb
│   ├── 06_Disagreement_Resolution.ipynb
│   └── 07_Meta_Review_Comparison.ipynb
├── src                              # Core source code
│   ├── consensus_resolution
│   │   ├── disagreement_resolution_deepseek.py  # DeepSeek-based resolution
│   │   └── meta_review_generation.py            # Meta review generation
│   ├── critique_points_extraction
│   │   ├── extract_critique_llm.py               # LLM-based extraction
│   │   └── extract_critique_nlp.py               # NLP-based extraction
│   ├── disagreement_detection
│   │   ├── compare_reviews_llm.py                # LLM-based comparison
│   │   └── compare_reviews_nlp.py                # NLP-based comparison
│   └── search_retrieval
│       ├── search_agent.ipynb                    # Search agent exploration
└       └── search_agent.py                       # Search agent implementation
```

## Methodology

Our approach follows a systematic workflow:

1. Data Collection and Preprocessing
2. Search-Augmented Review Analysis
3. Consensus Detection
4. Result Validation

For detailed workflow visualization, see [workflow_drawio.png](workflow_drawio.png).

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

Research Paper: "MetaSearch: Search-Augmented LLM with Reasoning for Consensus Resolution in Peer Review" ([Back to Top](https://github.com/AnonymousSubmission45/MetaSearch/tree/main?tab=readme-ov-file#metasearch-search-augmented-llm-with-reasoning-for-consensus-resolution-in-peer-review)🔝)
