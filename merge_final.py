import pandas as pd
df1 = pd.read_csv("data/processed/consensus_resolution/meta_reviews_final.csv")
df2 = pd.read_csv("data\processed\search\papers_with_search_final.csv")
merged_df_A = pd.merge(df1, df2, on='paper_id', how='inner')
merged_df_A.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], inplace=True)
merged_df_A.rename(columns={'meta_review_x': "deepseek_meta_review", 'meta_review_y': 'human_meta_review'}, inplace=True)
print(merged_df_A.info())
merged_df_A.to_csv("combined_final.csv")
