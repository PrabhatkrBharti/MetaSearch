import json
import pandas as pd

with open("data\processed\critique_points\critique_points_llm_2.json", "r", encoding="utf-8") as file:
    critiques = json.load(file)  

with open("data\processed\search\search_final_2.json", "r", encoding="utf-8") as file:
    search = json.load(file)

for critique in critiques:
    title = critique["paper_title"]
    if title in search:
        if "SoTA Results" in search[title]:
            critique["SoTA Results"] = search[title]["SoTA Results"]
        if "Combined Critiques" in search[title]:
            critique["Combined Critiques"] = search[title]["Combined Critiques"]
        if "Retrieved Evidence" in search[title]:
            critique["Retrieved Evidence"] = search[title]["Retrieved Evidence"]

with open("data\processed\search\papers_with_search_2.json", "w", encoding="utf-8") as file:
    json.dump(critiques, file, indent=4, ensure_ascii=False)

df = pd.DataFrame(critiques)
df.to_csv("data\processed\search\papers_with_search_2.csv", index=False, encoding="utf-8")
