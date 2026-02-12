# -*- coding: utf-8 -*-
"""
# Creating Search Agent using Langchain.

## Tools used:
## LLM: Gemini (gemini-2.0-flash)
## Search APIs: Semantic Scholar, arXiv, Google Serpic API (Google Scholar), Tavily Search

# Installing libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install --upgrade --quiet langchain-community langgraph langchain-anthropic tavily-python langgraph-checkpoint-sqlite semanticscholar google-search-results arxiv

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qU "langchain[groq]"

from dotenv import load_dotenv
load_dotenv()

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qU tavily-python

"""# Creating Peer Review Agent"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qU langchain-google-genai

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_retries=2,
)

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

semantic_scholar = SemanticScholarQueryRun()
google_scholar = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
arxiv_search = ArxivAPIWrapper()
tavily_search = TavilySearchResults(max_results=5)

search_tools = [tavily_search, semantic_scholar, google_scholar, arxiv_search]

from langchain.agents import initialize_agent, AgentExecutor
from langchain.tools import Tool

tools = [
    Tool(name="TavilySearch", func=tavily_search.run, description="Retrieves the latest State-of-the-Art (SoTA) research"),
    Tool(name="SemanticScholar", func=semantic_scholar.run, description="Find academic papers"),
    Tool(name="GoogleScholar", func=google_scholar.run, description="Search for scholarly articles"),
    Tool(name="ArxivSearch", func=arxiv_search.run, description="Find research papers on ArXiv"),
]

custom_prompt_general_sota_research = """
Answer the following question to the best of your ability. You have access to the following tools:

- TavilySearch(tool_input: 'str, str') → Retrieves the latest State-of-the-Art (SoTA) research on a given topic.
- SemanticScholar(tool_input: 'str, str') → Finds relevant academic papers.
- GoogleScholar(tool_input: 'str, str') → Searches for scholarly articles.
- ArxivSearch(tool_input: 'str, str') → Retrieves research papers from ArXiv.

Use the following structured format:

Question: The input question you must answer.
Thought: Analyze the question and determine the best approach.
Action: Use all four tools sequentially.
Action Input: Provide the appropriate input for each tool.
Observation: Capture and document the result of each action.

### Begin! ###

Question: Find the latest state-of-the-art research related to the paper '{paper_title}'.
Abstract: {paper_abstract}

Thought: I need to retrieve the latest research from all available sources.

Action: TavilySearch
Action Input: "{paper_title}", "{paper_abstract}"
Observation: {TavilySearch Output}

Action: SemanticScholar
Action Input: "{paper_title}", "{paper_abstract}"
Observation: {SemanticScholar Output}

Action: GoogleScholar
Action Input: "{paper_title}", "{paper_abstract}"
Observation: {GoogleScholar Output}

Action: ArxivSearch
Action Input: "{paper_title}", "{paper_abstract}"
Observation: {ArXiv Output}

Thought: I have gathered results from all sources. I will now summarize and rank the most relevant papers.
Final Answer: {Aggregated and ranked results}
"""

from langchain.agents import AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": custom_prompt_general_sota_research}
)

import json

def log_output(stage, data):
    with open("../../data/processed/search/logs.json", "a") as f:
        json.dump({stage: data}, f, indent=4)
        f.write("\n")

def search_relevant_research(paper_title, paper_abstract):
    query = f"Find latest state-of-the-art research related to the paper '{paper_title}'. Abstract: {paper_abstract}"

    try:
        response = agent.run(query)
        print(f"Raw Response: {response}")

        if not response.strip():
            raise ValueError("Received empty response from agent.")

        return response

    except Exception as e:
        print(f"⚠️ Output Parsing Failed: {e}\nRetrying with error handling...")
        log_output("SoTA_Search_Error", str(e))
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True
        )
        return agent_executor.run(query)

def combine_critiques(reviews):
    categories = ["Methodology", "Clarity", "Experiments", "Significance", "Novelty"]
    categorized_critiques = {cat: [] for cat in categories}

    for review in reviews:
        for category, critiques in review.items():
            if category in categorized_critiques:
                categorized_critiques[category].extend(critiques)

    for category in categories:
        categorized_critiques[category] = " ".join(categorized_critiques[category])

    return categorized_critiques

def retrieve_evidence(categorized_critiques):

    evidence_results = {}
    for category, critiques in categorized_critiques.items():
        if not critiques:
            continue

        query = f"Find research papers that support or contradict the following critiques related to {category}: " + "; ".join(critiques)
        result = agent.run(query)
        evidence_results[category] = result

    return evidence_results

import os

SAVE_PATH = "../../data/processed/search/search_intermediate_2.json"

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        processed_papers = json.load(f)
else:
    processed_papers = {}

def save_progress(paper_id, step, data):
    if paper_id not in processed_papers:
        processed_papers[paper_id] = {}

    processed_papers[paper_id][step] = data

    with open(SAVE_PATH, "w") as f:
        json.dump(processed_papers, f, indent=4)

def ai_agent_pipeline(paper):
    paper_id = paper["paper_title"]

    if paper_id in processed_papers and "Retrieved Evidence" in processed_papers[paper_id]:
        print(f"✅ Skipping {paper_id}, already fully processed.\n")
        return processed_papers[paper_id]

    title, abstract, reviews = paper["paper_title"], paper["paper_abstract"], paper["critique_points"]

    try:
        if paper_id in processed_papers and "SoTA Results" in processed_papers[paper_id]:
            sota_results = processed_papers[paper_id]["SoTA Results"]
        else:
            print("Step 1: General SoTA Search")
            sota_results = search_relevant_research(title, abstract)
            save_progress(paper_id, "SoTA Results", sota_results)
            print("✅ SoTA Search Completed\n")

        if paper_id in processed_papers and "Combined Critiques" in processed_papers[paper_id]:
            combined_critiques = processed_papers[paper_id]["Combined Critiques"]
        else:
            print("Step 2: Combining Critique Points")
            combined_critiques = combine_critiques(reviews)
            save_progress(paper_id, "Combined Critiques", combined_critiques)
            print("✅ Combining Completed\n")

        if paper_id in processed_papers and "Retrieved Evidence" in processed_papers[paper_id]:
            evidence = processed_papers[paper_id]["Retrieved Evidence"]
        else:
            print("Step 3: Retrieving Evidence for Critique Points")
            evidence = retrieve_evidence(combined_critiques)
            save_progress(paper_id, "Retrieved Evidence", evidence)
            print("✅ Evidence Retrieval Completed\n")

        return {
            "SoTA Results": sota_results,
            "Combined Critiques": combined_critiques,
            "Retrieved Evidence": evidence,
        }

    except Exception as e:
        print(f"❌ Error processing {paper_id}: {e}")
        return None

"""## Testing each function individually"""

paper_abstract = """
While the widespread use of Large Language
Models (LLMs) brings convenience, it also
raises concerns about the credibility of aca-
demic research and scholarly processes. To
better understand these dynamics, we evalu-
ate the penetration of LLMs across academic
workflows from multiple perspectives and di-
mensions, providing compelling evidence of
their growing influence. We propose a frame-
work with two components: ScholarLens, a
curated dataset of human-written and LLM-
generated content across scholarly writing and
peer review for multi-perspective evaluation,
and LLMetrica, a tool for assessing LLM pen-
etration using rule-based metrics and model-
based detectors for multi-dimensional evalua-
tion. Our experiments demonstrate the effec-
tiveness of LLMetrica, revealing the increas-
ing role of LLMs in scholarly processes. These
findings emphasize the need for transparency,
accountability, and ethical practices in LLM
usage to maintain academic credibility.
"""

paper_title = "Visual Correspondence Hallucination"

response = search_relevant_research(paper_title, paper_abstract)

sample_paper = {} # taking too much space, so removed it

ans = combine_critiques(sample_paper["critique_points"])

results = ai_agent_pipeline(sample_paper)

results["Retrieved Evidence"]

"""# Running pipeline on the whole dataset"""

import json
input_file = "../../data/processed/critique_points/critique_points_llm_2.json"
papers = json.load(open(input_file, "r"))

SAVE_PATH = "../../data/processed/search/search_intermediate_2.json"

import os

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        processed_papers = json.load(f)
else:
    processed_papers = {}

for paper in papers:
    paper = ai_agent_pipeline(paper)

output_file = "../../data/processed/search/search_results_2.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=4)

output_papers = json.load(open(output_file, "r"))
