"""
RAG query module: retrieve relevant chunks from the vector index
                  and generate a grounded answer via Llama 3.
"""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from mlflow.deployments import get_deploy_client
from databricks.vector_search.client import VectorSearchClient

load_dotenv()

ENDPOINT_NAME = os.environ["VECTOR_SEARCH_ENDPOINT"]
INDEX_NAME    = os.environ["VECTOR_SEARCH_INDEX"]
LLM_ENDPOINT  = os.environ["LLM_ENDPOINT"]
NUM_RESULTS   = int(os.environ["NUM_RESULTS"])

_PROMPTS_DIR = Path(__file__).parent / "prompts"

SYSTEM_PROMPT     = (_PROMPTS_DIR / "research_assistant_system_prompt.md").read_text().strip()
FEW_SHOT_EXAMPLES = json.loads((_PROMPTS_DIR / "few_shot_examples.json").read_text())


def retrieve(index, question: str) -> list:
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_text_string", "path"],
        num_results=NUM_RESULTS,
        query_type="hybrid",
    )
    return results["result"]["data_array"]


def generate(client, question: str, chunks: list) -> str:
    context = "\n\n".join(row[0] for row in chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
    ]
    response = client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={"messages": messages},
    )
    return response["choices"][0]["message"]["content"]


def research_assistant(question: str) -> tuple[str, pd.DataFrame]:
    """
    Returns:
        answer  – the LLM-generated answer string
        chunks  – DataFrame with columns [Rank, Document, Chunk, Similarity]
    """
    vsc   = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)

    client = get_deploy_client("databricks")

    chunks = retrieve(index, question)
    answer = generate(client, question, chunks)

    df = pd.DataFrame(chunks, columns=["Chunk", "Path", "Similarity"])
    df["Document"] = df["Path"].apply(os.path.basename)
    df = df.drop(columns=["Path"])
    df.insert(0, "Rank", range(1, len(df) + 1))
    df = df[["Rank", "Document", "Chunk", "Similarity"]]

    return answer, df


if __name__ == "__main__":
    question = "How is the Taguchi method used for robust design?"
    answer, df = research_assistant(question)
    print(answer)
    print()
    print(df.to_string(index=False))
