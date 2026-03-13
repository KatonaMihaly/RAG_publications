"""
Evaluation pipeline: load the truth table from Unity Catalog, run the RAG
assistant against each question, and score the results with MLflow using
Correctness and RelevanceToQuery metrics.

Results are logged to the Databricks MLflow experiment defined by
MLFLOW_EXPERIMENT in .env.

Run with:
    .venv/bin/python3 pipeline/evaluate_rag.py

Credentials are read from ~/.databrickscfg. Configuration from .env.
"""

import base64
import json
import os
import time
from pathlib import Path

import mlflow
import pandas as pd
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None  # noqa: E731
from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.vector_search.client import VectorSearchClient

load_dotenv()


def _param(key: str) -> str:
    """Read from Databricks job parameter (widget) on Databricks, or env var locally."""
    try:
        return dbutils.widgets.get(key)  # noqa: F821
    except NameError:
        return os.environ[key]


EVAL_TABLE             = _param("EVAL_TABLE")
MLFLOW_EXPERIMENT      = _param("MLFLOW_EXPERIMENT")
VECTOR_SEARCH_ENDPOINT = _param("VECTOR_SEARCH_ENDPOINT")
VECTOR_SEARCH_INDEX    = _param("VECTOR_SEARCH_INDEX")
LLM_ENDPOINT           = _param("LLM_ENDPOINT")
NUM_RESULTS            = int(_param("NUM_RESULTS"))

# Load prompts from local files when running locally; fall back to job
# parameters injected by deploy_job.py when running on Databricks.
try:
    _PROMPTS_DIR      = Path(__file__).parent / "prompts"
    SYSTEM_PROMPT     = (_PROMPTS_DIR / "research_assistant_system_prompt.md").read_text().strip()
    FEW_SHOT_EXAMPLES = json.loads((_PROMPTS_DIR / "few_shot_examples.json").read_text())
except (NameError, FileNotFoundError):
    SYSTEM_PROMPT     = _param("SYSTEM_PROMPT")
    FEW_SHOT_EXAMPLES = json.loads(base64.b64decode(_param("FEW_SHOT_EXAMPLES")).decode())

# Lazy-initialised once on the first call to predict()
_index  = None
_client = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _get_warehouse_id(w: WorkspaceClient) -> str:
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses found.")
    return warehouses[0].id


def _run_sql(w: WorkspaceClient, warehouse_id: str, sql: str) -> list:
    response = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="0s",
    )
    statement_id = response.statement_id
    while True:
        status = w.statement_execution.get_statement(statement_id)
        state = status.status.state
        if state == StatementState.SUCCEEDED:
            result = status.result
            return result.data_array if result and result.data_array else []
        elif state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            error = status.status.error
            raise RuntimeError(error.message if error else "unknown error")
        else:
            time.sleep(5)


def load_eval_data(w: WorkspaceClient, warehouse_id: str) -> pd.DataFrame:
    """
    Fetch the truth table from Unity Catalog as a pandas DataFrame.

    The table written by generate_truth_table.py uses STRING columns that
    already contain serialised JSON, so no to_json() conversion is needed.
    """
    rows = _run_sql(w, warehouse_id, f"SELECT inputs, expectations FROM {EVAL_TABLE}")

    records = []
    for inputs_json, expectations_json in rows:
        records.append({
            "inputs":       json.loads(inputs_json, strict=False),
            "expectations": json.loads(expectations_json, strict=False) if expectations_json else {},
        })

    df = pd.DataFrame(records)
    print(f"  {len(df)} evaluation examples loaded.")
    return df


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def _init_clients() -> None:
    global _index, _client
    if _index is None:
        vsc    = VectorSearchClient(disable_notice=True)
        _index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=VECTOR_SEARCH_INDEX,
        )
        _client = get_deploy_client("databricks")


def _retrieve(question: str) -> list:
    results = _index.similarity_search(
        query_text=question,
        columns=["chunk_text", "path"],
        num_results=NUM_RESULTS,
        query_type="hybrid",
    )
    return results["result"]["data_array"]


def _generate(question: str, chunks: list) -> str:
    context = "\n\n".join(row[0] for row in chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
    ]
    response = _client.predict(endpoint=LLM_ENDPOINT, inputs={"messages": messages})
    return response["choices"][0]["message"]["content"]


@mlflow.trace
def predict(query: str) -> dict:
    """
    Predict function for mlflow.genai.evaluate.

    The 'inputs' column in the truth table uses 'query' as the key, which
    mlflow unpacks and passes here as a keyword argument.

    Returns a dict so both the answer and the retrieved context are captured
    in the MLflow trace, enabling RelevanceToQuery and Correctness scoring.
    """
    _init_clients()
    chunks  = _retrieve(query)
    answer  = _generate(query, chunks)
    context = "\n\n".join(row[0] for row in chunks)
    return {"response": answer, "retrieved_context": context}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def evaluate_rag() -> None:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    w            = WorkspaceClient()
    warehouse_id = _get_warehouse_id(w)

    print("Loading evaluation data from truth table...")
    eval_data = load_eval_data(w, warehouse_id)

    print("Running MLflow evaluation (this calls the LLM for each row)...")
    with mlflow.start_run():
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict,
            scorers=[
                mlflow.genai.scorers.Correctness(metric_name="answer_correctness"),
                mlflow.genai.scorers.RelevanceToQuery(metric_name="context_relevance"),
            ],
        )

    print("\nEvaluation complete. Metrics:")
    for metric, value in results.metrics.items():
        formatted = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"  {metric}: {formatted}")


if __name__ == "__main__":
    evaluate_rag()
