"""
Truth table generation: fetch the chunked papers from Unity Catalog, use
databricks-agents to synthesise question/answer pairs, convert them to the
flat format expected by mlflow.genai.evaluate, and write the result back to
Unity Catalog.

Run with:
    .venv/bin/python3 pipeline/generate_truth_table.py

This is a one-off (or on-demand) step — re-run whenever new papers are added
and you want the truth table to cover the updated corpus.

Output table: EVAL_TABLE (default workspace.default.truth_table_converted)
Schema:
    inputs       STRING  — JSON: {"query": "<question>"}
    expectations STRING  — JSON: {"expected_facts": [...],
                                  "expected_retrieved_context": [...]}

Credentials are read from ~/.databrickscfg. Configuration from .env.
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None  # noqa: E731
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.agents.evals import generate_evals_df

load_dotenv()


def _param(key: str, default: str | None = None) -> str:
    """Read from Databricks job parameter (widget) on Databricks, or env var locally."""
    try:
        return dbutils.widgets.get(key)  # noqa: F821
    except NameError:
        return os.environ.get(key, default) if default is not None else os.environ[key]


CHUNKED_PAPERS_TPATH = _param("CHUNKED_PAPERS_TPATH")
EVAL_TABLE           = _param("EVAL_TABLE")
NUM_EVALS            = int(_param("NUM_EVALS", "20"))

try:
    _PROMPTS_DIR        = Path(__file__).parent / "prompts"
    AGENT_DESCRIPTION   = (_PROMPTS_DIR / "agent_description.md").read_text().strip()
    QUESTION_GUIDELINES = (_PROMPTS_DIR / "question_guidelines.md").read_text().strip()
except (NameError, FileNotFoundError):
    AGENT_DESCRIPTION   = _param("AGENT_DESCRIPTION")
    QUESTION_GUIDELINES = _param("QUESTION_GUIDELINES")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_warehouse_id(w: WorkspaceClient) -> str:
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses found.")
    return warehouses[0].id


def _run_sql(w: WorkspaceClient, warehouse_id: str, sql: str, label: str) -> list:
    response = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="0s",
    )
    statement_id = response.statement_id
    print(f"  [{label}] statement_id={statement_id}")
    while True:
        status = w.statement_execution.get_statement(statement_id)
        state = status.status.state
        if state == StatementState.SUCCEEDED:
            print(f"  [{label}] done.")
            result = status.result
            return result.data_array if result and result.data_array else []
        elif state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            error = status.status.error
            raise RuntimeError(f"[{label}] {state.value}: {error.message if error else 'unknown'}")
        else:
            print(f"  [{label}] {state.value} — waiting...")
            time.sleep(10)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def fetch_chunks(w: WorkspaceClient, warehouse_id: str) -> pd.DataFrame:
    """
    Fetch chunk_text and chunk_id from the silver table.
    Non-ASCII characters are stripped to avoid token-count issues in the
    generator (same pre-processing as the notebook).
    """
    rows = _run_sql(w, warehouse_id, f"""
        SELECT chunk_id, chunk_text_string
        FROM   {CHUNKED_PAPERS_TPATH}
        WHERE  chunk_text_string IS NOT NULL
    """, "fetch_chunks")

    df = pd.DataFrame(rows, columns=["doc_uri", "content"])
    # Strip non-ASCII (avoids tiktoken vocabulary fetch failures)
    df["content"] = df["content"].apply(lambda t: re.sub(r"[^\x00-\x7F]+", " ", t))
    print(f"  {len(df)} chunks fetched.")
    return df


def generate_evals(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Call databricks-agents to produce synthetic question/answer pairs."""
    print(f"  Generating {NUM_EVALS} synthetic eval(s) — this calls the LLM and may take a few minutes...")
    return generate_evals_df(
        chunks_df,
        num_evals=NUM_EVALS,
        agent_description=AGENT_DESCRIPTION,
        question_guidelines=QUESTION_GUIDELINES,
    )


def _extract_query(inputs_val) -> str:
    """Extract the user question from the inputs dict/string returned by generate_evals_df."""
    if isinstance(inputs_val, str):
        inputs_val = json.loads(inputs_val)
    # New databricks-agents format: {"question": "..."}
    if "question" in inputs_val:
        return inputs_val["question"]
    # Legacy format: {"messages": [{"role": "user", "content": "..."}]}
    messages = inputs_val.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def convert(raw_df: pd.DataFrame) -> list[dict]:
    """
    Flatten the generate_evals_df output to the format expected by
    mlflow.genai.evaluate:
        inputs       = {"query": "<question>"}
        expectations = {"expected_facts": [...], "expected_retrieved_context": [...]}
    """
    if raw_df.empty:
        print("  WARNING: generate_evals_df returned an empty DataFrame.")
        return []
    print(f"  raw_df columns: {list(raw_df.columns)}")
    print(f"  raw_df sample inputs[0]: {raw_df.iloc[0].get('inputs', '<missing>')}")

    records = []
    for _, row in raw_df.iterrows():
        query = _extract_query(row["inputs"])
        if not query:
            print(f"  WARNING: could not extract query from inputs: {row['inputs']}")
            continue
        expectations = row.get("expectations", {})
        if isinstance(expectations, str):
            expectations = json.loads(expectations)
        records.append({
            "inputs":       {"query": query},
            "expectations": expectations,
        })
    print(f"  {len(records)} rows after conversion.")
    return records


def write_truth_table(w: WorkspaceClient, warehouse_id: str, records: list[dict]) -> None:
    """
    Write the converted records to Unity Catalog as a Delta table with two
    plain STRING columns (JSON-serialised), replacing any prior version.
    """
    # 1. Create (or replace) the table with explicit STRING schema
    _run_sql(w, warehouse_id, f"""
        CREATE OR REPLACE TABLE {EVAL_TABLE} (
            inputs       STRING COMMENT 'JSON: {{"query": "..."}}',
            expectations STRING COMMENT 'JSON: {{"expected_facts": [...], "expected_retrieved_context": [...]}}'
        )
        USING DELTA
    """, "create_table")

    # 2. Batch INSERT all rows in a single statement
    if not records:
        raise RuntimeError("No eval records were generated — cannot write an empty truth table.")

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("'", "''")

    values = ",\n    ".join(
        f"('{_esc(json.dumps(r['inputs']))}', '{_esc(json.dumps(r['expectations']))}')"
        for r in records
    )
    _run_sql(w, warehouse_id, f"""
        INSERT INTO {EVAL_TABLE} (inputs, expectations)
        VALUES {values}
    """, "insert_rows")

    print(f"  {len(records)} rows written to '{EVAL_TABLE}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_truth_table() -> None:
    print(f"\n=== Truth Table Generation — {NUM_EVALS} evals from '{CHUNKED_PAPERS_TPATH}' ===\n")

    w            = WorkspaceClient()
    warehouse_id = _get_warehouse_id(w)
    print(f"Using warehouse: {warehouse_id}\n")

    print("[1/4] Fetching chunks from silver table...")
    chunks_df = fetch_chunks(w, warehouse_id)

    print("\n[2/4] Generating synthetic evals...")
    raw_df = generate_evals(chunks_df)

    print("\n[3/4] Converting to flat query/expectations format...")
    records = convert(raw_df)

    print("\n[4/4] Writing truth table to Unity Catalog...")
    write_truth_table(w, warehouse_id, records)

    print(f"\n=== Done. Run evaluate_rag.py to score the RAG pipeline. ===\n")


if __name__ == "__main__":
    generate_truth_table()
