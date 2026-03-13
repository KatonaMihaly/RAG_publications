"""
Silver layer: explode parsed papers into chunks and write to a CDF-enabled Delta table.

Runs SQL remotely on a Databricks SQL warehouse via the Statement Execution API.

Credentials are read automatically from ~/.databrickscfg.
"""

import os
import time
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None  # noqa: E731
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

load_dotenv()


def _param(key: str) -> str:
    """Read from Databricks job parameter (widget) on Databricks, or env var locally."""
    try:
        return dbutils.widgets.get(key)  # noqa: F821
    except NameError:
        return os.environ[key]


PARSED_PAPERS_TPATH  = _param("PARSED_PAPERS_TPATH")
CHUNKED_PAPERS_TPATH = _param("CHUNKED_PAPERS_TPATH")
MIN_CHUNK_SIZE       = _param("MIN_CHUNK_SIZE")

sql_create_table = f"""
CREATE TABLE IF NOT EXISTS {CHUNKED_PAPERS_TPATH} (
    path       STRING,
    chunk_id   STRING,
    chunk_text STRING,
    chunk_type STRING
) USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
"""

sql_delete_removed = f"""
DELETE FROM {CHUNKED_PAPERS_TPATH}
WHERE path NOT IN (SELECT DISTINCT path FROM {PARSED_PAPERS_TPATH})
"""

sql_insert_new = f"""
INSERT INTO {CHUNKED_PAPERS_TPATH}
SELECT
    path,
    uuid()                          AS chunk_id,
    exploded.value:content::string  AS chunk_text,
    exploded.value:type::string     AS chunk_type
FROM {PARSED_PAPERS_TPATH},
LATERAL VARIANT_EXPLODE(parsed_output:document.elements) AS exploded(pos, key, value)
WHERE exploded.value:content IS NOT NULL
  AND length(exploded.value:content::string) > {MIN_CHUNK_SIZE}
  AND path NOT IN (SELECT DISTINCT path FROM {CHUNKED_PAPERS_TPATH})
"""


def _get_warehouse_id(w: WorkspaceClient) -> str:
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses found in this workspace.")
    return warehouses[0].id


def _run_sql(w: WorkspaceClient, warehouse_id: str, sql: str, label: str) -> None:
    response = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="0s",
    )
    statement_id = response.statement_id
    print(f"  Statement ID: {statement_id}")

    while True:
        status = w.statement_execution.get_statement(statement_id)
        state = status.status.state

        if state == StatementState.SUCCEEDED:
            print(f"  {label} — done.")
            break
        elif state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            error = status.status.error
            raise RuntimeError(error.message if error else "unknown error")
        else:
            print(f"  {label} — {state.value}, waiting...")
            time.sleep(10)


def chunk_publications() -> None:
    w = WorkspaceClient()
    warehouse_id = _get_warehouse_id(w)
    print(f"Using warehouse {warehouse_id}")

    print(f"Step 1/3: Ensuring table '{CHUNKED_PAPERS_TPATH}' exists (CDF enabled)...")
    _run_sql(w, warehouse_id, sql_create_table, "Create table")

    print(f"Step 2/3: Removing chunks for deleted PDFs...")
    _run_sql(w, warehouse_id, sql_delete_removed, "Delete removed")

    print(f"Step 3/3: Inserting chunks for new paths from '{PARSED_PAPERS_TPATH}'...")
    _run_sql(w, warehouse_id, sql_insert_new, "Insert new")
    print(f"Done. '{CHUNKED_PAPERS_TPATH}' is ready for vector indexing.")


if __name__ == "__main__":
    chunk_publications()
