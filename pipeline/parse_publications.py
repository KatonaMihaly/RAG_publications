"""
Bronze layer: parse PDFs from Unity Catalog Volume into a structured Delta table using ai_parse_document.

Runs the SQL remotely on a Databricks SQL warehouse via the Statement Execution API.

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


VOLUME_PATH = _param("RAW_PAPERS_VPATH")
TARGET_TABLE = _param("PARSED_PAPERS_TPATH")

sql_create_table = f"""
CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
    path            STRING,
    parsed_output   VARIANT,
    modificationTime TIMESTAMP
) USING DELTA
"""

sql_delete_removed = f"""
DELETE FROM {TARGET_TABLE}
WHERE path NOT IN (
    SELECT path FROM READ_FILES('{VOLUME_PATH}', format => 'binaryFile')
)
"""

sql_insert_new = f"""
INSERT INTO {TARGET_TABLE}
SELECT
    path,
    ai_parse_document(content, map('version', '2.0')) AS parsed_output,
    modificationTime
FROM READ_FILES('{VOLUME_PATH}', format => 'binaryFile')
WHERE path NOT IN (SELECT path FROM {TARGET_TABLE})
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
            raise RuntimeError(f"Statement {state.value}: {error.message if error else 'unknown error'}")
        else:
            print(f"  Status: {state.value} — waiting...")
            time.sleep(10)


def parse_publications() -> None:
    w = WorkspaceClient()
    warehouse_id = _get_warehouse_id(w)
    print(f"Using warehouse {warehouse_id}")

    print(f"Step 1/3: Ensuring table '{TARGET_TABLE}' exists...")
    _run_sql(w, warehouse_id, sql_create_table, "Create table")

    print(f"Step 2/3: Removing rows for deleted PDFs...")
    _run_sql(w, warehouse_id, sql_delete_removed, "Delete removed")

    print(f"Step 3/3: Inserting new PDFs from '{VOLUME_PATH}'...")
    _run_sql(w, warehouse_id, sql_insert_new, "Insert new")
    print(f"Done. Table '{TARGET_TABLE}' is up to date.")


if __name__ == "__main__":
    parse_publications()
