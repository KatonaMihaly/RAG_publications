"""
Silver layer: explode parsed papers into chunks and write to a CDF-enabled Delta table.

Runs SQL remotely on a Databricks SQL warehouse via the Statement Execution API.

Credentials are read automatically from ~/.databrickscfg.
"""

import os
import time
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

load_dotenv()

PARSED_PAPERS_TPATH  = os.environ["PARSED_PAPERS_TPATH"]
CHUNKED_PAPERS_TPATH  = os.environ["CHUNKED_PAPERS_TPATH"]
MIN_CHUNK_SIZE = os.environ["MIN_CHUNK_SIZE"]

sql_command_chunk = f"""
CREATE OR REPLACE TABLE {CHUNKED_PAPERS_TPATH} AS
SELECT
    path,
    uuid()                          AS chunk_id,
    exploded.value:content::string  AS chunk_text,
    exploded.value:type::string     AS chunk_type
FROM {PARSED_PAPERS_TPATH},
LATERAL VARIANT_EXPLODE(parsed_output:document.elements) AS exploded(pos, key, value)
WHERE exploded.value:content IS NOT NULL
  AND length(exploded.value:content::string) > {MIN_CHUNK_SIZE}
"""

sql_command_enableChangeDataFeed = f"""
ALTER TABLE {CHUNKED_PAPERS_TPATH}
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
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

    print(f"Step 1/2: Chunking '{PARSED_PAPERS_TPATH}' → '{CHUNKED_PAPERS_TPATH}'...")
    _run_sql(w, warehouse_id, sql_command_chunk, "Chunking")

    print(f"Step 2/2: Enabling Change Data Feed on '{CHUNKED_PAPERS_TPATH}'...")
    _run_sql(w, warehouse_id, sql_command_enableChangeDataFeed, "CDF")

    print(f"Done. '{CHUNKED_PAPERS_TPATH}' is ready for vector indexing.")


if __name__ == "__main__":
    chunk_publications()
