"""
Bronze layer: parse PDFs from Unity Catalog Volume into a structured Delta table using ai_parse_document.

Runs the SQL remotely on a Databricks SQL warehouse via the Statement Execution API.

Credentials are read automatically from ~/.databrickscfg.
"""

import os
import time
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

load_dotenv()

VOLUME_PATH = os.environ["RAW_PAPERS_VPATH"]
TARGET_TABLE = os.environ["PARSED_PAPERS_TPATH"]

sql_command_parse = f"""
CREATE OR REPLACE TABLE {TARGET_TABLE} AS
SELECT
    path,
    ai_parse_document(content, map('version', '2.0')) AS parsed_output,
    modificationTime
FROM READ_FILES('{VOLUME_PATH}', format => 'binaryFile')
"""


def get_warehouse_id(w: WorkspaceClient) -> str:
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses found in this workspace.")
    return warehouses[0].id


def parse_publications() -> None:
    w = WorkspaceClient()
    warehouse_id = get_warehouse_id(w)

    print(f"Submitting parsing job to warehouse {warehouse_id}...")
    response = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql_command_parse,
        wait_timeout="0s"
    )

    statement_id = response.statement_id
    print(f"Statement ID: {statement_id}")

    while True:
        status = w.statement_execution.get_statement(statement_id)
        state = status.status.state

        if state == StatementState.SUCCEEDED:
            print(f"Done. Table '{TARGET_TABLE}' created successfully.")
            break
        elif state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            error = status.status.error
            raise RuntimeError(f"Statement {state.value}: {error.message if error else 'unknown error'}")
        else:
            print(f"  Status: {state.value} — waiting...")
            time.sleep(10)


if __name__ == "__main__":
    parse_publications(WorkspaceClient())
