"""
Bronze layer: parse PDFs from Unity Catalog Volume into a structured Delta table.

New files are ingested via Auto Loader (cloudFiles) with trigger(availableNow=True),
so ai_parse_document is only called on files not yet in the checkpoint — if nothing
changed, the stream exits immediately without modifying the table.

Deleted files are removed with a SQL DELETE after the stream completes.

Credentials are read automatically from ~/.databrickscfg.
"""

import os
import time
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None  # noqa: E731
from pyspark.sql.functions import expr  # noqa: F821 - spark available on Databricks
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

load_dotenv()


def _param(key: str) -> str:
    """Read from Databricks job parameter (widget) on Databricks, or env var locally."""
    try:
        return dbutils.widgets.get(key)  # noqa: F821
    except NameError:
        return os.environ[key]


VOLUME_PATH     = _param("RAW_PAPERS_VPATH")
TARGET_TABLE    = _param("PARSED_PAPERS_TPATH")
CHECKPOINT_PATH = _param("PARSE_CHECKPOINT_VPATH")

sql_delete_removed = f"""
DELETE FROM {TARGET_TABLE}
WHERE path NOT IN (
    SELECT path FROM READ_FILES('{VOLUME_PATH}', format => 'binaryFile')
)
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
    # Step 1: Stream new PDFs — Auto Loader only processes files not in the checkpoint
    print(f"Step 1/2: Streaming new PDFs from '{VOLUME_PATH}' into '{TARGET_TABLE}'...")
    (spark  # noqa: F821 - injected by Databricks runtime
        .readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .load(VOLUME_PATH)
        .select(
            "path",
            expr("ai_parse_document(content, map('version', '2.0'))").alias("parsed_output"),
            "modificationTime",
        )
        .writeStream
        .format("delta")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .outputMode("append")
        .trigger(availableNow=True)
        .toTable(TARGET_TABLE)
        .awaitTermination()
    )

    # Step 2: Remove rows for PDFs deleted from the volume
    print(f"Step 2/2: Removing rows for deleted PDFs...")
    w = WorkspaceClient()
    warehouse_id = _get_warehouse_id(w)
    _run_sql(w, warehouse_id, sql_delete_removed, "Delete removed")
    print(f"Done. Table '{TARGET_TABLE}' is up to date.")


if __name__ == "__main__":
    parse_publications()
