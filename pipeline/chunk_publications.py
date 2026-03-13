"""
Silver layer: explode parsed papers into chunks and write to a CDF-enabled Delta table.

New rows in parsed_papers are processed via Structured Streaming with
trigger(availableNow=True), so LATERAL VARIANT_EXPLODE only runs on newly
parsed PDFs. If parsed_papers has no new rows the stream exits immediately.

Deleted PDFs are removed with a SQL DELETE after the stream completes.

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
CHECKPOINT_PATH      = _param("CHUNK_CHECKPOINT_VPATH")

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


def _process_batch(batch_df, batch_id):
    """Explode each micro-batch of new parsed rows into chunks."""
    if batch_df.isEmpty():
        return
    batch_df.createOrReplaceTempView("_new_parsed")
    spark.sql(  # noqa: F821 - spark is a Databricks runtime global
        f"""
        INSERT INTO {CHUNKED_PAPERS_TPATH}
        SELECT
            path,
            uuid()                          AS chunk_id,
            exploded.value:content::string  AS chunk_text,
            exploded.value:type::string     AS chunk_type
        FROM _new_parsed,
        LATERAL VARIANT_EXPLODE(parsed_output:document.elements) AS exploded(pos, key, value)
        WHERE exploded.value:content IS NOT NULL
          AND length(exploded.value:content::string) > {MIN_CHUNK_SIZE}
          AND exploded.value:type::string IN (
              'paragraph', 'text', 'table', 'list_item', 'caption', 'formula'
          )
    """)


def chunk_publications() -> None:
    w = WorkspaceClient()
    warehouse_id = _get_warehouse_id(w)

    # Step 1: Ensure target table exists before the stream writes to it
    print(f"Step 1/3: Ensuring table '{CHUNKED_PAPERS_TPATH}' exists (CDF enabled)...")
    _run_sql(w, warehouse_id, sql_create_table, "Create table")

    # Step 2: Stream new rows from parsed_papers — only unprocessed rows per checkpoint
    print(f"Step 2/3: Streaming new chunks from '{PARSED_PAPERS_TPATH}'...")
    (spark  # noqa: F821 - injected by Databricks runtime
        .readStream
        .format("delta")
        .table(PARSED_PAPERS_TPATH)
        .writeStream
        .foreachBatch(_process_batch)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )

    # Step 3: Remove chunks for PDFs deleted from parsed_papers
    print(f"Step 3/3: Removing chunks for deleted PDFs...")
    _run_sql(w, warehouse_id, sql_delete_removed, "Delete removed")
    print(f"Done. '{CHUNKED_PAPERS_TPATH}' is ready for vector indexing.")


if __name__ == "__main__":
    chunk_publications()
