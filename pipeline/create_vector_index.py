"""
Gold layer: create a Mosaic AI Vector Search endpoint and Delta Sync index
            over the chunked papers silver table.

On subsequent runs (index already exists) the function triggers a sync instead
of re-creating, so this script is safe to call from a scheduled nightly job.

Credentials are read automatically from ~/.databrickscfg.
"""

import os
import time
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None  # noqa: E731
from databricks.vector_search.client import VectorSearchClient

load_dotenv()


def _param(key: str) -> str:
    """Read from Databricks job parameter (widget) on Databricks, or env var locally."""
    try:
        return dbutils.widgets.get(key)  # noqa: F821
    except NameError:
        return os.environ[key]


SOURCE_TABLE      = _param("CHUNKED_PAPERS_TPATH")
ENDPOINT_NAME     = _param("VECTOR_SEARCH_ENDPOINT")
INDEX_NAME        = _param("VECTOR_SEARCH_INDEX")
EMBEDDING_MODEL   = _param("EMBEDDING_MODEL")
EMBEDDING_COLUMN  = _param("EMBEDDING_COLUMN")
PRIMARY_KEY       = _param("PRIMARY_KEY")

# Maximum minutes to wait for index creation or sync to complete
INDEX_TIMEOUT_MIN = 30


def get_or_create_endpoint(client: VectorSearchClient, endpoint_name: str) -> None:
    existing = [e["name"] for e in client.list_endpoints().get("endpoints", [])]
    if endpoint_name in existing:
        print(f"  Endpoint '{endpoint_name}' already exists.")
        return
    print(f"  Creating endpoint '{endpoint_name}'...")
    client.create_endpoint_and_wait(name=endpoint_name, endpoint_type="STANDARD")
    print(f"  Endpoint '{endpoint_name}' ready.")


_FAILED_STATES = {"OFFLINE", "FAILED", "ONLINE_PIPELINE_FAILED"}


def _wait_for_index(client: VectorSearchClient) -> str:
    """
    Poll until the index reaches ONLINE or ONLINE_NO_PENDING_UPDATE.
    Returns the final detailed_state string.
    Raises RuntimeError if the index enters a failed state.
    """
    deadline = time.time() + INDEX_TIMEOUT_MIN * 60
    while time.time() < deadline:
        status = client.get_index(ENDPOINT_NAME, INDEX_NAME).describe()
        state = status.get("status", {}).get("detailed_state", "UNKNOWN")
        print(f"  Index state: {state}")
        if state in ("ONLINE_NO_PENDING_UPDATE", "ONLINE"):
            print("  Index is ready.")
            return state
        if state in _FAILED_STATES:
            raise RuntimeError(f"Index entered failed state: {state}")
        time.sleep(20)
    raise TimeoutError(f"Index did not become ready within {INDEX_TIMEOUT_MIN} minutes.")


def _create_fresh_index(client: VectorSearchClient) -> None:
    """Delete the index if it exists, then recreate it from scratch."""
    existing = [i["name"] for i in client.list_indexes(ENDPOINT_NAME).get("vector_indexes", [])]
    if INDEX_NAME in existing:
        print(f"  Deleting stale index '{INDEX_NAME}'...")
        client.delete_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
        # Wait until it disappears
        while True:
            existing = [i["name"] for i in client.list_indexes(ENDPOINT_NAME).get("vector_indexes", [])]
            if INDEX_NAME not in existing:
                break
            print("  Waiting for index deletion...")
            time.sleep(10)

    print(f"  Creating index '{INDEX_NAME}' (this may take several minutes)...")
    client.create_delta_sync_index_and_wait(
        endpoint_name=ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        primary_key=PRIMARY_KEY,
        pipeline_type="TRIGGERED",
        embedding_source_column=EMBEDDING_COLUMN,
        embedding_model_endpoint_name=EMBEDDING_MODEL,
    )


def create_vector_index() -> None:
    client = VectorSearchClient()

    print(f"Step 1/2: Ensuring endpoint '{ENDPOINT_NAME}' exists...")
    get_or_create_endpoint(client, ENDPOINT_NAME)

    existing_indexes = [
        i["name"] for i in client.list_indexes(ENDPOINT_NAME).get("vector_indexes", [])
    ]

    if INDEX_NAME in existing_indexes:
        print(f"Step 2/2: Index '{INDEX_NAME}' already exists — triggering sync...")
        client.get_index(ENDPOINT_NAME, INDEX_NAME).sync()
        try:
            _wait_for_index(client)
        except RuntimeError as e:
            print(f"  Sync failed ({e}). Recreating index to reset checkpoint...")
            _create_fresh_index(client)
            _wait_for_index(client)
    else:
        print(f"Step 2/2: Creating index '{INDEX_NAME}'...")
        _create_fresh_index(client)
        _wait_for_index(client)

    print(f"Done. Index '{INDEX_NAME}' is ready for querying.")


if __name__ == "__main__":
    create_vector_index()
