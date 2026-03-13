"""
Gold layer: create a Mosaic AI Vector Search endpoint and Delta Sync index
            over the chunked papers silver table.

Credentials are read automatically from ~/.databrickscfg.
"""

import os
from dotenv import load_dotenv
from databricks.vector_search.client import VectorSearchClient

load_dotenv()

SOURCE_TABLE      = os.environ["CHUNKED_PAPERS_TPATH"]
ENDPOINT_NAME     = os.environ["VECTOR_SEARCH_ENDPOINT"]
INDEX_NAME        = os.environ["VECTOR_SEARCH_INDEX"]
EMBEDDING_MODEL   = os.environ["EMBEDDING_MODEL"]
EMBEDDING_COLUMN  = os.environ["EMBEDDING_COLUMN"]
PRIMARY_KEY       = os.environ["PRIMARY_KEY"]


def get_or_create_endpoint(client: VectorSearchClient, endpoint_name: str) -> None:
    existing = [e["name"] for e in client.list_endpoints().get("endpoints", [])]
    if endpoint_name in existing:
        print(f"  Endpoint '{endpoint_name}' already exists.")
        return
    print(f"  Creating endpoint '{endpoint_name}'...")
    client.create_endpoint_and_wait(name=endpoint_name, endpoint_type="STANDARD")
    print(f"  Endpoint '{endpoint_name}' ready.")


def create_vector_index() -> None:
    client = VectorSearchClient()

    print(f"Step 1/2: Ensuring endpoint '{ENDPOINT_NAME}' exists...")
    get_or_create_endpoint(client, ENDPOINT_NAME)

    print(f"Step 2/2: Creating index '{INDEX_NAME}' (this may take several minutes)...")
    client.create_delta_sync_index_and_wait(
        endpoint_name=ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        primary_key=PRIMARY_KEY,
        pipeline_type="TRIGGERED",
        embedding_source_column=EMBEDDING_COLUMN,
        embedding_model_endpoint_name=EMBEDDING_MODEL,
    )
    print(f"Done. Index '{INDEX_NAME}' is ready for querying.")


if __name__ == "__main__":
    create_vector_index()
