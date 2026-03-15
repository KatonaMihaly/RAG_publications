"""
Package the research_assistant RAG chain as an MLflow pyfunc model,
register it in Unity Catalog, and deploy it to a Databricks Model Serving endpoint.
"""

import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    AiGatewayUsageTrackingConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from dotenv import load_dotenv
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

load_dotenv()

REGISTERED_MODEL_NAME    = os.environ["REGISTERED_MODEL_NAME"]
SERVING_ENDPOINT_NAME    = os.environ["SERVING_ENDPOINT_NAME"]
VECTOR_SEARCH_INDEX      = os.environ["VECTOR_SEARCH_INDEX"]
VECTOR_SEARCH_ENDPOINT   = os.environ["VECTOR_SEARCH_ENDPOINT"]
LLM_ENDPOINT             = os.environ["LLM_ENDPOINT"]

MODEL_FILE = str(Path(__file__).parent / "research_assistant_model.py")


def log_and_register_model() -> str:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("/Shared/research_assistant")

    resources = [
        DatabricksVectorSearchIndex(index_name=VECTOR_SEARCH_INDEX),
        DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    ]

    input_example  = {"messages": [{"role": "user", "content": "What is the Taguchi method?"}]}
    output_example = {
        "answer": "The Taguchi method is a robust design technique...",
        "chunks": [{"text": "Sample chunk text.", "path": "/Volumes/workspace/default/publications/paper.pdf"}],
    }
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="research_assistant",
            python_model=MODEL_FILE,
            artifacts={"prompts_dir": str(Path(__file__).parent / "prompts")},
            registered_model_name=REGISTERED_MODEL_NAME,
            resources=resources,
            signature=signature,
            input_example=input_example,
        )

    version = model_info.registered_model_version
    print(f"  Registered: {REGISTERED_MODEL_NAME} v{version}")
    return version


def deploy_endpoint(model_version: str) -> None:
    w        = WorkspaceClient()
    existing = {e.name for e in w.serving_endpoints.list()}

    served_entity = ServedEntityInput(
        entity_name=REGISTERED_MODEL_NAME,
        entity_version=model_version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars={
            "LLM_ENDPOINT":            os.environ["LLM_ENDPOINT"],
            "NUM_RESULTS":             os.environ["NUM_RESULTS"],
            "VECTOR_SEARCH_ENDPOINT":  os.environ["VECTOR_SEARCH_ENDPOINT"],
            "VECTOR_SEARCH_INDEX":     os.environ["VECTOR_SEARCH_INDEX"],
        },
    )

    if SERVING_ENDPOINT_NAME in existing:
        print(f"  Endpoint '{SERVING_ENDPOINT_NAME}' exists — updating...")
        try:
            w.serving_endpoints.update_config_and_wait(
                name=SERVING_ENDPOINT_NAME,
                served_entities=[served_entity],
            )
        except ResourceDoesNotExist:
            print(f"  Endpoint disappeared (stale listing) — creating instead...")
            w.serving_endpoints.create_and_wait(
                name=SERVING_ENDPOINT_NAME,
                config=EndpointCoreConfigInput(name=SERVING_ENDPOINT_NAME, served_entities=[served_entity]),
            )
    else:
        print(f"  Creating endpoint '{SERVING_ENDPOINT_NAME}'...")
        w.serving_endpoints.create_and_wait(
            name=SERVING_ENDPOINT_NAME,
            config=EndpointCoreConfigInput(name=SERVING_ENDPOINT_NAME, served_entities=[served_entity]),
        )

    print(f"  Enabling AI Gateway usage tracking...")
    w.serving_endpoints.put_ai_gateway(
        name=SERVING_ENDPOINT_NAME,
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
    )

    print(f"  Endpoint '{SERVING_ENDPOINT_NAME}' is ready.")


if __name__ == "__main__":
    print("Step 1/2: Logging and registering model...")
    version = log_and_register_model()

    print("Step 2/2: Deploying serving endpoint...")
    deploy_endpoint(version)
