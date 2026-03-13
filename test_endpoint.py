import os
import requests
from databricks.sdk.config import Config
from dotenv import load_dotenv

load_dotenv()

cfg = Config()
ENDPOINT_URL = f"{cfg.host}/serving-endpoints/{os.environ['SERVING_ENDPOINT_NAME']}/invocations"
HEADERS = {"Authorization": f"Bearer {cfg.token}", "Content-Type": "application/json"}


def ask(question: str) -> str:
    response = requests.post(
        ENDPOINT_URL,
        headers=HEADERS,
        json={"messages": [{"role": "user", "content": question}]},
    )
    response.raise_for_status()
    return response.json()["answer"]


if __name__ == "__main__":
    question = "What is the Taguchi method?"
    result = requests.post(
        ENDPOINT_URL,
        headers=HEADERS,
        json={"messages": [{"role": "user", "content": question}]},
    ).json()

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for i, chunk in enumerate(result.get("chunks", []), 1):
        print(f"  [{i}] {chunk['path']}")
        print(f"      {chunk['text'][:120].strip()}...")
