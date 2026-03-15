"""
ResearchAssistantModel — MLflow pyfunc model for the RAG research assistant.
Loaded via the models-from-code approach (file path passed to log_model).
"""

import json
import os
from pathlib import Path

import mlflow


class ResearchAssistantModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from mlflow.deployments import get_deploy_client
        from databricks.vector_search.client import VectorSearchClient

        self.llm_endpoint = os.environ["LLM_ENDPOINT"]
        self.num_results  = int(os.environ["NUM_RESULTS"])

        prompts_dir = Path(context.artifacts["prompts_dir"])
        self.system_prompt     = (prompts_dir / "research_assistant_system_prompt.md").read_text().strip()
        self.few_shot_examples = json.loads((prompts_dir / "few_shot_examples.json").read_text())

        vsc        = VectorSearchClient(disable_notice=True)
        self.index = vsc.get_index(
            endpoint_name=os.environ["VECTOR_SEARCH_ENDPOINT"],
            index_name=os.environ["VECTOR_SEARCH_INDEX"],
        )
        self.client = get_deploy_client("databricks")

    def predict(self, context, model_input, params=None):
        """Accepts {"messages": [...]} or {"question": "..."}."""
        if isinstance(model_input, dict):
            messages = model_input.get("messages", [])
            question = (
                next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
                if messages
                else model_input.get("question", "")
            )
        else:
            question = str(model_input)

        chunks  = self._retrieve(question)
        answer  = self._generate(question, chunks)
        return {
            "answer": answer,
            "chunks": [{"text": row[0], "path": row[1]} for row in chunks],
        }

    def _retrieve(self, question: str) -> list:
        results = self.index.similarity_search(
            query_text=question,
            columns=["chunk_text_string", "path"],
            num_results=self.num_results,
            query_type="hybrid",
        )
        return results["result"]["data_array"]

    def _generate(self, question: str, chunks: list) -> str:
        context_text = "\n\n".join(row[0] for row in chunks)
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.few_shot_examples,
            {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {question}"},
        ]
        response = self.client.predict(
            endpoint=self.llm_endpoint,
            inputs={"messages": messages},
        )
        return response["choices"][0]["message"]["content"]


mlflow.models.set_model(ResearchAssistantModel())
