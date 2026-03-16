# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python dependencies are managed via a local `.venv`. Always run scripts with:
```bash
.venv/bin/python3 <script.py>
```

Install packages into the venv:
```bash
.venv/bin/pip install <package>
```

Databricks credentials are stored in `~/.databrickscfg` (outside the repo). The `WorkspaceClient()` from `databricks-sdk` reads this automatically — no extra auth code needed.

All configuration lives in `.env` (local runs) and `job.yml` (Databricks job). Never hardcode resource names — always read from these files.

---

## Architecture

This is a **RAG pipeline** built entirely on Databricks, following the Medallion Architecture. All compute, storage, and AI runs inside the Databricks workspace — there is no local inference or local database.

### Data flow

```
publications/*.pdf  (local + Unity Catalog Volume)
        │
        ▼  ai_parse_document()
workspace.default.parsed_papers       ← Bronze table (Variant/JSON tree per PDF)
        │
        ▼  PySpark explode + filter (chunks < MIN_CHUNK_SIZE chars removed)
workspace.default.chunked_papers      ← Silver table (one row per chunk, CDF enabled)
        │
        ▼  Mosaic AI Vector Search (auto-embedding via databricks-gte-large-en)
workspace.default.publication_index   ← Vector index on academic_search_endpoint
        │
        ▼  Hybrid search (semantic + keyword) → Llama 3 70B via Model Serving
research_assistant(question)          ← Returns answer + retrieved chunks
        │
        ▼  MLflow pyfunc model (models-from-code)
workspace.default.research_assistant  ← Registered model in Unity Catalog
        │
        ▼  Databricks Model Serving
research_assistant_endpoint           ← REST API consumed by the Streamlit app
```

### Key Databricks resources

| Resource | Name |
|---|---|
| Catalog / Schema | `workspace.default` |
| Volume (source PDFs) | `/Volumes/workspace/default/publications` |
| Vector Search endpoint | `academic_search_endpoint` |
| Vector index | `workspace.default.publication_index` |
| Embedding model | `databricks-gte-large-en` |
| LLM endpoint | `databricks-meta-llama-3-3-70b-instruct` |
| Evaluation truth table | `workspace.default.truth_table_converted` |
| Registered model | `workspace.default.research_assistant` |
| Serving endpoint | `research_assistant_endpoint` |
| MLflow experiment | `/Shared/research_assistant` |

---

## Repository layout

```
.
├── .env                          # Local config (mirrors job.yml parameters)
├── job.yml                       # Databricks job definition (schedule, tasks, parameters)
├── deploy_job.py                 # Deploys/updates the nightly Databricks job
├── app.py                        # Streamlit chat UI — calls research_assistant_endpoint
├── check_volume.py               # Lists files in the Unity Catalog volume
├── pipeline/
│   ├── parse_publications.py     # Bronze: ai_parse_document → parsed_papers (streaming)
│   ├── chunk_publications.py     # Silver: explode chunks → chunked_papers (streaming, CDF)
│   ├── create_vector_index.py    # Sync chunked_papers → publication_index
│   ├── generate_truth_table.py   # Synthetic QA dataset via databricks-agents
│   ├── evaluate_rag.py           # MLflow evaluation (Correctness + RelevanceToQuery)
│   ├── research_assistant.py     # Standalone RAG function (local use / notebook)
│   ├── research_assistant_model.py  # MLflow PythonModel — packaged for serving
│   ├── deploy_serving_endpoint.py   # Registers model in UC + deploys serving endpoint
│   └── prompts/
│       ├── research_assistant_system_prompt.md
│       ├── few_shot_examples.json
│       ├── agent_description.md
│       └── question_guidelines.md
```

---

## Recreating the project from scratch

### Prerequisites

1. Databricks workspace with Unity Catalog enabled.
2. `~/.databrickscfg` with `host` and `token` set.
3. A SQL warehouse running in the workspace.
4. Cluster with DBR 14.x+ (for `ai_parse_document` and Variant type support).
5. Python 3.12 venv: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (or install manually: `databricks-sdk databricks-vectorsearch mlflow databricks-agents python-dotenv streamlit requests pyyaml`).

### Step 1 — Upload PDFs

Upload your PDF files to the Unity Catalog volume:
```
/Volumes/workspace/default/publications/
```
Use the Databricks UI, `check_volume.py`, or `databricks fs cp`.

### Step 2 — Create the Vector Search endpoint

In the Databricks UI: **Catalog → Vector Search → Create endpoint** named `academic_search_endpoint`. This is a one-time manual step.

### Step 3 — Deploy the nightly job

```bash
.venv/bin/python3 deploy_job.py
```

This uploads all `pipeline/*.py` scripts as notebooks to `/Users/<you>/rag_publications/` and creates the `rag_nightly_update` job (scheduled daily at 01:00 UTC). Edit `job.yml` to change parameters or schedule; re-run `deploy_job.py` to apply changes.

**Task DAG:**
```
parse_publications
        │
chunk_publications
        ├──────────────────────────┐
create_vector_index     generate_truth_table
        └──────────────────────────┘
                    │
              evaluate_rag
```

### Step 4 — Run the job manually (first time)

In the Databricks UI, trigger the job, or:
```bash
.venv/bin/python3 -c "from databricks.sdk import WorkspaceClient; WorkspaceClient().jobs.run_now(job_id=<id>)"
```
The job ID is printed by `deploy_job.py`.

### Step 5 — Register the model and deploy the serving endpoint

```bash
.venv/bin/python3 -c "
from pipeline.deploy_serving_endpoint import log_and_register_model, deploy_endpoint
version = log_and_register_model()
deploy_endpoint(version)
"
```

- `log_and_register_model()` logs the pyfunc model to MLflow and registers it as `workspace.default.research_assistant` in Unity Catalog.
- `deploy_endpoint(version)` creates `research_assistant_endpoint` on Databricks Model Serving.

**Important:** Databricks allows only **one serving endpoint** per workspace on the free tier. Delete the existing endpoint before creating a new one:
```bash
.venv/bin/python3 -c "
from databricks.sdk import WorkspaceClient; import os; from dotenv import load_dotenv; load_dotenv()
WorkspaceClient().serving_endpoints.delete(name=os.environ['SERVING_ENDPOINT_NAME'])
"
```

### Step 6 — Run the Streamlit app

```bash
.venv/bin/streamlit run app.py
```

---

## Key implementation details

### Parameter passing

All pipeline scripts use a `_param(key)` helper that reads from `dbutils.widgets.get(key)` on Databricks and falls back to `os.environ[key]` locally. This means the same script runs both in the Databricks job and locally with `.env`.

### Chunked papers schema

The Silver table `chunked_papers` has these columns:
- `chunk_id` — STRING, primary key (UUID)
- `chunk_text_string` — STRING, the text content (**not** `chunk_text`)
- `chunk_type` — STRING
- `path` — STRING, source PDF path

The Vector Search index must be built on `chunk_text_string` (embedding column) with `chunk_id` as the primary key. CDF must be enabled on `chunked_papers`.

### MLflow pyfunc serving — input format

When the model is deployed to a Databricks serving endpoint, MLflow always passes `model_input` to `predict` as a **pandas DataFrame**, not a dict. The `predict` method in `research_assistant_model.py` handles both cases:
- `pd.DataFrame` → `model_input.to_dict("records")[0]` to get the first row as a dict
- `dict` → used directly (local `mlflow.pyfunc.load_model` calls)

The numpy array truthiness pitfall: when `messages` arrives from a DataFrame row it may be a numpy array. Never use `if messages` or `messages or []` — use `_msgs is not None` and `len(messages) > 0` instead.

### Serving endpoint invocation format

`app.py` calls the endpoint with:
```python
json={"inputs": {"messages": messages}}
```
MLflow serving interprets `{"inputs": {"col": value}}` as named tensor inputs, creating a one-row DataFrame with a `messages` column.

### Serving endpoint cold start

`scale_to_zero_enabled=True` means the endpoint shuts down after ~10 minutes idle and takes 3–5 minutes to restart. The `requests.post` in `app.py` uses `timeout=300` to handle this. Set `scale_to_zero_enabled=False` in `deploy_serving_endpoint.py` for always-on behaviour.

---

## Databricks PAT scopes

When creating a token (Settings → Developer → Access Tokens → Generate, Beta scoped token feature), select: `all-apis`, `unity-catalog`, `sql`, `file.files`.

---

## Local utility script

`check_volume.py` — lists files in the Unity Catalog volume using `databricks-sdk`. Run with `.venv/bin/python3 check_volume.py`.
