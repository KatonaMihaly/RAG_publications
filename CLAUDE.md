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

## Architecture

This is a **RAG pipeline** built entirely on Databricks, following the Medallion Architecture. All compute, storage, and AI runs inside the Databricks workspace — there is no local inference or local database.

### Data flow

```
publications/*.pdf  (local + Unity Catalog Volume)
        │
        ▼  ai_parse_document()
workspace.default.parsed_papers       ← Bronze table (Variant/JSON tree per PDF)
        │
        ▼  PySpark explode + filter (chunks < 50 chars removed)
workspace.default.chunked_papers      ← Silver table (one row per chunk, CDF enabled)
        │
        ▼  Mosaic AI Vector Search (auto-embedding via databricks-gte-large-en)
workspace.default.publication_index   ← Vector index on academic_search_endpoint
        │
        ▼  Hybrid search (semantic + keyword) → Llama 3 70B via Model Serving
research_assistant(question)          ← Returns answer + retrieved chunks as DataFrame
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

### Main notebook

`research_assistant.ipynb` contains the full pipeline in sequence:
1. Bronze parsing (`ai_parse_document`)
2. Silver chunking (PySpark `lateralJoin` + `variant_explode`)
3. Vector index creation (done via Databricks UI — see notebook markdown cells)
4. Synthetic truth table generation (`databricks-agents`)
5. RAG function (`research_assistant`) with hybrid search + CoT prompting
6. MLflow evaluation (`mlflow.genai.evaluate` with Correctness + RelevanceToQuery scorers)

### Local utility script

`check_volume.py` — lists files in the Unity Catalog volume using `databricks-sdk`. Run with `.venv/bin/python3 check_volume.py`.

## Databricks PAT scopes

When creating a token (Settings → Developer → Access Tokens → Generate, Beta scoped token feature), select: `all-apis`, `unity-catalog`, `sql`, `file.files`.
