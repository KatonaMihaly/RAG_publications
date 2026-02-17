
# Retrieval-Augmented Generation - a Research Assistant

## 📖 Overview
This project is a **Retrieval-Augmented Generation (RAG)** application built on the **Databricks Data Intelligence Platform**.

I developed this tool to assist in writing my research. I came to the conclusion while writing my dissertation that over the past 5 years, I have accumulated and red numerous research papers, and recalling specific details across all of them became challenging. This AI assistant allows me to:
* **Summarize** answers to complex technical questions based *strictly* on my own open-access publications.
* **Locate** the exact source material by providing file paths and context chunks.
* **Validate** memories without manually re-reading hundreds of pages.

This project also serves as a practical implementation of concepts learned through the **Databricks Academy**.

---

## Technical Architecture
The pipeline follows the **Medallion Architecture** pattern to transform unstructured PDF data into a queryable knowledge base.

### 1. Ingestion & Parsing (Bronze Layer)
* **Source:** Open-access PDF publications stored in **Unity Catalog Volumes**.
* **Tool:** `ai_parse_document` (Databricks Intelligence).
* **Process:** Utilizes the "Nano Banana" model (via the parse function) to convert unstructured PDFs into a structured "Variant" format, extracting text, layout, and metadata.

### 2. Transformation & Chunking (Silver Layer)
* **Logic:** Uses **PySpark** to explode the parsed document structure into smaller, semantically meaningful chunks.
* **Cleaning:** Filters out NULL values and noise (chunks < 50 characters) to ensure high-quality embeddings.
* **Output:** A clean Delta Table ready for vectorization.

### 3. Vector Search & Indexing
* **Embedding Model:** `databricks-gte-large-en` (1024 dimensions).
* **Database:** **Mosaic AI Vector Search**.
* **Sync Mode:** Triggered sync with **Change Data Feed (CDF)** enabled for efficient, incremental updates.

### 4. Retrieval & Generation (The "Brain")
* **LLM:** **Llama 3 70B Instruct** (hosted via Model Serving).
* **Retrieval Strategy:** **Hybrid Search** (Keyword + Semantic Vectors). This is crucial for capturing specific technical acronyms (e.g., "PMSM") that pure vector search might miss.
* **Prompt Engineering:** Implemented **Chain-of-Thought (CoT)** and **Few-Shot Prompting** to strictly ground the model in the provided context and reduce hallucinations.

---

## Evaluation & Testing
To ensure the system provides accurate data, I implemented a robust evaluation pipeline using **MLflow**:

1.  **Synthetic Truth Table:** Used `databricks-agents` to generate a synthetic dataset of Questions, Answers, and Ground Truths derived directly from the source papers.
2.  **LLM-as-a-Judge:** Automated evaluation using MLflow's GenAI metrics:
    * **Correctness:** Is the answer factually accurate based on the documents?
    * **Relevance:** Did the bot answer the user's specific question?
---

## Usage
The core functionality is wrapped in a Python function `research_assistant(question)`.

**Example:**
```
df_result = research_assistant("In which application is asymmetric rotor topology advantageous?")
```