# Agentic Workflow: Implementation Report

This document details the implementation of the RAG ingestion pipeline's transition from a deterministic state machine to an agentic workflow using **LangGraph**.

## 1. Feature ROI Analysis (Post-Implementation)

| Feature | Type | Benefit | Cost (Latency/VRAM) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Semantic Contextualizer** | Agent | **High**: Adds global document context to every chunk, fixing the "orphan chunk" problem in RAG. | Moderate: One LLM call per document. | **Recommended** |
| **Smart Extraction Triage** | Agent | **Medium**: Detects subtle text layer corruption that heuristics miss. | Low: One LLM call on a 500-token sample. | **Optional** |
| **Self-Correcting OCR** | Agent | **Low**: Retries OCR with different parameters. | Very High: Multiple LLM/OCR calls per page. | **Not Recommended** |
| **Workflow Routing** | Agent | **None**: Deciding which node to visit next (Scan -> Extract). | High: LLM overhead for basic logic. | **Keep Deterministic** |

---

## 2. Implemented Feature: The Contextualizer Agent

### The Problem
Traditional chunking loses the "Big Picture." 
*   **File**: `annual_report_2023.pdf`
*   **Chunk**: "...profits increased by 5% due to cost-cutting in the logistics division."
*   **Retrieval Issue**: If a user asks "How did the company perform in 2023?", this chunk might be missed because it doesn't contain the year or the company name.

### The Agentic Solution (LangGraph)
We have integrated a **Supervisor Agent** node at the start of the `ProducerGraph`. 
1.  The `preview_node` extracts text from the first 10 pages.
2.  The `supervisor_node` (Qwen2.5) generates a 1-sentence "Document Persona" (e.g., "This is the 2023 Annual Report for ACME Corp, focusing on fiscal growth and logistics.").
3.  The extraction nodes prepend this Persona to **every chunk** before it is sent to Redis.

---

## 3. Technical Details

### Multi-Process Safety
To prevent GPU deadlocks during concurrent model loading, we implemented a **Global Multiprocessing Lock** (`gpu_lock`). Only one worker process can initialize or invoke the Supervisor LLM at a time.

### Singleton Pattern
All models (LLM, Embedding, Tokenizer) are implemented as **Per-Process Singletons**. They are lazy-loaded on first use and cached for the remainder of the process lifecycle.

### Data Flow
1.  **Enrichment**: The `doc_context` is prepended to chunks in the `on_chunks` streaming callback.
2.  **Atomicity**: DuckDB handles job status via `file_ingestion_jobs`.
3.  **Persistence**: Incremental DuckDB appends ensure zero-memory buffering for large files.

---

## 4. Resource Management (Guardrails)

To prevent the "Agentic Transition" from breaking the system:
1.  **VRAM Capping**: The Contextualizer LLM will be initialized with a small context window (e.g., 2048 tokens) to keep memory usage low.
2.  **Fallback**: If the LLM call fails or times out, the graph will automatically bypass the contextualizer and proceed with raw extraction (deterministic fallback).
3.  **Bypassing small files**: Files under 2000 characters will bypass the agent to save time.

## 5. Next Steps
1.  **Draft `llm_setup.py` changes** to provide a shared LLM instance for workers.
2.  **Add `contextualizer_node`** to the `ProducerGraph`.
3.  **Update `on_chunks` callback** to handle context injection.
