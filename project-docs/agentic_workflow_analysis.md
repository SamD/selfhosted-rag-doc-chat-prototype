# Agentic Workflow Analysis & Implementation Plan

This document evaluates the transition of the RAG ingestion pipeline from a deterministic state machine to an agentic workflow. It identifies specific areas where LLM reasoning provides a clear return on investment (ROI) without introducing unnecessary latency or complexity.

## 1. ROI Analysis

| Feature | Type | Benefit | Cost (Latency/VRAM) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Semantic Contextualizer** | Agent | **High**: Adds global document context to every chunk, fixing the "orphan chunk" problem in RAG. | Moderate: One LLM call per document. | **Recommended** |
| **Smart Extraction Triage** | Agent | **Medium**: Detects subtle text layer corruption that heuristics miss. | Low: One LLM call on a 500-token sample. | **Optional** |
| **Self-Correcting OCR** | Agent | **Low**: Retries OCR with different parameters. | Very High: Multiple LLM/OCR calls per page. | **Not Recommended** |
| **Workflow Routing** | Agent | **None**: Deciding which node to visit next (Scan -> Extract). | High: LLM overhead for basic logic. | **Keep Deterministic** |

---

## 2. Proposed Feature: The Contextualizer Agent

### The Problem
Traditional chunking loses the "Big Picture." 
*   **File**: `annual_report_2023.pdf`
*   **Chunk**: "...profits increased by 5% due to cost-cutting in the logistics division."
*   **Retrieval Issue**: If a user asks "How did the company perform in 2023?", this chunk might be missed because it doesn't contain the year or the company name.

### The Agentic Solution
We introduce a **Supervisor Agent** node at the start of the `ProducerGraph`. 
1.  The agent reads the first 1000 tokens of the document.
2.  It generates a 2-sentence "Document Persona" (e.g., "This is the 2023 Annual Report for ACME Corp, focusing on fiscal growth and logistics.").
3.  This Persona is added to the `IngestState`.
4.  The extraction nodes prepend this Persona to every chunk before it is sent to Redis.

---

## 3. Implementation Plan

### Step 1: LLM Integration
We will leverage the existing LLM configuration but wrap it in a LangGraph-compatible `ChatModel` (via `ChatLlamaCpp` or `ChatOllama`). We will use the **Singleton Pattern** in `llm_setup.py` to ensure the Producer doesn't fight the API for VRAM.

### Step 2: Define the "Contextualizer" Tool/Node
A new node in `producer_graph.py`:
```python
def contextualizer_node(state: IngestState) -> IngestState:
    sample_text = state["extracted_sample"]
    # Agent generates: "This document is [Topic] by [Author] regarding [Year]..."
    context_summary = llm.invoke(f"Summarize the context of this document: {sample_text}")
    return {**state, "doc_context": context_summary}
```

### Step 3: Modify `IngestState`
Add a `doc_context` field to the `TypedDict`.

### Step 4: Update Extraction Logic
Update `producer_utils.py` to accept the `doc_context` and prepend it during the `split_doc` phase:
```python
# Before
chunk = "profits increased by 5%..."
# After
chunk = "[Context: ACME Corp 2023 Report] profits increased by 5%..."
```

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
