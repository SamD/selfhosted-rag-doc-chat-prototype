# Final Implementation Plan: RAG-NiFi Transition

This plan details the step-by-step conversion of the Python/Redis pipeline to a native Apache NiFi 2.x architecture. It prioritizes **out-of-the-box components** to eliminate redundant code and relies on NiFi's internal repositories to replace Redis.

---

## Phase 1: Ingestion & Global Metadata (Built-in)
**Step 1: Monitoring**
- **Processor**: `ListFile`
- **Logic**: Monitors the `/staging` directory. Maintains its own state of which files have already been processed (replacing our custom directory scanning logic).
- **Confirmation**: No code needed.

**Step 2: Consumption**
- **Processor**: `FetchFile`
- **Logic**: Moves the file from disk into NiFi's internal Content Repository.
- **Confirmation**: No code needed.

**Step 3: Identity & Tracing**
- **Processor**: `UpdateAttribute`
- **Logic**: Sets `${trace_id}` using `${uuid}` and `${document_id}` using a MurmurHash3 calculation via Expression Language.
- **Confirmation**: Replaces our custom `trace_utils` for ID generation.

---

## Phase 2: Content-Aware Extraction (Built-in + Remote LXC)
**Step 4: Branching**
- **Processor**: `RouteOnAttribute`
- **Logic**: Routes based on `${filename:substringAfterLast('.')}` (pdf, mp3, mp4, txt).
- **Confirmation**: Built-in EL handles the "Gatekeeper" routing logic.

**Step 5: Distributed Extraction**
- **Processor**: `InvokeHTTP`
- **Logic**: Performs a `POST` to the respective LXC (Worker-02/03). 
- **Configuration**: Uses `multipart/form-data` to stream the FlowFile content directly to the remote API.
- **Confirmation**: Replaces custom `PDFHandler`, `MP3Handler`, etc. NiFi handles the network I/O and binary streaming.

---

## Phase 3: Semantic Refinement (Built-in + LLM)
**Step 6: Phi-4 Normalization**
- **Processor**: `InvokeHTTP`
- **Logic**: Sends the extracted text to the Refiner-01 LXC (`/v1/chat/completions`).
- **Confirmation**: We use NiFi's native retry logic (`RetryFlowFile`) here to handle LLM timeouts or concurrency limits.

---

## Phase 4: Zero-Loss Chunking (Custom Python Processor)
**Step 7: Semantic Splitting**
- **Processor**: `MarkdownSplitter` (Custom Python)
- **Why?**: This is the only part NiFi cannot do natively. It requires the `transformers` tokenizer and our specific **Zero-Loss Sub-Splitting** iterative logic.
- **Implementation**: 
    - A single class inheriting from `FlowFileTransform`.
    - Input: Cleaned Markdown.
    - Output: Multiple FlowFiles (one per chunk).
    - Metadata: Sets `fragment.identifier` (the trace_id), `fragment.index`, and `fragment.count`.
- **Confirmation**: This is where we port the `TextProcessor` logic verified today.

---

## Phase 5: Vectorization & Persistent Storage (Native Vector Store)
**Step 8: Embeddings**
- **Processor**: `InvokeHTTP`
- **Logic**: POSTs chunk text to Worker-01 (`e5-large-v2`).
- **Output**: Extracts the vector array into a FlowFile attribute.

**Step 9: Qdrant Persistence**
- **Processor**: `PutQdrant` (Native in NiFi 2.0)
- **Logic**: Map the `${vector}` attribute and other metadata attributes (`page`, `source_file`) directly into the Qdrant record.
- **Confirmation**: Replaces all our custom Qdrant client and `DatabaseService` persistence code.

---

## Phase 6: Flow Control & Reliability (The NiFi Advantage)
**Step 10: Backpressure**
- **Mechanism**: Connection Settings.
- **Logic**: We set the queue before `InvokeHTTP` (LLM) or `PutQdrant` to a specific limit (e.g., 5,000 objects). 
- **Result**: NiFi **automatically pauses** the upstream `ListFile` if the workers or vector DB cannot keep up. No custom Redis logic required.

**Step 11: Error Recovery**
- **Processor**: `RetryFlowFile`
- **Logic**: Any failed `InvokeHTTP` or `PutQdrant` call is routed through this to handle exponential backoff.
- **Final Failure**: Route to a `LogAttribute` and `PutFile` (Dead Letter Folder).

---

## Confirmation Summary for Tomorrow
1. **Zero Brokers**: Redis is gone. Persistence is handled by NiFi's FlowFile/Provenance repositories.
2. **Minimal Python**: We only write Python for the `MarkdownSplitter`. Everything else (HTTP, Files, Routing, Retries, Vector Storage) is native.
3. **Distributed LXC**: We only need to provide the LXCs with minimal FastAPI wrappers for the models.
