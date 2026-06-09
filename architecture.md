# Self-Hosted RAG Pipeline · Architecture

Interactive diagram for the selfhosted-rag-doc-chat-prototype. Click flow tabs to walk through scenarios step by step. Toggle between **Local** (Docker Compose) and **Remote** (HAProxy load-balanced) modes to see how the same pipeline adapts.

**Open the diagram:** [architecture.html](architecture.html) · just double-click to open in a browser.

---

## Components

The diagram uses 6 nodes representing the main system actors:

| Node | Role | Tech | Port (Local) | Port (Remote) |
|---|---|---|---|---|
| **User (Browser)** | `user` · mint | Astro v6 · Tailwind v4 · daisyUI | `http://localhost:4321` | `http://frontend:4321` |
| **FastAPI Gateway** | `orch` · sky | Python 3.12 · FastAPI · uvicorn | `http://localhost:8000` | `http://api:8000` |
| **Embedding Model** | `embed` · amber | e5-large-v2 · 1024-dim | `http://localhost:11434` | `http://haproxy_embd:11438/v1` |
| **Vector Database** | `vector` · violet | Qdrant v1.12.5 · COSINE | `http://localhost:6333` | `http://remote-qdrant:6333` |
| **Main LLM** | `compute` · magenta | llama-cpp-python · Phi-3.5-mini Q6_K | `http://localhost:11435` | `http://haproxy_supervisor:11437/v1` |
| **Worker Pipeline** | `seed` · orange | 6 workers · Redis queues · DuckDB state | (internal) | (internal) |

**Mode toggle:**

- **Local** — All services run in Docker Compose. Direct connections to localhost ports. No external dependencies.
- **Remote** — All services connect through HAProxy load balancers. Multiple backend instances supported via `*_ENDPOINTS` env vars.

---

## Flows

### 1. RAG Query (POST /api/v1/query) — 8 steps

The main pipeline. Demonstrates retrieval-augmented generation with citation tracking.

| # | From | To | What happens |
|---|---|---|---|
| 1 | User | FastAPI | `POST /api/v1/query` with `{query, session_id, limit}` |
| 2 | FastAPI | Embeddings | `POST /v1/embeddings` to convert query to 1024-dim vector |
| 3 | Embeddings | FastAPI | Returns query vector `[0.023, -0.156, ...]` |
| 4 | FastAPI | Vector DB | `POST /collections/rag/points/search` with COSINE similarity, k=5 |
| 5 | Vector DB | FastAPI | Returns top-5 chunks with `[DOC_XXXX]` IDs and scores |
| 6 | FastAPI | LLM | `POST /v1/chat/completions` with query + retrieved chunks as context |
| 7 | LLM | FastAPI | Returns grounded answer with `[docN]` inline citations |
| 8 | FastAPI | User | Returns `{answer, sources[], session_id}` with citation tags mapped to filenames |

**Latency breakdown:**
- Steps 1-3: ~110ms (network + embedding)
- Steps 4-5: ~30ms (vector search)
- Steps 6-7: 2-5s (LLM generation, temperature=0.0)
- Step 8: ~50ms (citation mapping + response)
- **Total: ~3-5s per query**

---

### 2. Document Ingest (PDF/Text) — 7 steps

Multi-stage pipeline for processing documents through 6 workers. Files flow: `staging/` → Gatekeeper → Producer → Consumer → `success/`.

| # | From | To | What happens |
|---|---|---|---|
| 1 | User | FastAPI | `POST /api/v1/stage` with multipart file upload |
| 2 | FastAPI | Workers | Gatekeeper atomically claims file from DuckDB (`NEW → PREPROCESSING`) |
| 3 | Workers | LLM | Supervisor LLM (Qwen3.5-4B) normalizes pages to Markdown with `### [INTERNAL_PAGE_X]` anchors |
| 4 | Workers | FastAPI | Producer chunks Markdown hierarchically (H1→H2→INTERNAL_PAGE→H3), assigns `[DOC_XXXX]` IDs via MurmurHash3 |
| 5 | Workers | Embeddings | Consumer batches chunks (32 at a time), embeds via e5-large-v2 |
| 6 | Embeddings | Workers | Returns 1024-dim vectors, Consumer stages to DuckDB until `file_end` sentinel |
| 7 | Workers | Vector DB | Atomic upsert to Qdrant with metadata, file moved to `success/` |

**Key details:**
- Atomic claims via DuckDB `UPDATE ... RETURNING *` with 20-retry exponential backoff
- Zero-loss chunking: oversized chunks are sub-split, not truncated
- Backpressure: Lua script blocks if Redis queue > 50K items
- File-level locking: any consumer can process any file's chunks once EOF arrives

---

### 3. Direct LLM (No RAG) — 4 steps

Baseline comparison. Same model, same temperature, NO context retrieval. Demonstrates hallucination risk.

| # | From | To | What happens |
|---|---|---|---|
| 1 | User | FastAPI | `POST /api/v1/direct` with `{query, temperature}` |
| 2 | FastAPI | LLM | `POST /v1/chat/completions` with ONLY the user's question (no context) |
| 3 | LLM | FastAPI | Returns ungrounded answer with no citations |
| 4 | FastAPI | User | Returns `{answer, sources: [], warning: "No context retrieved"}` |

**Why this matters:** Shows the difference between grounded (cited) and ungrounded (hallucinated) responses. The same Phi-3.5 model produces completely different output quality depending on whether it has retrieved context.

---

### 4. Media Ingest (Audio/Video) — 6 steps

WhisperX transcription for `.mp3`, `.wav`, `.mp4` files. Uses request-reply pattern via Redis ephemeral keys.

| # | From | To | What happens |
|---|---|---|---|
| 1 | User | FastAPI | `POST /api/v1/stage` with media file + language hint |
| 2 | FastAPI | Workers | Gatekeeper pushes job to `whisper_processing_job` queue with `reply_key` |
| 3 | Workers | LLM | WhisperX calls `/inference` with `Content-Type: video/mp4` |
| 4 | LLM | Workers | Returns streaming segments with word-level timestamps |
| 5 | Workers | FastAPI | Gatekeeper blocks on `BLPOP whisper_reply:{uuid}` until `done` message |
| 6 | FastAPI | Vector DB | Transcription treated as document, flows through Producer → Consumer → Vector DB |

**MIME type handling:** Content handlers declare `MIME_TYPE` class var, pass it through Redis job metadata, so WhisperX sends correct `Content-Type` header to the Whisper server.

---

## Mode Differences

The diagram shows how the same pipeline adapts to different deployment shapes:

| Component | Local Mode | Remote Mode |
|---|---|---|
| **FastAPI** | `http://localhost:8000` | `http://api:8000` (Docker network) |
| **Embeddings** | Direct to e5-large-v2 on port 11434 | Via HAProxy `haproxy_embd:11438` (round-robin across backends) |
| **Vector DB** | Local Qdrant on port 6333 | Remote Qdrant cluster |
| **LLM** | Local llama-cpp-python on port 11435 | Via HAProxy `haproxy_supervisor:11437` (load-balanced) |
| **Auth** | No auth (local dev) | Bearer token required |

**HAProxy load balancing:** When `*_ENDPOINTS` env vars are set, `run-compose.sh` auto-creates HAProxy containers. `roundrobin` + `httpclose` for even distribution. Health checks on `/models` or `/health`.

---

## Workshop Scenarios

### Scenario 1: "Show me the RAG pipeline"

1. Click **RAG Query** flow tab (Flow 1)
2. Press **Space** to play, or click **Next** to step through
3. Watch how the query flows: User → API → Embed → Vector → API → LLM → User
4. Toggle to **Remote** mode to see how URLs change (localhost → Docker service names → HAProxy)

### Scenario 2: "Why do we need RAG?"

1. Play through **Direct LLM** (Flow 3) to see ungrounded output
2. Then play through **RAG Query** (Flow 1) to see grounded output with citations
3. Notice the difference: Flow 1 has `[docN]` tags, Flow 3 has empty `sources[]`

### Scenario 3: "How does document ingestion work?"

1. Play through **Document Ingest** (Flow 2)
2. Pay attention to the worker pipeline node — it shows the 6-worker architecture
3. Notice how chunks get `[DOC_XXXX]` IDs (MurmurHash3) for deterministic citation tracking

### Scenario 4: "What about audio/video files?"

1. Play through **Media Ingest** (Flow 4)
2. See the request-reply pattern via Redis ephemeral keys (`whisper_reply:{uuid}`)
3. Notice how transcription output reuses the same Producer → Consumer pipeline

---

## Interactive Features

- **Click any node** — jumps to the first step where that node appears
- **Drag any node** — reposition on canvas (positions persist in localStorage)
- **Press R** — reset layout to original positions
- **Press F** — fullscreen mode (great for workshop projectors)
- **Press T** — toggle dark/light theme
- **Press Space** — play/pause auto-advance
- **Press ←/→** — previous/next step
- **Press 1-4** — jump to flow N
- **Press O** — toggle mode (Local ↔ Remote)

---

## Tech Stack Reference

- **Backend:** FastAPI, uvicorn, Python 3.12
- **Frontend:** Astro v6, Tailwind v4, daisyUI, Node.js v22.12.0+
- **Message Broker:** Redis 7.2 (with Lua scripts for atomic operations)
- **Vector DB:** Qdrant v1.12.5 (dual support for Chroma via `VECTOR_DB_PROFILE`)
- **Relational DB:** DuckDB (lifecycle state machine, chunk staging)
- **LLM:** llama-cpp-python (Phi-3.5-mini Q6_K GGUF) or remote OpenAI-compatible endpoints
- **Embeddings:** e5-large-v2 (1024-dim, HuggingFace or remote API)
- **OCR:** Docling (EasyOCR) local or docling-serve remote
- **Transcription:** WhisperX (CTranslate2 format)
- **Load Balancer:** HAProxy (multi-backend support)
- **Containerization:** Docker Compose with CUDA/CPU profiles

---

## Further Reading

- **Main README:** [`README.md`](README.md)
- **Agent instructions:** [`AGENTS.md`](AGENTS.md)
- **API documentation:** [`docs/quickstart.md`](docs/quickstart.md)
- **Architecture deep dive:** [`docs/deep-dive.md`](docs/deep-dive.md)
- **Operations runbook:** [`infra/operations/day-1.md`](infra/operations/day-1.md), [`infra/operations/day-2.md`](infra/operations/day-2.md)
