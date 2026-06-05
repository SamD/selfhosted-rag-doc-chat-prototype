## Why

Chat history is currently managed client-side — the full history array is sent with every HTTP request and grows unboundedly. This prevents stateless load balancing of the API, risks exceeding LLM context windows on long conversations, and provides no mechanism for session identification, persistence, or per-user isolation. Redis is already a core dependency (used for worker queues); storing chat history server-side in Redis solves all of these problems with minimal new infrastructure.

## What Changes

- **BREAKING**: Remove `chat_history` from `QueryRequest` — history is now managed server-side per session
- **NEW**: Add `session_id` to `QueryRequest` and `QueryResponse` — client sends session ID, backend manages the associated history
- **NEW**: `ChatSessionService` — stores/retrieves/truncates chat history in Redis using `session:{id}` key pattern
- **NEW**: Server-enforced max conversation turns with automatic oldest-first truncation
- **MODIFIED**: `QueryResponse` — replaces `chat_history` array with `session_id` (history is no longer echoed back)
- **MODIFIED**: Frontend — sends `session_id` + `query` instead of `query` + full history

## Capabilities

### New Capabilities
- `chat-session-management`: Redis-backed session storage for RAG chat conversations. Covers: session creation, history storage/retrieval with Redis lists, configurable max turns with oldest-first truncation, and session TTL expiry.

### Modified Capabilities
- `rag-chat`: The "Chat history accumulation" requirement changes from client-side unbounded history to server-side managed history with session IDs and configurable turn limits.

## Impact

- `doc-ingest-chat/models/query.py`: `QueryRequest` gains `session_id: str`, loses `chat_history: List[Dict]`. `QueryResponse` gains `session_id: str`, loses `chat_history: List[Dict]`.
- `doc-ingest-chat/api/endpoints.py`: `query_handler()` calls `ChatSessionService` to manage history before/after `answer_query()`
- `doc-ingest-chat/chat/chroma_chat.py`: `respond()` receives history from `ChatSessionService` instead of from the request
- `doc-ingest-chat/services/`: New `chat_session_service.py` module
- `astro-frontend/src/pages/index.astro`: Changed from sending `{ query, chat_history }` to `{ query, session_id }`. Client generates/manages `session_id` as a UUID in local storage.
- `openspec/specs/rag-chat/spec.md`: MODIFIED requirement — "Chat history accumulation" replaced with "Server-side chat history management"
- `openspec/specs/chat-session-management/spec.md`: NEW capability spec
