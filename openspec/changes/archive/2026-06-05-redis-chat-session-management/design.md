## Context

The RAG chat API currently receives the full chat_history array from the frontend with every request. This array grows unboundedly and is stored only in the browser's in-memory chatHistory variable. There is no session identification, no server-side persistence, and no history management.

Redis is already a core infrastructure dependency (used for worker job queues in the ingestion pipeline). The existing `redis_service.py` already provides a `get_redis_client()` singleton. Adding session storage uses no new infrastructure.

## Goals / Non-Goals

**Goals:**
- Store chat history server-side in Redis, keyed by session ID
- Frontend sends only `session_id` + `query` on each request
- Server enforces a configurable maximum number of conversation turns (default: 20), dropping oldest entries when exceeded
- Sessions expire after inactivity (configurable TTL, default: 24h)
- API becomes stateless between requests — any backend instance can serve any session
- Backward-compatible session_id generation: if omitted, server generates one

**Non-Goals:**
- No WebSocket or streaming — HTTP request/response remains unchanged
- No user authentication or authorization — session IDs identify conversations, not users
- No persistent long-term chat storage — Redis is the only store (ephemeral by design, TTL-based)
- No chat summarization or sliding window compression — just oldest-first truncation
- No changes to the RAG retrieval, citation, or LLM interaction logic

## Decisions

1. **Redis list for history storage**: Each session stores messages as a Redis list (`session:{id}`). New messages are `RPUSH`ed; when the list exceeds `MAX_SESSION_TURNS`, `LTRIM` drops oldest entries. This is O(1) for append and O(N) for trim. Alternative considered: separate keys per message (too many keys), JSON blob (no partial update).

2. **Session ID generated client-side as UUID v4**: The frontend generates a UUID on first load and persists it in `localStorage`. Alternative considered: server-generated (requires extra round-trip), hash-based (tying to browser fingerprint is fragile). Client-side UUID is stateless and simple.

3. **session_id in query param, not header**: Included in the request body alongside `query` for simplicity. Alternative: custom header (adds complexity on both sides for no benefit).

4. **Oldest-first truncation, not summarization**: When `MAX_SESSION_TURNS` is exceeded, the oldest messages are dropped. Alternative: LLM summarization (expensive, slow, adds latency to every turn). Simple truncation is predictable and fast.

5. **No echo of history in response**: Currently the API returns the updated `chat_history` array. With server-side storage, the frontend doesn't need it echoed back. The response includes `session_id` for confirmation but the frontend manages its display history independently.

## Risks / Trade-offs

- **Redis availability**: If Redis goes down, chat sessions are lost. Mitigation: the chat session service should fail open (fall back to empty history) rather than breaking the query endpoint. New sessions can still be created.
- **Memory usage**: Each session stores ~50KB-200KB of message history. At 1000 concurrent sessions, that's 50-200MB of Redis memory. Mitigation: TTL-based expiry and configurable max turns keep this bounded.
- **Migration**: Existing frontend sessions will break because they send `chat_history` but the new API expects `session_id`. Mitigation: the API transition should be explicit — this change marks `chat_history` as removed (BREAKING) and clients must be updated in lockstep.
