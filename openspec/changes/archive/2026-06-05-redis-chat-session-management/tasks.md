## 1. Backend: Chat Session Service

- [x] 1.1 Create `doc-ingest-chat/services/chat_session_service.py` with `ChatSessionService` class using `get_redis_client()` for Redis connection
- [x] 1.2 Implement `get_or_create_session(session_id: str) -> str` — validates existing session or creates new one, returns session_id
- [x] 1.3 Implement `append_history(session_id: str, user_msg: dict, assistant_msg: dict)` — RPUSHes messages, enforces MAX_SESSION_TURNS with LTRIM
- [x] 1.4 Implement `get_history(session_id: str) -> list[dict]` — retrieves and deserializes all messages for a session
- [x] 1.5 Implement graceful fallback: return empty history if Redis is unreachable

## 2. Backend: API Changes

- [x] 2.1 Update `models/query.py`: add `session_id: str = ""` to QueryRequest, replace `chat_history` with `session_id` in QueryResponse
- [x] 2.2 Update `api/endpoints.py`: inject `ChatSessionService`, call `get_history()` before `answer_query()`, call `append_history()` after
- [x] 2.3 Update `rag_service.py`: remove `chat_history` from response dict (history managed by ChatSessionService, not echoed back)
- [x] 2.4 Add `MAX_SESSION_TURNS` and `SESSION_TTL_HOURS` settings to shared config/env vars

## 3. Frontend: Session Management

- [x] 3.1 Update `index.astro`: generate UUID v4 session_id on first load, persist in localStorage
- [x] 3.2 Change `sendQuery()` to send `{ query, session_id }` instead of `{ query, chat_history }`
- [x] 3.3 Update local message rendering to manage display history independently from API history
- [x] 3.4 Update response handling to use `session_id` from API response (sync if server generated new one)

## 4. Code Quality and Testing

- [x] 4.1 Run ruff linting — all checks passed
- [x] 4.2 Run existing test suite — 121/121 passed, no regressions
- [x] 4.3 Write unit tests for ChatSessionService — 12 tests covering success, failure, and edge cases
- [x] 4.4 Run full test suite with new tests — 133/133 passed
