## MODIFIED Requirements

### Requirement: Chat history accumulation

**REPLACED BY**: Server-side chat history management.

The system SHALL manage chat history server-side in Redis, keyed by session ID. Each query and response pair SHALL be appended to a Redis list for the session. The system SHALL enforce a configurable maximum number of conversation turns (MAX_SESSION_TURNS, default: 20). When the limit is exceeded, the oldest messages SHALL be dropped (oldest-first truncation). Sessions SHALL expire after a configurable inactivity period (SESSION_TTL_HOURS, default: 24).

#### Scenario: History append to Redis
- **WHEN** a query is answered for a session
- **THEN** the query and response pair SHALL be RPUSHed to the Redis list at key session:{id}

#### Scenario: Oldest-first truncation
- **WHEN** the session history exceeds MAX_SESSION_TURNS after appending
- **THEN** the system SHALL LTRIM the Redis list to keep only the most recent MAX_SESSION_TURNS entries

#### Scenario: Session TTL expiry
- **WHEN** a session has no activity for SESSION_TTL_HOURS
- **THEN** the Redis key SHALL be automatically evicted

#### Scenario: History retrieval on query
- **WHEN** a query arrives with a session_id
- **THEN** the system SHALL retrieve the full history from Redis and pass it to the LLM for context

#### Scenario: New session creation
- **WHEN** a query arrives with an unknown or empty session_id
- **THEN** the system SHALL create a new session with empty history and return the new session_id in the response

#### Scenario: Redis unavailable
- **WHEN** Redis is unreachable during history retrieval
- **THEN** the system SHALL fall back to empty history and log the error, rather than failing the query
