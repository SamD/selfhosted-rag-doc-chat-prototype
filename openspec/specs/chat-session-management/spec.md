# Chat Session Management

## Purpose

The chat session management capability provides server-side session storage for RAG chat conversations using Redis. It handles session creation, history persistence with configurable turn limits, TTL-based expiry, and graceful degradation when Redis is unavailable. This enables stateless API load balancing and prevents unbounded chat history growth.

## Requirements

### Requirement: Session creation and identification

The system SHALL support session-based chat via a session_id field in the API request and response. If the client provides a session_id, the system SHALL use it. If the session_id is unknown, the system SHALL create a new empty session. If no session_id is provided, the system SHALL generate one and return it.

#### Scenario: Client provides a session_id
- **WHEN** a query includes a valid session_id
- **THEN** the system SHALL retrieve the associated chat history and use it for context

#### Scenario: Unknown session_id
- **WHEN** a query includes a session_id that does not exist in Redis
- **THEN** the system SHALL create a new session with that ID and empty history

#### Scenario: No session_id provided
- **WHEN** a query does not include a session_id
- **THEN** the system SHALL generate a UUID v4, create a new session, and return it in the response

### Requirement: Redis-backed history storage

The system SHALL store chat history in Redis using the key pattern `session:{id}`. Each entry SHALL be a JSON-serialized message with role and content fields. The service SHALL use `get_redis_client()` from the existing redis_service module for the Redis connection.

#### Scenario: History storage
- **WHEN** a query and response are processed
- **THEN** both messages SHALL be JSON-serialized and RPUSHed to the session:{id} list

#### Scenario: History retrieval
- **WHEN** a session_id is provided
- **THEN** all entries in session:{id} SHALL be retrieved, deserialized, and returned as a message list

### Requirement: Configurable session limits

The system SHALL support configurable session limits via environment variables. MAX_SESSION_TURNS SHALL control the maximum number of conversation turns (default: 20). SESSION_TTL_HOURS SHALL control the session expiry time (default: 24). These SHALL be loaded lazily via the shared settings system.

#### Scenario: Max turns enforcement
- **WHEN** the history length exceeds MAX_SESSION_TURNS
- **THEN** the oldest entries SHALL be trimmed using LTRIM

#### Scenario: Session expiry
- **WHEN** a session has no activity for SESSION_TTL_HOURS
- **THEN** Redis SHALL evict the session:{id} key

### Requirement: Frontend session management

The frontend SHALL generate a UUID v4 session_id on first load and persist it in localStorage. All subsequent requests SHALL include this session_id. The frontend SHALL NOT send chat_history in the request body.

#### Scenario: Session ID generation on first load
- **WHEN** the frontend loads and no session_id exists in localStorage
- **THEN** a UUID v4 SHALL be generated and stored in localStorage

#### Scenario: Session ID included in requests
- **WHEN** a query is sent
- **THEN** the request body SHALL include { query, session_id } and SHALL NOT include chat_history

#### Scenario: Frontend display history
- **WHEN** a response is received
- **THEN** the frontend SHALL append the query and response to its local display array for rendering, but SHALL NOT send it back to the API
