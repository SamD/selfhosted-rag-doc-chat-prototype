# RAG Chat

## Purpose

The RAG chat capability handles user queries by retrieving relevant chunks from the vector database, assembling them into a citation-enforced context, sending them to the LLM for response generation, and post-processing citations to map back to source filenames. It uses asymmetric prefix matching (query:/passage:) for retrieval and enforces strict citation rules on LLM output.

## Requirements

### Requirement: Vector database retrieval

The system SHALL retrieve relevant chunks from the configured vector database using asymmetric prefix matching. The user query SHALL be prefixed with "query: " before retrieval. Stored chunks SHALL be prefixed with "passage: " at ingest time. The system SHALL return the top-K most similar chunks.

#### Scenario: Query prefix matching
- **WHEN** a user submits a question
- **THEN** the query SHALL be prefixed with "query: " before being sent to the retriever

#### Scenario: Top-K retrieval
- **WHEN** a query is processed
- **THEN** the retriever SHALL return the top-K most similar chunks from the vector database

### Requirement: Context deduplication and assembly

The system SHALL deduplicate retrieved chunks by content (only unique content sent to LLM). Each chunk SHALL be annotated with its deterministic DOC_XXXX anchor and assigned a sequential [sourceN] citation tag. The assembled context SHALL include SOURCE_ID, CITATION_TAG, and CONTENT for each chunk.

#### Scenario: Content deduplication
- **WHEN** two retrieved chunks have identical page_content
- **THEN** only the first occurrence SHALL be included in the context sent to the LLM

#### Scenario: Citation tag assignment
- **WHEN** a unique chunk is included in the context
- **THEN** it SHALL be assigned a CITATION_TAG in the format [source1], [source2], etc., in sequential order

#### Scenario: DOC_XXXX anchor extraction
- **WHEN** a chunk's content contains a [DOC_XXXX] anchor
- **THEN** the anchor SHALL be extracted as the SOURCE_ID and the anchor prefix removed from the content sent to the LLM

### Requirement: LLM response with strict citations

The system SHALL enforce strict citation rules on LLM responses. Each factual sentence in the response MUST end with its CITATION_TAG (e.g., [source1]). If multiple sources apply, both tags MUST be used ([source1][source2]). If no relevant information is found, the LLM SHALL respond with "Data not found." No introductions, conclusions, or editorial commentary SHALL be included.

#### Scenario: Fact with single source
- **WHEN** the LLM uses information from one source
- **THEN** the sentence SHALL end with the corresponding [sourceN] tag

#### Scenario: Fact with multiple sources
- **WHEN** the LLM uses information from multiple sources for one sentence
- **THEN** multiple tags SHALL be used in sequence, e.g., [source1][source2]

#### Scenario: No relevant information
- **WHEN** the retrieved context does not contain an answer to the user's question
- **THEN** the LLM SHALL respond with exactly "Data not found."

### Requirement: Citation post-processing and filename mapping

The system SHALL map [sourceN] citation tags back to the original filenames and metadata from the retrieved documents. This enables the frontend to display clickable source links. The mapping SHALL be performed by replace_citation_labels() after the LLM response.

#### Scenario: Citation label replacement
- **WHEN** the LLM response contains [sourceN] tags
- **THEN** the system SHALL replace each tag with the corresponding filename and metadata from the source document

#### Scenario: No citation tags found
- **WHEN** the LLM response contains no valid citation tags
- **THEN** the system SHALL log a grounding failure warning and return the response as-is

### Requirement: Server-side chat history management

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
