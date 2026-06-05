## ADDED Requirements

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

### Requirement: Chat history accumulation

The system SHALL accumulate chat history across turns. Each query and response pair SHALL be appended to the chat_history list. The full history SHALL be sent with each subsequent query. There SHALL be no truncation, summarization, or sliding window applied to the history.

#### Scenario: History append
- **WHEN** a query is answered
- **THEN** the query and response SHALL be appended to chat_history as {"role": "user", "content": <query>} and {"role": "assistant", "content": <response>}

#### Scenario: History sent with subsequent queries
- **WHEN** a subsequent query is made
- **THEN** the complete chat_history array SHALL be included in the messages sent to the LLM
