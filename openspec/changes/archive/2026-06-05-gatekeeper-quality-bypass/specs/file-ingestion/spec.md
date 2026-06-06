## MODIFIED Requirements

### Requirement: Gatekeeper normalization

The system SHALL claim jobs in NEW status and transition them to PREPROCESSING. The Gatekeeper SHALL extract raw text from the file using the Chain of Responsibility handler system, batch the content, normalize it via the Supervisor LLM into clean Markdown with ### [INTERNAL_PAGE_X] anchors, and write the output to the ingestion directory. On success, status SHALL transition to PREPROCESSING_COMPLETE. On failure, status SHALL transition to INGEST_FAILED.

Before calling the Supervisor LLM, the system SHALL run `is_bad_ocr()` on the extracted text. If the quality check passes, the text SHALL be written directly as Markdown without LLM normalization. The output format (metadata anchor + content) SHALL be identical regardless of which path is taken.

#### Scenario: Gatekeeper claims a NEW job
- **WHEN** a job exists in NEW status
- **THEN** the Gatekeeper atomically claims it via UPDATE ... RETURNING * and transitions to PREPROCESSING

#### Scenario: Successful normalization
- **WHEN** the Supervisor LLM successfully normalizes all content batches into Markdown
- **THEN** the system writes a .md file to the ingestion directory and transitions to PREPROCESSING_COMPLETE

#### Scenario: Normalization failure
- **WHEN** the Supervisor LLM fails or any content handler throws an exception
- **THEN** the system transitions to INGEST_FAILED with an error log

#### Scenario: Quality check bypass
- **WHEN** a batch of extracted text passes `is_bad_ocr()` quality check
- **THEN** the text SHALL be written directly as Markdown with the standard metadata anchor, and the Supervisor LLM SHALL NOT be called

#### Scenario: Quality check failure routes to LLM
- **WHEN** a batch of extracted text fails `is_bad_ocr()` quality check
- **THEN** the text SHALL be sent to the Supervisor LLM for normalization as before
