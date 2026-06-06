## Why

The supervisor LLM normalizes every page unconditionally during gatekeeper processing, even when pdfplumber or OCR already produced clean, valid text. With Qwen running at 90 TPS on a single Arc card, each PDF page takes ~1-3 seconds of LLM inference. For documents where most pages extract cleanly (modern PDFs, well-OCR'd scans), this is wasted throughput. The existing `is_bad_ocr()` quality heuristic can determine whether extracted text needs LLM normalization or can be written directly as Markdown.

## What Changes

- **MODIFIED**: `process_chunk()` in `gatekeeper_logic.py` — before calling the supervisor LLM, run `is_bad_ocr()` on the extracted text. If quality passes, wrap the content in the standard Markdown/metadata format and return without calling the LLM. Only pages that fail the quality check proceed to normalization.
- **MODIFIED**: Test `test_gatekeeper_enforces_context_limit` — must mock `is_bad_ocr` to force the LLM path, since the test content is clean text that would otherwise bypass the LLM.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `file-ingestion`: The "Gatekeeper normalization" requirement changes — the supervisor LLM is no longer called unconditionally. Pages whose extracted text passes `is_bad_ocr()` quality check bypass the LLM and are written directly as Markdown with the same output format.

## Impact

- `doc-ingest-chat/workers/gatekeeper_logic.py`: `process_chunk()` gains a quality check branch before the LLM call. The anchor header and file write logic is duplicated for the bypass path — the output format (metadata header + content) must be identical regardless of the path taken.
- `doc-ingest-chat/tests/test_token_safety.py`: The context limit enforcement test must mock `is_bad_ocr` to return True, forcing the LLM path for test content that would otherwise be clean enough to bypass.
- No new dependencies, env vars, or configuration changes.
