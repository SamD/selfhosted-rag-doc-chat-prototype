## 1. Core Implementation

- [x] 1.1 Quality check added in `process_chunk()` — `is_bad_ocr()` runs before LLM call, bypasses with `⏭️` log line on pass

## 2. Test Updates

- [x] 2.1 `test_gatekeeper_enforces_context_limit` updated with `@patch('workers.gatekeeper_logic.is_bad_ocr')` returning True

## 3. Verification

- [x] 3.1 ruff check — clean
- [x] 3.2 pytest — 138/138 passed
- [x] 3.3 Bypass log line `⏭️ Batch ... quality check passed, skipping LLM` present in gatekeeper_logic.py
