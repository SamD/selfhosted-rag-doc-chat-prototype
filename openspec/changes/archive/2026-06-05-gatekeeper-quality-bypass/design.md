## Context

The gatekeeper's `process_chunk()` function unconditionally calls the supervisor LLM for every batch of extracted text. The LLM prompt asks it to convert raw text to clean Markdown — a task that is unnecessary when the extracted text is already clean.

The existing `is_bad_ocr()` function (from `utils.text_utils`) checks three conditions: gibberish detection (non-alpha character ratio), visible corruption (mojibake characters), and low token count. If a page passes all three checks, the text is already valid and doesn't need LLM normalization.

## Goals / Non-Goals

**Goals:**
- Skip the supervisor LLM call for pages where `is_bad_ocr()` passes
- Bypass path produces identical output format (metadata anchor + content)
- Log a clear indication when the LLM is bypassed vs called
- Test that the LLM path is still exercised when quality check fails

**Non-Goals:**
- No changes to the quality check logic itself — `is_bad_ocr()` remains as-is
- No new env vars or configuration
- No changes to the LLM normalization prompt or behavior
- No changes to the 3-tier extraction chain (pdfplumber → OCR → LLM)

## Decisions

1. **Quality check at the start of `process_chunk()`**: The check runs before any LLM-related work (tokenization, context limit enforcement). If it passes, the content is written directly and the function returns early. This minimizes overhead for the bypass path.

2. **Use existing `is_bad_ocr()` directly**: It's already imported at the module level in `gatekeeper_logic.py`. No new import or abstraction needed.

3. **Identical output format**: Both paths produce the same `anchor_header + content` structure and write to the same file with the same metadata. The consumer of the output (Producer) cannot distinguish which path was taken.

4. **Mock `is_bad_ocr` in the context limit test**: The test sends clean text ("word " * 1000). Without mocking, this text passes `is_bad_ocr()` and the LLM is bypassed — the test expects LLM truncation. Mocking `is_bad_ocr` to return True forces the LLM path and preserves the test's intent.

## Risks / Trade-offs

- **False positives**: `is_bad_ocr()` may pass for text that has subtle quality issues the LLM would fix. The existing heuristic was designed for OCR quality detection, not Markdown fitness. If bypassed text has quality issues downstream, the quality check may need tuning. Mitigation: monitor the `⏭️ Batch X: quality check passed, skipping LLM` log lines to audit bypass decisions.
