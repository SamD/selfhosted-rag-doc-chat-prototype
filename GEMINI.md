# Project Specification: Local RAG Ingestion Feature

## 1. Overview
* **Feature Status:** Active Implementation
* **Primary Objective:** Local document ingestion with environment-specific parsing strategies.
* **Target Environment:** Local / Air-Gapped (CPU Only)
* **Main Producer Worker:** Primary ingestion logic (Text-based).
* **OCR Worker (Fallback):** Docling with EasyOCR backend (Image/Scan-based).
* **Staging Environment:** phi-4-mini (Isolated staging/experimental parsing only).

---

## 2. Technical Architecture
### Logic Flow
1. **Producer Worker:** Main path for processing searchable document streams.
2. **OCR Worker (Fallback):** Triggered by the producer via `Docling[easyocr]` for non-searchable assets.
3. **Staging Environment:** Isolated `phi-4-mini` instance for testing high-reasoning extraction.
4. **Deduplication:** `MurmurHash` collision detection on normalized Markdown.
5. **Persistence:** Unique chunks pushed to local Redis.

---

## 3. Escalation Protocol (Agent Interaction)
When implementation is blocked or execution is required, utilize these specific agents:

* **@researcher:** Use for **External Knowledge**.
   * *Tasks:* Hunting specific library versions, finding `Docling` documentation, or solving `uv` dependency conflicts.
   * *Trigger:* "How do I configure EasyOCR for CPU-only in Docling 3.x?"

* **@analyzer:** Use for **Internal System Logic**.
   * *Tasks:* Debugging the `ocr_worker` hand-off, optimizing `MurmurHash` collision logic, or diagnosing Redis queue bottlenecks.
   * *Trigger:* "Why is the producer worker failing to hand off image-based PDFs to the ocr_worker?"

* **@coder:** Use for **Implementation & Refactoring**.
   * *Tasks:* Writing the actual Python code for the `ocr_worker`, updating `pyproject.toml` via `uv`, or refactoring the ingestion loop.
   * *Trigger:* "@coder, implement the Docling fallback logic in the ingestion service using the CPU-only configuration."

---

## 4. Implementation Plan
### Phase 1: Main Producer & OCR Worker
* [ ] Implement `Docling` + `EasyOCR` fallback logic within the `ocr_worker`.
* [ ] Suppress `EasyOCR` logging to prevent root logger hijacking.

### Phase 2: Staging Implementation
* [ ] Deploy `phi-4-mini` in the staging environment.
* [ ] benchmark SLM output against standard Markdown extraction.

### Phase 3: Core Pipeline
* [ ] Finalize `MurmurHash` integration for chunk-level deduplication.
* [ ] Verify stability in a fully air-gapped configuration.

### Phase 4: Modular Content Handlers
* [ ] Implement `BaseContentTypeHandler` (Chain of Responsibility).
* [ ] Implement `PDFContentTypeHandler` (Migrate existing logic).
* [ ] Implement `MP4ContentTypeHandler` (WhisperX for MP4).
* [ ] Refactor `gatekeeper_logic.py` to consume raw text streams.

---

## 5. Factual Constraints
* **Environment Isolation:** `phi-4-mini` is strictly for staging; do not leak SLM logic into the main producer worker.
* **No Cloud:** All tools must run locally on the M4 Neo (CPU Optimized).
* **Package Management:** Use `uv` for all dependency resolutions.

---

## 6. Revision History
* **2026-04-07:** Integrated Docling/EasyOCR fallback; set phi-4-mini to Staging only.
* **2026-04-07:** Defined Escalation Protocol for @researcher and @analyzer agents.