## Context

The self-hosted RAG pipeline consists of ~6 workers (Gatekeeper, Producer, Consumer x2, OCR, WhisperX) coordinated via Redis queues and a DuckDB-backed state machine. The system also includes a FastAPI backend, an Astro frontend, dual vector database support (Qdrant/Chroma), dual LLM support (Supervisor for normalization, RAG for chat), and HAProxy-based load balancing. Despite comprehensive documentation in AGENTS.md and docs/, there is no formal specification that defines component boundaries, data contracts, and system invariants in a machine-readable, change-tracked format.

## Goals / Non-Goals

**Goals:**
- Create OpenSpec capability specs for all 7 major system components
- Document normative requirements (SHALL/MUST) for each capability
- Define system boundaries, data contracts, and invariants
- Establish the spec-driven development baseline for all future changes

**Non-Goals:**
- No code changes, refactoring, or behavior modifications
- No changes to Docker configuration or deployment
- No changes to documentation files (AGENTS.md, docs/)
- No creation of delta specs (no existing specs to modify)

## Decisions

1. **Capability boundaries follow worker/component lines**: Each major worker or service group becomes its own capability spec. This maps cleanly to the deployment topology (each worker runs as a separate container) and aligns with the existing worker/service directory structure.

2. **Specs reference concrete implementation details**: Since this is an existing system being documented (not a greenfield design), specs will reference actual file paths, environment variable names, and queue names as they exist in the codebase.

3. **Requirements extracted from source code, not aspirational**: Specs describe the system as it IS, not as it SHOULD BE. This provides an accurate baseline. Future changes will use MODIFIED/REMOVED/ADDED requirements to track divergence.

4. **No cross-capability dependency modeling in this change**: Cross-capability contracts (e.g., what format Consumer expects from Producer) are documented within each spec but formal dependency links between specs are deferred to future changes.

## Risks / Trade-offs

- **Staleness risk**: Specs may drift from implementation if not kept in sync. Mitigation: The spec-driven workflow requires spec updates as part of every change.
- **Granularity risk**: 7 specs may be too coarse or too fine. Mitigation: Splitting/merging specs is straightforward in OpenSpec if the boundaries prove wrong.
- **Omission risk**: Some implementation details may be missed. Mitigation: Specs are living documents; omissions found during implementation of future changes become ADDED requirements.
