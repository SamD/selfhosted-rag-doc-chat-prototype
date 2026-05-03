## 1. Hardware-to-Service Mapping (The Inventory)

Define exactly which physical "islands" of compute handle which mathematical operations.

| **Node ID**    | **Physical Host** | **Hardware**        | **Primary Service**         | **Backend/Driver** |
|----------------|-------------------|---------------------|-----------------------------|--------------------|
| **Worker-01**  | Bee1 Mini PC      | Alder Lake-N (iGPU) | `llama-server` (e5-large)   | SYCL / OpenVINO    |
| **Worker-02**  | Bee2 Mini PC      | Alder Lake-N (iGPU) | `faster-whisper`            | OpenVINO / IPEX    |
| **Worker-03**  | Bee3 Mini PC      | Alder Lake-N (iGPU) | `docling` / EasyOCR         | IPEX / XPU         |
| **Refiner-01** | BOSGAME           | AMD 780M (iGPU)     | `llama-server` (Phi-4-mini) | Vulkan / ROCm      |
| **Brain-01**   | BOSGAME           | RTX 3060 (12GB)     | `llama-server` (Qwen/GLM)   | CUDA               |
| **Core-01**    | Mini PC           | Alder Lake-N (CPU)  | NiFi / Qdrant               | Native (JVM/Rust)  |

Export to Sheets

------------------------------------------------------------------------

## 2. Data Flow Architecture (The Pipeline)

Document the "travel" of a single piece of data from ingestion to storage.

1.  **Ingestion:** NiFi monitors local directories or API hooks.

2.  **Extraction (Branch A/B):**

    - **Audio:** NiFi POSTs to **Worker-02**; receives JSON transcript.

    - **Docs:** NiFi POSTs to **Worker-03**; receives raw Markdown.

3.  **Refinement:** NiFi sends raw text to **Refiner-01** (Phi-4).

    - *Instruction:* "Re-type into clean Markdown, fix table formatting, remove OCR artifacts."

4.  **Vectorization:** NiFi sends clean Markdown to **Worker-01** (e5-large).

    - *Result:* 1024-dimension vector.

5.  **Storage:** NiFi commits the **Vector + Refined Markdown + Original Metadata** to Qdrant on **Core-01**.

------------------------------------------------------------------------

## 3. API & Communication Contracts

Since you are replacing Redis, each LXC must expose a consistent interface.

- **Model Endpoints:** All `llama-server` instances must follow the OpenAI-compatible `/v1/embeddings` or `/v1/chat/completions` spec.

- **Worker Endpoints:** Use a lightweight Python wrapper (FastAPI) for Docling and Whisper to accept binary file uploads via `POST`.

- **State:** The "State of Truth" for any individual job exists only as a **FlowFile Attribute** in the NiFi queue.

------------------------------------------------------------------------

## 4. Implementation Constraints (The "Rules")

- **No Redis:** Use NiFi internal queues and Backpressure (set at 10,000 objects/10GB) for flow control.

- **Air-Gapped:** All Nix configurations must allow for offline builds or use a local binary cache.

- **Precision:** No "hallucination" in the Refinement stage. Phi-4 must be restricted to a temperature of 0.0 for markdown re-typing.

- **Formatting:** No em dashes (—) in output; strict Markdown format for all saved records.

------------------------------------------------------------------------

## 5. Phase 1 Setup Checklist (The "Next Steps")

1.  **Host Prep:** Enable IOMMU and GPU passthrough on all Proxmox nodes.

2.  **Base LXC Template:** Create a NixOS LXC template with `intel-compute-runtime` and `nix-command` enabled.

3.  **NiFi Blueprint:** Install NiFi on Node 4 and establish the basic Site-to-Site communication between nodes.

4.  **Service Validation:** Confirm `llama-server` can see the Alder Lake iGPU using the SYCL backend.

------------------------------------------------------------------------

### Suggested File Structure for this Plan

I recommend saving this as a `README.md` in your main infrastructure repository, structured like this:

- `/docs/architecture.md` (This plan)

- `/nix/hosts/` (Per-node configuration)

- `/nifi/templates/` (XML/JSON exports of the flows)

- `/scripts/wrappers/` (FastAPI code for Whisper/Docling)