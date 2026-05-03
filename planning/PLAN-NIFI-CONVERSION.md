I'm planning on making a NiFi version of this where the workers and graph remain Python but becomes Nifi Processors, the latest version of Nifi supports Python as a first class citizen.
This also means:
- Redis is dropped, backpressure retries etc. will be handled by Nifi Queues natively
- All of the models involved will be remote url accessed
    * both the supervisor (markdown writer) and llm server (RAG) are already done so no change there
    * e5 large will move to remote
    * whisperX will be remote, I will create a LXC container with FasterWhisper
    * docling easyocr will be remote, the same I will create a lxc container running docling

The current Python StateGraph might integrate well with Nifi.

**Before considering any plan you must:**
1. Review the latest Nifi Python documentation: https://nifi.apache.org/nifi-docs/python-developer-guide.html so that you have the latest version information and not rely upon something out of date.
2. Review the Expression Guide: https://nifi.apache.org/nifi-docs/expression-language-guide.html so you understand how Nifi attributes and conditions are handled when it comes to flowfiles and routing decisions
3. Review all the available "out of the box" components shipped with Nifi: https://nifi.apache.org/components/ so to leverage as much of what is provided for you.

You should use the `@researcher` agent whenever needed 


The following is to provide guidance consideration which might be helpful

---
---
### 1. System Philosophy

- **Orchestration:** Apache NiFi (replacing Kubernetes/Redis).
- 
- **Make use of what is provided:** Apache NiFi provides a massive amount of components out of the box for dealing with integrations across multiple cloud providers and local hosted / self deployed popular opensource tech stacks. When making a decision , it should be considered what can Nifi provide to make the task simpler. What can we use that Nifi gives us to make an data ingestion RAG flow decision easier to implement.

- **State Management:** Native NiFi FlowFile attributes and internal backpressure.

- **Formatting:** Strict Markdown; no em dashes; objective and factual.

- **Network:** Local, air-gapped, high-performance inter-node communication.


Define exactly which physical "islands" of compute handle which mathematical operations.

### 2. Proxmox LXC Worker Nodes Proposal

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

### 3. Data Flow Architecture (The Pipeline) Proposal

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

### 4. API & Communication Contracts Proposal

Since you are replacing Redis, each LXC must expose a consistent interface.

- **Model Endpoints:** All `llama-server` instances must follow the OpenAI-compatible `/v1/embeddings` or `/v1/chat/completions` spec.

- **Worker Endpoints:** Use a lightweight Python wrapper (FastAPI) for Docling and Whisper to accept binary file uploads via `POST`.

- **State:** The "State of Truth" for any individual job exists only as a **FlowFile Attribute** in the NiFi queue.

------------------------------------------------------------------------

### 5. Implementation Constraints (The "Rules") Propsal

- **No Redis:** Use NiFi internal queues and Backpressure (set at 10,000 objects/10GB) for flow control.

- **Air-Gapped:** All Nix configurations must allow for offline builds or use a local binary cache.

- **Precision:** No "hallucination" in the Refinement stage. Phi-4 must be restricted to a temperature of 0.0 for markdown re-typing.

- **Formatting:** No em dashes (—) in output; strict Markdown format for all saved records.

------------------------------------------------------------------------
