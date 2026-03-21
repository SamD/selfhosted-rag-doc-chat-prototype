# AI Behavior & Usage

## 🧮 Embedding Behavior and Bias

The embedding model (`e5-large-v2`) is used during ingestion to convert text chunks into semantic vectors. Unlike LLMs, it does not generate language — it only maps inputs to fixed-size embeddings deterministically.

This means there is **no stochasticity** and **no generation bias** during ingestion. However, any "bias" in this step comes from the embedding model's training data — it influences what the model considers similar or important. If a concept was underrepresented during training, its embeddings may cluster poorly or produce weak retrieval matches.

Understanding this helps clarify that while the LLM governs response generation, the embedding model governs what content even reaches that stage.

---

## 📤 How the LLM Is Used

The LLM (`llama-cpp`) is only used during the **Generation** phase of RAG. It does not participate in ingestion or retrieval.

- During **ingestion**, documents are embedded using `e5-large-v2`, a sentence transformer model — no LLM is invoked.
- During **retrieval**, Chroma/Qdrant uses vector similarity to return relevant chunks — again, no LLM is needed.
- Only during **generation** is the LLM called, where it receives the retrieved chunks as prompt context and produces a final response.

This separation improves performance and debuggability, ensuring that embedding and retrieval steps are deterministic and inspectable.
