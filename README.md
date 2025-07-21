> #### ⚠️ Disclaimer
> This project was developed as a learning exercise to gain hands-on experience with retrieval-augmented 
> generation (RAG) systems, document chunking, vector stores, and local LLM inference. 
> It intentionally avoids orchestration frameworks and emphasizes transparency over abstraction.
>
> **Note:** Performance will vary significantly depending on your hardware configuration (especially CPU vs GPU). 
> A CPU-only mode is available, but may be slow for larger models or documents.
>
> Since `llama-cpp` is used directly, all runtime configuration (e.g. model parameters, context size, batch size) 
> is controlled via environment variables defined in `ingest-svc.env`.

#### 🖥️ System Used for Testing
This prototype was tested locally on the following hardware:

- **CPU**: Intel Core i9-14900KF (24-core)
- **GPU**: NVIDIA RTX 4070 Dual 12GB
- **RAM**: 32 GB DDR5
- **Storage**: 1TB NVMe Gen4 SSD
- **OS**: Ubuntu 22.04 LTS (64-bit)
- **Python**: 3.11
- **Docker**: 24.0.5
- **Model backend**: llama-cpp (GGUF) + HuggingFace Transformers (e5-large-v2)

> ⚠️ Performance may vary significantly on CPU-only systems or with different model sizes.

------------------------------------------------------------------------

### PDF & HTML Document Chatbot (RAG System with Local LLM)

>A modular, transparent pipeline for retrieving and answering questions
>from document collections using local embeddings and LLMs.
------------------------------------------------------------------------

This project implements a retrieval-augmented generation (RAG) system
focused on hands-on learning and system-level control. It includes three
main stages:

- **Ingestion**: Documents are parsed, split into token-aware chunks,
  and embedded using `e5-large-v2`. Chunks are stored in Chroma for
  semantic search.

- **Retrieval**: At query time, top-k semantically similar chunks are
  fetched using cosine similarity.

- **Generation**: Retrieved context is passed to a local `llama-cpp` LLM
  (`Meta-Llama-3.1-8B-Instruct`) for contextual response generation.

This RAG pipeline is distributed and modular, designed to be less
abstract than frameworks like LangChain. It provides transparency and
direct access to each component.

The ingestion architecture processes documents (PDFs, HTML, scanned
images) via:

- **Producer**: Extracts and tokenizes text from documents.

- **Consumer**: Buffers and stores chunks into Chroma.

- **OCR Worker**: Uses Tesseract OCR to process scanned or image-based
  PDFs. Required for non-text PDFs.

Redis queues connect all services with coordinated backpressure,
retries, and transactional guarantees at the file level.

------------------------------------------------------------------------

### 🧠 Features

- Token-aware chunking using `transformers.AutoTokenizer`

- Fallback OCR (EasyOCR or Tesseract) for scanned PDFs

- Buffered Redis-based ingestion with file-level atomicity

- Distributed chunk batching, deduplication, and ingestion

- Vector store: https://github.com/chroma-core/chroma


🧭 Also see [arch.md](./project-docs/arch.md) for a system sequence diagram

------------------------------------------------------------------------

### 📦 Required Environment Variables

Set the following in your environment or a `.env` file:

Alternatively you can export them directly in the [run-compose.sh](./doc-ingest-chat/run-compose.sh) , you can
see an example in the file

    LLAMA_MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf
    E5_MODEL_PATH=/path/to/e5-large-v2
    INGEST_FOLDER=/absolute/path/to/your/docs

All other environment variables and defaults are defined in [ingest-svc.env](./doc-ingest-chat/ingest-svc.env)


#### 💾 Embedding and Model Setup

This system uses:

- **Embedding Model**: https://huggingface.co/intfloat/e5-large-v2

- **LLM**: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF

Models must be downloaded manually and referenced via absolute paths in
the environment file. These must be set before launching the system, as
both are critical for embedding and LLM-based response generation.


------------------------------------------------------------------------

### 🐳 Run and Test with Docker Compose

If your `INGEST_FOLDER` is empty of `PDF` or `HTML` files, you can copy files into that directory once running
, the directory is periodically scanned for new files

Use the provided `run-compose.sh` script to start the stack:

```shell
./doc-ingest-chat/run-compose.sh
```

If you are using the `$PROJECTDIR/Docs` directory for the `INGEST_FOLDER` you can copy your pdf and/or html file(s) there, 
for earlier testing I used:
* [The Outline Of History PT1](https://archive.org/download/outlineofhistory01welluoft/outlineofhistory01welluoft.pdf)
* [The Outline Of History PT2](https://archive.org/download/outlineofhistory02welluoft/outlineofhistory02welluoft.pdf)


> 🗂️ **NOTE** These PDF files contain several scanned pages which cannot be parsed by `pdfplumber`. In these cases 
> the `ocr` fallback is triggered which results in the chunk(s) sent to the `OCR_WORKER` via redis.
> This results in a much slower ingestion if there are many pages but the purpose is to demonstrate the fallback
> handling feature

Once running:
- The ingestion API will be available at `http://localhost:8000`

- \*The local Astro frontend will be served at
  \**`http://localhost:4321`*

The UI is a basic chatbox , queries for the session maintain conversation history. From here
you can ask relevant queries based on the documents you have ingested.


#### *CPU Mode Only*
Optionally a CPU only mode can be used if running on system with non-nvidia or lower end GPU. 
Note though this will likely run very slow and likely need to modify the llama and/or retriever 
properties in the `ingest-svc.env`
```sh
./doc-ingest-chat/run-compose-cpu.sh
```

### 📁 Output Artifacts

- **Parquet file of all embedded chunks** (`chunks.parquet`) and
  **DuckDB file** (`chunks.duckdb`) — DuckDB is the primary store and is
  updated first; the Parquet file is regenerated from DuckDB to support
  append-only archival and external analytics tools. DuckDB also allows
  efficient in-place querying of recent and historical chunks.

- `failed_files.txt` for skipped files

- `producer_failed_chunks.json` and `consumer_failed_chunks.json` for
  diagnostics
 
------------------------------------------------------------------------


### 🧮 Embedding Behavior and Bias

The embedding model (`e5-large-v2`) is used during ingestion to convert
text chunks into semantic vectors. Unlike LLMs, it does not generate
language — it only maps inputs to fixed-size embeddings
deterministically.

This means there is **no stochasticity** and **no generation bias**
during ingestion. However, any "bias" in this step comes from the
embedding model's training data — it influences what the model considers
similar or important. If a concept was underrepresented during training,
its embeddings may cluster poorly or produce weak retrieval matches.

Understanding this helps clarify that while the LLM governs response
generation, the embedding model governs what content even reaches that
stage.



------------------------------------------------------------------------


### 📤 How the LLM Is Used

The LLM (`llama-cpp`) is only used during the **Generation** phase of
RAG. It does not participate in ingestion or retrieval.

- During **ingestion**, documents are embedded using `e5-large-v2`, a
  sentence transformer model — no LLM is invoked.

- During **retrieval**, Chroma uses vector similarity to return relevant
  chunks — again, no LLM is needed.

- Only during **generation** is the LLM called, where it receives the
  retrieved chunks as prompt context and produces a final response.

This separation improves performance and debuggability, ensuring that
embedding and retrieval steps are deterministic and inspectable.


------------------------------------------------------------------------


### ❓ Comparison with LangChain

This system was developed as a learning-oriented ingestion pipeline,
with an emphasis on transparency and a hands-on understanding of each
component.
This setup was built without orchestration frameworks like LangChain, 
in order to provide full transparency into each stage of the retrieval 
and generation pipeline. The goal was to understand the underlying 
mechanics and retain control over data flow, chunking, and retrieval logic.
For example, parameters like `LLAMA_TOP_K`, `LLAMA_TEMPERATURE`, or `LLAMA_MAX_TOKENS` 
can be adjusted directly via environment variables without navigating framework abstractions. 
All llama-cpp runtime behavior is configurable via environment, enabling
transparent experimentation and reproducibility.

|                    |                                          |                                           |
|--------------------|------------------------------------------|-------------------------------------------|
| Tradeoff           | Custom Pipeline                          | LangChain                                 |
| Control            | Full                                     | Partial                                   |
| Abstraction        | Low (manual wiring)                      | High (simplified but opinionated)         |
| Flexibility        | High                                     | Moderate                                  |
| Setup complexity   | Higher initial effort                    | Easier to prototype                       |
| Performance tuning | Precise (token-aware chunking, batching) | Harder to optimize at low levels          |
| Ideal for          | Can provide ingestion at scale           | Rapid prototyping and low-code RAG setups |

LangChain is not used anywhere in this setup. All components are built
directly using Redis, HuggingFace, Chroma, and DuckDB.


------------------------------------------------------------------------


### 🛠️ Debugging and Inspection

#### Redis Queue Inspection

To inspect if ingestion is progressing:

- Use the Redis CLI or GUI to check the state of producer and consumer
  queues:

```bash

redis-cli -p 6379
redis-cli -p 6380

> LRANGE chunk_ingest_queue:0 0 -1
> LRANGE chunk_ingest_queue:1 0 -1
````

If the queues remain full or never drain, check consumer logs. Each
chunk is pushed with metadata, followed by a sentinel `file_end`
message.

#### DuckDB Inspection

To explore or debug ingested data:

```sql
    -- Connect to DuckDB shell
    duckdb /path/to/chunks.duckdb

    -- Sample queries:

    -- View chunk counts by engine type
    SELECT engine, COUNT(*) FROM chunks GROUP BY engine;

    -- Check if specific content (e.g. "Bretton") exists
    SELECT * FROM chunks WHERE text ILIKE '%bretton%';

    -- View per-file chunk totals
    SELECT source_file, COUNT(*) FROM chunks GROUP BY source_file ORDER BY COUNT(*) DESC;
```

These queries are helpful to verify ingestion completeness or to debug
why certain content isn't retrieved during RAG responses.

Additional examples:

#### Advanced Queries for Troubleshooting

```sql
    -- Show all chunk rows from files with 'Bretton' OR 'Woods' in them
    SELECT * FROM chunks WHERE text ILIKE '%bretton%' OR text ILIKE '%woods%';

    -- Find chunk counts for files that likely mention monetary systems
    SELECT source_file, COUNT(*) FROM chunks 
    WHERE text ILIKE '%currency%' OR text ILIKE '%exchange rate%' OR text ILIKE '%gold standard%'
    GROUP BY source_file ORDER BY COUNT(*) DESC;

    -- Check for empty or too-short text chunks
    SELECT * FROM chunks WHERE length(text) < 10;

    -- Check chunk length distribution to detect overly small or excessively large chunks
    SELECT length(text) AS token_count, COUNT(*) 
    FROM chunks 
    GROUP BY token_count 
    ORDER BY token_count DESC;

    -- Find chunks with unusual or null metadata (e.g., missing engine label)
    SELECT * 
    FROM chunks 
    WHERE engine IS NULL OR engine = '';

    -- Review a sample of chunks from a specific source
    SELECT * 
    FROM chunks 
    WHERE source_file LIKE '%bretton%' 
    LIMIT 10;

    -- 🔎 Inspect how often certain keywords appear in chunks
    SELECT 
      COUNT(*) FILTER (WHERE text ILIKE '%inflation%') AS inflation_hits,
      COUNT(*) FILTER (WHERE text ILIKE '%deflation%') AS deflation_hits,
      COUNT(*) FILTER (WHERE text ILIKE '%interest rate%') AS interest_rate_hits
    FROM chunks;

    -- Count how many chunks exist per unique document file
    SELECT source_file, COUNT(*) 
    FROM chunks 
    GROUP BY source_file 
    ORDER BY COUNT(*) DESC;

    -- Detect duplicate chunk texts (possible over-splitting or OCR duplication)
    SELECT text, COUNT(*) 
    FROM chunks 
    GROUP BY text 
    HAVING COUNT(*) > 1 
    ORDER BY COUNT(*) DESC 
    LIMIT 10;
```
