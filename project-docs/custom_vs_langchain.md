# Why Custom Ingestion Pipeline (vs LangChain Orchestration)

**What Uses LangChain**: Minimal wrappers for ChromaDB integration (`langchain_chroma.Chroma`) and HuggingFace embeddings (`langchain_huggingface.HuggingFaceEmbeddings`).

**What Doesn't**: The entire ingestion pipeline - producer/consumer/OCR workers, Redis queue coordination, chunking logic, quality detection, backpressure handling, atomicity guarantees.

This custom approach implements patterns needed for production that frameworks don't provide:

**1. Performance Optimization**
- Direct control over batching strategies (dynamic batch sizing based on token counts)
- Custom backpressure implementation preventing OOM errors
- Token-aware chunking using the actual embedding model's tokenizer
- Zero overhead from abstraction layers

**2. Operational Transparency**
- Every parameter exposed via environment variables for A/B testing
- Direct access to Redis queues, ChromaDB collections, and DuckDB for debugging
- Explicit retry logic and error handling (no hidden framework behavior)
- Full logging of pipeline stages with performance metrics

**3. Production Requirements**
- File-level transactional guarantees (atomicity not provided by LangChain)
- Distributed worker architecture with multiprocessing pools
- OCR fallback pipeline with quality detection and automatic retry
- Comprehensive failure tracking (per-file, per-chunk diagnostics)

**Comparison (Ingestion Pipeline):**

| Aspect | Custom Ingestion (This Project) | LangChain Document Loaders |
|--------|--------------------------------|---------------------------|
| **Setup Time** | 2-3 days initial build | Hours to prototype |
| **Control** | Full - every component configurable | Partial - framework constraints |
| **Performance Tuning** | Precise (batching, chunking, retries) | Limited without forking framework |
| **Production Readiness** | Atomicity, backpressure, monitoring built-in | Requires additional engineering |
| **Debugging** | Direct access to all components | Navigate framework abstractions |
| **Scaling** | Explicit worker pools, queue management | Single-threaded document loaders |
| **OCR Fallback** | Automatic quality detection + retry | Manual implementation required |
| **Ideal For** | Production deployments requiring control | Rapid prototyping, low-code demos |

**When to Use Each:**
- **Custom Ingestion**: Production systems processing 1000+ documents with mixed quality, SLAs, or need for distributed processing
- **LangChain Loaders**: Proof-of-concepts, demos, small document collections (<100 files)

**Note**: This project uses LangChain's ChromaDB and HuggingFace wrappers for convenience, but the ingestion pipeline (the complex part) is entirely custom. This demonstrates you can leverage helpful integrations while maintaining control over critical business logic.

---

## 🎯 Design Philosophy

**Custom Ingestion Pipeline**: The entire document processing pipeline (producer/consumer/OCR workers, Redis coordination, chunking, quality detection) is built from scratch without framework orchestration. 

This architecture maintains full control over:

**1. Chunking Strategy**
- Token-aware splitting using the actual embedding model's tokenizer (e5-large-v2)
- Configurable chunk size with automatic boundary detection
- Page-level metadata preservation for source attribution

**2. Quality Assurance**
- Multi-stage text quality detection (gibberish, corruption, encoding validation)
- Automatic OCR fallback with Tesseract when pdfplumber extraction fails
- Latin script ratio checking with configurable thresholds

**3. Production Guarantees**
- File-level atomicity: all chunks from a document commit together or not at all
- Redis-based backpressure to prevent memory overflow
- Distributed worker pools with multiprocessing
- Comprehensive error tracking and retry logic

**4. Transparency and Tunability**
- All parameters exposed via environment variables (`ingest-svc.env`)
- Direct access to vector DB, Redis queues, and DuckDB for inspection
- No hidden abstractions or magic behavior
- Full logging with emoji-coded event types for easy scanning
