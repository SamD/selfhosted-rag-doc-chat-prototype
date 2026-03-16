# Self-Hosted RAG System (Quarkus Port)

This is a Quarkus-based port of the Python `doc-ingest-chat` system. It replicates the same sharded, order-preserving ingestion pipeline and RAG chat logic using Java, LangChain4j, and Jlama for native local inference.

## Quickstart

### 1. Individual Maven Commands (Local Development)

To run the system locally without Docker, ensure you have Redis and Qdrant running, then start each service in its own terminal:

1.  **Build everything:**
    ```bash
    mvn clean install -DskipTests
    ```

2.  **Start Persistence App:**
    ```bash
    cd persistence-app
    mvn quarkus:dev
    ```

3.  **Start OCR Fallback App:**
    ```bash
    cd ocr-fallback-app
    mvn quarkus:dev
    ```

4.  **Start Ingestion App:**
    ```bash
    cd ingestion-app
    mvn quarkus:dev -DINGEST_FOLDER=/path/to/your/docs -DEMBEDDING_MODEL_PATH=/path/to/models/intfloat/e5-large-v2
    ```

5.  **Start Front-End API:**
    ```bash
    cd front-end
    mvn quarkus:dev -DEMBEDDING_MODEL_PATH=/path/to/models/intfloat/e5-large-v2
    ```

### 2. Individual Docker Commands

1.  **Build the images:**
    ```bash
    docker build -f front-end/Dockerfile.jvm -t rag-frontend .
    docker build -f ingestion-app/Dockerfile.jvm -t rag-ingestion .
    docker build -f persistence-app/Dockerfile.jvm -t rag-persistence .
    docker build -f ocr-fallback-app/Dockerfile.jvm -t rag-ocr .
    ```

2.  **Run a service (e.g., Front-End):**
    ```bash
    docker run -p 8080:8080 \
      -e EMBEDDING_MODEL_PATH=/models/embedding \
      -v /path/to/models/intfloat/e5-large-v2:/models/embedding:ro \
      rag-frontend
    ```

### 3. Docker Compose (Full Stack)

The easiest way to run the complete system is using Docker Compose.

1.  **Set your environment variables:**
    ```bash
    export INGEST_FOLDER=/path/to/your/docs
    export EMBEDDING_MODEL_PATH=/path/to/models/intfloat/e5-large-v2
    ```

2.  **Start the stack:**
    ```bash
    docker-compose up --build
    ```

The Front-End API will be available at `http://localhost:8080`.

## Architecture

The system consists of several specialized modules:

*   **`common-lib`**: Shared models, configurations, and utilities.
*   **`ingestion-app`**: Scans directories, performs text extraction/OCR, and enqueues chunks to Redis.
*   **`persistence-app`**: Consumes chunks from Redis and stores them in the Qdrant vector database.
*   **`ocr-fallback-app`**: Specialized worker for CPU/GPU-based OCR fallback.
*   **`front-end`**: REST API for chat retrieval and response generation using LangChain4j.

## Key Features

*   **Local Inference**: Uses Jlama for 100% Java-native execution of E5-large-v2 embeddings and Llama-3 chat models.
*   **Parity with Python**: Maintains exact chunking, prefixing (`query: `/`passage: `), and metadata schemas from the original Python version.
*   **Sharded Ingestion**: Uses Redis queues to ensure ordered processing of chunks even with multiple parallel workers.
