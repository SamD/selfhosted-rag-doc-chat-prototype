package com.selfhosted.rag.common.config;

import jakarta.enterprise.context.ApplicationScoped;
import lombok.Getter;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import java.util.Arrays;
import java.util.List;

/**
 * Configuration settings for the document ingestion system.
 *
 * Maps to Python: config/settings.py
 */
@ApplicationScoped
@Getter
public class AppConfig {

    @ConfigProperty(name = "REDIS_HOST", defaultValue = "localhost")
    String redisHost;

    @ConfigProperty(name = "REDIS_PORT", defaultValue = "6380")
    int redisPort;

    @ConfigProperty(name = "REDIS_OCR_JOB_QUEUE", defaultValue = "ocr_processing_job")
    String ocrJobQueue;

    @ConfigProperty(name = "REDIS_INGEST_QUEUE", defaultValue = "chunk_ingest_queue")
    String ingestQueue;

    @ConfigProperty(name = "QUEUE_NAMES", defaultValue = "chunk_ingest_queue:0,chunk_ingest_queue:1")
    String queueNames;

    @ConfigProperty(name = "MAX_QUEUE_LENGTH", defaultValue = "25")
    int maxQueueLength;

    @ConfigProperty(name = "POLL_INTERVAL", defaultValue = "0.5")
    double pollInterval;

    @ConfigProperty(name = "WAIT_WARN_THRESHOLD", defaultValue = "10.0")
    double waitWarnThreshold;

    @ConfigProperty(name = "VECTOR_DB_PROFILE", defaultValue = "qdrant")
    String vectorDbProfile;

    @ConfigProperty(name = "VECTOR_DB_HOST", defaultValue = "vector-db")
    String vectorDbHost;

    @ConfigProperty(name = "VECTOR_DB_PORT", defaultValue = "6333")
    int vectorDbPort;

    @ConfigProperty(name = "VECTOR_DB_COLLECTION", defaultValue = "vector_base_collection")
    String vectorDbCollection;

    @ConfigProperty(name = "QDRANT_RETRIEVER_K", defaultValue = "10")
    int qdrantRetrieverK;

    @ConfigProperty(name = "QDRANT_DENSE_WEIGHT", defaultValue = "0.3")
    double qdrantDenseWeight;

    @ConfigProperty(name = "QDRANT_SPARSE_WEIGHT", defaultValue = "0.7")
    double qdrantSparseWeight;

    @ConfigProperty(name = "CHUNK_TIMEOUT", defaultValue = "300")
    int chunkTimeout;

    @ConfigProperty(name = "MAX_CHUNKS", defaultValue = "20000")
    int maxChunks;

    @ConfigProperty(name = "MAX_CHROMA_BATCH_SIZE", defaultValue = "75")
    int maxBatchSize;

    @ConfigProperty(name = "MAX_TOKENS", defaultValue = "512")
    int maxTokens;

    @ConfigProperty(name = "EMBEDDING_MODEL_PATH", defaultValue = "/models/embedding")
    String embeddingModelPath;

    @ConfigProperty(name = "LLM_PATH", defaultValue = "/models/llm")
    String llmPath;

    @ConfigProperty(name = "LLAMA_USE_GPU", defaultValue = "true")
    boolean useGpu;

    @ConfigProperty(name = "INGEST_FOLDER", defaultValue = "/data/ingest")
    String ingestFolder;

    @ConfigProperty(name = "CHROMA_DATA_DIR", defaultValue = "/data/chroma")
    String chromaDataDir;

    @ConfigProperty(name = "SUPPORTED_EXTENSIONS", defaultValue = ".pdf,.html,.htm")
    String supportedExtensions;

    @ConfigProperty(name = "SUPPORTED_MEDIA_EXT", defaultValue = ".mp3,.wav,.m4a,.aac,.flac,.mp4,.mov,.mkv")
    String supportedMediaExtensions;

    @ConfigProperty(name = "FAILED_FILES", defaultValue = "failed_files.txt")
    String failedFiles;

    @ConfigProperty(name = "INGESTED_FILE", defaultValue = "ingested_files.txt")
    String ingestedFile;

    @ConfigProperty(name = "TRACK_FILE", defaultValue = "ingested_files.txt")
    String trackFile;

    @ConfigProperty(name = "PARQUET_FILE", defaultValue = "chunks.parquet")
    String parquetFile;

    @ConfigProperty(name = "DUCKDB_FILE", defaultValue = "chunks.duckdb")
    String duckdbFile;

    @ConfigProperty(name = "ALLOW_LATIN_EXTENDED", defaultValue = "true")
    boolean allowLatinExtended;

    @ConfigProperty(name = "LATIN_SCRIPT_MIN_RATIO", defaultValue = "0.7")
    double latinScriptMinRatio;

    @ConfigProperty(name = "DEBUG_IMAGE_DIR", defaultValue = "/tmp/ocr_debug")
    String ocrDebugImageDir;

    @ConfigProperty(name = "MAX_OCR_DIM", defaultValue = "3000")
    int maxOcrDim;

    @ConfigProperty(name = "TESSERACT_LANGS", defaultValue = "eng+lat")
    String tesseractLangs;

    @ConfigProperty(name = "TESSERACT_PSM", defaultValue = "6")
    int tesseractPsm;

    @ConfigProperty(name = "TESSERACT_OEM", defaultValue = "1")
    int tesseractOem;

    @ConfigProperty(name = "TESSERACT_USE_SCRIPT_LATIN", defaultValue = "true")
    boolean tesseractUseScriptLatin;

    @ConfigProperty(name = "TESSDATA_PREFIX", defaultValue = "/usr/share/tessdata")
    String tessdataPrefix;

    @ConfigProperty(name = "DEVICE", defaultValue = "cuda")
    String mediaDevice;

    @ConfigProperty(name = "MEDIA_BATCH_SIZE", defaultValue = "8")
    int mediaBatchSize;

    @ConfigProperty(name = "COMPUTE_TYPE", defaultValue = "float16")
    String mediaComputeType;

    @ConfigProperty(name = "USE_OLLAMA", defaultValue = "false")
    boolean useOllama;

    @ConfigProperty(name = "CONSUMER_ENABLED", defaultValue = "true")
    boolean consumerEnabled;

    @ConfigProperty(name = "PRODUCER_ENABLED", defaultValue = "true")
    boolean producerEnabled;

    @ConfigProperty(name = "OLLAMA_MODEL", defaultValue = "NeuralNet/openchat-3.6")
    String ollamaModel;

    @ConfigProperty(name = "OLLAMA_URL", defaultValue = "http://localhost:11434")
    String ollamaUrl;

    @ConfigProperty(name = "RETRIEVER_TOP_K", defaultValue = "20")
    int retrieverTopK;

    @ConfigProperty(name = "LLAMA_N_CTX", defaultValue = "32768")
    int llamaNCtx;

    @ConfigProperty(name = "LLAMA_N_GPU_LAYERS", defaultValue = "35")
    int llamaNGpuLayers;

    @ConfigProperty(name = "LLAMA_N_THREADS", defaultValue = "24")
    int llamaNThreads;

    @ConfigProperty(name = "LLAMA_N_BATCH", defaultValue = "512")
    int llamaNBatch;

    @ConfigProperty(name = "LLAMA_F16_KV", defaultValue = "true")
    boolean llamaF16Kv;

    @ConfigProperty(name = "LLAMA_TEMPERATURE", defaultValue = "0.3")
    double llamaTemperature;

    @ConfigProperty(name = "LLAMA_TOP_K", defaultValue = "25")
    int llamaTopK;

    @ConfigProperty(name = "LLAMA_TOP_P", defaultValue = "0.85")
    double llamaTopP;

    @ConfigProperty(name = "LLAMA_REPEAT_PENALTY", defaultValue = "1.2")
    double llamaRepeatPenalty;

    @ConfigProperty(name = "LLAMA_MAX_TOKENS", defaultValue = "512")
    int llamaMaxTokens;

    @ConfigProperty(name = "LLAMA_CHAT_FORMAT", defaultValue = "chatml")
    String llamaChatFormat;

    @ConfigProperty(name = "LLAMA_VERBOSE", defaultValue = "false")
    boolean llamaVerbose;

    @ConfigProperty(name = "LLAMA_SEED", defaultValue = "42")
    int llamaSeed;

    @ConfigProperty(name = "MAX_CHROMA_BATCH_SIZE_LIMIT", defaultValue = "5461")
    int maxBatchSizeLimit;

    @ConfigProperty(name = "METRICS_ENABLED", defaultValue = "true")
    boolean metricsEnabled;

    @ConfigProperty(name = "METRICS_LOG_FILE", defaultValue = "metrics.jsonl")
    String metricsLogFile;

    @ConfigProperty(name = "METRICS_LOG_TO_STDOUT", defaultValue = "true")
    boolean metricsLogToStdout;

    // Derived helpers

    public List<String> getQueueNamesList() { return Arrays.asList(queueNames.split(",")); }
    public boolean isUseQdrant() { return "qdrant".equalsIgnoreCase(vectorDbProfile); }
    public List<String> getSupportedExtensionsList() { return Arrays.asList(supportedExtensions.split(",")); }
    public List<String> getSupportedMediaExtensionsList() { return Arrays.asList(supportedMediaExtensions.split(",")); }
}
