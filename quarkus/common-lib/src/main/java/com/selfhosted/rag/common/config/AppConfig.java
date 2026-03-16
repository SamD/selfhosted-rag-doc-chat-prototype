package com.selfhosted.rag.common.config;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * Configuration settings for the document ingestion system.
 *
 * Maps to Python: config/settings.py
 */
@ApplicationScoped
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

    @ConfigProperty(name = "TESSDATA_PREFIX", defaultValue = "")
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

    // Getters

    public String getOcrJobQueue() { return ocrJobQueue; }
    public String getIngestQueue() { return ingestQueue; }
    public List<String> getQueueNamesList() { return Arrays.asList(queueNames.split(",")); }
    public int getMaxQueueLength() { return maxQueueLength; }
    public double getPollInterval() { return pollInterval; }
    public double getWaitWarnThreshold() { return waitWarnThreshold; }
    public String getVectorDbProfile() { return vectorDbProfile; }
    public boolean isUseQdrant() { return "qdrant".equalsIgnoreCase(vectorDbProfile); }
    public String getVectorDbHost() { return vectorDbHost; }
    public int getVectorDbPort() { return vectorDbPort; }
    public String getVectorDbCollection() { return vectorDbCollection; }
    public int getQdrantRetrieverK() { return qdrantRetrieverK; }
    public double getQdrantDenseWeight() { return qdrantDenseWeight; }
    public double getQdrantSparseWeight() { return qdrantSparseWeight; }
    public int getChunkTimeout() { return chunkTimeout; }
    public int getMaxChunks() { return maxChunks; }
    public int getMaxBatchSize() { return maxBatchSize; }
    public int getMaxTokens() { return maxTokens; }
    public String getEmbeddingModelPath() { return embeddingModelPath; }
    public String getLlmPath() { return llmPath; }
    public boolean isUseGpu() { return useGpu; }

    public String getIngestFolder() { return ingestFolder; }
    public List<String> getSupportedExtensionsList() { return Arrays.asList(supportedExtensions.split(",")); }
    public List<String> getSupportedMediaExtensionsList() { return Arrays.asList(supportedMediaExtensions.split(",")); }
    public String getFailedFiles() { return failedFiles; }
    public String getIngestedFile() { return ingestedFile; }
    public String getTrackFile() { return trackFile; }
    public String getParquetFile() { return parquetFile; }
    public String getDuckdbFile() { return duckdbFile; }

    public boolean isAllowLatinExtended() { return allowLatinExtended; }
    public double getLatinScriptMinRatio() { return latinScriptMinRatio; }

    public String getOcrDebugImageDir() { return ocrDebugImageDir; }
    public int getMaxOcrDim() { return maxOcrDim; }
    public String getTesseractLangs() { return tesseractLangs; }
    public int getTesseractPsm() { return tesseractPsm; }
    public int getTesseractOem() { return tesseractOem; }
    public boolean isTesseractUseScriptLatin() { return tesseractUseScriptLatin; }
    public String getTessdataPrefix() { return tessdataPrefix; }

    public String getMediaDevice() { return mediaDevice; }
    public int getMediaBatchSize() { return mediaBatchSize; }
    public String getMediaComputeType() { return mediaComputeType; }

    public boolean isUseOllama() { return useOllama; }
    public boolean isConsumerEnabled() { return consumerEnabled; }
    public boolean isProducerEnabled() { return producerEnabled; }
    public String getOllamaModel() { return ollamaModel; }
    public String getOllamaUrl() { return ollamaUrl; }

    public int getRetrieverTopK() { return retrieverTopK; }

    public int getLlamaNCtx() { return llamaNCtx; }
    public int getLlamaNGpuLayers() { return llamaNGpuLayers; }
    public int getLlamaNThreads() { return llamaNThreads; }
    public int getLlamaNBatch() { return llamaNBatch; }
    public boolean isLlamaF16Kv() { return llamaF16Kv; }
    public double getLlamaTemperature() { return llamaTemperature; }
    public int getLlamaTopK() { return llamaTopK; }
    public double getLlamaTopP() { return llamaTopP; }
    public double getLlamaRepeatPenalty() { return llamaRepeatPenalty; }
    public int getLlamaMaxTokens() { return llamaMaxTokens; }
    public String getLlamaChatFormat() { return llamaChatFormat; }
    public boolean isLlamaVerbose() { return llamaVerbose; }
    public int getLlamaSeed() { return llamaSeed; }

    public int getMaxBatchSizeLimit() { return maxBatchSizeLimit; }
}
