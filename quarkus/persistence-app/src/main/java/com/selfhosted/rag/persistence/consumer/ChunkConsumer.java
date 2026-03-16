package com.selfhosted.rag.persistence.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.selfhosted.rag.common.config.AppConfig;
import com.selfhosted.rag.common.model.ChunkEntry;
import com.selfhosted.rag.common.model.FileEndMessage;
import com.selfhosted.rag.common.service.FileTrackingService;
import com.selfhosted.rag.persistence.service.VectorStoreService;
import io.quarkus.redis.client.RedisClient;
import io.quarkus.runtime.Startup;
import io.vertx.redis.client.Response;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.context.ManagedExecutor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * Consumer worker that processes chunks from Redis and persists to vector DB.
 *
 * Maps to Python: workers/consumer_worker.py
 */
@ApplicationScoped
@Startup
public class ChunkConsumer {

    @Inject
    AppConfig appConfig;

    @Inject
    RedisClient redisClient;

    @Inject
    ObjectMapper objectMapper;

    @Inject
    VectorStoreService vectorStoreService;

    @Inject
    FileTrackingService fileTrackingService;

    @Inject
    ManagedExecutor managedExecutor;

    private final Map<String, List<ChunkEntry>> buffer = new ConcurrentHashMap<>();
    private final Map<String, Long> timestamps = new ConcurrentHashMap<>();
    private final AtomicBoolean shutdown = new AtomicBoolean(false);

    @PostConstruct
    void startConsumers() {
        if (!appConfig.isConsumerEnabled()) {
            System.out.println("⚠️ Consumer workers are disabled via configuration");
            return;
        }

        List<String> queues = appConfig.getQueueNamesList();
        System.out.println("🚀 Starting " + queues.size() + " consumer workers for: " + queues);
        for (String queue : queues) {
            managedExecutor.execute(() -> runConsumer(queue));
        }
    }

    @PreDestroy
    void stop() {
        shutdown.set(true);
    }

    private void runConsumer(String queueName) {
        System.out.println("🚀 Consumer started for: " + queueName);
        while (!shutdown.get()) {
            try {
                // BLPOP with 5s timeout
                Response response = redisClient.blpop(java.util.Arrays.asList(queueName, "5"));
                if (response != null && response.size() == 2) {
                    String json = response.get(1).toString();
                    processMessage(json, queueName);
                }
                
                // TTL check for buffers (simplified version of Python logic)
                checkBufferTimeouts();
                
            } catch (Exception e) {
                if (!shutdown.get()) {
                    System.err.println("Error in consumer worker for " + queueName + ": " + e.getMessage());
                    try { Thread.sleep(1000); } catch (InterruptedException ex) { break; }
                }
            }
        }
        System.out.println("✅ Consumer exiting for: " + queueName);
    }

    private void processMessage(String json, String queueName) throws IOException {
        if (json.contains("\"type\":\"file_end\"")) {
            FileEndMessage endMsg = objectMapper.readValue(json, FileEndMessage.class);
            handleFileEnd(endMsg, queueName);
        } else {
            ChunkEntry chunk = objectMapper.readValue(json, ChunkEntry.class);
            handleChunk(chunk, queueName);
        }
    }

    private void handleChunk(ChunkEntry chunk, String queueName) {
        String sourceFile = chunk.source_file();
        buffer.computeIfAbsent(sourceFile, k -> new ArrayList<>()).add(chunk);
        timestamps.put(sourceFile, System.currentTimeMillis());
        System.out.println("📥 [" + queueName + "] Received chunk " + chunk.id() + " from " + sourceFile);
    }

    private void handleFileEnd(FileEndMessage msg, String queueName) {
        String sourceFile = msg.source_file();
        List<ChunkEntry> chunks = buffer.remove(sourceFile);
        timestamps.remove(sourceFile);

        if (chunks == null || chunks.size() != msg.expected_chunks()) {
            System.err.println("❌ [" + queueName + "] Incomplete chunks for " + sourceFile);
            fileTrackingService.updateFailedFiles(sourceFile);
            return;
        }

        try {
            List<String> texts = chunks.stream().map(ChunkEntry::chunk).collect(Collectors.toList());
            List<Map<String, Object>> metadatas = chunks.stream().map(c -> {
                Map<String, Object> m = new HashMap<>();
                m.put("source_file", c.source_file());
                m.put("type", c.type());
                m.put("hash", c.hash());
                m.put("engine", c.engine());
                m.put("page", c.page());
                m.put("chunk_index", c.chunk_index());
                return m;
            }).collect(Collectors.toList());
            List<String> ids = chunks.stream().map(ChunkEntry::id).collect(Collectors.toList());

            vectorStoreService.addTexts(texts, metadatas, ids);
            fileTrackingService.updateIngestedFiles(sourceFile);
            System.out.println("✅ [" + queueName + "] Persisted " + sourceFile + " to vector DB");

        } catch (Exception e) {
            System.err.println("💥 [" + queueName + "] Failed to persist " + sourceFile + ": " + e.getMessage());
            fileTrackingService.updateFailedFiles(sourceFile);
        }
    }

    private void checkBufferTimeouts() {
        long now = System.currentTimeMillis();
        long timeoutMs = appConfig.getChunkTimeout() * 1000L;
        
        timestamps.forEach((file, firstSeen) -> {
            if (now - firstSeen > timeoutMs) {
                System.out.println("⌛ TTL expired for " + file + ", discarding buffer");
                buffer.remove(file);
                timestamps.remove(file);
                fileTrackingService.updateFailedFiles(file);
            }
        });
    }
}
