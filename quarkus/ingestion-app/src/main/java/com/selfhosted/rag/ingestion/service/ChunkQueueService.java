package com.selfhosted.rag.ingestion.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.selfhosted.rag.common.config.AppConfig;
import com.selfhosted.rag.common.model.ChunkEntry;
import com.selfhosted.rag.common.model.FileEndMessage;
import com.selfhosted.rag.common.service.FileTrackingService;
import com.selfhosted.rag.ingestion.processor.DocumentProcessor;
import com.selfhosted.rag.ingestion.processor.TextProcessor;
import io.quarkus.redis.client.RedisClient;
import io.vertx.redis.client.Response;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Service for enqueuing chunks to Redis.
 *
 * Maps to Python: workers/producer_worker.py
 */
@ApplicationScoped
public class ChunkQueueService {

    @Inject
    AppConfig appConfig;

    @Inject
    RedisClient redisClient;

    @Inject
    ObjectMapper objectMapper;

    @Inject
    DocumentProcessor documentProcessor;

    @Inject
    TextProcessor textProcessor;

    @Inject
    FileTrackingService fileTrackingService;

    private final AtomicInteger queueIndex = new AtomicInteger(0);

    private static final String PUSH_LUA_SCRIPT =
            "local queue = KEYS[1]\n" +
            "local max_len = tonumber(ARGV[1])\n" +
            "local new_items = {}\n" +
            "\n" +
            "for i = 2, #ARGV do\n" +
            "    table.insert(new_items, ARGV[i])\n" +
            "end\n" +
            "\n" +
            "local current_len = redis.call(\"LLEN\", queue)\n" +
            "if current_len + #new_items <= max_len then\n" +
            "    for _, item in ipairs(new_items) do\n" +
            "        redis.call(\"RPUSH\", queue, item)\n" +
            "    end\n" +
            "    return 1\n" +
            "else\n" +
            "    return 0\n" +
            "end";

    public String getNextQueue() {
        List<String> queues = appConfig.getQueueNamesList();
        int index = queueIndex.getAndIncrement() % queues.size();
        return queues.get(index);
    }

    public void pushWithBackpressure(String queueName, List<String> entries, int maxQueueLength, String relPath) {
        if (entries == null || entries.isEmpty()) return;
        
        long startWait = System.currentTimeMillis();
        boolean warned = false;
        long lastLogTime = startWait;

        while (true) {
            List<String> args = new ArrayList<>();
            args.add(PUSH_LUA_SCRIPT);
            args.add("1"); // numkeys
            args.add(queueName); // KEYS[1]
            args.add(String.valueOf(maxQueueLength)); // ARGV[1]
            args.addAll(entries); // ARGV[2..]

            try {
                Response response = redisClient.eval(args);
                int result = (response != null) ? response.toInteger() : 0;

                if (result == 1) {
                    if (warned) {
                        double elapsed = (System.currentTimeMillis() - startWait) / 1000.0;
                        System.out.printf("✅ Queue backpressure resolved after %.2fs — pushed %d entries to '%s' for %s%n",
                                elapsed, entries.size(), queueName, relPath);
                    }
                    return;
                }
            } catch (Exception e) {
                System.err.println("❌ Error executing Redis Lua script: " + e.getMessage());
                // Fallback to simple push if script fails
                for (String entry : entries) {
                    redisClient.rpush(Arrays.asList(queueName, entry));
                }
                return;
            }

            long now = System.currentTimeMillis();
            if (!warned && (now - startWait) > 10000) { // 10s warn
                int qlen = getQueueLength(queueName);
                System.out.printf("⏳ Queue '%s' length %d exceeds limit (%d) — backpressure delay on %s%n",
                        queueName, qlen, maxQueueLength, relPath);
                warned = true;
            }

            if (now - lastLogTime > 10000) { // log every 10s
                int qlen = getQueueLength(queueName);
                System.out.printf("🔁 Still waiting to enqueue %s (queue: %s, length: %d) [waited %.1fs]%n",
                        relPath, queueName, qlen, (now - startWait) / 1000.0);
                lastLogTime = now;
            }

            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    public int getQueueLength(String queueName) {
        Response response = redisClient.llen(queueName);
        return response != null ? response.toInteger() : 0;
    }

    /**
     * Ingest a single file.
     * Maps to: ingest_file(full_path, rel_path, job_id)
     */
    public int ingestFile(String fullPath, String relPath, String jobId) {
        System.out.println("📁 Processing file: " + relPath);
        
        try {
            List<DocumentProcessor.ChunkWithMetadata> chunksWithMetadata;
            String fileType;

            if (relPath.toLowerCase().endsWith(".pdf")) {
                fileType = "pdf";
                if (!new File(fullPath).exists()) return 1;
                chunksWithMetadata = documentProcessor.processPdfByPage(fullPath, relPath, fileType);
            } else if (relPath.toLowerCase().endsWith(".html") || relPath.toLowerCase().endsWith(".htm")) {
                fileType = "html";
                String text = documentProcessor.extractTextFromHtml(fullPath);
                if (text == null || text.isBlank()) return 1;
                List<String> chunks = textProcessor.splitDoc(text, relPath, fileType, -1);
                chunksWithMetadata = new ArrayList<>();
                for (int i = 0; i < chunks.size(); i++) {
                    chunksWithMetadata.add(new DocumentProcessor.ChunkWithMetadata(chunks.get(i), -1, i));
                }
            } else {
                return 1;
            }

            if (chunksWithMetadata.isEmpty()) {
                fileTrackingService.updateFailedFiles(relPath);
                return 1;
            }

            List<ChunkEntry> entries = new ArrayList<>();
            for (int i = 0; i < chunksWithMetadata.size(); i++) {
                DocumentProcessor.ChunkWithMetadata cm = chunksWithMetadata.get(i);
                ChunkEntry entry = ChunkEntry.builder()
                        .chunk(cm.text())
                        .id(textProcessor.makeChunkId(relPath, i, cm.text()))
                        .source_file(relPath)
                        .type(fileType)
                        .hash(fileTrackingService.md5(cm.text()))
                        .engine("quarkus")
                        .page(cm.pageNum())
                        .chunk_index(cm.chunkIndex())
                        .build();
                entries.add(entry);
            }

            String nextQueue = getNextQueue();
            List<String> jsonEntries = new ArrayList<>();
            for (ChunkEntry entry : entries) {
                jsonEntries.add(objectMapper.writeValueAsString(entry));
            }

            // Atomic enqueue to Redis with backpressure
            pushWithBackpressure(nextQueue, jsonEntries, 50000, relPath);

            // Send sentinel via SAME pushWithBackpressure to ensure order
            FileEndMessage sentinel = FileEndMessage.builder()
                    .source_file(relPath)
                    .expected_chunks(entries.size())
                    .build();
            pushWithBackpressure(nextQueue, Arrays.asList(objectMapper.writeValueAsString(sentinel)), 50000, relPath);

            System.out.println("📤 Done enqueuing " + entries.size() + " chunks for " + relPath);
            return 0;

        } catch (Exception e) {
            System.err.println("Error ingesting file " + relPath + ": " + e.getMessage());
            fileTrackingService.updateFailedFiles(relPath);
            return 1;
        }
    }
}
