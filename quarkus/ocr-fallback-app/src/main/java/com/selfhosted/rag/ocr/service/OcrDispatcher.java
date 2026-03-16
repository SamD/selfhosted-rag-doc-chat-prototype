package com.selfhosted.rag.ocr.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.selfhosted.rag.common.config.AppConfig;
import com.selfhosted.rag.common.model.OcrJob;
import com.selfhosted.rag.common.model.OcrResponse;
import io.quarkus.redis.client.RedisClient;
import io.quarkus.runtime.Startup;
import io.vertx.redis.client.Response;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.context.ManagedExecutor;

import java.io.IOException;
import java.util.Arrays;

/**
 * Dispatcher that listens for OCR jobs in Redis and dispatches to Tesseract service.
 *
 * Maps to Python: workers/ocr_worker.py - dispatcher()
 */
@ApplicationScoped
@Startup
public class OcrDispatcher {

    @Inject
    AppConfig appConfig;

    @Inject
    RedisClient redisClient;

    @Inject
    TesseractOcrService ocrService;

    @Inject
    ObjectMapper objectMapper;

    @Inject
    ManagedExecutor managedExecutor;

    @PostConstruct
    void startDispatcher() {
        int workerCount = appConfig.getOcrDebugImageDir() != null ? 2 : 1; // Simplified worker count logic
        System.out.println("🚀 Starting " + workerCount + " OCR dispatcher workers");
        for (int i = 0; i < workerCount; i++) {
            managedExecutor.execute(this::runDispatcher);
        }
    }

    private void runDispatcher() {
        System.out.println("🚀 OCR dispatcher worker started");
        String queueName = appConfig.getOcrJobQueue();
        
        while (true) {
            try {
                Response response = redisClient.brpop(Arrays.asList(queueName, "5"));
                if (response != null && response.size() == 2) {
                    String json = response.get(1).toString();
                    processJob(json);
                }
            } catch (Exception e) {
                System.err.println("Error in OCR dispatcher worker: " + e.getMessage());
                try { Thread.sleep(1000); } catch (InterruptedException ex) { break; }
            }
        }
    }

    private void processJob(String json) throws IOException {
        OcrJob job = objectMapper.readValue(json, OcrJob.class);
        System.out.println("📥 Job: " + job.getRel_path() + ", page " + job.getPage_num() + ", job_id " + job.getJob_id());
        
        String text = ocrService.extractTextFromBase64Image(job.getImage_base64());
        
        OcrResponse ocrResponse = OcrResponse.builder()
                .text(text)
                .rel_path(job.getRel_path())
                .page_num(job.getPage_num())
                .engine("tesseract-quarkus")
                .job_id(job.getJob_id())
                .build();
        
        String responseJson = objectMapper.writeValueAsString(ocrResponse);
        
        if (job.getReply_key() != null) {
            redisClient.lpush(Arrays.asList(job.getReply_key(), responseJson));
            redisClient.expire(job.getReply_key(), "300");
            System.out.println("📤 Response sent to " + job.getReply_key());
        }
    }
}
