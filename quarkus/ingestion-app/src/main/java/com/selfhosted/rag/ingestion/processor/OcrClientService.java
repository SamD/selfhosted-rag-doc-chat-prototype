package com.selfhosted.rag.ingestion.processor;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.selfhosted.rag.common.config.AppConfig;
import com.selfhosted.rag.common.model.OcrJob;
import com.selfhosted.rag.common.model.OcrResponse;
import io.quarkus.redis.client.RedisClient;
import io.vertx.redis.client.Response;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

/**
 * OCR client service for sending images to OCR worker via Redis.
 *
 * Maps to Python: workers/producer_worker.py - send_image_to_ocr()
 */
@ApplicationScoped
public class OcrClientService {

    @Inject
    AppConfig appConfig;

    @Inject
    RedisClient redisClient;

    @Inject
    ObjectMapper objectMapper;

    /**
     * Convert PDF page to image and send to OCR service, waiting for response.
     */
    public String fallbackOcrForPage(String fullPath, String relPath, int pageNum) {
        String jobId = UUID.randomUUID().toString();
        String replyKey = "ocr_reply_" + jobId;

        try (PDDocument document = Loader.loadPDF(new File(fullPath))) {
            PDFRenderer renderer = new PDFRenderer(document);
            BufferedImage image = renderer.renderImageWithDPI(pageNum - 1, 300); // 300 DPI for OCR

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(image, "png", baos);
            byte[] imageData = baos.toByteArray();
            String base64Image = Base64.getEncoder().encodeToString(imageData);

            OcrJob job = OcrJob.builder()
                    .job_id(jobId)
                    .rel_path(relPath)
                    .page_num(pageNum)
                    .image_shape(Arrays.asList(image.getHeight(), image.getWidth(), 3)) // Dummy shape as Python expects it
                    .image_dtype("uint8")
                    .image_base64(base64Image)
                    .reply_key(replyKey)
                    .build();

            String jobJson = objectMapper.writeValueAsString(job);
            
            // Push to OCR queue
            redisClient.lpush(Arrays.asList(appConfig.getOcrJobQueue(), jobJson));
            
            // Wait for response (BLPOP replyKey 30)
            Response response = redisClient.blpop(Arrays.asList(replyKey, "30"));
            
            if (response != null && response.size() == 2) {
                String responseJson = response.get(1).toString();
                OcrResponse ocrResponse = objectMapper.readValue(responseJson, OcrResponse.class);
                return ocrResponse.text();
            } else {
                System.err.println("OCR timeout or error for " + relPath + " page " + pageNum);
                return null;
            }

        } catch (IOException e) {
            System.err.println("Error rendering PDF page for OCR: " + e.getMessage());
            return null;
        }
    }
}
