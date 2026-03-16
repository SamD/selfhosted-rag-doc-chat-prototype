package com.selfhosted.rag.ocr.service;

import com.selfhosted.rag.common.config.AppConfig;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Base64;

/**
 * Tesseract OCR processing service.
 *
 * Maps to Python: workers/ocr_worker.py - fallback_to_tesseract()
 */
@ApplicationScoped
public class TesseractOcrService {

    @Inject
    AppConfig appConfig;

    private Tesseract tesseract;

    @PostConstruct
    void init() {
        this.tesseract = new Tesseract();
        
        // Set tessdata path if provided
        String tessdataPath = appConfig.getTessdataPrefix();
        if (tessdataPath != null && !tessdataPath.isBlank()) {
            tesseract.setDatapath(tessdataPath);
        }
        
        tesseract.setLanguage(appConfig.getTesseractLangs());
        tesseract.setPageSegMode(appConfig.getTesseractPsm());
        tesseract.setOcrEngineMode(appConfig.getTesseractOem());
        
        if (appConfig.isTesseractUseScriptLatin()) {
            tesseract.setTessVariable("tessedit_script", "Latin");
        }
    }

    public String extractTextFromBase64Image(String base64Image) {
        try {
            byte[] imageBytes = Base64.getDecoder().decode(base64Image);
            BufferedImage image = ImageIO.read(new ByteArrayInputStream(imageBytes));
            if (image == null) {
                System.err.println("Failed to decode image from base64");
                return null;
            }
            return tesseract.doOCR(image).strip();
        } catch (IOException | TesseractException e) {
            System.err.println("OCR error: " + e.getMessage());
            return null;
        }
    }
}
