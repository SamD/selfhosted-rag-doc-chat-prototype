package com.selfhosted.rag.ingestion.processor;

import com.selfhosted.rag.common.utils.TextUtils;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Document processing functionality.
 *
 * Maps to Python: processors/document_processor.py and workers/producer_worker.py
 */
@ApplicationScoped
public class DocumentProcessor {

    @Inject
    TextUtils textUtils;

    @Inject
    TextProcessor textProcessor;

    @Inject
    OcrClientService ocrClientService;

    /**
     * Extract text from HTML file.
     * Maps to: extract_text_from_html(full_path)
     */
    public String extractTextFromHtml(String fullPath) {
        try {
            File input = new File(fullPath);
            Document doc = Jsoup.parse(input, "UTF-8");
            return doc.text();
        } catch (IOException e) {
            System.err.println("Error extracting text from HTML: " + e.getMessage());
            return "";
        }
    }

    /**
     * Extract text from PDF using PDFBox.
     * Maps to: extract_text_with_pdfplumber(path)
     */
    public String extractTextFromPdf(String path) {
        try (PDDocument document = Loader.loadPDF(new File(path))) {
            PDFTextStripper stripper = new PDFTextStripper();
            return stripper.getText(document);
        } catch (IOException e) {
            System.err.println("Error extracting text from PDF: " + e.getMessage());
            return "";
        }
    }

    /**
     * Process PDF by page with OCR fallback.
     * Maps to: process_pdf_by_page(full_path, rel_path, file_type, redis_client, tokenizer)
     */
    public List<ChunkWithMetadata> processPdfByPage(String fullPath, String relPath, String fileType) {
        List<ChunkWithMetadata> results = new ArrayList<>();
        
        try (PDDocument document = Loader.loadPDF(new File(fullPath))) {
            int pageCount = document.getNumberOfPages();
            
            for (int i = 1; i <= pageCount; i++) {
                PDFTextStripper stripper = new PDFTextStripper();
                stripper.setStartPage(i);
                stripper.setEndPage(i);
                String text = stripper.getText(document);
                
                int tokenCount = textProcessor.getTokenCount(text);
                if (text == null || text.isBlank() || textUtils.isGibberish(text) || textUtils.isLowQuality(text, tokenCount) || textUtils.isVisiblyCorrupt(text)) {
                    // Fallback to OCR
                    System.out.println("⚠️ Low quality or corrupt text on page " + i + " of " + relPath + " (" + tokenCount + " tokens), falling back to OCR");
                    String ocrText = ocrClientService.fallbackOcrForPage(fullPath, relPath, i);
                    if (ocrText != null && !ocrText.isBlank()) {
                        text = ocrText;
                    }
                }
                
                if (text != null && !text.isBlank()) {
                    List<String> chunks = textProcessor.splitDoc(text, relPath, fileType, i);
                    for (int j = 0; j < chunks.size(); j++) {
                        results.add(new ChunkWithMetadata(chunks.get(j), i, j));
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error processing PDF by page: " + e.getMessage());
        }
        
        return results;
    }

    public static record ChunkWithMetadata(String text, int pageNum, int chunkIndex) {}
}
