package com.selfhosted.rag.ingestion.processor;

import com.selfhosted.rag.common.utils.TextUtils;
import io.quarkus.test.InjectMock;
import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

@QuarkusTest
public class DocumentProcessorTest {

    @Inject
    DocumentProcessor documentProcessor;

    @InjectMock
    TextUtils textUtils;

    @InjectMock
    TextProcessor textProcessor;

    @InjectMock
    OcrClientService ocrClientService;

    @Test
    void testExtractTextFromHtml(@TempDir Path tempDir) throws IOException {
        Path htmlFile = tempDir.resolve("test.html");
        Files.writeString(htmlFile, "<html><body><h1>Hello</h1><p>World!</p></body></html>");
        
        String text = documentProcessor.extractTextFromHtml(htmlFile.toString());
        
        assertThat(text).contains("Hello");
        assertThat(text).contains("World!");
    }

    @Test
    void testExtractTextFromEmptyHtml(@TempDir Path tempDir) throws IOException {
        Path htmlFile = tempDir.resolve("empty.html");
        Files.writeString(htmlFile, "");
        
        String text = documentProcessor.extractTextFromHtml(htmlFile.toString());
        
        assertThat(text).isEmpty();
    }
}
