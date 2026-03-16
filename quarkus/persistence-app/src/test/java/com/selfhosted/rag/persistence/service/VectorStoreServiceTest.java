package com.selfhosted.rag.persistence.service;

import com.selfhosted.rag.common.config.AppConfig;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import io.quarkus.test.InjectMock;
import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.*;

@QuarkusTest
public class VectorStoreServiceTest {

    @Inject
    VectorStoreService vectorStoreService;

    @InjectMock
    EmbeddingStore<TextSegment> embeddingStore;

    @InjectMock
    EmbeddingModel embeddingModel;

    @InjectMock
    AppConfig appConfig;

    @Test
    void testAddTexts() {
        // Mock embedding model response
        when(embeddingModel.embedAll(anyList())).thenReturn(Response.from(List.of()));
        
        vectorStoreService.addTexts(
                List.of("Hello world"),
                List.of(Map.of("source", "test.txt")),
                List.of("id-1")
        );
        
        verify(embeddingModel).embedAll(anyList());
        verify(embeddingStore).addAll(anyList(), anyList());
    }
}
