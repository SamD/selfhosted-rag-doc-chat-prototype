package com.selfhosted.rag.persistence.service;

import com.selfhosted.rag.common.config.AppConfig;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Vector database service for Qdrant operations using Quarkus LangChain4j.
 *
 * Maps to Python: services/database.py - DatabaseService
 */
@ApplicationScoped
public class VectorStoreService {

    @Inject
    EmbeddingStore<TextSegment> embeddingStore;

    @Inject
    EmbeddingModel embeddingModel;

    @Inject
    AppConfig appConfig;

    /**
     * Add texts to vector store with metadata.
     */
    public void addTexts(List<String> texts, List<Map<String, Object>> metadatas, List<String> ids) {
        List<TextSegment> segments = new ArrayList<>();
        for (int i = 0; i < texts.size(); i++) {
            Metadata metadata = new Metadata();
            if (metadatas != null && i < metadatas.size()) {
                metadatas.get(i).forEach((k, v) -> {
                    if (v instanceof String) metadata.put(k, (String) v);
                    else if (v instanceof Integer) metadata.put(k, (Integer) v);
                    else if (v instanceof Double) metadata.put(k, (Double) v);
                    else if (v instanceof Float) metadata.put(k, (Float) v);
                });
            }
            // Add ID to metadata if provided
            if (ids != null && i < ids.size()) {
                metadata.put("id", ids.get(i));
            }
            segments.add(TextSegment.from(texts.get(i), metadata));
        }

        // LangChain4j EmbeddingStore.addAll with EmbeddingModel will handle embedding automatically
        // in some stores, or we might need to embed explicitly if using a raw store.
        // The Quarkus extension usually injects an EmbeddingStore that handles it if an EmbeddingModel is available.
        embeddingStore.addAll(embeddingModel.embedAll(segments).content(), segments);
    }

    /**
     * Delete documents by source file.
     * Maps to Python logic: db.delete(where={"source_file": source_file})
     */
    public void deleteBySourceFile(String sourceFile) {
        // LangChain4j doesn't have a standard 'delete by metadata' in EmbeddingStore yet
        // but Qdrant usually supports it through the client.
        System.out.println("⚠️ Delete by source_file not fully implemented in Quarkus yet for " + sourceFile);
    }

    public long getCollectionCount() {
        // Count implementation depends on specific EmbeddingStore
        return -1;
    }
}
