package com.selfhosted.rag.common.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a text chunk with metadata.
 * Maps to Python: models/data_models.py - ChunkEntry
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ChunkEntry {
    private String chunk;
    private String id;
    private String source_file;
    private String type;
    private String hash;
    private String engine;
    private Integer page;
    private Integer chunk_index;
}
