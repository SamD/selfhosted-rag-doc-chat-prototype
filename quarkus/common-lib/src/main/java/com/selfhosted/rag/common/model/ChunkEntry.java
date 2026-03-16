package com.selfhosted.rag.common.model;

/**
 * Represents a text chunk with metadata.
 * Maps to Python: models/data_models.py - ChunkEntry
 * 
 * Using Record for Java 25 parity and boilerplate reduction.
 */
public record ChunkEntry(
    String chunk,
    String id,
    String source_file,
    String type,
    String hash,
    String engine,
    Integer page,
    Integer chunk_index
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String chunk;
        private String id;
        private String source_file;
        private String type;
        private String hash;
        private String engine;
        private Integer page;
        private Integer chunk_index;

        public Builder chunk(String chunk) { this.chunk = chunk; return this; }
        public Builder id(String id) { this.id = id; return this; }
        public Builder source_file(String source_file) { this.source_file = source_file; return this; }
        public Builder type(String type) { this.type = type; return this; }
        public Builder hash(String hash) { this.hash = hash; return this; }
        public Builder engine(String engine) { this.engine = engine; return this; }
        public Builder page(Integer page) { this.page = page; return this; }
        public Builder chunk_index(Integer chunk_index) { this.chunk_index = chunk_index; return this; }

        public ChunkEntry build() {
            return new ChunkEntry(chunk, id, source_file, type, hash, engine, page, chunk_index);
        }
    }
}
