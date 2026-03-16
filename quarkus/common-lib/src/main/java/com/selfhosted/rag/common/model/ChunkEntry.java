package com.selfhosted.rag.common.model;

/**
 * Represents a text chunk with metadata.
 * Maps to Python: models/data_models.py - ChunkEntry
 */
public class ChunkEntry {
    private String chunk;
    private String id;
    private String source_file;
    private String type;
    private String hash;
    private String engine;
    private Integer page;
    private Integer chunk_index;

    public ChunkEntry() {}

    public ChunkEntry(String chunk, String id, String source_file, String type, String hash, String engine, Integer page, Integer chunk_index) {
        this.chunk = chunk;
        this.id = id;
        this.source_file = source_file;
        this.type = type;
        this.hash = hash;
        this.engine = engine;
        this.page = page;
        this.chunk_index = chunk_index;
    }

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

    // Getters and Setters
    public String getChunk() { return chunk; }
    public void setChunk(String chunk) { this.chunk = chunk; }
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getSource_file() { return source_file; }
    public void setSource_file(String source_file) { this.source_file = source_file; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getHash() { return hash; }
    public void setHash(String hash) { this.hash = hash; }
    public String getEngine() { return engine; }
    public void setEngine(String engine) { this.engine = engine; }
    public Integer getPage() { return page; }
    public void setPage(Integer page) { this.page = page; }
    public Integer getChunk_index() { return chunk_index; }
    public void setChunk_index(Integer chunk_index) { this.chunk_index = chunk_index; }
}
