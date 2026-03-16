package com.selfhosted.rag.common.model;

/**
 * Represents a file end message.
 * Maps to Python: models/data_models.py - FileEndMessage
 */
public record FileEndMessage(
    String type,
    String source_file,
    Integer expected_chunks
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String type = "file_end";
        private String source_file = "";
        private Integer expected_chunks = 0;

        public Builder type(String type) { this.type = type; return this; }
        public Builder source_file(String source_file) { this.source_file = source_file; return this; }
        public Builder expected_chunks(Integer expected_chunks) { this.expected_chunks = expected_chunks; return this; }

        public FileEndMessage build() {
            return new FileEndMessage(type, source_file, expected_chunks);
        }
    }
}
