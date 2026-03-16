package com.selfhosted.rag.common.model;

import java.util.List;
import java.util.Map;

/**
 * Represents a chat query request.
 */
public record QueryRequest(
    String query,
    List<Map<String, Object>> chat_history
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String query;
        private List<Map<String, Object>> chat_history;

        public Builder query(String query) { this.query = query; return this; }
        public Builder chat_history(List<Map<String, Object>> chat_history) { this.chat_history = chat_history; return this; }

        public QueryRequest build() {
            return new QueryRequest(query, chat_history);
        }
    }
}
