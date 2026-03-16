package com.selfhosted.rag.common.model;

import java.util.List;
import java.util.Map;

/**
 * Represents a chat query request.
 */
public class QueryRequest {
    private String query;
    private List<Map<String, Object>> chat_history;

    public QueryRequest() {}

    public QueryRequest(String query, List<Map<String, Object>> chat_history) {
        this.query = query;
        this.chat_history = chat_history;
    }

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

    public String getQuery() {
        return query;
    }

    public void setQuery(String query) {
        this.query = query;
    }

    public List<Map<String, Object>> getChat_history() {
        return chat_history;
    }

    public void setChat_history(List<Map<String, Object>> chat_history) {
        this.chat_history = chat_history;
    }
}
