package com.selfhosted.rag.common.model;

import java.util.List;
import java.util.Map;

/**
 * Represents a chat query response.
 */
public class QueryResponse {
    private String answer;
    private List<Map<String, Object>> sources;
    private List<Map<String, Object>> chat_history;
    private Map<String, Object> debug;

    public QueryResponse() {}

    public QueryResponse(String answer, List<Map<String, Object>> sources, List<Map<String, Object>> chat_history, Map<String, Object> debug) {
        this.answer = answer;
        this.sources = sources;
        this.chat_history = chat_history;
        this.debug = debug;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String answer;
        private List<Map<String, Object>> sources;
        private List<Map<String, Object>> chat_history;
        private Map<String, Object> debug;

        public Builder answer(String answer) {
            this.answer = answer;
            return this;
        }

        public Builder sources(List<Map<String, Object>> sources) {
            this.sources = sources;
            return this;
        }

        public Builder chat_history(List<Map<String, Object>> chat_history) {
            this.chat_history = chat_history;
            return this;
        }

        public Builder debug(Map<String, Object> debug) {
            this.debug = debug;
            return this;
        }

        public QueryResponse build() {
            return new QueryResponse(answer, sources, chat_history, debug);
        }
    }

    // Getters and Setters
    public String getAnswer() { return answer; }
    public void setAnswer(String answer) { this.answer = answer; }
    public List<Map<String, Object>> getSources() { return sources; }
    public void setSources(List<Map<String, Object>> sources) { this.sources = sources; }
    public List<Map<String, Object>> getChat_history() { return chat_history; }
    public void setChat_history(List<Map<String, Object>> chat_history) { this.chat_history = chat_history; }
    public Map<String, Object> getDebug() { return debug; }
    public void setDebug(Map<String, Object> debug) { this.debug = debug; }
}
