package com.selfhosted.rag.common.model;

import java.util.List;
import java.util.Map;

/**
 * Represents a chat query response.
 */
public record QueryResponse(
    String answer,
    List<Map<String, Object>> sources,
    List<Map<String, Object>> chat_history,
    Map<String, Object> debug
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String answer;
        private List<Map<String, Object>> sources;
        private List<Map<String, Object>> chat_history;
        private Map<String, Object> debug;

        public Builder answer(String answer) { this.answer = answer; return this; }
        public Builder sources(List<Map<String, Object>> sources) { this.sources = sources; return this; }
        public Builder chat_history(List<Map<String, Object>> chat_history) { this.chat_history = chat_history; return this; }
        public Builder debug(Map<String, Object> debug) { this.debug = debug; return this; }

        public QueryResponse build() {
            return new QueryResponse(answer, sources, chat_history, debug);
        }
    }
}
