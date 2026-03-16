package com.selfhosted.rag.common.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;
import java.util.Map;

/**
 * Represents a chat query response.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class QueryResponse {
    private String answer;
    private List<Map<String, Object>> sources;
    private List<Map<String, Object>> chat_history;
    private Map<String, Object> debug;
}
