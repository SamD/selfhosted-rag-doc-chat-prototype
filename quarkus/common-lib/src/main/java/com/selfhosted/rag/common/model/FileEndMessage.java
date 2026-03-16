package com.selfhosted.rag.common.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a file end message.
 * Maps to Python: models/data_models.py - FileEndMessage
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FileEndMessage {
    @Builder.Default
    private String type = "file_end";
    @Builder.Default
    private String source_file = "";
    @Builder.Default
    private Integer expected_chunks = 0;
}
