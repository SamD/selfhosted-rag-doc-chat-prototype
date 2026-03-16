package com.selfhosted.rag.common.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents an OCR response.
 * Maps to Python: models/data_models.py - OCRResponse
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OcrResponse {
    private String text;
    private String rel_path;
    private Integer page_num;
    private String engine;
    private String job_id;
}
