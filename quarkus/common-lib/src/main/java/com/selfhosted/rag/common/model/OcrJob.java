package com.selfhosted.rag.common.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

/**
 * Represents an OCR job.
 * Maps to Python: models/data_models.py - OCRJob
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OcrJob {
    private String job_id;
    private String rel_path;
    private Integer page_num;
    private List<Integer> image_shape;
    private String image_dtype;
    private String image_base64;
    private String reply_key;
}
