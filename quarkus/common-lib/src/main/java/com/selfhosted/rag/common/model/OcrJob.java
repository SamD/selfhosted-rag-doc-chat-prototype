package com.selfhosted.rag.common.model;

import java.util.List;

/**
 * Represents an OCR job.
 * Maps to Python: models/data_models.py - OCRJob
 */
public record OcrJob(
    String job_id,
    String rel_path,
    Integer page_num,
    List<Integer> image_shape,
    String image_dtype,
    String image_base64,
    String reply_key
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String job_id;
        private String rel_path;
        private Integer page_num;
        private List<Integer> image_shape;
        private String image_dtype;
        private String image_base64;
        private String reply_key;

        public Builder job_id(String job_id) { this.job_id = job_id; return this; }
        public Builder rel_path(String rel_path) { this.rel_path = rel_path; return this; }
        public Builder page_num(Integer page_num) { this.page_num = page_num; return this; }
        public Builder image_shape(List<Integer> image_shape) { this.image_shape = image_shape; return this; }
        public Builder image_dtype(String image_dtype) { this.image_dtype = image_dtype; return this; }
        public Builder image_base64(String image_base64) { this.image_base64 = image_base64; return this; }
        public Builder reply_key(String reply_key) { this.reply_key = reply_key; return this; }

        public OcrJob build() {
            return new OcrJob(job_id, rel_path, page_num, image_shape, image_dtype, image_base64, reply_key);
        }
    }
}
