package com.selfhosted.rag.common.model;

import java.util.List;

/**
 * Represents an OCR job.
 * Maps to Python: models/data_models.py - OCRJob
 */
public class OcrJob {
    private String job_id;
    private String rel_path;
    private Integer page_num;
    private List<Integer> image_shape;
    private String image_dtype;
    private String image_base64;
    private String reply_key;

    public OcrJob() {}

    public OcrJob(String job_id, String rel_path, Integer page_num, List<Integer> image_shape, String image_dtype, String image_base64, String reply_key) {
        this.job_id = job_id;
        this.rel_path = rel_path;
        this.page_num = page_num;
        this.image_shape = image_shape;
        this.image_dtype = image_dtype;
        this.image_base64 = image_base64;
        this.reply_key = reply_key;
    }

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

    // Getters and Setters
    public String getJob_id() { return job_id; }
    public void setJob_id(String job_id) { this.job_id = job_id; }
    public String getRel_path() { return rel_path; }
    public void setRel_path(String rel_path) { this.rel_path = rel_path; }
    public Integer getPage_num() { return page_num; }
    public void setPage_num(Integer page_num) { this.page_num = page_num; }
    public List<Integer> getImage_shape() { return image_shape; }
    public void setImage_shape(List<Integer> image_shape) { this.image_shape = image_shape; }
    public String getImage_dtype() { return image_dtype; }
    public void setImage_dtype(String image_dtype) { this.image_dtype = image_dtype; }
    public String getImage_base64() { return image_base64; }
    public void setImage_base64(String image_base64) { this.image_base64 = image_base64; }
    public String getReply_key() { return reply_key; }
    public void setReply_key(String reply_key) { this.reply_key = reply_key; }
}
