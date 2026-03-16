package com.selfhosted.rag.common.model;

/**
 * Represents an OCR response.
 * Maps to Python: models/data_models.py - OCRResponse
 */
public class OcrResponse {
    private String text;
    private String rel_path;
    private Integer page_num;
    private String engine;
    private String job_id;

    public OcrResponse() {}

    public OcrResponse(String text, String rel_path, Integer page_num, String engine, String job_id) {
        this.text = text;
        this.rel_path = rel_path;
        this.page_num = page_num;
        this.engine = engine;
        this.job_id = job_id;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String text;
        private String rel_path;
        private Integer page_num;
        private String engine;
        private String job_id;

        public Builder text(String text) { this.text = text; return this; }
        public Builder rel_path(String rel_path) { this.rel_path = rel_path; return this; }
        public Builder page_num(Integer page_num) { this.page_num = page_num; return this; }
        public Builder engine(String engine) { this.engine = engine; return this; }
        public Builder job_id(String job_id) { this.job_id = job_id; return this; }

        public OcrResponse build() {
            return new OcrResponse(text, rel_path, page_num, engine, job_id);
        }
    }

    // Getters and Setters
    public String getText() { return text; }
    public void setText(String text) { this.text = text; }
    public String getRel_path() { return rel_path; }
    public void setRel_path(String rel_path) { this.rel_path = rel_path; }
    public Integer getPage_num() { return page_num; }
    public void setPage_num(Integer page_num) { this.page_num = page_num; }
    public String getEngine() { return engine; }
    public void setEngine(String engine) { this.engine = engine; }
    public String getJob_id() { return job_id; }
    public void setJob_id(String job_id) { this.job_id = job_id; }
}
