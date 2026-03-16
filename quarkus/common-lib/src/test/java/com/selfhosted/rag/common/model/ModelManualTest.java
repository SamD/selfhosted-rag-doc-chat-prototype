package com.selfhosted.rag.common.model;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.Map;

public class ModelManualTest {

    @Test
    public void testOcrJobManual() {
        OcrJob job = OcrJob.builder()
                .job_id("test-id")
                .rel_path("path/to/file")
                .page_num(1)
                .build();

        assertThat(job.job_id()).isEqualTo("test-id");
        assertThat(job.rel_path()).isEqualTo("path/to/file");
        assertThat(job.page_num()).isEqualTo(1);
    }

    @Test
    public void testChunkEntryManual() {
        ChunkEntry entry = ChunkEntry.builder()
                .chunk("some text")
                .id("chunk-1")
                .build();

        assertThat(entry.chunk()).isEqualTo("some text");
        assertThat(entry.id()).isEqualTo("chunk-1");
    }

    @Test
    public void testQueryRequestManual() {
        QueryRequest request = QueryRequest.builder()
                .query("what is rag?")
                .chat_history(List.of(Map.of("role", "user", "content", "hello")))
                .build();

        assertThat(request.query()).isEqualTo("what is rag?");
        assertThat(request.chat_history()).hasSize(1);
    }

    @Test
    public void testFileEndMessageManual() {
        FileEndMessage message = FileEndMessage.builder()
                .source_file("doc.pdf")
                .build();

        assertThat(message.source_file()).isEqualTo("doc.pdf");
        assertThat(message.type()).isEqualTo("file_end"); // default value
        assertThat(message.expected_chunks()).isEqualTo(0); // default value
    }
}
