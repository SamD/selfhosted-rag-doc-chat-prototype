package com.selfhosted.rag.api.resource;

import com.selfhosted.rag.api.service.RagService;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import io.quarkus.test.InjectMock;
import io.quarkus.test.junit.QuarkusTest;
import io.restassured.http.ContentType;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static io.restassured.RestAssured.given;
import static org.hamcrest.CoreMatchers.is;

@QuarkusTest
public class ChatResourceTest {

    @InjectMock
    RagService ragService;

    @InjectMock
    com.selfhosted.rag.common.config.AppConfig config;

    @InjectMock
    ChatLanguageModel chatModel;

    @InjectMock
    ContentRetriever contentRetriever;

    @Test
    public void testHealthEndpoint() {
        given()
          .when().get("/api/v1/health")
          .then()
             .statusCode(200)
             .body("status", is("healthy"));
    }

    @Test
    public void testStatusEndpoint() {
        given()
          .when().get("/api/v1/status")
          .then()
             .statusCode(200)
             .body("status", is("operational"));
    }
}
