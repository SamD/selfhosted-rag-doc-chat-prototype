package com.selfhosted.rag.api.resource;

import com.selfhosted.rag.api.service.RagService;
import com.selfhosted.rag.common.model.QueryRequest;
import com.selfhosted.rag.common.model.QueryResponse;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;

import java.util.HashMap;
import java.util.Map;

/**
 * REST API resource for RAG queries.
 *
 * Maps to Python: api/endpoints.py
 */
@Path("/api/v1")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class ChatResource {

    @Inject
    RagService ragService;

    @GET
    @Path("/health")
    public Map<String, String> health() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "healthy");
        response.put("message", "API is running");
        return response;
    }

    @GET
    @Path("/status")
    public Map<String, Object> status() {
        // Return operational status
        Map<String, Object> response = new HashMap<>();
        response.put("status", "operational");
        response.put("collection_count", 0);
        response.put("model_info", new HashMap<>());
        return response;
    }

    @POST
    @Path("/query")
    public QueryResponse query(QueryRequest req) {
        System.out.println("Received query: " + req.query());
        Map<String, Object> result = ragService.answerQuery(req.query(), req.chat_history());
        
        return QueryResponse.builder()
                .answer((String) result.get("answer"))
                .sources((java.util.List) result.get("sources"))
                .chat_history((java.util.List) result.get("chat_history"))
                .debug(Map.of("message", result.get("debug")))
                .build();
    }
}
