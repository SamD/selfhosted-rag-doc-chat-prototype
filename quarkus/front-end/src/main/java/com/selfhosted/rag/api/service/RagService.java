package com.selfhosted.rag.api.service;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingStore;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Inject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * RAG service implementation for the chat system.
 * Replicates logic from Python: chat/chroma_chat.py, utils/llm_setup.py, and services/rag_service.py
 */
@ApplicationScoped
public class RagService {

    @Inject
    ChatLanguageModel chatModel;

    @Inject
    ContentRetriever contentRetriever;

    @Produces
    @ApplicationScoped
    public ContentRetriever createRetriever(EmbeddingStore<TextSegment> embeddingStore, EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(20)
                .build();
    }

    /**
     * Replicates Python's RagService.answer_query()
     */
    public Map<String, Object> answerQuery(String query, List<Map<String, Object>> chatHistory) {
        // Updated logic to match Python's respond() returns
        Map<String, Object> respondResult = respond(query, chatHistory);
        
        List<Map<String, Object>> updatedHistory = (List<Map<String, Object>>) respondResult.get("chat_history");
        String debug = (String) respondResult.get("debug");
        String answer = (String) updatedHistory.get(updatedHistory.size() - 1).get("content");

        Map<String, Object> response = new HashMap<>();
        response.put("answer", answer);
        response.put("chat_history", updatedHistory);
        response.put("debug", debug);
        response.put("sources", respondResult.get("sources"));

        return response;
    }

    /**
     * Replicates Python's ChromaChat.respond()
     */
    private Map<String, Object> respond(String query, List<Map<String, Object>> chatHistory) {
        // 1. Simulate conversational chain (retrieval + LLM call)
        Map<String, Object> chainResult = simulateConversationalChain(query, chatHistory);
        String output = (String) chainResult.get("answer");
        List<TextSegment> docs = (List<TextSegment>) chainResult.get("docs");

        // 2. Citation validation logic
        // Replicates Python: found_bracket_tags, found_paren_tags, validation
        Set<String> foundBracketTags = findTags(output, "\\[source\\d+\\]");
        Set<String> foundParenTags = findTags(output, "\\(source\\d+\\)");
        
        Set<String> foundTags = new java.util.HashSet<>(foundBracketTags);
        foundTags.addAll(foundParenTags);

        Set<String> expectedTagsBracket = IntStream.range(0, docs.size())
                .mapToObj(i -> "[source" + (i + 1) + "]")
                .collect(Collectors.toSet());
        Set<String> expectedTagsParen = IntStream.range(0, docs.size())
                .mapToObj(i -> "(source" + (i + 1) + ")")
                .collect(Collectors.toSet());

        boolean citationTagsPresent = false;
        for (int i = 0; i < docs.size(); i++) {
            int sourceNum = i + 1;
            if (foundTags.contains("[source" + sourceNum + "]") || foundTags.contains("(source" + sourceNum + ")")) {
                citationTagsPresent = true;
                break;
            }
        }

        Set<String> allExpectedTags = new java.util.HashSet<>(expectedTagsBracket);
        allExpectedTags.addAll(expectedTagsParen);
        
        Set<String> invalidTags = new java.util.HashSet<>(foundTags);
        invalidTags.removeAll(allExpectedTags);

        String debugStr;
        if (!invalidTags.isEmpty()) {
            if (!citationTagsPresent) {
                debugStr = "❌ Invalid citation tags found — response rejected.";
            } else {
                debugStr = "✅ Citation tags were used (with some hallucinations: " + invalidTags + ").";
            }
        } else {
            debugStr = citationTagsPresent ? "✅ Citation tags were used." : "⚠️ No valid citation tags found in model output.";
        }

        // 3. Replace labels if valid tags present
        String finalOutput = output;
        if (citationTagsPresent) {
            finalOutput = replaceCitationLabels(output, docs);
        }

        // 4. Update chat history
        List<Map<String, Object>> updatedHistory = new ArrayList<>(chatHistory != null ? chatHistory : new ArrayList<>());
        updatedHistory.add(Map.of("role", "user", "content", query));
        updatedHistory.add(Map.of("role", "assistant", "content", finalOutput));

        Map<String, Object> result = new HashMap<>();
        result.put("chat_history", updatedHistory);
        result.put("debug", debugStr);
        result.put("sources", formatSourcesForResponse(docs));
        return result;
    }

    /**
     * Replicates Python's ChromaChat.simulate_conversational_chain()
     */
    private Map<String, Object> simulateConversationalChain(String query, List<Map<String, Object>> chatHistory) {
        Query q = Query.from("query: " + query);
        List<Content> retrievedContent = contentRetriever.retrieve(q);
        List<TextSegment> docs = retrievedContent.stream()
                .map(Content::textSegment)
                .collect(Collectors.toList());

        String context = formatChunksWithCitations(docs);
        String citationList = docs.isEmpty() ? "" : 
                IntStream.range(0, docs.size())
                        .mapToObj(i -> "[source" + (i + 1) + "]")
                        .collect(Collectors.joining(", "));

        String systemPromptTemplate = """
                You are a factual and precise assistant.
                
                **Rules:**
                1. Only answer using information found in <context> and the prior messages.
                2. Do not speculate, summarize loosely, or add opinions. Do not use phrases like 'however', 'some believe', 'it is important to note', unless they appear **verbatim** in the context.
                3. You MUST include all relevant and only relevant citation tags like {{citationList}} inline after each factual statement. Do not invent or omit tags.
                4. Cite multiple chunks if more than one supports a fact. Do not cherry-pick.
                5. If no relevant context exists, respond: 'Not enough information in the provided sources.'
                
                <context>
                {{context}}
                </context>
                """;
        
        Map<String, Object> variables = new HashMap<>();
        variables.put("citationList", citationList);
        variables.put("context", context);
        Prompt prompt = PromptTemplate.from(systemPromptTemplate).apply(variables);
        
        List<ChatMessage> messages = new ArrayList<>();
        messages.add(SystemMessage.from(prompt.text()));
        
        if (chatHistory != null) {
            for (Map<String, Object> msg : chatHistory) {
                String role = (String) msg.get("role");
                String content = (String) msg.get("content");
                if ("user".equalsIgnoreCase(role)) {
                    messages.add(UserMessage.from(content));
                } else if ("assistant".equalsIgnoreCase(role)) {
                    messages.add(AiMessage.from(content));
                }
            }
        }
        messages.add(UserMessage.from(query));

        Response<AiMessage> aiResponse = chatModel.generate(messages);
        
        Map<String, Object> result = new HashMap<>();
        result.put("answer", aiResponse.content().text());
        result.put("docs", docs);
        return result;
    }

    private Set<String> findTags(String text, String regex) {
        Set<String> tags = new java.util.HashSet<>();
        Matcher matcher = Pattern.compile(regex).matcher(text);
        while (matcher.find()) {
            tags.add(matcher.group());
        }
        return tags;
    }

    private String formatChunksWithCitations(List<TextSegment> docs) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < docs.size(); i++) {
            TextSegment doc = docs.get(i);
            String citation = "[source" + (i + 1) + "]";
            String source = doc.metadata().getString("source_file");
            if (source == null) source = "unknown";
            Integer page = doc.metadata().getInteger("page");
            String pageStr = (page != null) ? page.toString() : "N/A";
            
            sb.append(citation).append(" (Source: ").append(source).append(", Page: ").append(pageStr).append(")\n---\n")
              .append(doc.text().trim()).append("\n\n");
        }
        return sb.toString().trim();
    }

    private String replaceCitationLabels(String output, List<TextSegment> docs) {
        Map<String, Boolean> seen = new LinkedHashMap<>();
        String currentOutput = output;

        for (int i = 0; i < docs.size(); i++) {
            int sourceNum = i + 1;
            String bracketLabel = "[source" + sourceNum + "]";
            String parenLabel = "(source" + sourceNum + ")";

            TextSegment doc = docs.get(i);
            String source = doc.metadata().getString("source_file");
            if (source == null) source = "unknown";
            Integer page = doc.metadata().getInteger("page");

            String resolved;
            if (page != null && page >= 0) {
                resolved = "[" + source + ", page " + page + "]";
            } else {
                resolved = "[" + source + "]";
            }

            seen.put(resolved, true);

            currentOutput = currentOutput.replace(bracketLabel, resolved);
            currentOutput = currentOutput.replace(parenLabel, resolved);
        }

        if (!seen.isEmpty()) {
            currentOutput += "\n\n---\n**Sources:** " + String.join(", ", seen.keySet());
        }

        return currentOutput.trim();
    }

    private List<Map<String, Object>> formatSourcesForResponse(List<TextSegment> docs) {
        return docs.stream().map(doc -> {
            Map<String, Object> info = new HashMap<>();
            info.put("text", doc.text());
            info.put("source", doc.metadata().getString("source_file"));
            info.put("page", doc.metadata().getInteger("page"));
            return info;
        }).collect(Collectors.toList());
    }
}
