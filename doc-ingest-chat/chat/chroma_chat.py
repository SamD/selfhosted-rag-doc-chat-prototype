import logging
import re

from config.settings import USE_OLLAMA
from utils.chat_utils import replace_citation_labels
from utils.llm_setup import get_chain_or_llama, get_retriever, get_vectorstore

log = logging.getLogger(__name__)


class ChromaChat:
    """Core chat logic for conversational retrieval and response as static methods."""

    @staticmethod
    def simulate_conversational_chain(query, chat_history):
        vectorstore = get_vectorstore()
        retriever = get_retriever(vectorstore)
        _, llama_model = get_chain_or_llama(retriever)

        # Retrieval query: Ensure the 'query: ' prefix matches the 'passage: ' used in ingest
        docs = retriever.invoke(f"query: {query}")

        # --- DEDUPLICATE CONTEXT ---
        # Ensure we only send unique facts to the LLM
        unique_contents = {}
        context_chunks = []

        for i, doc in enumerate(docs):
            content = doc.page_content
            if content in unique_contents:
                continue
            unique_contents[content] = True

            # 1. Extract deterministic [DOC_XXXX] anchor using UNDERSCORE-AWARE regex
            doc_id = "UNKNOWN"
            anchor_match = re.search(r"\[(DOC_[a-fA-F0-9_]+)\]", content)
            if anchor_match:
                doc_id = anchor_match.group(1)
                # Clean content for the LLM
                content = re.sub(r"^passage: \[DOC_[a-fA-F0-9_]+\]\s*", "", content)

            chunk_text = f"SOURCE_ID: {doc_id}\nCITATION_TAG: [source{len(context_chunks) + 1}]\nCONTENT: {content}"
            context_chunks.append(chunk_text)

        context_str = "\n\n---\n\n".join(context_chunks)

        # 2. UNIFIED USER PROMPT (No System Message for 0.5B focus)
        user_prompt = (
            "You are a factual extraction engine. Answer ONLY using the provided <context>.\n\n"
            "RULES:\n"
            "1. NO PROSE. Do not add introductions or conclusions.\n"
            "2. CITE: Every single sentence MUST end with its CITATION_TAG like [source1].\n"
            "3. If multiple sources apply, use both: [source1][source2].\n"
            "4. If the answer is not in the context, respond: 'Data not found.'\n\n"
            "<context>\n"
            f"{context_str}\n"
            "</context>\n\n"
            f"QUESTION: {query}"
        )

        # Build messages: History + Unified User Prompt
        messages = chat_history + [{"role": "user", "content": user_prompt}]

        log.info(f"🛰️ Context loaded: {len(context_chunks)} unique chunks. Anchors: {[c.splitlines()[0] for c in context_chunks]}")

        try:
            result = llama_model.create_chat_completion(
                messages=messages,
            )
            return result["choices"][0]["message"]["content"].strip(), docs
        except Exception as e:
            log.error(f"LLM call failed: {e}", exc_info=True)
            raise

    @staticmethod
    def respond(query, chat_history):
        if USE_OLLAMA:
            vectorstore = get_vectorstore()
            retriever = get_retriever(vectorstore)
            chain, _ = get_chain_or_llama(retriever)
            result = chain.invoke({"question": query, "chat_history": [(m["content"], a["content"]) for m, a in zip(chat_history[::2], chat_history[1::2])]})
            output = result["answer"].strip()
            docs = result.get("source_documents", [])
            new_history = chat_history + [{"role": "user", "content": query}, {"role": "assistant", "content": output}]
        else:
            # We strip the current query from history because simulate_conversational_chain
            # adds it as the unified 'user_prompt' at the end.
            output, docs = ChromaChat.simulate_conversational_chain(query, chat_history)
            new_history = chat_history + [{"role": "user", "content": query}, {"role": "assistant", "content": output}]

        # Post-process: Map [sourceN] tags back to filenames/metadata
        found_tags = set(re.findall(r"\[source\d+\]|\(source\d+\)", output))

        if found_tags:
            output_with_citations = replace_citation_labels(output, docs)
            new_history[-1]["content"] = output_with_citations
            debug_str = f"✅ Grounding successful. {len(found_tags)} citations mapped."
        else:
            debug_str = "⚠️ Grounding failure: No valid citation tags found."

        return (new_history, debug_str)


# --- EXPORT ---
respond = ChromaChat.respond

if __name__ == "__main__":
    ChromaChat.use_cli()
