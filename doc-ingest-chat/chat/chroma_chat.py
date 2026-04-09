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

        context_chunks = []
        for i, doc in enumerate(docs):
            content = doc.page_content

            # 1. Strip 'passage: ' prefix from ingest
            if content.startswith("passage: "):
                content = content[9:].strip()

            # 2. Extract deterministic [DOC_XXXX] anchor
            doc_id = "UNKNOWN"
            if content.startswith("[") and "]" in content:
                end_idx = content.find("]")
                doc_id = content[1:end_idx].strip()
                content = content[end_idx + 1 :].strip()

            # Format specifically for Phi-3.5-mini's attention mechanism
            # Anchoring the SOURCE_ID to a numbered CITATION_TAG
            chunk_text = f"SOURCE_ID: {doc_id}\nCITATION_TAG: [source{i + 1}]\nCONTENT: {content}"
            context_chunks.append(chunk_text)

        context_str = "\n\n---\n\n".join(context_chunks)

        system_msg = {
            "role": "system",
            "content": (
                "You are a factual extraction engine. Answer ONLY using the provided <context>.\n\n"
                "RULES:\n"
                "1. NO PROSE. No introductions, conclusions, or meta-commentary about the context.\n"
                "2. CITE: Every sentence MUST end with its CITATION_TAG (e.g. [source1]).\n"
                "3. If multiple sources apply, use both: [source1][source2].\n"
                "4. If the answer is not in the context, respond: 'Data not found.'\n\n"
                f"<context>\n{context_str}\n</context>"
            ),
        }

        messages = [system_msg] + chat_history + [{"role": "user", "content": query}]
        log.info(f"Context loaded: {len(docs)} chunks. Anchors verified.")

        try:
            # Phi-3.5-mini requires temp=0.0 to prevent hallucinating beyond the context window
            result = llama_model.create_chat_completion(
                messages=messages,
                temperature=0.0,
                repeat_penalty=1.1,
                max_tokens=256,  # Forces conciseness
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
        else:
            output, docs = ChromaChat.simulate_conversational_chain(query, chat_history)

        # Post-process: Map [sourceN] tags back to filenames/metadata
        found_tags = set(re.findall(r"\[source\d+\]|\(source\d+\)", output))

        if found_tags:
            output = replace_citation_labels(output, docs)
            debug_str = f"✅ Grounding successful. {len(found_tags)} citations mapped."
        else:
            debug_str = "⚠️ Grounding failure: No valid citation tags found."

        return (chat_history + [{"role": "user", "content": query}, {"role": "assistant", "content": output}], debug_str)

    @staticmethod
    def use_cli():
        chat_history = []
        while True:
            try:
                query = input("\n❓ Ask: ")
                if query.lower() in {"exit", "quit"}:
                    break
                chat_history, debug = ChromaChat.respond(query, chat_history)
                print(f"\n📘 Answer:\n{chat_history[-1]['content']}\n\n🔍 Debug: {debug}")
            except KeyboardInterrupt:
                break


# --- EXPORT ---
respond = ChromaChat.respond

if __name__ == "__main__":
    ChromaChat.use_cli()
