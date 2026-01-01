"""
Core chat logic for conversational retrieval and response.
"""

#!/usr/bin/env python
import re

from config.settings import USE_OLLAMA
from utils.chat_utils import format_chunks_with_citations, get_env_boolean, replace_citation_labels
from utils.llm_setup import get_chain_or_llama, get_retriever, get_vectorstore
from utils.logging_config import setup_logging

log = setup_logging("chroma_chat.log")

vectorstore = get_vectorstore()
log.info(f"Vectorstore contains {vectorstore.get_collection_count()} documents")
retriever = get_retriever(vectorstore)

chain, llama_model = get_chain_or_llama(retriever)


class ChromaChat:
    """Core chat logic for conversational retrieval and response as static methods."""

    @staticmethod
    def simulate_conversational_chain(query, chat_history):
        docs = retriever.invoke(f"query: {query}")
        context_chunks = format_chunks_with_citations(docs)
        context_str = "\n\n".join(context_chunks)
        citation_list = ", ".join(f"[source{i + 1}]" for i in range(len(docs)))
        system_msg = {
            "role": "system",
            "content": (
                "You are a factual and precise assistant.\n\n"
                "**Rules:**\n"
                "1. Only answer using information found in <context> and the prior messages.\n"
                "2. Do not speculate, summarize loosely, or add opinions. Do not use phrases like 'however', 'some believe', 'it is important to note', unless they appear **verbatim** in the context.\n"
                f"3. You MUST include all relevant and only relevant citation tags like {citation_list} inline after each factual statement. Do not invent or omit tags.\n"
                "4. Cite multiple chunks if more than one supports a fact. Do not cherry-pick.\n"
                "5. If no relevant context exists, respond: 'Not enough information in the provided sources.'\n"
                f"\n<context>\n{context_str}\n</context>"
            ),
        }
        messages = [system_msg] + chat_history + [{"role": "user", "content": query}]
        log.info(f"Calling LLM with {len(messages)} messages")
        try:
            result = llama_model.create_chat_completion(messages=messages)
            log.info("LLM response received")
            return result["choices"][0]["message"]["content"].strip(), docs
        except Exception as e:
            log.error(f"LLM call failed: {e}", exc_info=True)
            raise

    @staticmethod
    def respond(query, chat_history):
        if USE_OLLAMA:
            result = chain.invoke({"question": query, "chat_history": [(m["content"], a["content"]) for m, a in zip(chat_history[::2], chat_history[1::2])]})
            output = result["answer"].strip()
            docs = result.get("source_documents", [])
        else:
            output, docs = ChromaChat.simulate_conversational_chain(query, chat_history)
        for i, doc in enumerate(docs):
            log.debug(f"Expecting [source{i + 1}] for: {doc.metadata.get('source_file')} | Page {doc.metadata.get('page')}")
            log.debug(f"\n🧪 Raw model output:\n{output}")
        log.debug("\n📄 CONTEXT:\n" + "\n\n".join(format_chunks_with_citations(docs)))
        log.debug(f"\n⚙️ MODEL OUTPUT (raw):\n{output}")
        # Find both [sourceX] and (sourceX) formats
        found_bracket_tags = set(re.findall(r"\[source\d+\]", output))
        found_paren_tags = set(re.findall(r"\(source\d+\)", output))
        found_tags = found_bracket_tags | found_paren_tags
        # Check if any expected tags (in either format) are found
        expected_tags_bracket = {f"[source{i + 1}]" for i in range(len(docs))}
        expected_tags_paren = {f"(source{i + 1})" for i in range(len(docs))}
        # A citation is valid if we find the bracket or paren version
        citation_tags_present = False
        for i in range(len(docs)):
            source_num = i + 1
            bracket_tag = f"[source{source_num}]"
            paren_tag = f"(source{source_num})"
            if bracket_tag in found_tags or paren_tag in found_tags:
                citation_tags_present = True
                log.debug(f"🔎 Found citation for source {source_num}: {bracket_tag if bracket_tag in found_tags else paren_tag}")
        # Check for invalid/unexpected tags
        all_expected_tags = expected_tags_bracket | expected_tags_paren
        invalid_tags = found_tags - all_expected_tags
        if invalid_tags:
            log.warning(f"🚫 Model hallucinated unexpected citation tags: {sorted(invalid_tags)}")
            # Don't reject if we also have valid tags
            if not citation_tags_present:
                debug_str = "❌ Invalid citation tags found — response rejected."
                log.warning("⚠️ Skipping citation replacement — only invalid tags found.")
                return (chat_history + [{"role": "user", "content": query}, {"role": "assistant", "content": output}], debug_str)
        debug_str = "✅ Citation tags were used." if citation_tags_present else "⚠️ No valid citation tags found in model output."
        if citation_tags_present:
            output = replace_citation_labels(output, docs)
        else:
            log.warning("⚠️ Skipping citation replacement — no valid tags matched.")
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
                print("\n📘 Answer:\n" + chat_history[-1]["content"])
                print("\n🔍 Debug Info:\n" + debug)
            except KeyboardInterrupt:
                break

    @staticmethod
    def use_fast_api():
        pass


if __name__ == "__main__":
    if get_env_boolean("USE_FASTAPI", False):
        ChromaChat.use_fast_api()
    else:
        ChromaChat.use_cli()

respond = ChromaChat.respond
