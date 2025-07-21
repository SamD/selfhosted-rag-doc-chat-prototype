#!/usr/bin/env python3
"""
Utility functions for chat formatting and citation.
"""
from collections import OrderedDict

from langchain_core.documents import Document
import os

def format_chunks_with_citations(docs: list[Document]) -> list[str]:
    formatted = []
    for i, doc in enumerate(docs):
        citation = f"[source{i+1}]"
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        formatted.append(f"{citation} (Source: {source}, Page: {page})\n---\n{content}")
    return formatted

def replace_citation_labels(output: str, docs: list[Document]) -> str:
    """
    Replaces [source1], [source2], ... with human-readable citations like [filename.pdf, page X],
    and appends a deduplicated source summary at the end.
    """
    seen = OrderedDict()

    for i, doc in enumerate(docs):
        label = f"[source{i+1}]"
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "N/A")

        if str(page).isdigit() and int(page) >= 0:
            resolved = f"[{source}, page {page}]"
        else:
            resolved = f"[{source}]"

        # Save first occurrence for deduped summary
        if resolved not in seen:
            seen[resolved] = True

        # Replace all [sourceX] with resolved label
        output = output.replace(label, resolved)

    # Append deduplicated list of sources at the end
    if seen:
        output += "\n\n---\n**Sources:** " + ", ".join(seen.keys())

    return output.strip()

def format_hermes_chat(chat_history):
    prompt = ""
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def get_env_boolean(env_var_name, default_value=False):
    env_value = os.getenv(env_var_name)
    if env_value is None:
        return default_value
    return env_value.lower() == 'true' 