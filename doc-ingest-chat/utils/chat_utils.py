#!/usr/bin/env python3
"""
Utility functions for chat formatting and citation.
"""

import os
from collections import OrderedDict

from langchain_core.documents import Document


def format_chunks_with_citations(docs: list[Document]) -> list[str]:
    formatted = []
    for i, doc in enumerate(docs):
        citation = f"[source{i + 1}]"
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        formatted.append(f"{citation} (Source: {source}, Page: {page})\n---\n{content}")
    return formatted


def replace_citation_labels(output: str, docs: list[Document]) -> str:
    """
    Replaces [source1], [source2], ... with clickable Markdown links
    pointing to the original PDF in the /files/ route.
    """
    # Use OrderedDict to maintain discovery order for filenames
    seen_files = OrderedDict()

    # Base URL for file access (linked to FastAPI backend)
    # We use localhost:8000 by default since that's where api_gpu serves /files
    api_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    for i, doc in enumerate(docs):
        source_num = i + 1
        bracket_label = f"[source{source_num}]"
        paren_label = f"(source{source_num})"

        # Handle source file mapping: if it's the .md, we want to link to the .pdf
        source_path = doc.metadata.get("source_file", "unknown")
        filename = os.path.basename(source_path)

        # If it's the normalized markdown, map it back to the original PDF
        # e.g. outline-of-history-pt1-4e82d6f5.md -> outline_of_history_pt1.pdf
        if filename.endswith(".md"):
            # We can check if the original filename is in the metadata
            original = doc.metadata.get("original_filename")
            if original:
                filename = original
            else:
                # Fallback: hope it matches the standard naming convention or keep .md
                pass

        page = doc.metadata.get("page", "N/A")
        file_url = f"{api_url}/files/{filename}"

        # Inline citation with Markdown Link
        if str(page).isdigit() and int(page) >= 0:
            resolved = f"[[{filename}, page {page}]({file_url})]"
        else:
            resolved = f"[[{filename}]({file_url})]"

        # Record file for unique list at bottom
        if filename not in seen_files:
            seen_files[filename] = file_url

        # Replace tags in content
        output = output.replace(bracket_label, resolved)
        output = output.replace(paren_label, resolved)

    # Append deduplicated list of unique source links at the end
    if seen_files:
        links = [f"[{name}]({url})" for name, url in seen_files.items()]
        output += "\n\n---\n**Sources:** " + ", ".join(links)

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
    return env_value.lower() == "true"
