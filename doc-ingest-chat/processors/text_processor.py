#!/usr/bin/env python3
"""
Text processing functionality.
"""

import logging
import re

import mmh3
import yaml
from config.settings import EMBEDDING_MODEL_PATH, MAX_TOKENS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

log = logging.getLogger("ingest.text_processor")


class TextProcessor:
    """Text processing functionality as static methods."""

    @staticmethod
    def split_markdown_doc(text: str, rel_path: str, tokenizer=None, budget=None, prefix="passage: ", document_id=None):
        """
        Parses Markdown with YAML metadata headers and performs hierarchical splitting.
        Zero-Drop Policy: Hard truncates chunks to ensure they ALWAYS pass validation.
        """
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
        if budget is None:
            budget = MAX_TOKENS

        # 1. Separate YAML metadata from Markdown body
        header_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if header_match:
            header_yaml = header_match.group(1)
            try:
                raw_yaml = yaml.safe_load(header_yaml)
                file_metadata = {k.lower(): v for k, v in raw_yaml.items()} if isinstance(raw_yaml, dict) else {}
            except Exception as e:
                log.error(f"💥 Failed to parse YAML header in {rel_path}: {e}")
                file_metadata = {}
            content_body = text[header_match.end() :]
        else:
            file_metadata = {}
            content_body = text

        # 2. Define Markdown splitting hierarchy
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("### [INTERNAL_PAGE_", "Internal_Page"),
            ("###", "Header_3"),
        ]

        log.info(f"📄 Splitting Markdown: {rel_path} (Budget: {budget} tokens)")

        # Mandatory RAG prefix for length calculation
        meta_id = document_id if document_id else "DOC_UNKNOWN"
        enrichment_prefix = f"{prefix}[{meta_id}] "
        prefix_tokens = tokenizer.encode(enrichment_prefix, add_special_tokens=True)
        prefix_len = len(prefix_tokens)

        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = md_splitter.split_text(content_body)

        # 3. Apply secondary recursive splitting for large sections
        def token_len(text):
            return prefix_len + len(tokenizer.encode(text, add_special_tokens=False))

        # SAFETY: Target 450 tokens to minimize truncation events.
        safe_budget = min(450, budget - prefix_len)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=safe_budget,
            chunk_overlap=50,
            length_function=token_len,
            separators=["\n\n", "\n", " ", ""],
        )

        final_chunks = text_splitter.split_documents(sections)

        chunks = []
        metadata = []

        total_chunks = len(final_chunks)
        current_page = 1
        for idx, chunk in enumerate(final_chunks):
            # EXTRACTION OF PAGE FROM ANCHOR
            for key, value in chunk.metadata.items():
                if "[INTERNAL_PAGE_" in str(value):
                    page_match = re.search(r"(\d+)", str(value))
                    if page_match:
                        current_page = int(page_match.group(1))
                        break

            chunk_text = chunk.page_content

            # --- HARD TRUNCATION SAFETY NET ---
            # If the splitter failed to stay under the budget, we truncate now.
            full_encoded = tokenizer.encode(f"{enrichment_prefix}{chunk_text}", add_special_tokens=True)
            if len(full_encoded) > MAX_TOKENS:
                log.warning(f"🔨 Hard-truncating chunk {idx} ({len(full_encoded)} -> {MAX_TOKENS})")
                # Truncate to 510 to allow for [SEP] token at the end
                truncated_tokens = full_encoded[: MAX_TOKENS - 1]
                chunk_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                # Remove the enrichment_prefix from the decoded text as it is re-added later
                chunk_text = re.sub(rf"^{re.escape(enrichment_prefix)}", "", chunk_text)

            chunks.append(chunk_text)
            meta = {
                **file_metadata,
                **chunk.metadata,
                "page": current_page,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "source_file": rel_path,
            }
            # Cleanup internal markers
            for k in list(meta.keys()):
                if "Internal_Page" in k or (isinstance(meta[k], str) and "[INTERNAL_PAGE_" in meta[k]):
                    meta.pop(k, None)
            metadata.append(meta)

        log.info(f"✅ Finished splitting {rel_path} → {len(chunks)} chunks")
        return chunks, metadata

    @staticmethod
    def get_document_id(file_bytes: bytes) -> str:
        """Deterministically generates an 8-char MurmurHash3 from binary bytes."""
        m_hash = hex(mmh3.hash(file_bytes) & 0xFFFFFFFF)[2:].upper().zfill(8)
        return f"DOC_{m_hash}"

    @staticmethod
    def make_chunk_id(rel_path: str, idx: int, chunk: str, document_id: str = None) -> str:
        """Deterministic ID based on content using MurmurHash3."""
        context = document_id if document_id else rel_path
        c_hash = hex(mmh3.hash(chunk.encode()) & 0xFFFFFFFF)[2:].lower().zfill(8)
        return f"{context}_{c_hash}"

    @staticmethod
    def make_chunk(prefix_tokens, start, full_tokens, tokenizer, budget=512, overlap=50, decode_step=5, log_prefix=""):
        """Create a chunk from tokens ensuring it fits within the token budget."""
        prefix_len = len(prefix_tokens)
        available_budget = budget - prefix_len

        if available_budget <= 0:
            log.error(f"{log_prefix}💥 Prefix tokens ({prefix_len}) exceed budget ({budget})!")
            return 0, ""

        end = min(start + available_budget, len(full_tokens))
        content_chunk = full_tokens[start:end]

        full_chunk_tokens = prefix_tokens + content_chunk
        decoded = tokenizer.decode(full_chunk_tokens, skip_special_tokens=True).strip()
        return len(content_chunk), decoded

    @staticmethod
    def split_doc(text: str, rel_path: str, file_type: str, tokenizer=None, prefix="passage: ", budget=None, overlap=50, page_num=None, document_id=None):
        """Split document into chunks with aggressive safety margin."""
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
        if budget is None:
            budget = 450

        log.info(f"📄 Splitting file: {rel_path}, Page: {page_num} (Safety Budget: {budget} tokens)")
        content_tokens = tokenizer.encode(text, add_special_tokens=False)

        meta_id = document_id if document_id else "DOC_UNKNOWN"
        enrichment_prefix = f"{prefix}[{meta_id}] "
        prefix_tokens = tokenizer.encode(enrichment_prefix, add_special_tokens=True)

        chunks = []
        i = 0
        while i < len(content_tokens):
            last, chunk_str = TextProcessor.make_chunk(prefix_tokens, i, content_tokens, tokenizer, 512, overlap)
            if not chunk_str or last <= 0:
                i += 1
                continue
            chunks.append(chunk_str)
            i += max(1, last - overlap)

        metadata = [{"source_file": rel_path, "type": file_type, "chunk_index": idx, "page": page_num} for idx in range(len(chunks))]
        return chunks, metadata

    @staticmethod
    def normalize_metadata(entry: dict, default_values: dict = None) -> dict:
        """Normalize metadata entry with expected keys."""
        expected_keys = ["chunk", "id", "source_file", "document_id", "trace_id", "type", "hash", "engine", "page", "chunk_index"]
        default_values = default_values or {}
        normalized = {}
        for key in expected_keys:
            normalized[key] = entry.get(key, default_values.get(key, None))
        return normalized

    @staticmethod
    def validate_chunk(chunk: str, tokenizer) -> str:
        """Zero-Drop Validator: Truncates oversized chunks instead of dropping them."""
        if not isinstance(chunk, str):
            return ""

        tokens = tokenizer.encode(chunk, add_special_tokens=True)
        if len(tokens) > MAX_TOKENS:
            log.warning(f"🔨 Validator hard-truncating oversized chunk ({len(tokens)} -> {MAX_TOKENS})")
            # Truncate to 511 to allow for the final separator token
            truncated_tokens = tokens[: MAX_TOKENS - 1]
            return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        return chunk


# Convenience aliases
split_doc = TextProcessor.split_doc
split_markdown_doc = TextProcessor.split_markdown_doc
make_chunk_id = TextProcessor.make_chunk_id
normalize_metadata = TextProcessor.normalize_metadata
validate_chunk = TextProcessor.validate_chunk
get_document_id = TextProcessor.get_document_id
