#!/usr/bin/env python3
"""
Text processing functionality.
"""

import logging
import re
from typing import List

import mmh3
import yaml
from config.settings import MAX_TOKENS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from utils.text_utils import get_tokenizer

log = logging.getLogger("ingest.text_processor")


class TextProcessor:
    """Text processing functionality as static methods."""

    @staticmethod
    def split_markdown_doc(text: str, rel_path: str, tokenizer=None, budget=None, prefix="passage: ", document_id=None):
        """
        Parses Markdown with YAML metadata headers and performs hierarchical splitting.
        Zero-Drop Policy: Hard truncates chunks to ensure they ALWAYS pass validation.
        """
        tokenizer = tokenizer or get_tokenizer()
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

        # DYNAMIC SAFETY BUDGET: Target ~85% of total tokens to minimize overflow events.
        # This ensures we scale correctly whether MAX_TOKENS is 512 or 256.
        safe_budget = int(budget * 0.85) - prefix_len

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=safe_budget,
            chunk_overlap=50,
            length_function=token_len,
            separators=["\n\n", "\n", " ", ""],
        )

        final_chunks = text_splitter.split_documents(sections)

        chunks = []
        metadata = []

        for idx, chunk in enumerate(final_chunks):
            # EXTRACTION OF PAGE FROM ANCHOR
            current_page = 1
            for key, value in chunk.metadata.items():
                if "[INTERNAL_PAGE_" in str(value):
                    page_match = re.search(r"(\d+)", str(value))
                    if page_match:
                        current_page = int(page_match.group(1))
                        break

            chunk_text = chunk.page_content
            
            # --- NON-DESTRUCTIVE OVERFLOW HANDLING ---
            # If the splitter somehow produced an oversized chunk, sub-split it 
            # instead of truncating.
            full_encoded = tokenizer.encode(f"{enrichment_prefix}{chunk_text}", add_special_tokens=True)
            
            if len(full_encoded) <= MAX_TOKENS:
                # Fits perfectly
                chunks.append(chunk_text)
                meta = TextProcessor._create_metadata(file_metadata, chunk.metadata, current_page, len(chunks)-1, rel_path, chunk_text, meta_id)
                metadata.append(meta)
            else:
                # Oversized! Sub-split using a sliding window approach
                log.warning(f"⚠️ Chunk {idx} is oversized ({len(full_encoded)} tokens). Sub-splitting to prevent data loss.")
                
                # Encode content without prefix and special tokens for manual slicing
                content_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
                available_budget = MAX_TOKENS - prefix_len - 2 # -2 for [CLS]/[SEP] safety
                
                start_tok = 0
                while start_tok < len(content_tokens):
                    end_tok = min(start_tok + available_budget, len(content_tokens))
                    sub_chunk_tokens = content_tokens[start_tok:end_tok]
                    sub_chunk_text = tokenizer.decode(sub_chunk_tokens, skip_special_tokens=True).strip()
                    
                    if sub_chunk_text:
                        chunks.append(sub_chunk_text)
                        meta = TextProcessor._create_metadata(file_metadata, chunk.metadata, current_page, len(chunks)-1, rel_path, sub_chunk_text, meta_id)
                        metadata.append(meta)
                    
                    start_tok = end_tok

        # Update total_chunks in metadata after potential sub-splitting
        actual_total = len(chunks)
        for meta in metadata:
            meta["total_chunks"] = actual_total

        log.info(f"✅ Finished splitting {rel_path} → {actual_total} chunks (Zero-Loss)")
        return chunks, metadata

    @staticmethod
    def _create_metadata(file_meta, chunk_meta, page, idx, rel_path, text, doc_id):
        """Helper to construct clean metadata for a chunk, including ID and Hash."""
        # Generate Deterministic ID
        c_id = TextProcessor.make_chunk_id(rel_path, idx, text, doc_id)
        # Generate Content Hash
        c_hash = hex(mmh3.hash(text) & 0xFFFFFFFF)[2:].upper().zfill(8)

        meta = {
            **file_meta,
            **chunk_meta,
            "id": c_id,
            "hash": c_hash,
            "page": page,
            "chunk_index": idx,
            "source_file": rel_path,
        }
        if "document_id" not in meta:
            meta["document_id"] = doc_id
        # Cleanup internal markers
        for k in list(meta.keys()):
            if "Internal_Page" in k or (isinstance(meta[k], str) and "[INTERNAL_PAGE_" in meta[k]):
                meta.pop(k, None)
        return meta

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
        tokenizer = tokenizer or get_tokenizer()
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
    def validate_chunk(chunk: str, tokenizer) -> List[str]:
        """
        Zero-Loss Validator: If a chunk is oversized, it sub-splits it into 
        valid pieces instead of truncating.
        Returns a List[str] of valid chunks.
        """
        if not isinstance(chunk, str):
            return []

        tokens = tokenizer.encode(chunk, add_special_tokens=True)
        if len(tokens) <= MAX_TOKENS:
            return [chunk]

        log.warning(f"⚠️ Validator detected oversized chunk ({len(tokens)} tokens). Sub-splitting to prevent loss.")
        
        # Non-destructive sub-split
        content_tokens = tokenizer.encode(chunk, add_special_tokens=False)
        # Conservative budget for sub-splitting without knowing prefix here
        budget = MAX_TOKENS - 4 
        
        sub_chunks = []
        start = 0
        while start < len(content_tokens):
            end = min(start + budget, len(content_tokens))
            decoded = tokenizer.decode(content_tokens[start:end], skip_special_tokens=True).strip()
            if decoded:
                sub_chunks.append(decoded)
            start = end
            
        return sub_chunks


# Convenience aliases
split_doc = TextProcessor.split_doc
split_markdown_doc = TextProcessor.split_markdown_doc
make_chunk_id = TextProcessor.make_chunk_id
normalize_metadata = TextProcessor.normalize_metadata
validate_chunk = TextProcessor.validate_chunk
get_document_id = TextProcessor.get_document_id
