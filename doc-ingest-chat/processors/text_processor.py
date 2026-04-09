#!/usr/bin/env python3
"""
Text processing functionality.
"""

import hashlib
import logging

import mmh3
from config.settings import EMBEDDING_MODEL_PATH, MAX_TOKENS
from transformers import AutoTokenizer
from utils.text_utils import is_bad_ocr

log = logging.getLogger("ingest.text_processor")


class TextProcessor:
    """Text processing functionality as static methods."""

    @staticmethod
    def get_document_id(file_bytes: bytes) -> str:
        """
        Generates an 8-char MurmurHash3 directly from raw binary checksum.
        No supervisor, no text sampling, no hallucinations.
        """
        # Deterministic 8-char Hash of the binary file bytes
        m_hash = hex(mmh3.hash(file_bytes) & 0xFFFFFFFF)[2:].upper().zfill(8)
        return f"DOC_{m_hash}"

    @staticmethod
    def make_chunk_id(rel_path: str, idx: int, chunk: str) -> str:
        """Generate a unique chunk ID."""
        digest = hashlib.md5(chunk.encode()).hexdigest()[:8]
        return f"{rel_path}_chunk_{idx}_{digest}"

    @staticmethod
    def make_chunk(prefix_tokens, start, full_tokens, tokenizer, budget=512, overlap=50, decode_step=5, log_prefix=""):
        """
        Create a chunk from tokens ensuring it fits within the token budget.
        
        Args:
            prefix_tokens: Tokens to prepend to every chunk (e.g. enrichment anchor).
            start: Start index in full_tokens.
            full_tokens: The complete tokenized document content.
            tokenizer: The tokenizer instance.
            budget: Maximum total tokens allowed (prefix + content).
            overlap: Token overlap between chunks.
            decode_step: Step size for checking budget (performance optimization).
            log_prefix: Prefix for logs.
            
        Returns:
            (consumed_count, decoded_string)
        """
        # Prefix is always included
        prefix_len = len(prefix_tokens)
        available_budget = budget - prefix_len
        
        if available_budget <= 0:
            log.error(f"{log_prefix}💥 Prefix tokens ({prefix_len}) exceed budget ({budget})!")
            return 0, ""

        end = min(start + available_budget, len(full_tokens))
        content_chunk = full_tokens[start:end]
        
        # Combined tokens for decoding
        full_chunk_tokens = prefix_tokens + content_chunk
        decoded = tokenizer.decode(full_chunk_tokens, skip_special_tokens=True).strip()
        
        log.debug(f"{log_prefix}🧩 Created chunk: {len(full_chunk_tokens)} tokens, {len(decoded)} chars")
        
        return len(content_chunk), decoded

    @staticmethod
    def split_doc(text: str, rel_path: str, file_type: str, tokenizer=None, prefix="passage: ", budget=None, overlap=50, page_num=None, document_id=None):
        """Split document into chunks with fast hash-based enrichment."""
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
        
        # Use global MAX_TOKENS if budget not provided
        if budget is None:
            budget = MAX_TOKENS

        log.info(f"📄 Splitting file: {rel_path}, Page: {page_num} (Budget: {budget} tokens)")
        content_tokens = tokenizer.encode(text, add_special_tokens=False)

        # Final Enrichment Tag: [DOC_A1B2]
        meta_id = document_id if document_id else "DOC_UNKNOWN"
        enrichment_prefix = f"{prefix}[{meta_id}] "

        log.info(f"✨ [Enrichment] Applying anchor: [{meta_id}] to {rel_path} Page {page_num}")

        prefix_tokens = tokenizer.encode(enrichment_prefix, add_special_tokens=False)

        chunks = []
        i = 0
        while i < len(content_tokens):
            # last is the number of CONTENT tokens consumed
            last, chunk_str = TextProcessor.make_chunk(prefix_tokens, i, content_tokens, tokenizer, budget, overlap)

            if not chunk_str or last <= 0:
                i += 1
                continue

            chunks.append(chunk_str)
            
            # Move index forward by (consumed - overlap)
            # Ensure we always move forward by at least 1
            i += max(1, last - overlap)

        log.info(f"✅ Finished splitting {rel_path}, Page {page_num} → {len(chunks)} chunks")

        metadata = [
            {
                "source_file": rel_path,
                "type": file_type,
                "chunk_index": idx,
                "page": page_num,
            }
            for idx in range(len(chunks))
        ]

        return chunks, metadata

    @staticmethod
    def normalize_metadata(entry: dict, default_values: dict = None) -> dict:
        """Normalize metadata entry with expected keys."""
        expected_keys = [
            "chunk",  # The actual text
            "id",  # Chunk ID
            "source_file",  # Original file relative path
            "document_id",  # Fast mmh3 ID (e.g. DOC_A1B2)
            "type",  # File type: pdf/html/video
            "hash",  # Content hash
            "engine",  # OCR engine used
            "page",  # Page number
            "chunk_index",  # Index within the file
        ]

        default_values = default_values or {}

        normalized = {}
        for key in expected_keys:
            normalized[key] = entry.get(key, default_values.get(key, None))

        return normalized

    @staticmethod
    def validate_chunk(chunk: str, tokenizer) -> bool:
        """Validate if a chunk is acceptable based on token length."""
        if not isinstance(chunk, str):
            return False

        # Use actual tokenizer to check length
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        token_len = len(tokens)
        
        if token_len > MAX_TOKENS:
            log.warning(f"⚠️ Chunk exceeds {MAX_TOKENS} tokens ({token_len}) — dropping")
            return False

        return not is_bad_ocr(chunk, tokenizer)


# Expose static methods as module-level functions after class definition
split_doc = TextProcessor.split_doc
make_chunk_id = TextProcessor.make_chunk_id
normalize_metadata = TextProcessor.normalize_metadata
validate_chunk = TextProcessor.validate_chunk
get_document_id = TextProcessor.get_document_id
