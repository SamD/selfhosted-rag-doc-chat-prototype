#!/usr/bin/env python3
"""
Text processing functionality.
"""
import hashlib

from config.settings import E5_MODEL_PATH, MAX_TOKENS
from transformers import AutoTokenizer
from utils.text_utils import is_bad_ocr


class TextProcessor:
    """Text processing functionality as static methods."""
    
    @staticmethod
    def make_chunk_id(rel_path: str, idx: int, chunk: str) -> str:
        """Generate a unique chunk ID."""
        digest = hashlib.md5(chunk.encode()).hexdigest()[:8]
        return f"{rel_path}_chunk_{idx}_{digest}"

    @staticmethod
    def make_chunk(prefix_tokens, start, full_tokens, tokenizer, budget=512, overlap=50, decode_step=5, log_prefix=""):
        """Create a chunk from tokens."""
        chunk = prefix_tokens.copy()
        end = start
        last_valid_chunk = ""
        decode_now = True

        print(f"{log_prefix}🧩 Starting chunk at token {start}")

        for i in range(start, len(full_tokens)):
            chunk.append(full_tokens[i])
            end = i + 1

            decode_now = (i - start) % decode_step == 0 or len(chunk) >= (budget - 10)

            if decode_now:
                decoded = tokenizer.decode(chunk, skip_special_tokens=True).strip()
                print(f"{log_prefix}🔎 Decoded at token {i}: {len(decoded)} chars")

                if len(decoded) > budget:
                    print(f"{log_prefix}❌ Chunk at token {i} exceeded {budget} chars — backtracking")

                    # Backtrack until within budget
                    for j in range(i, start - 1, -1):
                        test_chunk = prefix_tokens + full_tokens[start:j]
                        decoded = tokenizer.decode(test_chunk, skip_special_tokens=True).strip()
                        if len(decoded) <= budget:
                            print(f"{log_prefix}✅ Found valid chunk ending at token {j - 1} ({len(decoded)} chars)")
                            return j - start, decoded

                    print(f"{log_prefix}💥 Could not fit any tokens from position {start} under {budget}-char budget")
                    return 0, ""

                else:
                    last_valid_chunk = decoded

        print(f"{log_prefix}✅ Final chunk: {len(last_valid_chunk)} chars, {end - start} tokens")
        return end - start, last_valid_chunk

    @staticmethod
    def split_doc(text: str, rel_path: str, file_type: str, tokenizer=None, prefix="passage: ", budget=512, overlap=50, page_num=None):
        """Split document into chunks."""
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(E5_MODEL_PATH)

        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        content_tokens = tokenizer.encode(text, add_special_tokens=False)

        print(f"📄 Splitting file: {rel_path}, Page: {page_num}")
        print(f"🔢 Total content tokens: {len(content_tokens)} | Prefix tokens: {len(prefix_tokens)}")

        chunks = []
        i = 0
        while i < len(content_tokens):
            last, chunk_str = TextProcessor.make_chunk(prefix_tokens, i, content_tokens, tokenizer, budget, overlap)
            chunks.append(chunk_str)
            i += last

        print(f"✅ Finished splitting {rel_path}, Page {page_num} → {len(chunks)} chunks")

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
            "type",  # File type: pdf/html/video
            "hash",  # Content hash
            "engine"  # OCR engine used: "easyocr" or "tesseract"
        ]

        default_values = default_values or {}

        normalized = {}
        for key in expected_keys:
            normalized[key] = entry.get(key, default_values.get(key, None))

        return normalized

    @staticmethod
    def validate_chunk(chunk: str, tokenizer) -> bool:
        """Validate if a chunk is acceptable."""
        if not isinstance(chunk, str):
            return False
        
        token_len = len(chunk)
        if token_len > MAX_TOKENS:
            print(f"⚠️ Chunk exceeds {MAX_TOKENS} tokens ({token_len}) — dropping")
            return False
        
        return not is_bad_ocr(chunk, tokenizer) 

# Expose static methods as module-level functions after class definition
split_doc = TextProcessor.split_doc
make_chunk_id = TextProcessor.make_chunk_id
normalize_metadata = TextProcessor.normalize_metadata
validate_chunk = TextProcessor.validate_chunk 