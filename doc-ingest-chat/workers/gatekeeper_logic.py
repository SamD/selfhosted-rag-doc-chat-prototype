import logging
import os
import re
import time
import unicodedata
import uuid
import gc
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import List, Optional

import duckdb
import pdfplumber
from config import settings
from llama_cpp import Llama, LlamaGrammar
from pdf2image import convert_from_path
from utils.producer_utils import preprocess_image, send_image_to_ocr
from utils.text_utils import is_bad_ocr, is_valid_pdf

# Set CUDA optimization
os.environ["GGML_CUDA_GRAPH_OPT"] = "1"

log = logging.getLogger("ingest.gatekeeper_logic")

_MODEL = None
_CHUNK0_GRAMMAR = None

def get_llm():
    global _MODEL, _CHUNK0_GRAMMAR
    if _MODEL is None:
        log.info(f"🚀 Loading GateKeeper Model: {settings.SUPERVISOR_LLM_PATH}")
        _MODEL = Llama(
            model_path=settings.SUPERVISOR_LLM_PATH, 
            n_gpu_layers=0, # CPU-only for maximum stability with large files
            n_ctx=4096, 
            n_batch=512, 
            flash_attn=True, 
            seed=42, 
            verbose=False
        )
        _CHUNK0_GRAMMAR = LlamaGrammar.from_string(CHUNK0_GBNF_STR)
    return _MODEL, _CHUNK0_GRAMMAR


# GBNF for Chunk 0 completion (starting after the pre-filled "# ")
# Matches: Title + Body
CHUNK0_GBNF_STR = r"""
root    ::= title body
title   ::= [^\n]+ "\n\n"
body    ::= [^\t\r\n]* ("\n" [^\t\r\n]*)*
"""


def get_slug(text: str) -> str:
    """Sanitized, collision-resistant slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    suffix = blake2b(text.encode(), digest_size=4).hexdigest()
    return f"{text[:50]}-{suffix}"


def assemble_metadata(file_path: str, slug: str, chunk_idx: int, total_chunks: int):
    return {
        "id": str(uuid.uuid4()),
        "slug": slug,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_type": "pdf_ocr_raw" if file_path.endswith(".pdf") else "text_raw",
        "tier": 3,
        "chunk_index": chunk_idx,
        "total_chunks": total_chunks,
        "schema_version": "2026.04.07",
        "raw_path": os.path.abspath(file_path),
    }


def sliding_window_chunks(raw_text: str, chunk_size: int = 6000, overlap: int = 600) -> List[str]:
    """
    Yields overlapping raw text chunks from a large string.
    """
    if not raw_text:
        return []

    total_len = len(raw_text)
    chunks = []

    # First chunk
    end_pos_0 = min(chunk_size, total_len)
    chunks.append(raw_text[0:end_pos_0])

    if total_len <= chunk_size:
        return chunks

    start_pos = end_pos_0 - overlap

    while start_pos < total_len:
        end_pos = min(start_pos + chunk_size, total_len)
        if end_pos <= start_pos + overlap:
            break
        chunks.append(raw_text[start_pos:end_pos])
        if end_pos == total_len:
            break
        start_pos = end_pos - overlap

    return chunks


def sliding_window_normalize(file_path: str, chunk_size: int = 6000, overlap: int = 600) -> List[str]:
    """
    Extracts raw text from file and returns sliding window chunks.
    Mainly used for tests or non-streaming workflows.
    """
    raw_text = ""
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    raw_text += t + "\n\n"
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

    return sliding_window_chunks(raw_text, chunk_size=chunk_size, overlap=overlap)


def gatekeeper_extract_and_normalize(file_path: str, metadata_base: Optional[dict] = None) -> bool:
    """
    Entry point for normalization.
    Separates Extraction and Normalization to manage peak memory usage.
    """
    try:
        # 1. Setup paths and slugs
        file_slug = get_slug(Path(file_path).stem)
        final_md_path = os.path.join(settings.INGEST_FOLDER, f"{file_slug}.md")
        os.makedirs(settings.INGEST_FOLDER, exist_ok=True)

        raw_text_buffer = ""
        
        # 2. EXTRACTION PHASE (No LLM in memory)
        if file_path.lower().endswith(".pdf"):
            if not is_valid_pdf(file_path):
                raise ValueError(f"Invalid PDF: {file_path}")
                
            log.info(f"📄 Extracting raw text from {file_path}...")
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    try:
                        t = page.extract_text()
                    except Exception as e:
                        log.warning(f"⚠️ Page {page_num} extraction failed: {e}")
                        t = None

                    if not t or is_bad_ocr(t):
                        log.info(f"📸 Page {page_num}/{total_pages} delegating to OCR worker...")
                        # Use lower DPI (200) to save memory during large file conversion
                        images = convert_from_path(file_path, dpi=200, first_page=page_num, last_page=page_num)
                        if images:
                            np_image = preprocess_image(images[0])
                            if np_image is not None:
                                ocr_text, _, _, engine, _, _ = send_image_to_ocr(np_image, file_path, page_num)
                                t = ocr_text
                            else:
                                log.error(f"💥 Failed to preprocess page {page_num}")
                            
                            # Explicitly close image objects
                            for img in images:
                                img.close()
                        else:
                            log.error(f"💥 Failed to convert page {page_num} to image")

                    if t:
                        raw_text_buffer += t + "\n\n"
            
            # Close PDF and force GC before starting LLM inference
            gc.collect()
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text_buffer = f.read()

        if not raw_text_buffer.strip():
            log.error("❌ No text extracted from document. Normalization aborted.")
            return False

        # 3. NORMALIZATION PHASE (Load LLM)
        log.info(f"🧠 Normalizing content for {file_slug}...")
        get_llm() # Lazy load model here
        
        chunks = sliding_window_chunks(raw_text_buffer, chunk_size=6000, overlap=600)
        
        for idx, chunk_content in enumerate(chunks):
            process_chunk(idx, chunk_content, file_path, file_slug, final_md_path)

        log.info(f"✅ Document normalized -> {final_md_path}")
        return True

    except Exception as e:
        log.error(f"❌ Normalization failed: {e}", exc_info=True)
        return False


def process_chunk(idx, raw_content, file_path, slug, final_md_path):
    """Normalizes a chunk and writes immediately to file."""
    meta = assemble_metadata(file_path, slug, idx, 9999)  # total_chunks unknown yet

    if idx == 0:
        anchor_header = (
            f"---\n"
            f"ID: {meta['id']}\n"
            f"Slug: {meta['slug']}\n"
            f"Processed-At: {meta['processed_at']}\n"
            f"Source-Type: {meta['source_type']}\n"
            f"Extraction-Tier: {meta['tier']}\n"
            f"Chunk-Index: {meta['chunk_index']}\n"
            f"Schema-Version: {meta['schema_version']}\n"
            f"Raw-Path: {meta['raw_path']}\n"
            f"---\n\n# "
        )
        use_grammar = True
        system_msg = "You are a strict Markdown normalizer. Provide an accurate document title and normalized Markdown body. Do NOT repeat the metadata block. Use proper headers (##, ###)."
        stop_tokens = ["<|im_end|>", "---"]
        max_tokens = 4096
    else:
        anchor_header = f"\n\n<!-- CHUNK {idx} -->\n\n"
        use_grammar = False
        system_msg = "You are a document normalizer. Convert raw text into clean Markdown body. No metadata, no titles. Smooth transition."
        stop_tokens = ["<|im_end|>", "---", "metadata:", "ID:", "Slug:", "# "]
        max_tokens = 6144

    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{raw_content}<|im_end|>\n<|im_start|>assistant\n{anchor_header}"

    log.info(f"🧠 Processing Chunk {idx} (Inference)...")
    
    llm, grammar = get_llm()
    response = llm(
        prompt, 
        max_tokens=max_tokens, 
        grammar=grammar if use_grammar else None, 
        stop=stop_tokens, 
        temperature=0.1
    )

    content = response["choices"][0]["text"]
    final_text = anchor_header + content

    # Use 'w' for first chunk, 'a' for others
    mode = "w" if idx == 0 else "a"
    with open(final_md_path, mode, encoding="utf-8") as f:
        f.write(final_text)
        if not final_text.endswith("\n"):
            f.write("\n")
        f.flush()
        os.fsync(f.fileno()) # Hard flush to disk
        
    log.info(f"📝 Wrote Chunk {idx} to {final_md_path}")


def generate_uuid():
    return str(uuid.uuid4())


def generate_slug(file_path):
    return get_slug(Path(file_path).stem)


def log_to_failure_sink(metadata, error_msg, raw_content):
    db_path = settings.GATEKEEPER_FAILURE_DB
    try:
        con = duckdb.connect(db_path)
        con.execute("CREATE TABLE IF NOT EXISTS failures (slug VARCHAR, timestamp TIMESTAMP, reason VARCHAR, raw_data TEXT)")
        con.execute("INSERT INTO failures VALUES (?, ?, ?, ?)", [metadata.get("slug"), datetime.now(timezone.utc), error_msg, raw_content])
        con.close()
    except Exception as e:
        log.error(f"Failed to log to failure sink: {e}")
