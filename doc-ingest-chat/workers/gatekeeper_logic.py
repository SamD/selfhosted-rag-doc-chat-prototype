import gc
import logging
import os
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pdfplumber
from config import settings
from llama_cpp import LlamaGrammar
from pdf2image import convert_from_path
from utils.llm_setup import RemoteLlama, get_supervisor_llm
from utils.ocr_utils import preprocess_image, send_image_to_ocr
from utils.text_utils import is_bad_ocr, is_valid_pdf

# Set CUDA optimization
os.environ["GGML_CUDA_GRAPH_OPT"] = "1"

log = logging.getLogger("ingest.gatekeeper_logic")


def get_llm_and_grammar():
    """centralized getter for supervisor and grammar."""
    llm = get_supervisor_llm()
    # Grammar depends on whether model is local (needs object) or remote (needs string)
    if isinstance(llm, RemoteLlama):
        return llm, CHUNK0_GBNF_STR
    else:
        return llm, LlamaGrammar.from_string(CHUNK0_GBNF_STR)


# GBNF for Chunk 0 completion (starting after the pre-filled "# ")
# Matches: Title + Body
CHUNK0_GBNF_STR = r"""
root    ::= title body
title   ::= [^\n]+ "\n"+
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


def gatekeeper_extract_and_normalize(job_id: str, pdf_path: str, md_path: str) -> Tuple[bool, Optional[dict]]:
    """
    Core normalization logic for a claimed job.
    Processes pages in fixed batches for improved throughput.
    """
    try:
        file_slug = get_slug(Path(pdf_path).stem)
        log.info(f"🔍 Starting extraction and batch normalization for {pdf_path}")

        # Ensure we have the model ready
        get_llm_and_grammar()

        chunk_idx = 0
        first_chunk_meta = None

        if not is_valid_pdf(pdf_path):
            raise ValueError(f"Invalid PDF: {pdf_path}")

        log.info(f"📄 Extracting {pdf_path} in batches of {settings.GATEKEEPER_BATCH_SIZE} pages...")
        batch_text = []
        batch_start_page = 1

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                try:
                    try:
                        t = page.extract_text()
                    except Exception as e:
                        log.warning(f"⚠️ Page {page_num} extraction failed: {e}")
                        t = None

                    if not t or is_bad_ocr(t):
                        log.info(f"📸 Page {page_num}/{total_pages} delegating to OCR worker...")
                        images = convert_from_path(pdf_path, dpi=200, first_page=page_num, last_page=page_num)
                        if images:
                            np_image = preprocess_image(images[0])
                            if np_image is not None:
                                ocr_text, _, _, engine, _, _ = send_image_to_ocr(np_image, pdf_path, page_num)
                                t = ocr_text

                            for img in images:
                                img.close()

                    if not t:
                        log.warning(f"⚠️ No text could be extracted for page {page_num}. Adding placeholder.")
                        t = f"[DOCUMENT PAGE {page_num} EXTRACTION FAILED OR PAGE IS EMPTY]"

                    # TAG EVERY PAGE: Critical for Producer to extract accurate metadata
                    tagged_text = f"### [INTERNAL_PAGE_{page_num}]\n{t}"
                    batch_text.append(tagged_text)

                    # PROCESS BATCH: Trigger when we hit the size limit
                    if len(batch_text) >= settings.GATEKEEPER_BATCH_SIZE:
                        log.info(f"📊 Normalizing Batch (Pages {batch_start_page}-{page_num})...")
                        full_content = "\n\n".join(batch_text)
                        meta = process_chunk(chunk_idx, full_content, pdf_path, file_slug, md_path)
                        if chunk_idx == 0:
                            first_chunk_meta = meta

                        chunk_idx += 1
                        batch_text = []  # CLEAR COMPLETELY
                        batch_start_page = page_num + 1

                except Exception as page_err:
                    log.error(f"💥 Critical error on page {page_num}: {page_err}. Skipping page but continuing...")
                    continue

            # FINAL BATCH: Handle remaining pages
            if batch_text:
                log.info(f"📊 Normalizing Final Batch (Pages {batch_start_page}-{total_pages})...")
                full_content = "\n\n".join(batch_text)
                meta = process_chunk(chunk_idx, full_content, pdf_path, file_slug, md_path)
                if chunk_idx == 0:
                    first_chunk_meta = meta

        gc.collect()
        log.info(f"✅ Document normalization finished: {md_path}")
        return True, first_chunk_meta

    except Exception as e:
        log.error(f"❌ Normalization failed: {e}", exc_info=True)
        return False, None


def process_chunk(idx, raw_content, file_path, slug, md_path):
    """Stateless normalization using the exact prompt verified by the user."""
    meta = assemble_metadata(file_path, slug, idx, 9999)

    # 1. Define the Anchor
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
            f"---\n\n"
        )
    else:
        anchor_header = "\n\n\n\n"

    # 2. VERIFIED 'Markdown Formatter' Prompt
    user_msg = f"""
    You are a Markdown formatter. Convert RAW_TEXT below into valid Markdown. 
    Preserve all content and structure as much as possible. 
    Use headings, bullet points, numbered lists, code blocks, and tables only when they fit the input. 
    DO NOT summarize, infer, or add new information. 
    Return only Markdown, with no preface or explanation.
    Remove any OCR gibberish and unreadable characters.
    
    RAW_TEXT:
    {raw_content}
    """

    # 3. Payload (Unified User Role)
    messages = [{"role": "user", "content": user_msg}]

    log.info(f"🧠 Normalizing Batch {idx} (High-Fidelity Verified Prompt)...")

    llm, _ = get_llm_and_grammar()

    response = llm.create_chat_completion(
        messages=messages,
    )

    content = response["choices"][0]["message"]["content"]

    # 3. Final text construction
    final_text = anchor_header + content

    mode = "w" if idx == 0 else "a"
    with open(md_path, mode, encoding="utf-8") as f:
        f.write(final_text)
        if not final_text.endswith("\n"):
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    log.info(f"📝 Wrote Chunk {idx} to {md_path}")
    return meta


def log_gatekeeper_result(slug: str, status: str, metadata: Optional[dict] = None, error_msg: Optional[str] = None):
    """
    Deprecated in favor of JobService lifecycle tracking,
    but kept for schema compatibility during migration.
    """
    db_path = settings.GATEKEEPER_FAILURE_DB
    import json

    try:
        con = duckdb.connect(db_path)
        con.execute("CREATE TABLE IF NOT EXISTS gatekeeper_history (slug VARCHAR, timestamp TIMESTAMP, status VARCHAR, metadata TEXT, error VARCHAR)")
        con.execute("INSERT INTO gatekeeper_history VALUES (?, ?, ?, ?, ?)", [slug, datetime.now(timezone.utc), status, json.dumps(metadata) if metadata else None, error_msg])
        con.close()
    except Exception as e:
        log.error(f"Failed to log gatekeeper result: {e}")
