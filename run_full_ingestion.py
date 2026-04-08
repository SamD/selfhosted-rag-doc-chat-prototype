import datetime
import logging
import os
import re
import uuid
import time
from pathlib import Path

import duckdb
import mmh3
from outlines import models, regex, Generator
from llama_cpp import Llama
import pdfplumber
from config import settings
from utils.producer_utils import fallback_ocr
from utils.text_utils import is_valid_pdf

# Setup dedicated logger for the full run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("gatekeeper_full_run.log"), logging.StreamHandler()]
)
log = logging.getLogger("ingest.gatekeeper_full")

# markdown_polymorphic_regex definition for Outlines
markdown_full_regex = r"""---
ID: [a-f0-9\-]+
Slug: [a-z0-9\-]+
Source-Type: (document|web|video)
---

# [^\n]+

(## [^\n]+\n\n|### [^\n]+\n\n|(\* [^\n]+\n)+|(\[ [0-9]{2}:[0-9]{2} \] [^\n]+\n\n)|(\| ([^|]+\|)+\n\| (--- \|)+\n(\| ([^|]+\|)+\n)+)|[^\n]+\n\n)+
""".strip()

# Regex for subsequent batches (just the blocks)
markdown_content_regex = r"""(## [^\n]+\n\n|### [^\n]+\n\n|(\* [^\n]+\n)+|(\[ [0-9]{2}:[0-9]{2} \] [^\n]+\n\n)|(\| ([^|]+\|)+\n\| (--- \|)+\n(\| ([^|]+\|)+\n)+)|[^\n]+\n\n)+
""".strip()

_MODEL = None
_GEN_FULL = None
_GEN_CONTENT = None

def get_model():
    global _MODEL
    if _MODEL is None:
        log.info(f"🚀 Loading Model: {settings.SUPERVISOR_LLM_PATH}")
        llm = Llama(
            model_path=settings.SUPERVISOR_LLM_PATH,
            n_gpu_layers=-1,
            n_ctx=8192,
            seed=42,
            verbose=False
        )
        _MODEL = models.LlamaCpp(llm, chat_mode=False)
    return _MODEL

def get_generator(mode="full"):
    global _GEN_FULL, _GEN_CONTENT
    model = get_model()
    if mode == "full":
        if _GEN_FULL is None:
            log.info("⚙️ Compiling Full Schema FSM (this takes a moment)...")
            _GEN_FULL = Generator(model, regex(markdown_full_regex))
        return _GEN_FULL
    else:
        if _GEN_CONTENT is None:
            log.info("⚙️ Compiling Content Schema FSM...")
            _GEN_CONTENT = Generator(model, regex(markdown_content_regex))
        return _GEN_CONTENT

def is_at_boundary(text: str) -> bool:
    if not text: return True
    text = text.strip()
    return any(text.endswith(char) for char in [".", "!", "?", "\n\n"])

def normalize_batch(raw_content: str, metadata: dict, is_first: bool):
    generator = get_generator("full" if is_first else "content")
    
    if is_first:
        prompt = f"""<|im_start|>system
Normalize this RAW CONTENT into Markdown. Start with metadata header '---'.<|im_end|>
<|im_start|>user
ID: {metadata['id']}
Slug: {metadata['slug']}
Source-Type: {metadata['source_type']}
RAW CONTENT:
{raw_content}
<|im_end|>
<|im_start|>assistant
"""
    else:
        prompt = f"""<|im_start|>system
Continue normalizing into Markdown blocks. NO header.<|im_end|>
<|im_start|>user
RAW CONTENT:
{raw_content}
<|im_end|>
<|im_start|>assistant
"""

    output = generator(prompt, max_tokens=4096, temperature=0.1)
    if not output.endswith("\n\n"): output += "\n\n"
    return output

def process_whole_pdf(file_path, output_path):
    metadata = {
        "id": str(uuid.uuid4()),
        "slug": "outline-of-history-complete",
        "source_type": "document",
        "extraction_tier": "pdfplumber"
    }
    
    batch_size = 5
    current_text = ""
    pages_in_batch = 0
    is_first = True
    
    with pdfplumber.open(file_path) as pdf:
        total = len(pdf.pages)
        for i in range(total):
            page_text = pdf.pages[i].extract_text()
            if page_text:
                current_text += page_text + "\n"
                pages_in_batch += 1
            
            should_proc = (i == total - 1) or (pages_in_batch >= batch_size and is_at_boundary(current_text)) or (pages_in_batch >= batch_size * 2)
            
            if should_proc and current_text.strip():
                log.info(f"⏳ Processing pages {i-pages_in_batch+2} to {i+1} of {total}...")
                start_time = time.time()
                normalized = normalize_batch(current_text, metadata, is_first)
                
                with open(output_path, "a" if not is_first else "w") as f:
                    f.write(normalized)
                
                log.info(f"✅ Batch completed in {time.time()-start_time:.1f}s")
                is_first = False
                current_text = ""
                pages_in_batch = 0

if __name__ == "__main__":
    os.environ["SUPERVISOR_LLM_PATH"] = "/home/samueldoyle/AI_LOCAL/Models/Qwen-3/Qwen3-8B-Q6_K.gguf"
    process_whole_pdf("Docs/outline_of_history_pt1.pdf", "FINAL_outline_of_history.md")
