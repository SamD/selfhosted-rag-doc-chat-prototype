#!/usr/bin/env python3
import base64
import fcntl
import gc
import hashlib
import json
import math
import multiprocessing
import os
import re
import signal
import time
import traceback
import uuid
from multiprocessing import Lock, Value

import cv2
import numpy as np
import pdfplumber
import redis
import torch

# from faster_whisper import WhisperModel
import whisperx
from bs4 import BeautifulSoup
from charset_normalizer import from_path

# Import configuration
from config.settings import (
    ALL_SUPPORTED_EXT,
    COMPUTE_TYPE,
    DEBUG_IMAGE_DIR,
    DEVICE,
    EMBEDDING_MODEL_PATH,
    FAILED_FILES,
    INGEST_FOLDER,
    MAX_CHROMA_BATCH_SIZE_LIMIT,
    MAX_OCR_DIM,
    MEDIA_BATCH_SIZE,
    QUEUE_NAMES,
    REDIS_HOST,
    REDIS_OCR_JOB_QUEUE,
    REDIS_PORT,
    SUPPORTED_MEDIA_EXT,
    TRACK_FILE,
)
from pdf2image import convert_from_path
from PIL import Image
from processors.text_processor import TextProcessor, make_chunk_id, split_doc
from transformers import AutoTokenizer
from utils.file_utils import load_tracked, normalize_rel_path
from utils.logging_config import setup_logging
from utils.metrics import FileMetrics
from utils.text_utils import is_gibberish, is_low_quality, is_valid_pdf, is_visibly_corrupt

os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Image.MAX_IMAGE_PIXELS = 500_000_000
MAX_CHROMA_BATCH_SIZE = MAX_CHROMA_BATCH_SIZE_LIMIT

# Use queue names from configuration
queue_names = QUEUE_NAMES

queue_lock = None
queue_index = None

try:
    Resample = Image.Resampling  # Pillow ≥ 10
except AttributeError:
    Resample = Image  # Pillow < 10


def get_next_queue():
    global queue_lock, queue_index
    with queue_lock:
        i = queue_index.value
        queue_index.value = (i + 1) % len(queue_names)
        return queue_names[i]


def update_failed_files(file):
    with open(FAILED_FILES, "a", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(file + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Define global queues
# ocr_queue = multiprocessing.Queue(maxsize=100)
# ocr_result_queue = multiprocessing.Queue()

# multiprocessing.set_start_method('fork')


# Use this to capture loggers producing messages
# class CaptureLoggerNames(logging.Handler):
#     def emit(self, record):
#         print(f"🕵️ Captured logger: {record.name} → {record.getMessage()}")

# logging.basicConfig(level=logging.INFO)
# logging.getLogger().addHandler(CaptureLoggerNames())

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler(), logging.FileHandler("ingest_producer.log")]
# )

_DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH, use_fast=True, trust_remote_code=True, local_files_only=True)
# WHISPER_MODEL = whisper.load_model("medium")

device = DEVICE
batch_size = MEDIA_BATCH_SIZE  # reduce if low on GPU mem
compute_type = COMPUTE_TYPE  # change to "int8" if low on GPU mem (may reduce accuracy)

log = setup_logging("ingest_producer.log", include_default_filters=True)


# log.error(traceback.format_stack())


def get_whisper_model():
    return whisperx.load_model("large-v2", device, compute_type=compute_type)
    # return WhisperModel("large-v3", device=device, compute_type=compute_type)


def is_bad_ocr(text):
    return (
        not text
        or not text.strip()
        # or not is_mostly_printable_ascii(text)
        or is_gibberish(text)
        or is_visibly_corrupt(text)
        or is_low_quality(text, _DEFAULT_TOKENIZER)
    )


def put_or_cpu_fallback(q, item, cpu_fallback_fn, queue_name="GPU OCR queue"):
    if q.full():
        log.warning(f"⚠️ {queue_name} is full — falling back to CPU OCR (no retries)")
        try:
            text = cpu_fallback_fn(*item)  # item = (pill_image, rel_path, page_num, job_id)
            return ("cpu", text)
        except Exception as e:
            log.error(f"💥 CPU fallback OCR failed: {e}")
            return ("cpu", None)
    else:
        try:
            q.put(item, block=True)  # block until space is available
            return "gpu"
        except Exception as e:
            log.error(f"💥 Failed to enqueue to {queue_name}: {e}")
            return ("gpu_failed", None)


def wait_for_queue_space(rclient, queue_name, max_length, poll_interval=0.5):
    while True:
        qlen = rclient.llen(queue_name)
        if qlen < max_length:
            break
        time.sleep(poll_interval)


def blocking_push_with_backpressure(rclient, queue_name: str, entries: list[str], max_queue_length: int = 1000, poll_interval: float = 0.5, warn_after: float = 10.0, rel_path: str = "unknown"):
    push_script = rclient.register_script("""
    local queue = KEYS[1]
    local max_len = tonumber(ARGV[1])
    local new_items = {}

    for i = 2, #ARGV do
        table.insert(new_items, ARGV[i])
    end

    local current_len = redis.call("LLEN", queue)
    if current_len + #new_items <= max_len then
        for _, item in ipairs(new_items) do
            redis.call("RPUSH", queue, item)
        end
        return 1
    else
        return 0
    end
    """)

    start_wait = time.time()
    warned = False
    total_wait_time = 0
    attempt = 0

    log.debug(f"📤 Attempting to push {len(entries)} entries to queue '{queue_name}' for {rel_path}")

    while True:
        attempt += 1
        result = push_script(keys=[queue_name], args=[max_queue_length] + entries)

        if result == 1:
            elapsed = time.time() - start_wait
            if warned:
                log.info(f"✅ Queue backpressure resolved after {elapsed:.2f}s — pushed {len(entries)} entries to '{queue_name}' for {rel_path}")
            else:
                log.debug(f"✅ Enqueued {len(entries)} entries to '{queue_name}' for {rel_path} on attempt {attempt}")
            return  # success

        if not warned and (time.time() - start_wait) > warn_after:
            qlen = rclient.llen(queue_name)
            log.warning(f"⏳ Queue '{queue_name}' length {qlen} exceeds limit ({max_queue_length}) — backpressure delay on {rel_path}")
            warned = True

        time.sleep(poll_interval)
        total_wait_time += poll_interval
        if total_wait_time % 10 < poll_interval:  # log every 10s
            qlen = rclient.llen(queue_name)
            log.debug(f"🔁 Still waiting to enqueue {rel_path} (queue: {queue_name}, length: {qlen}) [waited {total_wait_time:.1f}s]")


def preprocess_image(pil_image):
    w, h = pil_image.size

    # Rescale only if very large
    if max(w, h) > MAX_OCR_DIM:
        scale = MAX_OCR_DIM / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size, Resample.LANCZOS)

    np_image = np.array(pil_image)
    log.info(f"PIL size: w: {w}, h: {h}")
    log.info(f"NumPy shape: {np_image.shape}")

    # Convert to grayscale
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

    # Skip denoise, binarization, deskew for now
    return np_image


def preprocess_image_v1(pil_image):
    w, h = pil_image.size

    # Too big → downscale
    if max(w, h) > MAX_OCR_DIM:
        scale = MAX_OCR_DIM / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size, Resample.BICUBIC)

    # Too small → upscale
    if w < 800:
        pil_image = pil_image.resize((w * 2, h * 2), Resample.BICUBIC)

    # Convert to NumPy for OpenCV
    np_image = np.array(pil_image)
    log.info(f"PIL size: w: {w}, h: {h}")
    log.info(f"NumPy shape: {np_image.shape}")

    # Convert to grayscale
    start = time.time()
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    log.info(f"⏱ Grayscale: {time.time() - start:.3f}s")

    # Denoise
    start = time.time()
    np_image = cv2.fastNlMeansDenoising(np_image, h=30)
    log.info(f"⏱ Denoise: {time.time() - start:.3f}s")

    # Binarize
    start = time.time()
    np_image = cv2.adaptiveThreshold(np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    log.info(f"⏱ Binarize: {time.time() - start:.3f}s")

    # Deskew
    start = time.time()
    coords = np.column_stack(np.where(np_image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = np_image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    np_image = cv2.warpAffine(np_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    log.info(f"⏱ Deskew: {time.time() - start:.3f}s")

    return np_image


def extract_text_from_html(full_path: str) -> str:
    try:
        match = from_path(full_path).best()
        if not match:
            raise ValueError(f"[ERROR] Could not detect encoding for: {full_path}")

        html = str(match)  # Decoded text (charset-normalizer >= 3.x)
        soup = BeautifulSoup(html, "html5lib")  # Most forgiving parser

        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # Collapse extra blank lines
        return text

    except Exception as e:
        log.error(f"[ERROR] extract_text_from_html failed for {full_path}: {e}", exc_info=True)
        return None


def extract_text_with_pdfplumber(path):
    try:
        with pdfplumber.open(path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                if len(full_text.strip()) < 10:
                    raise ValueError("Extracted text too short; likely not useful.")
        if full_text.strip() == "":
            raise ValueError("No extractable text found; likely a scanned PDF.")
        return full_text
    except Exception as e:
        log.info(f"[OCR Fallback] pdfplumber failed: {e}")
        return None


def extract_text_from_media(filepath):
    if not filepath.lower().endswith(SUPPORTED_MEDIA_EXT):
        raise ValueError(f"Unsupported file type: {filepath}")

    log.info(f" 🎥 Processing media {filepath}")

    # Use Whisper directly; it internally extracts audio from video
    try:
        audio = whisperx.load_audio(filepath)
        model = get_whisper_model()
        # result = model.transcribe(audio, batch_size=batch_size)

        result = model.transcribe(audio, batch_size=batch_size)
        return result["segments"]
    except Exception as e:
        log.error(f"Transcription failed for {filepath}: {e}", exc_info=True)
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        del model


def send_image_to_ocr(np_image, rel_path, page_num):
    start_time = time.perf_counter()

    job_id = str(uuid.uuid4())
    reply_key = f"ocr_reply:{job_id}"

    job = {
        "job_id": job_id,
        "rel_path": rel_path,
        "page_num": page_num,
        "image_shape": np_image.shape,
        "image_dtype": str(np_image.dtype),
        "image_base64": base64.b64encode(np_image.tobytes()).decode(),
        "reply_key": reply_key,
    }

    redis_client.lpush(REDIS_OCR_JOB_QUEUE, json.dumps(job))
    result = redis_client.blpop(reply_key, timeout=300)
    if not result:
        raise TimeoutError(f"OCR timeout for {rel_path} page {page_num}")

    ocr_roundtrip_ms = (time.perf_counter() - start_time) * 1000.0

    _, data = result
    result = json.loads(data)
    return (result.get("text"), result.get("rel_path"), result.get("page_num"), result.get("engine"), result.get("job_id"), ocr_roundtrip_ms)


def fallback_ocr(full_path, rel_path=None, job_id=None, metrics=None):
    log.warning(f"🔍 Running OCR fallback for: {full_path}")
    chunks = []

    try:
        with pdfplumber.open(full_path) as pdf:
            for i, page in enumerate(pdf.pages):
                log.info(f"📄 Processing page {i + 1} of {rel_path or full_path}")

                # Try extracting text first
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    log.warning(f"⚠️ pdfplumber extract_text() failed on page {i + 1}: {e}")
                    text = ""

                if not is_bad_ocr(text):
                    log.info(f"✅ Used pdfplumber for page {i + 1} of {rel_path}")
                    sub_chunks, _ = TextProcessor.split_doc(text.strip(), rel_path or full_path, "pdfplumber", page_num=i + 1)
                    for c in sub_chunks:
                        chunks.append((c, "pdfplumber", i + 1))
                    continue
                else:
                    log.info(f"✅ Text unable to be processed by pdfplumber for page {i + 1} of {rel_path}, falling back to OCR")

                pill_image = convert_from_path(full_path, dpi=300, first_page=i + 1, last_page=i + 1)[0]
                # preprocess image once and pass to ocr and tesseract instead of each doing it
                np_image = preprocess_image(pill_image)

                # Render single page image for OCR
                log.info(f"🔁 Sending page {i + 1} to OCR")
                result = send_image_to_ocr(np_image, rel_path, i + 1)
                text, rel_path, page_num, engine, job_id, ocr_roundtrip_ms = result

                # Track OCR operation in metrics if provided
                if metrics:
                    metrics.add_ocr_operation(page=page_num, ocr_roundtrip_time_ms=ocr_roundtrip_ms, engine=engine, success=not (engine and engine.startswith("notext")))

                if engine and engine.startswith("notext"):
                    log.warning(f"⚠️ OCR Worker reported short text: engine {engine} doc {rel_path} page {page_num} job_id {job_id}")
                    continue

                if not text or not isinstance(text, str) or is_bad_ocr(text):
                    log.error(f"⚠️ Text is empty for for {rel_path} page {page_num} job_id {job_id} ")
                    # raise ValueError("OCR Failed")
                    log.warning(f"⚠️ Bad text or garbage: engine {engine} doc {rel_path} page {page_num} job_id {job_id} text -- {text}")
                    continue
                else:
                    if engine != "error":
                        log.info(f"✅ EasyOCR succeeded for {rel_path} page {page_num} job_id {job_id} ")
                        try:
                            sub_chunks, _ = TextProcessor.split_doc(text.strip(), rel_path or full_path, "pdf", page_num=i + 1)
                            for c in sub_chunks:
                                chunks.append((c, engine, i + 1))
                        except Exception as e:
                            log.error(f"💥 Failed to split OCR text for {rel_path or full_path} page {i + 1}: {e}, text: {text.strip()}", exc_info=True)
                    else:
                        log.error(f"❌ EasyOCR failed for {rel_path} page {page_num} job_id {job_id} ")
                        raise ValueError("OCR Failed")
    except Exception as e:
        log.error(f"💥 Failed to process PDF {full_path}: {e}", traceback.format_exc())
        return []

    if not chunks:
        log.warning(f"⚠️ No usable pdfplumber text; queued pages for OCR: {rel_path}")

    return chunks


# Reserve space for padding/special tokens (e.g. 32 tokens)
PAD_RESERVE = 32


# def make_chunk(prefix_tokens, start, full_tokens, tokenizer, budget=512, overlap=50):
#     last_chunk, chunk_str = "",""
#     chunk = prefix_tokens.copy()
#     i,end = 0,0
#     total_budget = budget - overlap
#
#     for i in range(start, len(full_tokens)):
#         chunk.append(full_tokens[i])
#         chunk_str = tokenizer.decode(chunk, skip_special_tokens=True).strip()
#         if len(chunk_str) > total_budget:
#             # log.error(f"*** RETURNING END: {end}, *** CHUNK: {last_chunk}")
#             return end, last_chunk
#         last_chunk = chunk_str
#         end = i
#
#     if len(chunk_str) > total_budget:
#         return end, last_chunk
#
#     return end, chunk_str


def process_pdf_by_page(full_path, rel_path, file_type):
    chunks = []
    metadatas = []

    try:
        with pdfplumber.open(full_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    log.warning(f"⚠️ pdfplumber failed on page {page_num} of {rel_path}: {e}")
                    text = ""

                if not text.strip() or is_bad_ocr(text):
                    log.info(f"🔁 Falling back to OCR for page {page_num} of {rel_path}")

                    try:
                        pill_image = convert_from_path(full_path, dpi=300, first_page=page_num + 1, last_page=page_num + 1)[0]
                        np_image = preprocess_image(pill_image)
                        result = send_image_to_ocr(np_image, rel_path, page_num + 1)
                        text, rel_path, page_num_ocr, engine, job_id = result

                        if not text or not isinstance(text, str) or is_bad_ocr(text):
                            log.warning(f"⚠️ OCR returned garbage for {rel_path} page {page_num + 1}")
                            continue

                        chunk_texts, metadata = split_doc(text.strip(), rel_path, file_type, page_num=page_num + 1)
                        chunks.extend(chunk_texts)
                        metadatas.extend(metadata)

                    except Exception as e:
                        log.error(f"💥 OCR failed on {rel_path} page {page_num + 1}: {e}")
                        continue
                else:
                    # Good text, use pdfplumber output
                    chunk_texts, metadata = split_doc(text.strip(), rel_path, file_type, page_num=page_num + 1)
                    chunks.extend(chunk_texts)
                    metadatas.extend(metadata)

    except Exception as e:
        log.error(f"💥 Failed to open PDF {rel_path} for per-page processing: {e}", exc_info=True)

    return chunks, metadatas


def process_pdf_by_page_nofallback(full_path, rel_path, file_type):
    chunks = []
    metadatas = []

    with pdfplumber.open(full_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text or not text.strip():
                log.warning(f"⚠️ Empty or unreadable page {page_num} in {rel_path}")
                continue

            page_chunks, page_metadata = split_doc(
                text,
                rel_path,
                file_type,
                page_num=page_num,  # ✅ pass page number
            )
            chunks.extend(page_chunks)
            metadatas.extend(page_metadata)

    return chunks, metadatas


def md5_from_int_list(int_list):
    """
    Generates an MD5 hash from a list of integers.

    Args:
      int_list: A list of integers.

    Returns:
      A string representing the MD5 hash in hexadecimal format.
    """
    # Convert the list of integers to a JSON string and encode it to bytes
    json_string = json.dumps(int_list, sort_keys=True).encode("utf-8")

    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Update the hash object with the encoded JSON string
    md5_hash.update(json_string)

    # Return the hexadecimal representation of the hash
    return md5_hash.hexdigest()


def ingest_file(full_path, rel_path, job_id):
    # Initialize metrics collection
    metrics = FileMetrics(worker="producer", file=rel_path)
    pages_processed = 0

    try:
        log.info(f"📁 FULL: {full_path}, REL: {rel_path}, JOB_ID: {job_id}")

        with metrics.timer("total_processing"):
            if full_path.lower().endswith(".pdf"):
                file_type = "pdf"

                if not is_valid_pdf(full_path):
                    log.warning(f"⚠️ Invalid PDF {full_path}")
                    update_failed_files(rel_path)
                    return 1

                try:
                    # Try fast whole-doc extract first
                    with metrics.timer("text_extraction"):
                        chunks, metadatas = process_pdf_by_page(full_path, rel_path, file_type)
                    chunks_with_engine = [(chunk, "pdfplumber", meta["page"]) for chunk, meta in zip(chunks, metadatas) if not is_bad_ocr(chunk)]
                    pages_processed = len(set(meta["page"] for meta in metadatas))
                    log.info(f"📄 Used full-doc pdfplumber extraction for {rel_path} ({len(chunks_with_engine)} chunks)")
                except Exception:
                    log.info(f"🔁 Falling back to per-page extraction and OCR for {rel_path}")
                    chunks_with_engine = []
                    with metrics.timer("text_extraction"):
                        ocr_chunks = fallback_ocr(full_path, rel_path, job_id, metrics=metrics)
                    for chunk, engine, page in ocr_chunks:
                        chunks_with_engine.append((chunk, engine, page))
                    pages_processed = len(set(page for _, _, page in ocr_chunks))

            elif full_path.lower().endswith((".html", ".htm")):
                file_type = "html"
                with metrics.timer("text_extraction"):
                    text = extract_text_from_html(full_path)
                if not text or len(text.strip()) < 10:
                    log.warning(f"⚠️ Skipping {rel_path}, empty HTML content")
                    update_failed_files(rel_path)
                    return 1

                tokens = _DEFAULT_TOKENIZER.encode(text, add_special_tokens=False)
                if len(tokens) < 5:
                    log.warning(f"⚠️ Skipping {rel_path}, HTML tokenized to only {len(tokens)} tokens")
                    update_failed_files(rel_path)
                    return 1

                chunks, _ = split_doc(text, rel_path, file_type, page_num=-1)
                chunks_with_engine = [(c, "html", -1) for c in chunks]
                pages_processed = 1
            # elif full_path.lower().endswith(SUPPORTED_MEDIA_EXT):
            #     file_type = "video"
            #     text = extract_text_from_media(full_path)
            #     if not text or len(text.strip()) < 10:
            #         log.warning(f"⚠️ Skipping {rel_path}, empty video content")
            #         update_failed_files(rel_path)
            #         # with open(FAILED_FILES, "a", encoding="utf-8") as f:
            #         #     f.write(rel_path + "\n")
            #         return 1
            #
            #     tokens = _DEFAULT_TOKENIZER.encode(text, add_special_tokens=False)
            #     if len(tokens) < 5:
            #         log.warning(f"⚠️ Skipping {rel_path}, VIDEO tokenized to only {len(tokens)} tokens")
            #         update_failed_files(rel_path)
            #         return 1
            #
            #     chunks, _ = split_doc(text, rel_path, file_type, page_num=-1)
            #     chunks_with_engine = [(c, "video", -1) for c in chunks]
            else:
                log.warning(f"⚠️ Skipping unknown file type: {rel_path}")
                update_failed_files(rel_path)
                # with open(FAILED_FILES, "a", encoding="utf-8") as f:
                #     f.write(rel_path + "\n")
                return 1

        if not chunks_with_engine:
            log.warning(f"⚠️ All chunks for {rel_path} were garbage — skipping")
            update_failed_files(rel_path)
            # with open(FAILED_FILES, "a", encoding="utf-8") as f:
            #     f.write(rel_path + "\n")
            return 1

        # Generate IDs using make_chunk_id
        for idx, tup in enumerate(chunks_with_engine):
            if not isinstance(tup, tuple) or len(tup) != 3:
                raise RuntimeError(f"💣 Invalid chunk structure at index {idx} in {rel_path}: {tup}")

        ids = [make_chunk_id(rel_path, i, chunk) for i, (chunk, _, _) in enumerate(chunks_with_engine)]

        # log.error(f"ids: {ids}")

        # Prepare chunk entries
        chunk_entries = []
        for i, (chunk, engine, page) in enumerate(chunks_with_engine):
            if not isinstance(page, int):
                try:
                    page = int(page) if str(page).isdigit() else -1
                except Exception:
                    page = -1

            entry = {
                "chunk": chunk,
                "id": ids[i],
                "source_file": rel_path,
                "type": file_type,
                "hash": hashlib.md5(chunk.encode()).hexdigest(),
                "engine": engine,
                "page": page,  # real page number
                "chunk_index": i,
            }
            chunk_entries.append(TextProcessor.normalize_metadata(entry))

        # Atomic enqueue to Redis
        try:
            total_chunks = len(chunk_entries)

            if total_chunks == 0:
                log.warning(f"⚠️ No chunks to enqueue for {rel_path}")
                return 1

            # Calculate dynamic batch size
            num_batches = math.ceil(total_chunks / MAX_CHROMA_BATCH_SIZE)
            dynamic_batch_size = math.ceil(total_chunks / num_batches)

            log.info(f"📤 Enqueuing {total_chunks} chunks in {num_batches} batches (batch size: {dynamic_batch_size}) for {rel_path}")

            with metrics.timer("redis_enqueue"):
                next_queue = get_next_queue()
                for i in range(0, total_chunks, dynamic_batch_size):
                    batch = chunk_entries[i : i + dynamic_batch_size]
                    json_batch = [json.dumps(entry) for entry in batch]

                    blocking_push_with_backpressure(rclient=redis_client, queue_name=next_queue, entries=json_batch, max_queue_length=50000, poll_interval=0.5, warn_after=10.0, rel_path=rel_path)

                # Send sentinel
                blocking_push_with_backpressure(
                    rclient=redis_client, queue_name=next_queue, entries=[json.dumps({"type": "file_end", "source_file": rel_path, "expected_chunks": total_chunks})], max_queue_length=50000, poll_interval=0.5, warn_after=10.0, rel_path=rel_path
                )

            log.info(f"📤 Done enqueuing {total_chunks} chunks for {rel_path}")

            # Add metrics counters
            metrics.add_counter("chunks_produced", total_chunks)
            metrics.add_counter("pages_processed", pages_processed)

        except Exception as e:
            log.error(f"❌ Atomic enqueue failed for file {rel_path}: {e}")
            update_failed_files(rel_path)
            return 1

        # Emit metrics
        metrics.emit(log)
        return 0

    except Exception as e:
        log.error(f"💥 Error processing {rel_path}: {e}\n{traceback.format_exc()}")
        update_failed_files(rel_path)
        # with open(FAILED_FILES, "a", encoding="utf-8") as f:
        #     f.write(rel_path + "\n")
        return 1

    finally:
        gc.collect()


def run_ingest(job_tuple):
    job_id, source_file, rel_path = job_tuple
    return ingest_file(source_file, rel_path, job_id)


def init_worker(lock_obj, index_obj):
    global queue_lock, queue_index
    queue_lock = lock_obj
    queue_index = index_obj


lock = Lock()
index = Value("i", 0)

SHUTDOWN = multiprocessing.Event()


def signal_handler(sig, frame):
    log.warning(f"💥 Received signal {sig}, initiating shutdown...")
    SHUTDOWN.set()


def run_tree_watcher(scan_interval=30):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass  # Already set

    while not SHUTDOWN.is_set():
        try:
            track = load_tracked(TRACK_FILE)
            skipped = load_tracked(FAILED_FILES)

            jobs = []
            job_id = 0

            for root, _, files in os.walk(INGEST_FOLDER):
                for fname in files:
                    if fname.lower().endswith(ALL_SUPPORTED_EXT):
                        full_path = os.path.join(root, fname)
                        full_path = full_path.decode() if isinstance(full_path, bytes) else full_path
                        _ingest_folder = INGEST_FOLDER.decode() if isinstance(INGEST_FOLDER, bytes) else INGEST_FOLDER
                        rel_path = normalize_rel_path(os.path.relpath(full_path, _ingest_folder))
                        if rel_path in track or rel_path in skipped:
                            continue
                        jobs.append((job_id, full_path, rel_path))
                        job_id += 1

            if jobs:
                log.info(f"📦 Found {len(jobs)} new file(s) to ingest")
                success = 0

                pool = None
                try:
                    pool = multiprocessing.Pool(
                        processes=min(4, os.cpu_count()),
                        initializer=init_worker,
                        initargs=(
                            lock,
                            index,
                        ),
                        maxtasksperchild=1,
                    )

                    for result in pool.imap_unordered(run_ingest, jobs, chunksize=1):
                        if SHUTDOWN.is_set():
                            log.warning("⛔ Shutdown requested mid-processing")
                            break
                        if result:
                            success += 1

                except Exception as e:
                    log.error(f"Error during multiprocessing: {e}")
                finally:
                    if pool:
                        pool.terminate()
                        pool.join()
                        log.info(f"✅ Ingested {success}/{len(jobs)} file(s) this cycle")

            else:
                log.info("🔍 No new files found this cycle")

            if not SHUTDOWN.is_set():
                time.sleep(scan_interval)

        except Exception as e:
            log.error(f"Unhandled error in producer loop: {e}")
            time.sleep(5)

    log.info("🛑 Tree walker exiting cleanly.")


def main():
    run_tree_watcher()


if __name__ == "__main__":
    main()
