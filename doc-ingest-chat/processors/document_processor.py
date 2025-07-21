#!/usr/bin/env python3
"""
Document processing functionality.
"""
import base64
import json
import re
import uuid
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from charset_normalizer import from_path
from pdf2image import convert_from_path

from config.settings import (
    SUPPORTED_MEDIA_EXT, MAX_OCR_DIM,
    DEVICE, MEDIA_BATCH_SIZE, COMPUTE_TYPE
)
from processors.text_processor import TextProcessor
from utils.text_utils import is_bad_ocr


class DocumentProcessor:
    """Document processing functionality as static methods."""
    
    @staticmethod
    def extract_text_from_html(full_path: str) -> Optional[str]:
        """Extract text from HTML file."""
        try:
            match = from_path(full_path).best()
            if not match:
                raise ValueError(f"[ERROR] Could not detect encoding for: {full_path}")

            html = str(match)  # Decoded text (charset-normalizer >= 3.x)
            soup = BeautifulSoup(html, "html5lib")  # Most forgiving parser

            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r'\n\s*\n+', '\n\n', text)  # Collapse extra blank lines
            return text

        except Exception as e:
            print(f"[ERROR] extract_text_from_html failed for {full_path}: {e}")
            return None

    @staticmethod
    def extract_text_with_pdfplumber(path: str) -> Optional[str]:
        """Extract text from PDF using pdfplumber."""
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                full_text = ''
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + '\n'
                    if len(full_text.strip()) < 10:
                        raise ValueError("Extracted text too short; likely not useful.")
            if full_text.strip() == '':
                raise ValueError("No extractable text found; likely a scanned PDF.")
            return full_text
        except Exception as e:
            print(f"[OCR Fallback] pdfplumber failed: {e}")
            return None

    @staticmethod
    def extract_text_from_media(filepath: str) -> Optional[List]:
        """Extract text from media files using Whisper."""
        if not filepath.lower().endswith(SUPPORTED_MEDIA_EXT):
            raise ValueError(f"Unsupported file type: {filepath}")

        print(f" 🎥 Processing media {filepath}")

        # Use Whisper directly; it internally extracts audio from video
        try:
            import whisperx
            audio = whisperx.load_audio(filepath)
            model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE)
            result = model.transcribe(audio, batch_size=MEDIA_BATCH_SIZE)
            return result["segments"]
        except Exception as e:
            print(f"Transcription failed for {filepath}: {e}")
            return None

    @staticmethod
    def preprocess_image(pil_image: Image.Image) -> np.ndarray:
        """Preprocess image for OCR."""
        w, h = pil_image.size

        # Rescale only if very large
        if max(w, h) > MAX_OCR_DIM:
            scale = MAX_OCR_DIM / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        np_image = np.array(pil_image)
        print(f"PIL size: w: {w}, h: {h}")
        print(f"NumPy shape: {np_image.shape}")

        # Convert to grayscale
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        return np_image

    @staticmethod
    def send_image_to_ocr(np_image: np.ndarray, rel_path: str, page_num: int, redis_client) -> Tuple[
        Optional[str], str, int, str, str]:
        """Send image to OCR service."""
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

        redis_client.lpush("ocr_processing_job", json.dumps(job))
        result = redis_client.blpop(reply_key, timeout=300)
        if not result:
            raise TimeoutError(f"OCR timeout for {rel_path} page {page_num}")

        _, data = result
        result = json.loads(data)
        return (
            result.get("text"),
            result.get("rel_path"),
            result.get("page_num"),
            result.get("engine"),
            result.get("job_id")
        )

    @staticmethod
    def process_pdf_by_page(full_path: str, rel_path: str, file_type: str, redis_client, tokenizer) -> Tuple[
        List[str], List[dict]]:
        """Process PDF by page with OCR fallback."""
        chunks = []
        metadatas = []

        try:
            import pdfplumber
            with pdfplumber.open(full_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                    except Exception as e:
                        print(f"⚠️ pdfplumber failed on page {page_num} of {rel_path}: {e}")
                        text = ""

                    if not text.strip() or is_bad_ocr(text, tokenizer):
                        print(f"🔁 Falling back to OCR for page {page_num} of {rel_path}")

                        try:
                            pill_image = \
                            convert_from_path(full_path, dpi=300, first_page=page_num + 1, last_page=page_num + 1)[0]
                            np_image = DocumentProcessor.preprocess_image(pill_image)
                            result = DocumentProcessor.send_image_to_ocr(np_image, rel_path, page_num + 1, redis_client)
                            text, rel_path, page_num_ocr, engine, job_id = result

                            if not text or not isinstance(text, str) or is_bad_ocr(text, tokenizer):
                                print(f"⚠️ OCR returned garbage for {rel_path} page {page_num + 1}")
                                continue

                            chunk_texts, metadata = TextProcessor.split_doc(text.strip(), rel_path, file_type, tokenizer,
                                                                          page_num=page_num + 1)
                            chunks.extend(chunk_texts)
                            metadatas.extend(metadata)

                        except Exception as e:
                            print(f"💥 OCR failed on {rel_path} page {page_num + 1}: {e}")
                            continue
                    else:
                        # Good text, use pdfplumber output
                        chunk_texts, metadata = TextProcessor.split_doc(text.strip(), rel_path, file_type, tokenizer,
                                                                      page_num=page_num + 1)
                        chunks.extend(chunk_texts)
                        metadatas.extend(metadata)

        except Exception as e:
            print(f"💥 Failed to open PDF {rel_path} for per-page processing: {e}")

        return chunks, metadatas

    @staticmethod
    def process_pdf_by_page_nofallback(full_path: str, rel_path: str, file_type: str, tokenizer) -> Tuple[
        List[str], List[dict]]:
        """Process PDF by page without OCR fallback."""
        chunks = []
        metadatas = []

        import pdfplumber
        with pdfplumber.open(full_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    print(f"⚠️ Empty or unreadable page {page_num} in {rel_path}")
                    continue

                page_chunks, page_metadata = TextProcessor.split_doc(
                    text,
                    rel_path,
                    file_type,
                    tokenizer,
                    page_num=page_num
                )
                chunks.extend(page_chunks)
                metadatas.extend(page_metadata)

        return chunks, metadatas

# Expose static methods as module-level functions after class definition
extract_text_from_html = DocumentProcessor.extract_text_from_html