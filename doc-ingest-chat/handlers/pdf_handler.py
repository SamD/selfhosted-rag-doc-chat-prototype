from typing import Generator

import pdfplumber
from config.settings import PDF_FORCE_OCR
from pdf2image import convert_from_path
from utils.ocr_utils import preprocess_image, send_image_to_ocr
from utils.text_utils import is_bad_ocr
from utils.trace_utils import get_logger, get_trace_id

from .base_handler import BaseContentTypeHandler

log = get_logger("ingest.handlers.pdf")


class PDFContentTypeHandler(BaseContentTypeHandler):
    """
    Handler for PDF files using pdfplumber and OCR fallback.
    """

    MIME_TYPE = "application/pdf"

    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")

    def stream_content(self, file_path: str) -> Generator[str, None, None]:
        """
        Extracts raw text from PDF pages, falling back to OCR if needed.
        """
        log.info(f"📄 Extracting PDF content from {file_path}")
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    t = None
                    
                    if not PDF_FORCE_OCR:
                        try:
                            t = page.extract_text()
                        except Exception as e:
                            log.warning(f"⚠️ Page {page_num} extraction failed: {e}")
                            t = None

                    if PDF_FORCE_OCR or not t or is_bad_ocr(t):
                        log.info(f"📸 Page {page_num}/{total_pages} delegating to OCR worker (Reason: {'Forced' if PDF_FORCE_OCR else 'Bad Extraction'})...")
                        images = convert_from_path(file_path, dpi=200, first_page=page_num, last_page=page_num)
                        if images:
                            np_image = preprocess_image(images[0])
                            if np_image is not None:
                                ocr_text, _, _, engine, _, _ = send_image_to_ocr(np_image, file_path, page_num, trace_id=get_trace_id())
                                if ocr_text:
                                    # Check if text is garbled (too many chars per word)
                                    word_count = len(ocr_text.split())
                                    if word_count > 0 and len(ocr_text) / word_count > 1000:
                                        log.warning(f"⚠️ Page {page_num}: OCR returned garbled text ({len(ocr_text)} chars, {word_count} words) — discarding")
                                        t = ""
                                    else:
                                        t = ocr_text
                                else:
                                    t = ""

                            for img in images:
                                img.close()

                    if not t:
                        log.error(f"❌ Extraction failed for {file_path} page {page_num}. Stopping.")
                        raise RuntimeError(f"Failed to extract text for page {page_num}")

                    yield t

        except Exception as e:
            log.error(f"❌ PDF extraction failed: {e}")
            raise
