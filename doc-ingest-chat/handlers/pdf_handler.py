from typing import Generator

import pdfplumber
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
                    try:
                        t = page.extract_text()
                    except Exception as e:
                        log.warning(f"⚠️ Page {page_num} extraction failed: {e}")
                        t = None

                    if not t or is_bad_ocr(t):
                        log.info(f"📸 Page {page_num}/{total_pages} delegating to OCR worker...")
                        images = convert_from_path(file_path, dpi=200, first_page=page_num, last_page=page_num)
                        if images:
                            np_image = preprocess_image(images[0])
                            if np_image is not None:
                                ocr_text, _, _, engine, _, _ = send_image_to_ocr(np_image, file_path, page_num, trace_id=get_trace_id())
                                t = ocr_text

                            for img in images:
                                img.close()

                    if not t:
                        log.warning(f"⚠️ No text could be extracted for page {page_num}. Adding placeholder.")
                        t = f"[DOCUMENT PAGE {page_num} EXTRACTION FAILED OR PAGE IS EMPTY]"

                    yield t

        except Exception as e:
            log.error(f"❌ PDF extraction failed: {e}")
            raise
