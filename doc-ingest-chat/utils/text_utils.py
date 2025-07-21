#!/usr/bin/env python3
"""
Text processing utility functions.
"""
import re
import string
import unicodedata


class TextUtils:
    """Text processing utility functions as static methods."""
    
    @staticmethod
    def is_visibly_corrupt(text: str) -> bool:
        """Check if text contains corruption indicators."""
        return re.search(r'[âã¢£™žœÂÃ]', text) is not None

    @staticmethod
    def is_gibberish(text: str) -> bool:
        """Check if text appears to be gibberish."""
        if not text or not text.strip():
            return True
        normalized = unicodedata.normalize("NFKD", text)
        printable = ''.join(c for c in normalized if c.isprintable())
        total = len(printable)
        if total == 0:
            return True
        non_alpha = sum(1 for c in printable if not c.isalpha() and c not in (' ', '\n'))
        ratio = non_alpha / total
        return ratio > 0.6

    @staticmethod
    def is_mostly_printable_ascii(text: str, threshold: float = 0.75) -> bool:
        """
        Returns True if at least `threshold` fraction of characters are printable ASCII.
        """
        printable = set(string.printable)
        if not text:
            return False
        printable_count = sum(1 for c in text if c in printable)
        ratio = printable_count / len(text)
        return ratio >= threshold

    @staticmethod
    def is_low_quality(text: str, tokenizer, min_tokens: int = 5) -> bool:
        """Check if text has too few tokens."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens) < min_tokens

    @staticmethod
    def is_bad_ocr(text: str, tokenizer) -> bool:
        """Check if OCR text is of poor quality."""
        return (
            not text
            or not text.strip()
            or TextUtils.is_gibberish(text)
            or TextUtils.is_visibly_corrupt(text)
            or TextUtils.is_low_quality(text, tokenizer)
        )

    @staticmethod
    def is_invalid_text(text: str) -> bool:
        """Check if text is invalid for processing."""
        return not text.strip() or len(text.strip()) < 20 or not TextUtils.is_mostly_printable_ascii(text)

    @staticmethod
    def is_valid_pdf(path: str) -> bool:
        """
        Validates whether a file is a structurally valid PDF.
        Does not attempt to extract text or check semantic quality.
        """
        try:
            with open(path, "rb") as f:
                header = f.read(4)
                if header != b"%PDF":
                    raise ValueError("Not a valid PDF file header")

            # Import here to avoid circular imports
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                if not pdf.pages:
                    raise ValueError("PDF has no pages")
                _ = pdf.pages[0]  # Try accessing the first page

            return True

        except Exception:
            return False 

is_invalid_text = TextUtils.is_invalid_text 
is_gibberish = TextUtils.is_gibberish
is_visibly_corrupt = TextUtils.is_visibly_corrupt
is_low_quality = TextUtils.is_low_quality
is_valid_pdf = TextUtils.is_valid_pdf
is_bad_ocr = TextUtils.is_bad_ocr 