#!/usr/bin/env python3
"""
Text processing utility functions.
"""

import logging
import re
import string
import unicodedata
from typing import Any

import regex  # third-party regex with Unicode script support
from config.settings import ALLOW_LATIN_EXTENDED, EMBEDDING_MODEL_PATH, LATIN_SCRIPT_MIN_RATIO
from ftfy import fix_text
from transformers import AutoTokenizer

log = logging.getLogger("ingest.text_utils")

# Global tokenizer cache (Lazy initialized per process)
_CACHED_TOKENIZER = None


def get_tokenizer():
    """Lazy initializer for the shared tokenizer."""
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is None:
        try:
            log.info(f"🚀 Loading tokenizer from {EMBEDDING_MODEL_PATH}")
            _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH, use_fast=True, trust_remote_code=True, local_files_only=True)
        except Exception as e:
            log.error(f"💥 Failed to load tokenizer: {e}")
            return None
    return _CACHED_TOKENIZER


class TextUtils:
    """Text processing utility functions as static methods."""

    @staticmethod
    def fix_mojibake(text: str) -> str:
        """Use ftfy to fix common mojibake/encoding issues before checks."""
        try:
            return fix_text(text)
        except Exception:
            return text

    @staticmethod
    def latin_script_ratio(text: str) -> float:
        """Return fraction of characters that are Latin letters or combining marks."""
        if not text:
            return 0.0
        matches = regex.findall(r"\p{Latin}|\p{M}", text)
        return (len(matches) / len(text)) if text else 0.0

    @staticmethod
    def is_visibly_corrupt(text: str) -> bool:
        """Check if text contains corruption indicators.

        When ALLOW_LATIN_EXTENDED is True, do not treat Latin ligatures/diacritics
        like 'œ' as corruption.
        """
        pattern = r"[âã¢£™žÂÃ]" if ALLOW_LATIN_EXTENDED else r"[âã¢£™žœÂÃ]"
        return re.search(pattern, text) is not None

    @staticmethod
    def is_gibberish(text: str) -> bool:
        """Check if text appears to be gibberish.

        If ALLOW_LATIN_EXTENDED is True, preserve composed characters (NFC) and
        ignore combining marks when computing noise.
        """
        if not text or not text.strip():
            return True
        # Normalize and clean text first
        text = TextUtils.fix_mojibake(text)
        normalized = unicodedata.normalize("NFC" if ALLOW_LATIN_EXTENDED else "NFKD", text)
        printable = "".join(c for c in normalized if c.isprintable())
        total = len(printable)
        if total == 0:
            return True
        # If allowing Latin extended, quickly gate by script ratio
        if ALLOW_LATIN_EXTENDED and TextUtils.latin_script_ratio(printable) >= LATIN_SCRIPT_MIN_RATIO:
            # Likely Latin script; be more lenient about non-alpha characters
            noise_denominator = max(1, total)
            non_alpha = 0
            for c in printable:
                if unicodedata.category(c) == "Mn":
                    continue
                if not (c.isalpha() or c in (" ", "\n", "\t", "-", "–", "—", "·", ".", ",", ";", ":", "(", ")", "[", "]", "'", '"')):
                    non_alpha += 1
            ratio = non_alpha / noise_denominator
            return ratio > 0.75  # more tolerant threshold for Latin script
        non_alpha = 0
        for c in printable:
            if ALLOW_LATIN_EXTENDED and unicodedata.category(c) == "Mn":
                # Ignore combining diacritical marks
                continue
            if not (c.isalpha() or c in (" ", "\n")):
                non_alpha += 1
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
        if not tokenizer:
            return False
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens) < min_tokens

    @staticmethod
    def is_bad_ocr(text: str, tokenizer: Any = None) -> bool:
        """Check if OCR text is of poor quality.

        Args:
            text: The text to check.
            tokenizer: Optional tokenizer instance. If not provided, the global one is used.
        """
        if not text or not text.strip():
            return True
        if not tokenizer:
            tokenizer = get_tokenizer()
        return TextUtils.is_gibberish(text) or TextUtils.is_visibly_corrupt(text) or TextUtils.is_low_quality(text, tokenizer)

    @staticmethod
    def is_invalid_text(text: str) -> bool:
        """Check if text is invalid for processing.

        When ALLOW_LATIN_EXTENDED is True, replace the ASCII-printable requirement
        with a Unicode-printable ratio so Latin extended characters are accepted.
        """
        if not text or not text.strip() or len(text.strip()) < 20:
            return True

        if ALLOW_LATIN_EXTENDED:
            text = TextUtils.fix_mojibake(text)
            printable_count = sum(1 for c in text if c.isprintable())
            ratio = printable_count / len(text)
            # Also require a minimum Latin script ratio if we are using Latin-extended mode
            if ratio < 0.6:
                return True
            return TextUtils.latin_script_ratio(text) < LATIN_SCRIPT_MIN_RATIO

        return not TextUtils.is_mostly_printable_ascii(text)

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
