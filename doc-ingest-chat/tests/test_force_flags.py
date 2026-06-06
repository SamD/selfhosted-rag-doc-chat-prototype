import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.getcwd(), "doc-ingest-chat"))

from handlers.pdf_handler import PDFContentTypeHandler
from workers.gatekeeper_logic import process_chunk


class TestForceMarkdownLLM(unittest.TestCase):

    @patch('os.fsync')
    @patch('workers.gatekeeper_logic.get_llm_and_grammar')
    @patch('workers.gatekeeper_logic.is_bad_ocr')
    @patch('utils.text_utils.get_tokenizer')
    @patch('builtins.open', new_callable=MagicMock)
    def test_force_markdown_llm_bypasses_quality_check(self, mock_open, mock_get_tokenizer, mock_bad_ocr, mock_get_llm_grammar, mock_fsync):
        """When FORCE_MARKDOWN_LLM is true, quality check is skipped and LLM is always called."""
        mock_bad_ocr.return_value = False  # Would normally bypass LLM

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1] * 10
        mock_tokenizer.decode.return_value = "processed"
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_llm = MagicMock()
        mock_get_llm_grammar.return_value = (mock_llm, "fake-grammar")
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "LLM processed"}}]
        }

        with patch('config.settings.FORCE_MARKDOWN_LLM', True):
            meta, content = process_chunk(0, "clean text", "test.pdf", "test-slug", "/tmp/test.md")

            # LLM should have been called despite clean text
            mock_llm.create_chat_completion.assert_called_once()
            self.assertEqual(content, "LLM processed")

    @patch('os.fsync')
    @patch('workers.gatekeeper_logic.get_llm_and_grammar')
    @patch('workers.gatekeeper_logic.is_bad_ocr')
    @patch('utils.text_utils.get_tokenizer')
    @patch('builtins.open', new_callable=MagicMock)
    def test_default_bypasses_llm_when_quality_passes(self, mock_open, mock_get_tokenizer, mock_bad_ocr, mock_get_llm_grammar, mock_fsync):
        """When FORCE_MARKDOWN_LLM is false (default), clean text bypasses the LLM."""
        mock_bad_ocr.return_value = False  # Quality check passes

        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_llm = MagicMock()
        mock_get_llm_grammar.return_value = (mock_llm, "fake-grammar")

        with patch('config.settings.FORCE_MARKDOWN_LLM', False):
            meta, content = process_chunk(0, "clean text", "test.pdf", "test-slug", "/tmp/test.md")

            # LLM should NOT have been called
            mock_llm.create_chat_completion.assert_not_called()
            self.assertEqual(content, "clean text")


class TestPDFForceOCR(unittest.TestCase):

    @patch('handlers.pdf_handler.preprocess_image')
    @patch('handlers.pdf_handler.convert_from_path')
    @patch('handlers.pdf_handler.send_image_to_ocr')
    @patch('handlers.pdf_handler.is_bad_ocr')
    @patch('handlers.pdf_handler.pdfplumber')
    def test_pdf_force_ocr_skips_pdfplumber(self, mock_pdfplumber, mock_is_bad_ocr, mock_send_ocr, mock_convert, mock_preprocess):
        """When PDF_FORCE_OCR is true, pdfplumber is not called and all pages go to OCR."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "valid text"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        mock_preprocess.return_value = MagicMock()
        mock_convert.return_value = [MagicMock()]
        mock_send_ocr.return_value = ("ocr text", None, None, "docling", None, None)
        mock_is_bad_ocr.return_value = True

        handler = PDFContentTypeHandler()

        with patch('handlers.pdf_handler.PDF_FORCE_OCR', True):
            result = list(handler.stream_content("/tmp/test.pdf"))
            mock_page.extract_text.assert_not_called()
            self.assertEqual(result, ["ocr text"])

    @patch('handlers.pdf_handler.preprocess_image')
    @patch('handlers.pdf_handler.convert_from_path')
    @patch('handlers.pdf_handler.send_image_to_ocr')
    @patch('handlers.pdf_handler.is_bad_ocr')
    @patch('handlers.pdf_handler.pdfplumber')
    def test_pdf_default_uses_pdfplumber(self, mock_pdfplumber, mock_is_bad_ocr, mock_send_ocr, mock_convert, mock_preprocess):
        """When PDF_FORCE_OCR is false (default), pdfplumber is tried first."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "valid text"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        mock_is_bad_ocr.return_value = False

        handler = PDFContentTypeHandler()

        with patch('handlers.pdf_handler.PDF_FORCE_OCR', False):
            result = list(handler.stream_content("/tmp/test.pdf"))
            mock_page.extract_text.assert_called_once()
            self.assertEqual(result, ["valid text"])


if __name__ == "__main__":
    unittest.main()
