from unittest.mock import MagicMock, patch

from utils import producer_utils


def test_is_bad_ocr_true():
    with patch("utils.text_utils.AutoTokenizer.from_pretrained") as mock_get:
        mock_tokenizer = MagicMock()
        mock_get.return_value = mock_tokenizer
        with patch("utils.text_utils.TextUtils.is_gibberish", return_value=True), patch("utils.text_utils.TextUtils.is_visibly_corrupt", return_value=False), patch("utils.text_utils.TextUtils.is_low_quality", return_value=False):
            assert producer_utils.is_bad_ocr("nonsense") is True


def test_is_bad_ocr_false():
    with patch("utils.text_utils.AutoTokenizer.from_pretrained") as mock_get:
        mock_tokenizer = MagicMock()
        mock_get.return_value = mock_tokenizer
        with patch("utils.text_utils.TextUtils.is_gibberish", return_value=False), patch("utils.text_utils.TextUtils.is_visibly_corrupt", return_value=False), patch("utils.text_utils.TextUtils.is_low_quality", return_value=False):
            assert producer_utils.is_bad_ocr("some good text") is False


def test_extract_text_from_html(tmp_path):
    html_file = tmp_path / "test.html"
    html_file.write_text("<html><body><p>Hello World</p></body></html>")
    text = producer_utils.extract_text_from_html(str(html_file))
    assert "Hello World" in text
