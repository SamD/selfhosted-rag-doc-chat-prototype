from utils import producer_utils


def test_extract_text_from_html(tmp_path):
    html_file = tmp_path / "test.html"
    html_file.write_text("<html><body><p>Hello World</p></body></html>")
    text = producer_utils.extract_text_from_html(str(html_file))
    assert "Hello World" in text
