from handlers.mp4_handler import MP4ContentTypeHandler
from handlers.mp3_handler import MP3ContentTypeHandler
from handlers.pdf_handler import PDFContentTypeHandler
from handlers.text_handler import TextContentTypeHandler


def test_handler_chain_pdf():
    text_handler = TextContentTypeHandler()
    mp4_handler = MP4ContentTypeHandler(next_handler=text_handler)
    pdf_handler = PDFContentTypeHandler(next_handler=mp4_handler)

    # PDF should be handled by PDF handler
    assert pdf_handler.can_handle("test.pdf") is True
    assert mp4_handler.can_handle("test.pdf") is False

    # Chain traversal
    stream = pdf_handler.handle("test.pdf")
    assert stream is not None
    assert hasattr(stream, "__iter__")


def test_handler_chain_mp4():
    text_handler = TextContentTypeHandler()
    mp4_handler = MP4ContentTypeHandler(next_handler=text_handler)
    pdf_handler = PDFContentTypeHandler(next_handler=mp4_handler)

    assert pdf_handler.can_handle("test.mp4") is False
    assert mp4_handler.can_handle("test.mp4") is True

    stream = pdf_handler.handle("test.mp4")
    assert stream is not None


def test_handler_chain_text():
    text_handler = TextContentTypeHandler()
    mp4_handler = MP4ContentTypeHandler(next_handler=text_handler)
    pdf_handler = PDFContentTypeHandler(next_handler=mp4_handler)

    assert pdf_handler.can_handle("test.txt") is False
    assert mp4_handler.can_handle("test.txt") is False
    assert text_handler.can_handle("test.txt") is True

    stream = pdf_handler.handle("test.txt")
    assert stream is not None


def test_handler_chain_unsupported():
    text_handler = TextContentTypeHandler()
    mp4_handler = MP4ContentTypeHandler(next_handler=text_handler)
    pdf_handler = PDFContentTypeHandler(next_handler=mp4_handler)

    stream = pdf_handler.handle("test.unknown")
    assert stream is None


def test_handler_mime_type_class_vars():
    """Each handler declares the correct MIME_TYPE."""
    assert MP4ContentTypeHandler.MIME_TYPE == "video/mp4"
    assert PDFContentTypeHandler.MIME_TYPE == "application/pdf"
    assert MP3ContentTypeHandler.MIME_TYPE is None  # uses get_mime_type fallback
    assert TextContentTypeHandler.MIME_TYPE is None  # uses get_mime_type fallback


def test_handler_get_mime_type_fixed():
    """Handlers with MIME_TYPE set always return that value."""
    handler = MP4ContentTypeHandler()
    assert handler.get_mime_type("anything.mp4") == "video/mp4"
    assert handler.get_mime_type("anything.mkv") == "video/mp4"  # always returns class var

    handler = PDFContentTypeHandler()
    assert handler.get_mime_type("anything.pdf") == "application/pdf"


def test_handler_get_mime_type_guessed():
    """Handlers without MIME_TYPE guess from extension."""
    handler = MP3ContentTypeHandler()
    assert handler.get_mime_type("test.mp3") == "audio/mpeg"
    assert handler.get_mime_type("test.wav") == "audio/x-wav"
    assert handler.get_mime_type("test.m4a") == "audio/mp4"
    assert handler.get_mime_type("test.flac") == "audio/flac"

    handler = TextContentTypeHandler()
    assert handler.get_mime_type("test.txt") == "text/plain"
    assert handler.get_mime_type("test.md") == "text/markdown"
    assert handler.get_mime_type("test.html") == "text/html"
