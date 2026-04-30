from handlers.mp4_handler import MP4ContentTypeHandler
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
    # We can't easily check which one handled it without mocking,
    # but we can check if handle returns a generator or None
    # For a non-existent file, it should still return the generator if can_handle passes
    # because the generator is only exhausted when iterated.
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
