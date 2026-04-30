import os
import tempfile
import time

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption
from PIL import Image


def test_ocr():
    print("🚀 Initializing Docling...")
    ocr_options = EasyOcrOptions(lang=["en"], use_gpu=True)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        },
    )

    # Create a small image with some text
    img = Image.new('RGB', (200, 100), color = (73, 109, 137))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.text((10,10), "Hello World", fill=(255,255,0))
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name

    print(f"🔄 Converting image {tmp_path}...")
    start = time.perf_counter()
    try:
        result = converter.convert(tmp_path)
        text = result.document.export_to_text().strip()
        elapsed = time.perf_counter() - start
        print(f"✅ Extracted in {elapsed:.2f}s: {text}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    test_ocr()
