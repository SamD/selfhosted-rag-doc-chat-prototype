import logging
import os
import sys

# 1. UNIFIED BUILD-TIME CACHE
# We bake EVERYTHING into this path so it's immutable and offline at runtime.
CACHE_PATH = "/usr/local/model_cache"
os.makedirs(CACHE_PATH, exist_ok=True)

os.environ["HF_HOME"] = CACHE_PATH
os.environ["TORCH_HOME"] = CACHE_PATH
os.environ["XDG_CACHE_HOME"] = CACHE_PATH
os.environ["HF_HUB_OFFLINE"] = "0" # Allow download during build

# 2. DUMMY ENV VARS FOR SETTINGS.PY
os.environ["DEFAULT_DOC_INGEST_ROOT"] = "/tmp"
os.environ["INGEST_FOLDER"] = "/tmp"
os.environ["STAGING_FOLDER"] = "/tmp"
os.environ["EMBEDDING_MODEL_PATH"] = "/tmp"
os.environ["LLM_PATH"] = "http://localhost"
os.environ["SUPERVISOR_LLM_PATH"] = "http://localhost"
os.environ["WHISPER_MODEL_PATH"] = "/tmp"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("warmup")

def warmup():
    log.info("🔥 Starting COMPREHENSIVE Build-Time Model Warmup...")
    
    try:
        # --- A. DOCLING WARMUP ---
        log.info("📥 Downloading Docling/EasyOCR acceleration models...")
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        ocr_options = EasyOcrOptions(lang=["en"], download_enabled=True)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = ocr_options
        pipeline_options.do_table_structure = True 
        
        _ = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )
        
        log.info("✅ COMPREHENSIVE Warmup complete. All assets baked into /usr/local/model_cache")
    except Exception as e:
        log.error(f"❌ Warmup failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    warmup()
