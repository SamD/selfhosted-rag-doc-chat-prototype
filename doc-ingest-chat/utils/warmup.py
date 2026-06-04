import logging
import os

# 1. UNIFIED BUILD-TIME CACHE
# We bake EVERYTHING into this path so it's immutable and offline at runtime.
CACHE_PATH = "/usr/local/model_cache"
os.makedirs(CACHE_PATH, exist_ok=True)

os.environ["HF_HOME"] = CACHE_PATH
os.environ["TORCH_HOME"] = CACHE_PATH
os.environ["XDG_CACHE_HOME"] = CACHE_PATH
os.environ["EASYOCR_MODULE_PATH"] = CACHE_PATH
os.environ["HF_HUB_OFFLINE"] = "0" # Allow download during build

# 2. DUMMY ENV VARS FOR SETTINGS.PY
os.environ["DEFAULT_DOC_INGEST_ROOT"] = "/tmp"
os.environ["INGEST_FOLDER"] = "/tmp"
os.environ["STAGING_FOLDER"] = "/tmp"
os.environ["EMBEDDING_MODEL_PATH"] = "/tmp"
os.environ["LLM_PATH"] = "http://localhost"
os.environ["SUPERVISOR_LLM_PATH"] = "http://localhost"
os.environ["WHISPER_MODEL_ENDPOINTS"] = "/tmp"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("warmup")

def warmup():
    log.info("🔥 Starting COMPREHENSIVE Build-Time Model Warmup...")
    
    ocr_path = os.environ.get("OCR_ENDPOINTS", "LOCAL")

    if ocr_path.startswith(("http://", "https://")):
        log.info(f"⏭️ OCR_ENDPOINTS is remote ({ocr_path}), skipping Docling warmup")
    else:
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
            
            log.info("✅ Docling warmup complete")
        except Exception as e:
            log.warning(f"⚠️ Docling warmup failed (non-fatal): {e}")

    try:
        # --- B. TOKENIZER WARMUP ---
        log.info("📥 Downloading E5 Tokenizer (fallback model)...")
        from transformers import AutoTokenizer
        fallback_dir = os.path.join(CACHE_PATH, "e5-fallback")
        tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
        tokenizer.save_pretrained(fallback_dir)
        log.info(f"✅ Tokenizer warmup complete. Files saved to {fallback_dir}")
    except Exception as e:
        log.warning(f"⚠️ Tokenizer warmup failed (non-fatal): {e}")

    log.info("✅ Warmup complete")

if __name__ == "__main__":
    warmup()
