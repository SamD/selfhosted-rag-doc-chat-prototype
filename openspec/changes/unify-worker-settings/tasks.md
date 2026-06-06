## 1. WhisperX Worker

- [ ] 1.1 Add sys.path fix and `from config.settings import REDIS_HOST, REDIS_PORT, REDIS_WHISPER_JOB_QUEUE, DEVICE, COMPUTE_TYPE, MEDIA_BATCH_SIZE, WHISPER_MODEL_ENDPOINTS`
- [ ] 1.2 Replace all 6 module-level `os.getenv()` calls with the imported settings

## 2. OCR Worker

- [ ] 2.1 Replace `os.getenv('EMBEDDING_ENDPOINTS')` and `os.getenv('OCR_ENDPOINTS', 'LOCAL')` debug prints with `from config.settings import EMBEDDING_ENDPOINTS, OCR_ENDPOINTS`

## 3. Verification

- [ ] 3.1 Run `ruff check --fix` on all changed files
- [ ] 3.2 Run full pytest suite — no regressions
