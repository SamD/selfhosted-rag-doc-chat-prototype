from typing import Generator

from utils.trace_utils import get_logger, get_trace_id
from utils.whisper_utils import send_media_to_whisperx

from .base_handler import BaseContentTypeHandler

log = get_logger("ingest.handlers.mp3")

class MP3ContentTypeHandler(BaseContentTypeHandler):
    """
    Handler for MP3 files using a dedicated WhisperX worker.
    """

    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith(".mp3") or file_path.lower().endswith(".wav")

    def stream_content(self, file_path: str) -> Generator[str, None, None]:
        """
        Delegates transcription to the WhisperX worker.
        """
        log.info(f"🎵 Transcribing Audio via dedicated worker: {file_path}")
        try:
            yield from send_media_to_whisperx(file_path, trace_id=get_trace_id())
        except Exception as e:
            log.error(f"❌ Audio transcription delegation failed: {e}")
            raise
