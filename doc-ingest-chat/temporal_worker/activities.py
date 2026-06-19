#!/usr/bin/env python3
"""
Temporal Activity for WhisperX transcription.
Wraps the existing RemoteWhisper pattern from whisperx_worker.py.
"""

import logging
import mimetypes
import os

from models.transcription_input import TranscriptionInput, TranscriptionResult
from temporalio import activity

log = logging.getLogger("ingest.temporal")

@activity.defn(name="transcribe_media")
async def transcribe_media(input: TranscriptionInput) -> TranscriptionResult:
    """Transcribe media file as a Temporal Activity."""
    file_path = input.file_path
    language = input.language
    mime_type = input.mime_type
    print(f"[ACTIVITY] transcribe_media called with file_path={file_path}", flush=True)
    
    try:
        # Check file exists — non-retryable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        # Determine if using remote or local whisper
        from config.settings import WHISPER_MODEL_ENDPOINTS
        
        if WHISPER_MODEL_ENDPOINTS.startswith(("http://", "https://")):
            # Use RemoteWhisper pattern (import from existing worker)
            import requests
            with open(file_path, "rb") as audio_file:
                files = {"file": (os.path.basename(file_path), audio_file, mime_type)}
                data = {"temperature": "0.0", "response_format": "json", "language": language}
                response = requests.post(WHISPER_MODEL_ENDPOINTS, files=files, data=data, timeout=300)
                response.raise_for_status()
                result = response.json()
            # DEBUG: print response keys for troubleshooting
            print(f"[ACTIVITY] WhisperX response keys: {list(result.keys())}, text_len={len(result.get('text', ''))}", flush=True)
            if 'error' in result:
                print(f"[ACTIVITY] WhisperX error: {result['error']}", flush=True)
        else:
            # Local whisperx path — load model and transcribe
            import whisperx
            from config.settings import COMPUTE_TYPE, DEVICE, MEDIA_BATCH_SIZE
            batch_size = MEDIA_BATCH_SIZE
            audio = whisperx.load_audio(file_path)
            result = whisperx.load_model(WHISPER_MODEL_ENDPOINTS, DEVICE, compute_type=COMPUTE_TYPE).transcribe(audio, batch_size=batch_size, language=language)
        
        segments = [{"text": result.get("text", "")}] if result.get("text", "").strip() else []
        source_file = os.path.basename(file_path)
        return TranscriptionResult(segments=segments, source_file=source_file, job_id="")
    except Exception as e:
        print(f"[ACTIVITY] Exception in transcribe_media: {e}", flush=True)
        raise