from unittest.mock import patch

from handlers.mp3_handler import MP3ContentTypeHandler
from handlers.mp4_handler import MP4ContentTypeHandler


def test_mp4_handler_delegation():
    """Verify MP4 handler delegates to whisper_utils with trace_id."""
    handler = MP4ContentTypeHandler()
    with patch("handlers.mp4_handler.send_media_to_whisperx") as mock_send, \
         patch("handlers.mp4_handler.get_trace_id", return_value="trace-abc"):
        mock_send.return_value = iter(["Segment 1", "Segment 2"])
        results = list(handler.stream_content("video.mp4"))
        assert results == ["Segment 1", "Segment 2"]
        mock_send.assert_called_once_with("video.mp4", trace_id="trace-abc")

def test_mp3_handler_delegation():
    """Verify MP3 handler delegates to whisper_utils with trace_id."""
    handler = MP3ContentTypeHandler()
    with patch("handlers.mp3_handler.send_media_to_whisperx") as mock_send, \
         patch("handlers.mp3_handler.get_trace_id", return_value="trace-abc"):
        mock_send.return_value = iter(["Audio Segment"])
        results = list(handler.stream_content("audio.mp3"))
        assert results == ["Audio Segment"]
        mock_send.assert_called_once_with("audio.mp3", trace_id="trace-abc")
