import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from workers.ocr_graph import run_ocr_graph


@pytest.fixture
def mock_ocr_job():
    img = np.zeros((10, 10), dtype=np.uint8)
    img_base64 = base64.b64encode(img.tobytes()).decode()
    return {"job_id": "ocr-job-1", "rel_path": "test.pdf", "page_num": 1, "image_base64": img_base64, "image_shape": [10, 10], "image_dtype": "uint8", "reply_key": "ocr_reply:test"}


def test_decode_image_node(mock_ocr_job):
    from workers.ocr_graph import decode_image_node

    state = decode_image_node(mock_ocr_job)
    assert state["np_image"] is not None
    assert state["np_image"].shape == (10, 10)


@patch("workers.ocr_graph.run_ocr")
def test_ocr_node_success(mock_run, mock_ocr_job):
    from workers.ocr_graph import ocr_node

    mock_ocr_job["np_image"] = np.zeros((10, 10))
    mock_ocr_job["status"] = "processing"
    mock_run.return_value = ("Extracted Text", "docling_easyocr", 100.0)
    state = ocr_node(mock_ocr_job)
    assert state["text"] == "Extracted Text"
    assert state["status"] == "success"
    assert state["engine"] == "docling_easyocr"


@patch("workers.ocr_graph.get_redis_client")
def test_respond_node(mock_redis, mock_ocr_job):
    from workers.ocr_graph import respond_node

    mock_ocr_job["text"] = "Extracted Text"
    mock_ocr_job["engine"] = "docling_easyocr"
    mock_ocr_job["status"] = "success"
    respond_node(mock_ocr_job)
    mock_redis.return_value.lpush.assert_called_once()


def test_run_ocr_graph_integration(mock_ocr_job):
    with patch("workers.ocr_graph.get_ocr_app") as mock_get:
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"status": "success"}
        mock_get.return_value = mock_app
        assert run_ocr_graph(mock_ocr_job) is True
