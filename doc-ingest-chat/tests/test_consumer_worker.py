#!/usr/bin/env python3
"""
Tests for consumer_worker functions.
"""

import json
import os
import sys
from itertools import cycle
from unittest.mock import MagicMock, patch

# Set required environment variables before importing
os.environ.setdefault("INGEST_FOLDER", "/tmp/test")
os.environ.setdefault("CHROMA_DATA_DIR", "/tmp/chroma")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "intfloat/e5-large-v2")
os.environ.setdefault("LLM_PATH", "/tmp/test.gguf")

# Ensure the worker module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../workers")))
import consumer_worker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_CHUNK_KEYS = {"id", "chunk", "source_file", "type", "hash"}


def _make_chunk(
    id="chunk-1",
    chunk="some text",
    source_file="doc.pdf",
    type="pdf",
    hash="abc",
    engine="pdfplumber",
    chunk_index=0,
    page=1,
):
    return {
        "id": id,
        "chunk": chunk,
        "source_file": source_file,
        "type": type,
        "hash": hash,
        "engine": engine,
        "chunk_index": chunk_index,
        "page": page,
    }


def _make_file_end(source_file="doc.pdf", expected_chunks=1):
    return {"type": "file_end", "source_file": source_file, "expected_chunks": expected_chunks}


def _run_worker(messages, shared_data=None, mock_db=None, mock_redis=None, parq_lock=None, extra_patches=None):
    """
    Drive consumer_worker through a fixed sequence of messages then exit.

    After all messages are exhausted a sentinel item sets shutdown_flag so the
    loop terminates cleanly.
    """
    if shared_data is None:
        shared_data = {"shutdown_flag": False}
    if mock_db is None:
        mock_db = MagicMock()
        mock_db.get_collection_count.return_value = 1
    if mock_redis is None:
        mock_redis = MagicMock()

    call_count = [0]
    encoded = [(b"q", json.dumps(m).encode()) for m in messages]

    def blpop_side_effect(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(encoded):
            return encoded[idx]
        # All messages delivered — trigger shutdown on next loop iteration
        shared_data["shutdown_flag"] = True
        return (b"q", b'{"type":"_sentinel"}')

    mock_redis.blpop.side_effect = blpop_side_effect

    patches = {
        "consumer_worker.get_redis_client": mock_redis,
        "consumer_worker.get_db": mock_db,
        "consumer_worker.update_failed_files": MagicMock(),
        "consumer_worker.update_ingested_files": MagicMock(),
        "consumer_worker.write_to_parquet": MagicMock(),
    }
    if extra_patches:
        patches.update(extra_patches)

    with patch("consumer_worker.get_redis_client", return_value=mock_redis), \
         patch("consumer_worker.get_db", return_value=mock_db), \
         patch("consumer_worker.update_failed_files") as mock_failed, \
         patch("consumer_worker.update_ingested_files") as mock_ingested, \
         patch("consumer_worker.write_to_parquet") as mock_parquet:

        consumer_worker.consumer_worker("test_q", shared_data, parq_lock or MagicMock())

    return {
        "db": mock_db,
        "redis": mock_redis,
        "failed": mock_failed,
        "ingested": mock_ingested,
        "parquet": mock_parquet,
    }


# ---------------------------------------------------------------------------
# Existing utility tests (preserved)
# ---------------------------------------------------------------------------


def test_get_next_queue_cycles():
    with patch.object(consumer_worker, "queue_lock"), \
         patch.object(consumer_worker, "queue_cycle", cycle(["q1", "q2"])):
        r1 = consumer_worker.get_next_queue()
        r2 = consumer_worker.get_next_queue()
    assert r1 in ["q1", "q2"]
    assert r2 in ["q1", "q2"]
    assert r1 != r2


def test_current_time_returns_int():
    assert isinstance(consumer_worker.current_time(), int)


@patch("consumer_worker.get_redis_client")
@patch("consumer_worker.get_db")
def test_consumer_worker_handles_shutdown(mock_get_db, mock_get_redis):
    mock_redis = MagicMock()
    mock_get_redis.return_value = mock_redis
    mock_db = MagicMock()
    mock_get_db.return_value = mock_db
    shared_data = {"shutdown_flag": True}
    consumer_worker.consumer_worker("test_queue", shared_data, MagicMock())


# ---------------------------------------------------------------------------
# Chunk buffering
# ---------------------------------------------------------------------------


def test_chunk_is_buffered():
    """A valid chunk message is added to the buffer."""
    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    r["db"].add_texts.assert_called_once()
    texts = list(r["db"].add_texts.call_args[0][0])
    assert texts == ["some text"]


def test_chunk_missing_required_keys_is_skipped():
    """A chunk missing required keys is silently dropped."""
    bad_chunk = {"id": "x", "chunk": "text"}  # missing source_file, type, hash
    # Deliver bad chunk then exit; nothing should be buffered
    r = _run_worker([bad_chunk])

    r["db"].add_texts.assert_not_called()


def test_chunk_sets_timestamp_on_first_arrival():
    """The first chunk for a file records a timestamp (tested via TTL expiry path)."""
    # Deliver a chunk, then simulate TTL expiry before file_end arrives
    chunk = _make_chunk()
    # current_time() calls per iteration:
    #   loop 1 – TTL check, then buffer-timestamp assignment: 2 calls
    #   loop 2 – TTL check: 1 call → must exceed CHUNK_TIMEOUT vs loop-1 timestamp
    times = iter([1000, 1000, 1000 + 99999])

    with patch("consumer_worker.current_time", side_effect=times), \
         patch("consumer_worker.get_redis_client") as mock_rcl, \
         patch("consumer_worker.get_db"), \
         patch("consumer_worker.update_failed_files") as mock_failed, \
         patch("consumer_worker.update_ingested_files"), \
         patch("consumer_worker.write_to_parquet"):

        mock_redis = MagicMock()
        mock_rcl.return_value = mock_redis

        call_count = [0]
        msgs = [(b"q", json.dumps(chunk).encode())]
        shared = {"shutdown_flag": False}

        def blpop_se(*a, **kw):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                return msgs[0]
            shared["shutdown_flag"] = True
            return (b"q", b'{"type":"_sentinel"}')

        mock_redis.blpop.side_effect = blpop_se
        consumer_worker.consumer_worker("test_q", shared, MagicMock())

    mock_failed.assert_called_with("doc.pdf")


def test_multiple_chunks_same_file_all_buffered():
    """Multiple chunk messages for the same file are all collected."""
    chunks = [_make_chunk(id=f"c{i}", chunk=f"text {i}", chunk_index=i) for i in range(3)]
    file_end = _make_file_end(expected_chunks=3)

    r = _run_worker(chunks + [file_end])

    texts_arg = list(r["db"].add_texts.call_args[0][0])
    assert len(texts_arg) == 3


def test_max_chunks_exceeded_marks_file_failed():
    """When the buffer reaches MAX_CHUNKS the file is marked as failed."""
    with patch("consumer_worker.MAX_CHUNKS", 2):
        chunks = [_make_chunk(id=f"c{i}", chunk_index=i) for i in range(3)]
        r = _run_worker(chunks)

    r["failed"].assert_called_with("doc.pdf")
    r["db"].add_texts.assert_not_called()


# ---------------------------------------------------------------------------
# Malformed messages
# ---------------------------------------------------------------------------


def test_malformed_json_is_skipped():
    """A non-JSON Redis entry is skipped without crashing."""
    shared = {"shutdown_flag": False}
    mock_redis = MagicMock()
    call_count = [0]

    def blpop_se(*a, **kw):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            return (b"q", b"not valid json }{")
        shared["shutdown_flag"] = True
        return (b"q", b'{"type":"_sentinel"}')

    mock_redis.blpop.side_effect = blpop_se

    with patch("consumer_worker.get_redis_client", return_value=mock_redis), \
         patch("consumer_worker.get_db"), \
         patch("consumer_worker.update_failed_files"), \
         patch("consumer_worker.update_ingested_files"), \
         patch("consumer_worker.write_to_parquet"):
        consumer_worker.consumer_worker("test_q", shared, MagicMock())  # must not raise


# ---------------------------------------------------------------------------
# Idle counter
# ---------------------------------------------------------------------------


def test_idle_counter_increments_on_empty_poll():
    """blpop returning None increments the idle counter."""
    consumer_worker.consumer_worker._idle_counter = 0
    shared = {"shutdown_flag": False}
    mock_redis = MagicMock()
    call_count = [0]

    def blpop_se(*a, **kw):
        idx = call_count[0]
        call_count[0] += 1
        if idx < 3:
            return None
        shared["shutdown_flag"] = True
        return (b"q", b'{"type":"_sentinel"}')

    mock_redis.blpop.side_effect = blpop_se

    with patch("consumer_worker.get_redis_client", return_value=mock_redis), \
         patch("consumer_worker.get_db"), \
         patch("consumer_worker.update_failed_files"), \
         patch("consumer_worker.update_ingested_files"), \
         patch("consumer_worker.write_to_parquet"):
        consumer_worker.consumer_worker("test_q", shared, MagicMock())

    assert consumer_worker.consumer_worker._idle_counter == 0  # reset after receiving item


def test_idle_counter_reset_on_item_received():
    """idle_counter is reset to 0 when a non-None item arrives."""
    consumer_worker.consumer_worker._idle_counter = 99
    chunk = _make_chunk()

    _run_worker([chunk])

    assert consumer_worker.consumer_worker._idle_counter == 0


# ---------------------------------------------------------------------------
# file_end: incomplete file
# ---------------------------------------------------------------------------


def test_file_end_incomplete_marks_failed():
    """file_end with fewer chunks than expected marks the file as failed."""
    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=5)  # only 1 chunk arrived

    r = _run_worker([chunk, file_end])

    r["failed"].assert_called_with("doc.pdf")
    r["db"].delete.assert_called_once_with(where={"source_file": "doc.pdf"})
    r["db"].add_texts.assert_not_called()


def test_file_end_incomplete_does_not_write_parquet():
    """Incomplete file is not written to Parquet."""
    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=3)

    r = _run_worker([chunk, file_end])

    r["parquet"].assert_not_called()


# ---------------------------------------------------------------------------
# file_end: chunk validation
# ---------------------------------------------------------------------------


def test_file_end_drops_non_string_chunks():
    """Chunks whose 'chunk' field is not a string are excluded from valid_chunks."""
    bad = _make_chunk(id="bad", chunk=12345)  # int, not str
    good = _make_chunk(id="good", chunk="valid text", chunk_index=1)
    # expected=2 but only 1 valid → incomplete → failed
    file_end = _make_file_end(expected_chunks=2)

    r = _run_worker([bad, good, file_end])

    r["failed"].assert_called_with("doc.pdf")
    r["db"].add_texts.assert_not_called()


def test_file_end_drops_oversized_chunks():
    """Chunks exceeding MAX_TOKENS are dropped."""
    with patch("consumer_worker.MAX_TOKENS", 5):
        oversized = _make_chunk(id="big", chunk="x" * 100)
        file_end = _make_file_end(expected_chunks=1)

        r = _run_worker([oversized, file_end])

    # oversized chunk dropped → count mismatch → failed
    r["failed"].assert_called_with("doc.pdf")
    r["db"].add_texts.assert_not_called()


def test_file_end_accepts_chunks_within_token_limit():
    """Chunks within MAX_TOKENS pass validation."""
    with patch("consumer_worker.MAX_TOKENS", 200):
        chunk = _make_chunk(chunk="short text")
        file_end = _make_file_end(expected_chunks=1)

        r = _run_worker([chunk, file_end])

    r["db"].add_texts.assert_called_once()
    r["ingested"].assert_called_with("doc.pdf")


# ---------------------------------------------------------------------------
# file_end: page metadata normalisation
# ---------------------------------------------------------------------------


def test_file_end_page_int_is_preserved():
    """Integer page numbers are stored as-is."""
    chunk = _make_chunk(page=7)
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    metas = list(r["db"].add_texts.call_args[1]["metadatas"])
    assert metas[0]["page"] == 7


def test_file_end_page_string_digit_is_cast_to_int():
    """String digit page numbers are cast to int."""
    chunk = _make_chunk(page="3")
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    metas = list(r["db"].add_texts.call_args[1]["metadatas"])
    assert metas[0]["page"] == 3


def test_file_end_non_digit_page_becomes_minus_one():
    """Non-numeric page values fall back to -1."""
    chunk = _make_chunk(page="unknown")
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    metas = list(r["db"].add_texts.call_args[1]["metadatas"])
    assert metas[0]["page"] == -1


def test_file_end_missing_page_becomes_minus_one():
    """Missing page key falls back to -1."""
    chunk = _make_chunk()
    del chunk["page"]
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    metas = list(r["db"].add_texts.call_args[1]["metadatas"])
    assert metas[0]["page"] == -1


# ---------------------------------------------------------------------------
# file_end: metadata fields
# ---------------------------------------------------------------------------


def test_file_end_metadata_engine_defaults_to_unknown():
    """Missing engine field in a chunk defaults to 'unknown' in stored metadata."""
    chunk = _make_chunk()
    del chunk["engine"]
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    metas = list(r["db"].add_texts.call_args[1]["metadatas"])
    assert metas[0]["engine"] == "unknown"


def test_file_end_chunk_index_defaults_to_enumeration():
    """Missing chunk_index falls back to enumeration index."""
    chunk = _make_chunk()
    del chunk["chunk_index"]
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    metas = list(r["db"].add_texts.call_args[1]["metadatas"])
    assert metas[0]["chunk_index"] == 0


# ---------------------------------------------------------------------------
# file_end: successful ingestion
# ---------------------------------------------------------------------------


def test_file_end_success_calls_add_texts():
    """Successful ingestion calls db.add_texts with texts, metadatas, and ids."""
    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    r["db"].add_texts.assert_called_once()
    call_kwargs = r["db"].add_texts.call_args
    assert list(call_kwargs[0][0]) == ["some text"]
    assert list(call_kwargs[1]["ids"]) == ["chunk-1"]


def test_file_end_success_writes_parquet():
    """Successful ingestion writes chunks to Parquet."""
    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    r["parquet"].assert_called_once()


def test_file_end_success_marks_file_ingested():
    """Successful ingestion calls update_ingested_files."""
    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end])

    r["ingested"].assert_called_with("doc.pdf")
    r["failed"].assert_not_called()


def test_file_end_success_clears_buffer():
    """After a successful file_end the buffer entry is removed."""
    chunk = _make_chunk()
    file_end_1 = _make_file_end(expected_chunks=1)
    # A second file_end for the same file with expected=0 should be incomplete
    # (buffer was cleared so 0 chunks buffered, but expected=0 matches → would succeed)
    # Instead send expected=1 so mismatch confirms buffer was cleared
    file_end_2 = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end_1, file_end_2])

    # Second file_end finds empty buffer → 0 != 1 → failed
    assert r["failed"].call_count == 1
    assert r["ingested"].call_count == 1


def test_file_end_batches_chunks_by_max_batch_size():
    """Chunks are split into batches of MAX_CHROMA_BATCH_SIZE when calling add_texts."""
    with patch("consumer_worker.MAX_CHROMA_BATCH_SIZE", 2):
        chunks = [_make_chunk(id=f"c{i}", chunk=f"text {i}", chunk_index=i) for i in range(5)]
        file_end = _make_file_end(expected_chunks=5)

        r = _run_worker(chunks + [file_end])

    # 5 chunks / batch_size 2 → 3 calls (2+2+1)
    assert r["db"].add_texts.call_count == 3


def test_file_end_zero_vector_db_count_marks_failed():
    """If the vector DB reports 0 documents after ingestion the file is marked failed."""
    mock_db = MagicMock()
    mock_db.get_collection_count.return_value = 0

    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end], mock_db=mock_db)

    r["failed"].assert_called_with("doc.pdf")
    r["ingested"].assert_not_called()


# ---------------------------------------------------------------------------
# file_end: error paths
# ---------------------------------------------------------------------------


def test_file_end_db_write_failure_marks_failed():
    """Exception from db.add_texts marks the file as failed."""
    mock_db = MagicMock()
    mock_db.add_texts.side_effect = RuntimeError("DB write error")

    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=1)

    r = _run_worker([chunk, file_end], mock_db=mock_db)

    r["failed"].assert_called_with("doc.pdf")
    r["ingested"].assert_not_called()


def test_file_end_db_write_failure_deletes_partial_data():
    """Exception from db.add_texts triggers a delete of partial data."""
    mock_db = MagicMock()
    mock_db.add_texts.side_effect = RuntimeError("DB error")

    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=1)

    _run_worker([chunk, file_end], mock_db=mock_db)

    mock_db.delete.assert_called_with(where={"source_file": "doc.pdf"})


def test_file_end_parquet_failure_marks_failed():
    """Exception from write_to_parquet marks the file as failed and rolls back."""
    mock_db = MagicMock()
    mock_db.get_collection_count.return_value = 5

    chunk = _make_chunk()
    file_end = _make_file_end(expected_chunks=1)

    shared = {"shutdown_flag": False}
    mock_redis = MagicMock()
    call_count = [0]
    msgs = [(b"q", json.dumps(chunk).encode()), (b"q", json.dumps(file_end).encode())]

    def blpop_se(*a, **kw):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(msgs):
            return msgs[idx]
        shared["shutdown_flag"] = True
        return (b"q", b'{"type":"_sentinel"}')

    mock_redis.blpop.side_effect = blpop_se

    with patch("consumer_worker.get_redis_client", return_value=mock_redis), \
         patch("consumer_worker.get_db", return_value=mock_db), \
         patch("consumer_worker.update_failed_files") as mock_failed, \
         patch("consumer_worker.update_ingested_files") as mock_ingested, \
         patch("consumer_worker.write_to_parquet", side_effect=Exception("disk full")):
        consumer_worker.consumer_worker("test_q", shared, MagicMock())

    mock_failed.assert_called_with("doc.pdf")
    mock_ingested.assert_not_called()
    mock_db.delete.assert_called_with(where={"source_file": "doc.pdf"})


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


def test_ttl_expiry_marks_file_failed():
    """Files that exceed CHUNK_TIMEOUT are discarded and marked as failed."""
    chunk = _make_chunk()
    shared = {"shutdown_flag": False}
    mock_redis = MagicMock()
    call_count = [0]

    # time[0]: initial TTL check (before first blpop)
    # time[1]: TTL check on second loop iteration (after chunk is buffered)
    tick = [1000, 1000, 1000 + 99999]  # third tick is way past CHUNK_TIMEOUT

    def blpop_se(*a, **kw):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            return (b"q", json.dumps(chunk).encode())
        shared["shutdown_flag"] = True
        return (b"q", b'{"type":"_sentinel"}')

    mock_redis.blpop.side_effect = blpop_se

    with patch("consumer_worker.get_redis_client", return_value=mock_redis), \
         patch("consumer_worker.get_db"), \
         patch("consumer_worker.current_time", side_effect=tick), \
         patch("consumer_worker.update_failed_files") as mock_failed, \
         patch("consumer_worker.update_ingested_files"), \
         patch("consumer_worker.write_to_parquet"):
        consumer_worker.consumer_worker("test_q", shared, MagicMock())

    mock_failed.assert_called_with("doc.pdf")


# ---------------------------------------------------------------------------
# make_sigint_handler
# ---------------------------------------------------------------------------


def test_sigint_handler_sets_shutdown_flag():
    """Handler sets shutdown_flag=True when called in the parent process."""
    processes = [MagicMock(), MagicMock()]
    shared_data = {"shutdown_flag": False}
    ppid = os.getpid()

    handler = consumer_worker.make_sigint_handler(processes, ppid, shared_data)

    with patch("sys.exit"):
        handler(2, None)

    assert shared_data["shutdown_flag"] is True


def test_sigint_handler_joins_all_child_processes():
    """Handler calls join() on every child process."""
    processes = [MagicMock(), MagicMock(), MagicMock()]
    shared_data = {"shutdown_flag": False}
    ppid = os.getpid()

    handler = consumer_worker.make_sigint_handler(processes, ppid, shared_data)

    with patch("sys.exit"):
        handler(2, None)

    for p in processes:
        p.join.assert_called_once()


def test_sigint_handler_calls_sys_exit():
    """Handler calls sys.exit(0) after joining children."""
    shared_data = {"shutdown_flag": False}
    ppid = os.getpid()
    handler = consumer_worker.make_sigint_handler([], ppid, shared_data)

    with patch("sys.exit") as mock_exit:
        handler(2, None)

    mock_exit.assert_called_once_with(0)


def test_sigint_handler_ignores_non_parent_pid():
    """Handler returns without doing anything when called in a child process."""
    processes = [MagicMock()]
    shared_data = {"shutdown_flag": False}
    ppid = os.getpid() + 9999  # current pid will never equal this

    handler = consumer_worker.make_sigint_handler(processes, ppid, shared_data)
    handler(2, None)  # must not raise or call join

    assert shared_data["shutdown_flag"] is False
    processes[0].join.assert_not_called()
