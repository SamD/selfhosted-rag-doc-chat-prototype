import json
from unittest.mock import MagicMock, patch

import pytest

from services.chat_session_service import ChatSessionService

FAKE_SESSION = "test-session-uuid-1234"


@patch("services.chat_session_service.get_redis_client")
def test_get_or_create_session_with_existing_id(mock_get_redis):
    mock_redis = MagicMock()
    mock_redis.exists.return_value = True
    mock_get_redis.return_value = mock_redis

    result = ChatSessionService.get_or_create_session(FAKE_SESSION)

    assert result == FAKE_SESSION
    mock_redis.exists.assert_called_once_with(f"session:{FAKE_SESSION}")


@patch("services.chat_session_service.get_redis_client")
def test_get_or_create_session_new_id(mock_get_redis):
    mock_redis = MagicMock()
    mock_redis.exists.return_value = False
    mock_get_redis.return_value = mock_redis

    result = ChatSessionService.get_or_create_session(FAKE_SESSION)

    assert result == FAKE_SESSION
    mock_redis.expire.assert_called_once()


@patch("services.chat_session_service.get_redis_client")
def test_get_or_create_session_empty_id_generates_uuid(mock_get_redis):
    mock_redis = MagicMock()
    mock_get_redis.return_value = mock_redis

    result = ChatSessionService.get_or_create_session("")

    assert result != ""
    assert isinstance(result, str)
    assert len(result) > 10


@patch("services.chat_session_service.get_redis_client")
def test_get_or_create_session_redis_unavailable(mock_get_redis):
    mock_get_redis.side_effect = Exception("Connection refused")

    result = ChatSessionService.get_or_create_session(FAKE_SESSION)

    assert result == FAKE_SESSION


@patch("services.chat_session_service.get_redis_client")
def test_get_history_returns_messages(mock_get_redis):
    mock_redis = MagicMock()
    mock_redis.lrange.return_value = [
        json.dumps({"role": "user", "content": "hello"}),
        json.dumps({"role": "assistant", "content": "hi there"}),
    ]
    mock_get_redis.return_value = mock_redis

    history = ChatSessionService.get_history(FAKE_SESSION)

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"
    assert history[1]["role"] == "assistant"
    mock_redis.lrange.assert_called_once_with(f"session:{FAKE_SESSION}", 0, -1)


@patch("services.chat_session_service.get_redis_client")
def test_get_history_empty_session_id(mock_get_redis):
    history = ChatSessionService.get_history("")
    assert history == []
    mock_get_redis.assert_not_called()


@patch("services.chat_session_service.get_redis_client")
def test_get_history_redis_unavailable(mock_get_redis):
    mock_get_redis.side_effect = Exception("Connection refused")

    history = ChatSessionService.get_history(FAKE_SESSION)

    assert history == []


@patch("services.chat_session_service.get_redis_client")
def test_get_history_skips_corrupted_json(mock_get_redis):
    mock_redis = MagicMock()
    mock_redis.lrange.return_value = [
        json.dumps({"role": "user", "content": "valid"}),
        "not-json",
    ]
    mock_get_redis.return_value = mock_redis

    history = ChatSessionService.get_history(FAKE_SESSION)

    assert len(history) == 1
    assert history[0]["content"] == "valid"


@patch("services.chat_session_service.get_redis_client")
def test_append_history_rpush_and_expire(mock_get_redis):
    mock_redis = MagicMock()
    mock_redis.llen.return_value = 2
    mock_get_redis.return_value = mock_redis

    ChatSessionService.append_history(
        FAKE_SESSION,
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    )

    assert mock_redis.rpush.call_count == 1
    assert mock_redis.expire.call_count == 1
    args, _ = mock_redis.rpush.call_args
    assert args[0] == f"session:{FAKE_SESSION}"


@patch("services.chat_session_service.get_redis_client")
def test_append_history_trims_when_exceeded(mock_get_redis):
    mock_redis = MagicMock()
    mock_redis.llen.return_value = 50
    mock_get_redis.return_value = mock_redis

    ChatSessionService.append_history(
        FAKE_SESSION,
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    )

    mock_redis.lpop.assert_called_once()


@patch("services.chat_session_service.get_redis_client")
def test_append_history_noop_for_empty_session(mock_get_redis):
    ChatSessionService.append_history("", {"role": "user"}, {"role": "assistant"})
    mock_get_redis.assert_not_called()


@patch("services.chat_session_service.get_redis_client")
def test_append_history_redis_unavailable(mock_get_redis):
    mock_get_redis.side_effect = Exception("Connection refused")

    ChatSessionService.append_history(FAKE_SESSION, {"role": "user"}, {"role": "assistant"})

    assert True
