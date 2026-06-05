import json
import logging
import uuid

from config.settings import MAX_SESSION_TURNS, SESSION_TTL_HOURS
from services.redis_service import get_redis_client

log = logging.getLogger("ingest.chat_session_service")

SESSION_KEY_PREFIX = "session:"


class ChatSessionService:
    @staticmethod
    def get_or_create_session(session_id: str) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())
        key = f"{SESSION_KEY_PREFIX}{session_id}"
        try:
            client = get_redis_client()
            exists = client.exists(key)
            if not exists:
                client.expire(key, SESSION_TTL_HOURS * 3600)
        except Exception as e:
            log.warning(f"Redis unavailable during session lookup: {e}")
        return session_id

    @staticmethod
    def get_history(session_id: str) -> list[dict]:
        if not session_id:
            return []
        key = f"{SESSION_KEY_PREFIX}{session_id}"
        try:
            client = get_redis_client()
            raw_messages = client.lrange(key, 0, -1)
            history = []
            for raw in raw_messages:
                try:
                    history.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
            return history
        except Exception as e:
            log.warning(f"Redis unavailable during history retrieval: {e}")
            return []

    @staticmethod
    def append_history(session_id: str, user_msg: dict, assistant_msg: dict):
        if not session_id:
            return
        key = f"{SESSION_KEY_PREFIX}{session_id}"
        try:
            client = get_redis_client()
            client.rpush(key, json.dumps(user_msg), json.dumps(assistant_msg))
            client.expire(key, SESSION_TTL_HOURS * 3600)
            history_len = client.llen(key)
            max_turns = MAX_SESSION_TURNS
            if history_len > max_turns * 2:
                excess = history_len - (max_turns * 2)
                client.lpop(key, excess)
        except Exception as e:
            log.warning(f"Redis unavailable during history append: {e}")
