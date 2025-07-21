#!/usr/bin/env python3
"""
Redis service for queue operations.
"""
import json
import time
import redis
from typing import List, Optional, Tuple
from config.settings import REDIS_HOST, REDIS_PORT


class RedisService:
    """Service for Redis operations."""
    
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
    
    def push_to_queue(self, queue_name: str, data: dict) -> None:
        """Push data to a Redis queue."""
        self.client.lpush(queue_name, json.dumps(data))
    
    def pop_from_queue(self, queue_name: str, timeout: int = 5) -> Optional[Tuple[str, str]]:
        """Pop data from a Redis queue."""
        return self.client.blpop(queue_name, timeout=timeout)
    
    def get_queue_length(self, queue_name: str) -> int:
        """Get the length of a queue."""
        return self.client.llen(queue_name)
    
    def push_reply(self, reply_key: str, data: dict, expire: int = 300) -> None:
        """Push reply data to a reply key."""
        self.client.lpush(reply_key, json.dumps(data))
        self.client.expire(reply_key, expire)
    
    def blocking_push_with_backpressure(
        self,
        queue_name: str,
        entries: List[str],
        max_queue_length: int = 1000,
        poll_interval: float = 0.5,
        warn_after: float = 10.0,
        rel_path: str = "unknown"
    ) -> None:
        """Push entries to queue with backpressure handling."""
        push_script = self.client.register_script("""
        local queue = KEYS[1]
        local max_len = tonumber(ARGV[1])
        local new_items = {}

        for i = 2, #ARGV do
            table.insert(new_items, ARGV[i])
        end

        local current_len = redis.call("LLEN", queue)
        if current_len + #new_items <= max_len then
            for _, item in ipairs(new_items) do
                redis.call("RPUSH", queue, item)
            end
            return 1
        else
            return 0
        end
        """)

        start_wait = time.time()
        warned = False
        total_wait_time = 0
        attempt = 0

        while True:
            attempt += 1
            result = push_script(keys=[queue_name], args=[max_queue_length] + entries)

            if result == 1:
                elapsed = time.time() - start_wait
                if warned:
                    print(f"✅ Queue backpressure resolved after {elapsed:.2f}s — pushed {len(entries)} entries to '{queue_name}' for {rel_path}")
                return  # success

            if not warned and (time.time() - start_wait) > warn_after:
                qlen = self.client.llen(queue_name)
                print(f"⏳ Queue '{queue_name}' length {qlen} exceeds limit ({max_queue_length}) — backpressure delay on {rel_path}")
                warned = True

            time.sleep(poll_interval)
            total_wait_time += poll_interval
            if total_wait_time % 10 < poll_interval:  # log every 10s
                qlen = self.client.llen(queue_name)
                print(f"🔁 Still waiting to enqueue {rel_path} (queue: {queue_name}, length: {qlen}) [waited {total_wait_time:.1f}s]")


def get_redis_client() -> redis.Redis:
    """Get Redis client instance."""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True) 