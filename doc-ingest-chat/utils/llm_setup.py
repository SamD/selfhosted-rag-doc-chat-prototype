#!/usr/bin/env python3
"""
LLM and chain setup for chat system.
Uses singleton pattern to ensure models are only loaded once per process.
Supports local GGUF via llama-cpp-python or remote llama.cpp server via API.
"""

import logging
import os
import time
from typing import Any, Optional, Union

import openai
from config.llama_strategy import LlamaParamStrategy
from config.settings import (
    LLAMA_REMOTE_TIMEOUT,
    RETRIEVER_TOP_K,
    SUPERVISOR_LLM_ENDPOINTS,
    SUPERVISOR_REMOTE_MODEL_NAME,
)
from langchain_core.embeddings import Embeddings
from llama_cpp import Llama
from services.database import get_vectorstore as get_vdb

log = logging.getLogger("ingest.llm_setup")

# Singleton caches (Per-Process)
_LLAMA_MODEL_CACHE: Optional[Union[Llama, "RemoteLlama"]] = None
_SUPERVISOR_LLM_CACHE: Optional[Any] = None
_PID_OWNER: Optional[int] = None


def _check_fork():
    """Reset caches if we are in a new process (fork-safety)."""
    global _LLAMA_MODEL_CACHE, _SUPERVISOR_LLM_CACHE, _PID_OWNER
    current_pid = os.getpid()
    if _PID_OWNER is not None and _PID_OWNER != current_pid:
        log.info(f"🔄 Detected process fork (PID {_PID_OWNER} -> {current_pid}). Resetting LLM clients.")
        _LLAMA_MODEL_CACHE = None
        _SUPERVISOR_LLM_CACHE = None
    _PID_OWNER = current_pid


class RemoteEmbeddings(Embeddings):
    """
    Wrapper for remote OpenAI-compatible embedding server.
    Inherits from LangChain Embeddings for compatibility.
    """

    def __init__(self, base_url: str, model_name: str = "local-model"):
        # Sanitize URL
        sanitized_url = base_url.rstrip("/")
        changed = True
        while changed:
            changed = False
            for suffix in ["/v1", "/embeddings"]:
                if sanitized_url.endswith(suffix):
                    sanitized_url = sanitized_url[: -len(suffix)]
                    changed = True
                    break

        self.base_url = f"{sanitized_url}/v1"
        self.model_name = model_name or "local-model"

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="sk-no-key-required",
            timeout=LLAMA_REMOTE_TIMEOUT,
            max_retries=3,
        )
        log.info(f"🌐 Remote Embeddings Client: {self.base_url} (Model: {self.model_name})")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents with internal micro-batching for stability."""
        try:
            results = []
            # Use a conservative micro-batch size of 2 for remote servers with small physical batch limits
            MICRO_BATCH_SIZE = 2
            
            from more_itertools import chunked
            for micro_batch in chunked(texts, MICRO_BATCH_SIZE):
                response = self.client.embeddings.create(input=micro_batch, model=self.model_name)
                results.extend([data.embedding for data in response.data])
            return results
        except Exception as e:
            log.error(f"💥 Remote embedding failed: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        try:
            response = self.client.embeddings.create(input=[text], model=self.model_name)
            return response.data[0].embedding
        except Exception as e:
            log.error(f"💥 Remote query embedding failed: {e}")
            raise


class RemoteLlama:
    """
    Wrapper for remote llama.cpp or OpenAI-compatible server.
    Mimics the llama-cpp-python Llama interface.
    """

    def __init__(self, base_url: str):
        # Sanitize URL: Remove common suffixes to get the base API URL
        sanitized_url = base_url.rstrip("/")

        # Strip all known suffixes to get to the root
        changed = True
        while changed:
            changed = False
            for suffix in ["/v1", "/completions", "/chat/completions", "/chat"]:
                if sanitized_url.endswith(suffix):
                    sanitized_url = sanitized_url[: -len(suffix)]
                    changed = True
                    break

        self.base_url = f"{sanitized_url}/v1"

        # Initialize client with timeout
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="sk-no-key-required",
            timeout=LLAMA_REMOTE_TIMEOUT,
            max_retries=3,
        )

        # AUTO-DETECT Model Name: Crucial for strict servers
        self.model_name = SUPERVISOR_REMOTE_MODEL_NAME
        if not self.model_name:
            try:
                models = self.client.models.list()
                if models.data:
                    self.model_name = models.data[0].id
                    log.info(f"🤖 Auto-detected remote model: {self.model_name}")
            except Exception as e:
                log.warning(f"⚠️ Could not auto-detect model name: {e}. Falling back to 'local-model'.")
                self.model_name = "local-model"

        log.info(f"🌐 Remote Llama Client: {self.base_url} (Model: {self.model_name}, Timeout: {LLAMA_REMOTE_TIMEOUT}s)")

    def __call__(self, prompt: str, **kwargs):
        """
        Simulates completion interface but uses Chat API internally.
        Parses the instructional prompt into messages using ChatML tags.
        """
        # Parse ChatML tags
        messages = []

        # Extract System
        if "<|im_start|>system" in prompt:
            system_part = prompt.split("<|im_start|>system")[1].split("<|im_end|>")[0].strip()
            messages.append({"role": "system", "content": system_part})

        # Extract User
        if "<|im_start|>user" in prompt:
            user_part = prompt.split("<|im_start|>user")[1].split("<|im_end|>")[0].strip()
            messages.append({"role": "user", "content": user_part})
        else:
            # Fallback for simple prompts
            if not messages:
                messages.append({"role": "system", "content": "You are a helpful assistant."})
            messages.append({"role": "user", "content": prompt})

        log.info(f"🛰️ Sending remote chat request (Model: {self.model_name}, {len(messages)} msgs)...")
        start_request = time.perf_counter()
        try:
            # ONLY pass what's explicitly in kwargs to trust server defaults
            api_params = {
                "model": self.model_name,
                "messages": messages,
            }
            if "max_tokens" in kwargs:
                api_params["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                api_params["temperature"] = kwargs["temperature"]
            
            # Support for jinja template passing if requested in kwargs
            if "jinja" in kwargs:
                api_params["extra_body"] = {"jinja": kwargs["jinja"]}

            response = self.client.chat.completions.create(**api_params)
            elapsed = time.perf_counter() - start_request
            text = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
            log.info(f"✅ Received {len(text)} chars in {elapsed:.1f}s. Reason: {finish_reason}.")

            # Return in a format compatible with the gatekeeper's dict access
            # AND conform to standard response structure
            return {
                "id": response.id,
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": text}
                    }
                ],
                "usage": response.usage.model_dump() if hasattr(response, "usage") and response.usage else {}
            }
        except Exception as e:
            log.error(f"💥 Remote chat request failed: {e}")
            raise

    def create_chat_completion(self, messages: list, **kwargs):
        """
        Chat completion call used in Gatekeeper and ChromaChat.
        Refactored to trust server settings and return full compliant dict.
        """
        log.info(f"🛰️ Sending remote chat request to {self.model_name}...")
        try:
            # Build API parameters dynamically
            api_params = {
                "model": self.model_name,
                "messages": messages,
            }

            # Map standard OpenAI params
            standard_params = ["temperature", "max_tokens", "top_p", "stream", "stop", "presence_penalty", "frequency_penalty", "logit_bias", "user"]
            
            extra_body = {}
            for k, v in kwargs.items():
                if k in standard_params:
                    api_params[k] = v
                elif k != "extra_body":
                    # Put non-standard params (like 'jinja', 'repeat_penalty', 'top_k') in extra_body
                    extra_body[k] = v
            
            # Merge with existing extra_body if provided
            if "extra_body" in kwargs:
                extra_body.update(kwargs["extra_body"])

            if extra_body:
                api_params["extra_body"] = extra_body

            response = self.client.chat.completions.create(**api_params)
            content = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
            
            log.info(f"✅ Received {len(content)} chars from remote chat.")
            
            # Return full standard response dict
            return {
                "id": response.id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason
                    }
                ],
                "usage": response.usage.model_dump() if hasattr(response, "usage") and response.usage else {}
            }
        except Exception as e:
            log.error(f"💥 Remote chat completion failed: {e}")
            raise


def get_vectorstore():
    return get_vdb()


def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})


def get_chain_or_llama(retriever):
    global _LLAMA_MODEL_CACHE
    _check_fork()
    if _LLAMA_MODEL_CACHE is None:
        params = LlamaParamStrategy().get_params()
        model_path = params.get("model_path")
        from utils.exceptions import ConfigurationError

        if not model_path:
            raise ConfigurationError("LLM_PATH was not set")

        if model_path.startswith(("http://", "https://")):
            log.info(f"🚀 Connecting to Main Remote Llama: {model_path}")
            _LLAMA_MODEL_CACHE = RemoteLlama(base_url=model_path)
        else:
            if not os.path.exists(model_path):
                raise ConfigurationError(f"LLM_PATH file not found at {model_path}")

            log.info(f"🚀 Loading Main Local Llama: {model_path}")
            _LLAMA_MODEL_CACHE = Llama(**params)

    return None, _LLAMA_MODEL_CACHE


def get_supervisor_llm() -> Any:
    """
    Get the Supervisor LLM (singleton per process).
    Supports local GGUF via Llama or remote llama.cpp server via RemoteLlama.
    """
    global _SUPERVISOR_LLM_CACHE
    _check_fork()
    if _SUPERVISOR_LLM_CACHE is None:
        from utils.exceptions import ConfigurationError

        if not SUPERVISOR_LLM_ENDPOINTS:
            raise ConfigurationError("SUPERVISOR_LLM_ENDPOINTS was not set")

        if SUPERVISOR_LLM_ENDPOINTS.startswith(("http://", "https://")):
            log.info(f"🧠 Connecting to Remote Supervisor LLM: {SUPERVISOR_LLM_ENDPOINTS}")
            _SUPERVISOR_LLM_CACHE = RemoteLlama(base_url=SUPERVISOR_LLM_ENDPOINTS)
        else:
            if not os.path.exists(SUPERVISOR_LLM_ENDPOINTS):
                raise ConfigurationError(f"SUPERVISOR_LLM_ENDPOINTS file not found at {SUPERVISOR_LLM_ENDPOINTS}")

            log.info(f"🧠 Loading Local Supervisor LLM: {SUPERVISOR_LLM_ENDPOINTS}")
            _SUPERVISOR_LLM_CACHE = Llama(
                model_path=SUPERVISOR_LLM_ENDPOINTS,
                n_gpu_layers=0,  # CPU-only for stability during high-res OCR
                n_ctx=4096,
                n_batch=512,
                flash_attn=True,
                seed=42,
                verbose=False,
            )
    return _SUPERVISOR_LLM_CACHE
