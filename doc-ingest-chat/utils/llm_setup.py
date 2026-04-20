#!/usr/bin/env python3
"""
LLM and chain setup for chat system.
Uses singleton pattern to ensure models are only loaded once per process.
Supports local GGUF via llama-cpp-python or remote llama.cpp server via API.
"""

import logging
from typing import Any, Optional, Union

import openai
from config.llama_strategy import LlamaParamStrategy
from config.settings import (
    LLAMA_N_GPU_LAYERS,
    LLAMA_N_THREADS,
    LLAMA_SEED,
    LLAMA_VERBOSE,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RETRIEVER_TOP_K,
    SUPERVISOR_LLM_PATH,
    SUPERVISOR_TEMPERATURE,
    SUPERVISOR_TOP_K,
    USE_OLLAMA,
)

try:
    from langchain.chains import ConversationalRetrievalChain
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_classic.chains import ConversationalRetrievalChain
    except (ImportError, ModuleNotFoundError):
        # Last resort fallback for some distributions
        from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.llms import LlamaCpp
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from llama_cpp import Llama
from prompts.chat_prompts import SHARED_CHAT_PROMPT
from services.database import get_db

log = logging.getLogger("ingest.llm_setup")

# Singleton caches (Per-Process)
_LLAMA_MODEL_CACHE: Optional[Union[Llama, "RemoteLlama"]] = None
_SUPERVISOR_LLM_CACHE: Optional[Any] = None


class RemoteLlama:
    """
    Wrapper for remote llama.cpp server that mimics the llama-cpp-python Llama interface.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url if base_url.endswith("/v1") else f"{base_url.rstrip('/')}/v1"
        self.client = openai.OpenAI(base_url=self.base_url, api_key="sk-no-key-required")
        log.info(f"🌐 Remote Llama Client initialized at {self.base_url}")

    def __call__(self, prompt: str, **kwargs):
        """Standard completion call used in gatekeeper_logic.py."""
        grammar = kwargs.pop("grammar", None)
        extra_body = {}
        if isinstance(grammar, str):
            extra_body["grammar"] = grammar

        response = self.client.completions.create(
            model="local-model",
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.1),
            stop=kwargs.get("stop", []),
            extra_body=extra_body,
        )
        return {"choices": [{"text": response.choices[0].text}]}

    def create_chat_completion(self, messages: list, **kwargs):
        """Chat completion call used in ChromaChat."""
        response = self.client.chat.completions.create(
            model="local-model",
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 256),
        )
        return {"choices": [{"message": {"content": response.choices[0].message.content}}]}


def get_vectorstore():
    return get_db()


def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})


def get_chain_or_llama(retriever):
    global _LLAMA_MODEL_CACHE
    if USE_OLLAMA:
        llm = ChatOllama(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
        return (
            ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True, prompt=SHARED_CHAT_PROMPT),
            None,
        )
    else:
        if _LLAMA_MODEL_CACHE is None:
            params = LlamaParamStrategy().get_params()
            model_path = params["model_path"]

            if model_path.startswith(("http://", "https://")):
                log.info(f"🚀 Connecting to Main Remote Llama: {model_path}")
                _LLAMA_MODEL_CACHE = RemoteLlama(base_url=model_path)
            else:
                log.info(f"🚀 Loading Main Local Llama: {model_path}")
                _LLAMA_MODEL_CACHE = Llama(**params)

        return None, _LLAMA_MODEL_CACHE


def get_supervisor_llm() -> Any:
    """
    Get the Supervisor LLM (singleton per process).
    Supports local GGUF via LlamaCpp or remote llama.cpp server via ChatOpenAI.
    """
    global _SUPERVISOR_LLM_CACHE
    if _SUPERVISOR_LLM_CACHE is None:
        if not SUPERVISOR_LLM_PATH:
            raise ValueError("❌ SUPERVISOR_LLM_PATH is not set in environment. Please set it to the absolute path or URL of your Supervisor model.")

        if SUPERVISOR_LLM_PATH.startswith(("http://", "https://")):
            log.info(f"🧠 Connecting to Remote Supervisor LLM: {SUPERVISOR_LLM_PATH}")
            base_url = SUPERVISOR_LLM_PATH if SUPERVISOR_LLM_PATH.endswith("/v1") else f"{SUPERVISOR_LLM_PATH.rstrip('/')}/v1"
            _SUPERVISOR_LLM_CACHE = ChatOpenAI(
                base_url=base_url,
                api_key="sk-no-key-required",
                temperature=SUPERVISOR_TEMPERATURE,
                max_tokens=None,  # Allow dynamic response length
            )
        else:
            log.info(f"🧠 Loading Local Supervisor LLM: {SUPERVISOR_LLM_PATH}")
            _SUPERVISOR_LLM_CACHE = LlamaCpp(
                model_path=SUPERVISOR_LLM_PATH,
                n_ctx=2048,
                n_gpu_layers=LLAMA_N_GPU_LAYERS,
                n_threads=LLAMA_N_THREADS,
                temperature=SUPERVISOR_TEMPERATURE,
                top_k=SUPERVISOR_TOP_K,
                verbose=LLAMA_VERBOSE,
                seed=LLAMA_SEED,
            )
    return _SUPERVISOR_LLM_CACHE
