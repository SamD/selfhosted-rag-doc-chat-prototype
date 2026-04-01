#!/usr/bin/env python3
"""
LLM and chain setup for chat system.
Uses singleton pattern to ensure models are only loaded once per process.
"""

import logging
from typing import Optional

from config.llama_strategy import LlamaParamStrategy
from config.settings import LLAMA_N_GPU_LAYERS, LLAMA_N_THREADS, LLAMA_SEED, LLAMA_VERBOSE, OLLAMA_MODEL, OLLAMA_URL, RETRIEVER_TOP_K, SUPERVISOR_LLM_PATH, SUPERVISOR_TEMPERATURE, SUPERVISOR_TOP_K, USE_OLLAMA
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import LlamaCpp
from langchain_ollama import ChatOllama
from llama_cpp import Llama
from prompts.chat_prompts import SHARED_CHAT_PROMPT
from services.database import get_db

log = logging.getLogger("ingest.llm_setup")

# Singleton caches (Per-Process)
_LLAMA_MODEL_CACHE: Optional[Llama] = None
_SUPERVISOR_LLM_CACHE: Optional[LlamaCpp] = None


def get_vectorstore():
    return get_db()


def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})


def get_chain_or_llama(retriever):
    global _LLAMA_MODEL_CACHE
    if USE_OLLAMA:
        llm = ChatOllama(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True, prompt=SHARED_CHAT_PROMPT), None
    else:
        if _LLAMA_MODEL_CACHE is None:
            params = LlamaParamStrategy().get_params()
            log.info(f"🚀 Loading Main Llama: {params['model_path']}")
            _LLAMA_MODEL_CACHE = Llama(**params)
        return None, _LLAMA_MODEL_CACHE


def get_supervisor_llm() -> LlamaCpp:
    """
    Get the Supervisor LLM (singleton per process).
    Note: Calling process must manage its own multiprocessing locks for GPU safety.
    """
    global _SUPERVISOR_LLM_CACHE
    if _SUPERVISOR_LLM_CACHE is None:
        if not SUPERVISOR_LLM_PATH:
            raise ValueError("❌ SUPERVISOR_LLM_PATH is not set in environment. Please set it to the absolute path of your Supervisor GGUF model.")
        log.info(f"🧠 Loading Supervisor LLM: {SUPERVISOR_LLM_PATH}")
        _SUPERVISOR_LLM_CACHE = LlamaCpp(model_path=SUPERVISOR_LLM_PATH, n_ctx=2048, n_gpu_layers=LLAMA_N_GPU_LAYERS, n_threads=LLAMA_N_THREADS, temperature=SUPERVISOR_TEMPERATURE, top_k=SUPERVISOR_TOP_K, verbose=LLAMA_VERBOSE, seed=LLAMA_SEED)
    return _SUPERVISOR_LLM_CACHE
