#!/usr/bin/env python3
"""
LLM and chain setup for chat system.
Uses singleton pattern to ensure models are only loaded once per process.
"""

import threading
from typing import Optional

from config.llama_strategy import LlamaParamStrategy
from config.settings import LLAMA_N_GPU_LAYERS, LLAMA_N_THREADS, LLAMA_SEED, LLAMA_VERBOSE, OLLAMA_MODEL, OLLAMA_URL, RETRIEVER_TOP_K, SUPERVISOR_LLM_PATH, SUPERVISOR_TEMPERATURE, SUPERVISOR_TOP_K, USE_OLLAMA
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import LlamaCpp
from langchain_ollama import ChatOllama
from llama_cpp import Llama
import logging
from prompts.chat_prompts import SHARED_CHAT_PROMPT
from services.database import get_db

log = logging.getLogger("ingest.llm_setup")

# Singleton caches
_LLAMA_MODEL_CACHE: Optional[Llama] = None
_SUPERVISOR_LLM_CACHE: Optional[LlamaCpp] = None
_LLM_LOCK = threading.Lock()

def get_vectorstore():
    """Get the vector database instance."""
    return get_db()

def get_retriever(vectorstore):
    """Get a retriever from the vectorstore."""
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

def get_chain_or_llama(retriever):
    """Get the conversation chain or raw Llama model (singleton)."""
    global _LLAMA_MODEL_CACHE
    
    if USE_OLLAMA:
        llm = ChatOllama(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            return_source_documents=True, 
            prompt=SHARED_CHAT_PROMPT
        )
        return chain, None
    else:
        if _LLAMA_MODEL_CACHE is None:
            with _LLM_LOCK:
                if _LLAMA_MODEL_CACHE is None:
                    params = LlamaParamStrategy().get_params()
                    log.info(f"🚀 Loading Main Llama: {params['model_path']}")
                    _LLAMA_MODEL_CACHE = Llama(**params)
        return None, _LLAMA_MODEL_CACHE

def get_supervisor_llm() -> LlamaCpp:
    """Get or initialize the Supervisor (Metadata Extractor) LLM (singleton)."""
    global _SUPERVISOR_LLM_CACHE
    if _SUPERVISOR_LLM_CACHE is None:
        with _LLM_LOCK:
            if _SUPERVISOR_LLM_CACHE is None:
                log.info(f"🧠 Loading Supervisor LLM: {SUPERVISOR_LLM_PATH}")
                # We use a smaller context window for the supervisor to save VRAM
                _SUPERVISOR_LLM_CACHE = LlamaCpp(
                    model_path=SUPERVISOR_LLM_PATH,
                    n_ctx=2048, 
                    n_gpu_layers=LLAMA_N_GPU_LAYERS,
                    n_threads=LLAMA_N_THREADS,
                    temperature=SUPERVISOR_TEMPERATURE,
                    top_k=SUPERVISOR_TOP_K,
                    verbose=LLAMA_VERBOSE,
                    seed=LLAMA_SEED
                )
    return _SUPERVISOR_LLM_CACHE
