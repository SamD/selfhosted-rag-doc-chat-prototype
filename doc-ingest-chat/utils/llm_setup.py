#!/usr/bin/env python3
"""
LLM and chain setup for chat system.
"""

from config.llama_strategy import LlamaParamStrategy
from config.settings import OLLAMA_MODEL, OLLAMA_URL, RETRIEVER_TOP_K, USE_OLLAMA
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from llama_cpp import Llama
from prompts.chat_prompts import SHARED_CHAT_PROMPT
from services.database import get_db


def get_vectorstore():
    return get_db()


def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})


def get_chain_or_llama(retriever):
    if USE_OLLAMA:
        llm = ChatOllama(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True, prompt=SHARED_CHAT_PROMPT)
        return chain, None
    else:
        llama_model = Llama(**LlamaParamStrategy().get_params())

        # llama_model = Llama(
        #     model_path=LLM_PATH,
        #     n_ctx=LLAMA_N_CTX,
        #     n_gpu_layers=LLAMA_N_GPU_LAYERS,
        #     n_threads=LLAMA_N_THREADS,
        #     n_batch=LLAMA_N_BATCH,
        #     f16_kv=LLAMA_F16_KV,
        #     temperature=LLAMA_TEMPERATURE,
        #     top_k=LLAMA_TOP_K,
        #     top_p=LLAMA_TOP_P,
        #     repeat_penalty=LLAMA_REPEAT_PENALTY,
        #     max_tokens=LLAMA_MAX_TOKENS,
        #     chat_format=LLAMA_CHAT_FORMAT,
        #     verbose=LLAMA_VERBOSE,
        #     seed=LLAMA_SEED,
        # )
        return None, llama_model
