"""
RAG service implementation for the chat system.
"""
# services/rag_pipeline.py

from chat.chroma_chat import respond  # ✅ import directly

class RagService:
    def __init__(self):
        pass  # No retriever/LLM injected yet

    def answer_query(self, query: str, chat_history: list[dict]) -> dict:
        updated_history, debug = respond(query, chat_history)
        return {
            "answer": updated_history[-1]["content"],
            "chat_history": updated_history,
            "debug": debug
        }
