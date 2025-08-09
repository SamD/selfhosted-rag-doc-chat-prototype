#!/usr/bin/env python3
# apimain.py
# To run this app, use: uvicorn apimain:app --reload

import os
import sys

import uvicorn
from config.env_strategy import get_env_strategy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    # Ensure package imports work when running this file directly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Import endpoints lazily to keep module-level imports at the top
    from api.endpoints import router as api_router

    app = FastAPI(
        title="Document Chat API",
        description="API for querying documents using RAG (Retrieval-Augmented Generation)",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/")
    def root():
        return {"message": "Document Chat API is running. Use /api/v1/query to chat with your documents."}

    return app


app = create_app()


def main() -> None:
    # Apply environment strategy before starting the server
    get_env_strategy().apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
