# apimain.py
# To run this app, use: uvicorn main:app --reload
# Or run as module: python -m uvicorn main:app --reload
from config.env_strategy import get_env_strategy
get_env_strategy().apply()

import sys
import os
import uvicorn

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Use absolute imports to work when running apimain.py directly
from api.endpoints import router as api_router

app = FastAPI(
    title="Document Chat API",
    description="API for querying documents using RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
@app.get("/")
def root():
    return {"message": "Document Chat API is running. Use /api/v1/query to chat with your documents."}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
