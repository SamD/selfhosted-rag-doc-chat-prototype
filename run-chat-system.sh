#!/bin/bash

# AI Chat System - Start Script
# This script starts both the FastAPI backend and Astro frontend

echo "🚀 Starting AI Chat System..."

# Function to cleanup background processes on exit
cleanup() {
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start FastAPI backend
echo "📡 Starting FastAPI backend..."
cd doc-ingest-chat
source /home/samueldoyle/Projects/virtualenv/ollama/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start Astro frontend
echo "🎨 Starting Astro frontend..."
cd astro-frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Services started!"
echo "📡 Backend: http://localhost:8000"
echo "🎨 Frontend: http://localhost:4321"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for background processes
wait 