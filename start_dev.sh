#!/bin/bash

echo "🚀 Starting WorldRAG Engine Services..."

# Activate virtual environment if it exists
# if [ -d ".venv" ]; then
#     echo "📦 Activating virtual environment..."
#     source .venv/bin/activate
# fi

# Ensure that if this script is killed, all background processes are also killed
trap 'kill 0' SIGINT SIGTERM EXIT

echo "🌐 Starting Uvicorn backend on port 8000 (Logging to uvicorn_backend.log)..."
python3.10 -m uvicorn tsc.web.app:app --host 0.0.0.0 --port 8000 --reload --loop asyncio > uvicorn_backend.log 2>&1 &

# Move to frontend directory
cd predictive_ui || exit

echo "🔌 Starting Node WebSocket server..."
node server/websocket.js &

echo "🎨 Starting Vite Frontend..."
npm run dev &

echo "✅ All services are starting up! Press Ctrl+C to stop."

# Wait for all background processes
wait
