#!/bin/bash

echo "🚀 Starting WorldRAG Engine Services..."

# Activate virtual environment if it exists
# if [ -d ".venv" ]; then
#     echo "📦 Activating virtual environment..."
#     source .venv/bin/activate
# fi

# Ensure that if this script is killed, all background processes are also killed
trap 'kill 0' SIGINT SIGTERM EXIT

# ── DEADLOCK PREVENTION: macOS gRPC/Torch/Abseil ──────────────────────────────
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export GRPC_ENABLE_FORK_SUPPORT=false
export GRPC_POLL_STRATEGY=poll
export GRPC_DNS_RESOLVER=native
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONMALLOC=malloc

echo "🌐 Starting Uvicorn backend on port 8000 (Logging to uvicorn_backend.log)..."
python3.10 tsc/web/run_server.py > uvicorn_backend.log 2>&1 &

# Move to frontend directory
cd predictive_ui || exit

echo "🔌 Starting Node WebSocket server..."
node server/websocket.js &

echo "🎨 Starting Vite Frontend..."
npm run dev &

echo "✅ All services are starting up! Press Ctrl+C to stop."

# Wait for all background processes
wait
