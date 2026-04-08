#!/bin/bash
# ── OASIS macOS Deadlock Immunity Orchestrator (Final SOTA) ─────────────────
#
# This script ensures that gRPC's native poller is inhibited AND Python uses
# the system malloc AND conflicting backends are disabled BEFORE the 
# interpreter loads its first module. This is the ONLY definitive way to # ── DEADLOCK PREVENTION: macOS gRPC/Torch/Abseil ──────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────

# 2. Disable Conflicting Backends to prevent Abseil collisions
# Transformers/Torch will use the native Torch backend instead.
export TRANSFORMERS_VERBOSITY=error
export USE_ONNX=0
export USE_TENSORFLOW=0
export TF_CPP_MIN_LOG_LEVEL=3

# 3. Optimized Threading for macOS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# 4. Path Setup + Load .env
export PYTHONPATH=".:$PYTHONPATH"
# Export GEMINI_API_KEY from .env if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | grep 'GEMINI_API_KEY' | xargs)
fi

# 5. Run Simulation
echo "🚀 Starting Deadlock-Free OASIS Simulation (Alloc: Malloc | Core: Torch)..."
python3 "$@"
