#!/bin/bash
# Worker Wrapper to fix gRPC/Torch deadlocks on macOS

# ── macOS-Specific Fork Safety ──────────────────────────────────────────────
# CRITICAL: Prevents ObjC runtime crash when Python forks on macOS (Sequoia+)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# ── gRPC Thread Safety ───────────────────────────────────────────────────────
# Prevents gRPC mutex deadlock when running inside asyncio + spawned subprocess
export GRPC_ENABLE_FORK_SUPPORT=false
export GRPC_POLL_STRATEGY=poll

# ── Threading Caps (prevent torch oversubscription) ─────────────────────────
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# ── Python Paths ─────────────────────────────────────────────────────────────
export PYTHONPATH=$PYTHONPATH:.

# Execute the worker with unbuffered output (no -v to avoid verbose import logging)
exec python3 -u "$@"
