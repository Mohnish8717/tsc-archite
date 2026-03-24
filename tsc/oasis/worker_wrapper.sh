#!/bin/bash
# Worker Wrapper to fix gRPC/Torch deadlocks on macOS
export GRPC_ENABLE_FORK_SUPPORT=false
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export GRPC_POLL_STRATEGY=poll
export PYTHONPATH=$PYTHONPATH:.

# Execute the worker with unbuffered and verbose output
exec python3 -u -v "$@"
