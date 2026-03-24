import os
import multiprocessing

# Global gRPC/Multi-process hardening for OASIS
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
