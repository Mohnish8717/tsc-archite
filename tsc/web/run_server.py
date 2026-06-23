import multiprocessing
import os
import sys
import logging

if __name__ == "__main__":
    # 1. Force spawn
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # 2. Set environment variables
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
    os.environ["GRPC_POLL_STRATEGY"] = "poll"
    os.environ["GRPC_DNS_RESOLVER"] = "native"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["USE_TF"] = "0"
    os.environ["USE_JAX"] = "0"
    os.environ["USE_TORCH"] = "1"

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    os.makedirs("log", exist_ok=True)
    
    file_handler = logging.FileHandler("log/backend.log", mode="a")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(levelname)s:     %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logger = logging.getLogger("run_server")

    # 3. Pre-warm PyTorch
    logger.info("Pre-warming PyTorch models BEFORE Uvicorn starts to prevent macOS deadlocks...")
    try:
        from tsc.memory.world_rag import _get_embedder, _get_reranker
        _get_embedder()
        _get_reranker()
        logger.info("PyTorch pre-warmed successfully.")
    except Exception as e:
        logger.error(f"Failed to prewarm: {e}")

    # 4. Start Uvicorn
    logger.info("Starting Uvicorn...")
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG
    
    log_config = LOGGING_CONFIG.copy()
    log_config["handlers"]["file"] = {
        "class": "logging.FileHandler",
        "filename": "log/backend.log",
        "formatter": "default",
    }
    log_config["loggers"]["uvicorn"]["handlers"] = ["default", "file"]
    log_config["loggers"]["uvicorn.error"]["handlers"] = ["default", "file"]
    log_config["loggers"]["uvicorn.access"]["handlers"] = ["access", "file"]
    
    uvicorn.run("tsc.web.app:app", host="0.0.0.0", port=8000, loop="asyncio", log_config=log_config)
