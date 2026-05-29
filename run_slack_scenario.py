import asyncio
import os
import sys
from pathlib import Path
import multiprocessing

# Fix macOS ObjC + gRPC + asyncio deadlocks
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "poll"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_TORCH"] = "1"

# SOTA BGE-M3 is 2.3GB and slow to download. Switch to the fully-cached all-MiniLM-L6-v2.
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["EMBEDDING_DIM"] = "384"

# MUST be spawn to prevent PyTorch/gRPC fork deadlocks on macOS Apple Silicon
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Add project root to sys.path
PROJECT_ROOT = Path("/Users/mohnish/Downloads/tsc architecture")
sys.path.insert(0, str(PROJECT_ROOT))

import dotenv
dotenv.load_dotenv(PROJECT_ROOT / ".env")
# os.environ["TSC_LLM_PROVIDER"] = "google"
# os.environ["TSC_LLM_MODEL"] = "gemini-2.5-flash"
os.environ["TSC_GM_LLM_PROVIDER"] = "ollama"
os.environ["TSC_GM_LLM_MODEL"] = "gemma4:e2b"
os.environ["HINDSIGHT_URL"] = "" # Disable Hindsight to avoid 402 API errors
os.environ["GEMINI_FREE_RPM"] = "12"
os.environ["TSC_GEMINI_RPM_LIMIT"] = "12"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("slack_scenario_runner")

async def run_scenario():
    logger.info("Pre-warming PyTorch models BEFORE any gRPC imports to prevent macOS deadlocks...")
    # This MUST happen before create_llm_client() imports google.genai (which uses grpc/httpx)
    from tsc.memory.world_rag import _get_embedder, _get_reranker
    _get_embedder()
    _get_reranker()

    logger.info("Starting Slack AI Controversy Scenario...")
    from tsc.pipeline.orchestrator import TSCPipeline
    pipeline = TSCPipeline()
    data_dir = PROJECT_ROOT / "slack_ai_scenario"
    
    try:
        recommendation = await pipeline.evaluate(
            interviews=str(data_dir / "customer_interviews.txt"),
            support=str(data_dir / "support_tickets.txt"),
            analytics=str(data_dir / "analytics.json"),
            context=str(data_dir / "company_context.json"),
            proposal=str(data_dir / "feature_proposal.json"),
            num_simulations=10, # requested by user to spawn 10 agents
            use_legacy_personas=False
        )
        
        logger.info("="*60)
        logger.info("PIPELINE RESULT")
        logger.info(f"Verdict: {recommendation.final_verdict}")
        logger.info(f"Summary: {recommendation.summary_for_leadership}")
        logger.info(f"Adopt Rate: {recommendation.projected_adoption_rate}")
        logger.info(f"Debate Consensus: {recommendation.boardroom_consensus}")
        logger.info("="*60)
        
        output_file = PROJECT_ROOT / "slack_scenario_result.json"
        output_file.write_text(recommendation.model_dump_json(indent=2))
        logger.info(f"Full recommendation saved to {output_file}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_scenario())
