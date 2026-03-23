import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("/Users/mohnish/Downloads/tsc architecture")
sys.path.append(str(PROJECT_ROOT))

# Configure Environment for Groq (Provided by User)
os.environ["TSC_LLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
os.environ["TSC_LLM_MODEL"] = "llama-3.1-8b-instant"

# macOS Stability Fixes
# os.environ["TSC_MOCK_EMBEDDINGS"] = "1"
os.environ["GRPC_PYTHON_FORK_SUPPORT_ONLY"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

# Import Pipeline Components
from tsc.pipeline.orchestrator import TSCPipeline
from tsc.config import settings

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("production_pipeline")

async def run_production_flow():
    logger.info("Starting Production-Grade TSC Pipeline Run...")
    
    # Initialize Pipeline
    pipeline = TSCPipeline()
    
    # Define Data Paths
    data_dir = PROJECT_ROOT / "production_data"
    
    try:
        recommendation = await pipeline.evaluate(
            interviews=str(data_dir / "customer_interviews.txt"),
            support=str(data_dir / "support_tickets.txt"),
            context=str(data_dir / "company_context.json"),
            proposal=str(data_dir / "feature_proposal.json"),
            num_simulations=300 # Use our optimized Monte Carlo
        )
        
        logger.info("="*40)
        logger.info("PIPELINE RESULT")
        logger.info(f"Verdict: {recommendation.final_verdict}")
        logger.info(f"Summary: {recommendation.summary_for_leadership[:200]}...")
        logger.info("="*40)
        
        # Save full result to disk
        output_file = PROJECT_ROOT / "production_recommendation.json"
        output_file.write_text(recommendation.model_dump_json(indent=2))
        logger.info(f"Full recommendation saved to {output_file}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_production_flow())
