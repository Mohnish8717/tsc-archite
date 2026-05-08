import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("/Users/mohnish/Downloads/tsc architecture")
sys.path.append(str(PROJECT_ROOT))

import dotenv
dotenv.load_dotenv(PROJECT_ROOT / ".env")
os.environ["TSC_LLM_PROVIDER"] = "google"
os.environ["TSC_LLM_MODEL"] = "gemma-4-31b-it" # or gemini-3-pro if preferred, sticking to standard

os.environ["GRPC_PYTHON_FORK_SUPPORT_ONLY"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

from tsc.pipeline.orchestrator import TSCPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("slack_scenario_runner")

async def run_scenario():
    logger.info("Starting Slack AI Controversy Scenario...")
    pipeline = TSCPipeline()
    data_dir = PROJECT_ROOT / "slack_ai_scenario"
    
    try:
        recommendation = await pipeline.evaluate(
            interviews=str(data_dir / "customer_interviews.txt"),
            support=str(data_dir / "support_tickets.txt"),
            analytics=str(data_dir / "analytics.json"),
            context=str(data_dir / "company_context.json"),
            proposal=str(data_dir / "feature_proposal.json"),
            num_simulations=20, # requested by user
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
