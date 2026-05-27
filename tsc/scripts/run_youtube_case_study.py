import asyncio
import os
import sys
import logging
from uuid import uuid4

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tsc.pipeline.orchestrator import TSCPipeline
from tsc.config.settings import TSCConfig
from tsc.models.inputs import FeatureProposal, CompanyContext

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("YT_CaseStudy")

async def run_youtube_case_study():
    logger.info("Starting Real-World Case Study: YouTube Dislike Button Removal (2021)")
    
    # 1. Load the highly optimized inputs
    base_dir = os.path.join(PROJECT_ROOT, "tsc/data/optimized_inputs/youtube_dislike")
    
    with open(os.path.join(base_dir, "proposal.json"), "r") as f:
        proposal_data = f.read()
        
    with open(os.path.join(base_dir, "context.json"), "r") as f:
        context_data = f.read()
        
    with open(os.path.join(base_dir, "community_feedback.txt"), "r") as f:
        feedback_data = f.read()

    # 2. Configure Pipeline
    # Using the standard orchestrator entry point to ensure it hits OASISUserPersonaGenerator
    cfg = TSCConfig(
        environment="test",
        llm_provider="google",
        llm_model="gemini-1.5-pro",
        max_concurrent_simulations=10,
        cache_enabled=False # Force fresh generation
    )
    
    pipeline = TSCPipeline(cfg=cfg)
    
    # Optional: Attach a progress callback to see what's happening
    def on_progress(step, msg, status):
        logger.info(f"[Step {step}] {msg} - {status}")
    pipeline.set_progress_callback(on_progress)

    # 3. Execute
    logger.info("Executing Pipeline with massive historical dataset...")
    result = await pipeline.evaluate(
        interviews=None,
        support=feedback_data, # Feed the 3000-word feedback here
        analytics=None,
        context=context_data,
        proposal=proposal_data,
        num_simulations=10, # Generate 10 OASIS agents based on this data
        use_legacy_personas=False
    )
    
    logger.info("\n=== PREDICTION RESULTS ===")
    logger.info(f"Verdict: {result.verdict}")
    logger.info(f"Confidence: {result.confidence}")
    logger.info("==========================\n")
    logger.info("Check the latest run in `log/oasis_runs/` to view the generated OASIS personas and ensure they are historically accurate!")

if __name__ == "__main__":
    asyncio.run(run_youtube_case_study())
