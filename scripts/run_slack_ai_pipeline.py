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

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

async def main():
    print("Pre-warming PyTorch models BEFORE any gRPC imports to prevent macOS deadlocks...")
    # This MUST happen before create_llm_client() imports google.generativeai (which uses grpc)
    from tsc.memory.world_rag import _get_embedder, _get_reranker
    _get_embedder()
    _get_reranker()

    print("Initializing Predictive Reality Engine Pipeline (Slack AI Scenario)...")
    
    # Import everything else now that PyTorch has safely initialized
    from tsc.pipeline.orchestrator import TSCPipeline
    from tsc.config import settings
    
    # 20 agents and 10 timesteps (10 is the default in OASISSimulationConfig)
    NUM_AGENTS = 20
    
    # Paths to slack AI scenario files
    scenario_dir = Path("slack_ai_scenario")
    interviews = str(scenario_dir / "customer_interviews.txt")
    support = str(scenario_dir / "support_tickets.txt")
    analytics = str(scenario_dir / "analytics.json")
    context = str(scenario_dir / "company_context.json")
    proposal = str(scenario_dir / "feature_proposal.json")
    
    pipeline = TSCPipeline()
    
    # Print progress to console
    def on_progress(layer, name, status, details):
        emoji = "✅" if status == "done" else "⏳"
        info = ""
        if details:
            info = " [" + ", ".join(f"{k}: {v}" for k, v in details.items()) + "]"
        print(f"{emoji} Layer {layer}/8: {name}{info}")
        
    pipeline.set_progress_callback(on_progress)
    
    print(f"Running pipeline with {NUM_AGENTS} agents and 10 timesteps...")
    
    try:
        result = await pipeline.evaluate(
            interviews=interviews,
            support=support,
            analytics=analytics,
            context=context,
            proposal=proposal,
            num_simulations=NUM_AGENTS,
        )
        print("\n=== PIPELINE COMPLETE ===")
        print(f"Verdict: {result.final_verdict}")
        print(f"Confidence: {result.overall_confidence}")
        print("Check the predictive dashboard at http://localhost:5173 to view the real-time simulation replay.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
