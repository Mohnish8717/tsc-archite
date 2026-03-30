import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path("/Users/mohnish/Downloads/tsc architecture")
sys.path.append(str(PROJECT_ROOT))

# Configure Environment
os.environ["TSC_LLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
os.environ["TSC_LLM_MODEL"] = "llama-3.1-8b-instant"

# Import OASIS Components
from tsc.oasis.models import OASISSimulationConfig, OASISAgentProfile
from tsc.oasis.simulation_engine import RunOASISSimulation
from tsc.models.inputs import FeatureProposal, CompanyContext

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sim_test")

async def test_standalone_simulation():
    logger.info("🚀 Starting Hardened OASIS Simulation Test (100 Agents)...")
    
    # 1. Setup Mock Profiles
    profiles = []
    for i in range(100):
        profiles.append(OASISAgentProfile(
            agent_id=i,
            source_persona_id=f"persona_{i}",
            agent_type="customer_segment",
            user_info_dict={
                "name": f"Agent_{i}",
                "profile": {
                    "user_profile": "A tech-savvy early adopter interested in AI efficiency.",
                    "mbti": "INTJ",
                    "other_info": {"role": "Developer"}
                }
            }
        ))
    
    # 2. Setup Config
    config = OASISSimulationConfig(
        simulation_name=f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        num_timesteps=5,
        platform_type="twitter",
        population_size=100
    )
    
    # 3. Setup Context
    feature = FeatureProposal(
        title="AI-Powered Code Reviewer",
        description="A tool that uses LLMs to provide real-time feedback on pull requests."
    )
    context = CompanyContext(
        company_name="TSC Corp",
        mission="Accelerating software development with AI."
    )
    
    # 4. Run Simulation
    try:
        logger.info(f"Running simulation: {config.simulation_name}")
        result = await RunOASISSimulation(
            config=config,
            agent_profiles=profiles,
            feature=feature,
            context=context,
            base_dir="/tmp/oasis_runs"
        )
        
        logger.info("✅ Simulation completed successfully without deadlocks!")
        logger.info(f"Simulation ID: {config.simulation_name}")
        
        # Verify Trace Table
        db_path = f"/tmp/oasis_runs/{config.simulation_name}/{config.simulation_name}.sqlite"
        if os.path.exists(db_path):
            logger.info(f"📊 Database persisted at: {db_path}")
        else:
            logger.error("❌ Database file not found!")
            
    except Exception as e:
        logger.exception(f"❌ Simulation failed: {e}")

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not found in environment.")
        sys.exit(1)
    asyncio.run(test_standalone_simulation())
