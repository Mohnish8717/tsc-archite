import asyncio
import os
import sys
import time
from typing import List

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tsc.oasis.models import OASISSimulationConfig, OASISAgentProfile, OpinionVector
from tsc.models.inputs import FeatureProposal
from tsc.models.inputs import CompanyContext
from tsc.oasis.simulation_runner import SimulationRunner

async def test_orchestration():
    print("🚀 Starting OASIS Hardened Orchestration Test...")
    
    # 1. Mock Data
    config = OASISSimulationConfig(
        num_agents=3,
        num_timesteps=3,
        simulation_name="test_run_hardened"
    )
    
    agent_profiles = [
        OASISAgentProfile(
            agent_id=i,
            source_persona_id=f"p_{i}",
            agent_type="USER",
            user_info_dict={
                "user_name": f"test_user_{i}",
                "bio": f"I am a tester for agent {i}. I like efficiency.",
                "platform": "generic"
            },
            initial_belief=OpinionVector(
                technical_feasibility=0.5,
                market_demand=0.5,
                resource_alignment=0.5,
                risk_tolerance=0.5,
                adoption_velocity=0.5,
                confidence=1.0,
                source_persona_id=f"p_{i}"
            ),
            influence_strength=0.1,
            receptiveness=0.8
        ) for i in range(3)
    ]
    
    feature = FeatureProposal(
        title="Automated Subprocess Isolation",
        description="A system that runs AI simulations in isolated processes to prevent database locks."
    )
    
    context = CompanyContext(company_name="Antigravity Corp")
    
    # 2. Run Orchestration
    runner = SimulationRunner("test_verification_run")
    print(f"📦 Workspace: {runner.workspace}")
    
    runner.start_simulation(config, agent_profiles, feature, context)
    
    # 3. Monitor for a few seconds
    for _ in range(5):
        status = runner.get_status()
        print(f"Status: {status}")
        
        actions_file = os.path.join(runner.workspace, "actions.jsonl")
        if os.path.exists(actions_file):
            with open(actions_file, 'r') as f:
                lines = f.readlines()
                print(f"  Action count: {len(lines)}")
        
        if status["status"] in ["COMPLETED", "FAILED"]:
            break
            
        await asyncio.sleep(2)
    
    # 4. Cleanup
    print("🛑 Cleaning up...")
    runner.stop()
    
    final_status = runner.get_status()
    print(f"Final Status: {final_status}")
    
    if os.path.exists(runner.result_file):
        print("✅ Success: Result file found.")
    else:
        print("⚠️ Note: Result file not found (might have stopped early).")

if __name__ == "__main__":
    asyncio.run(test_orchestration())
