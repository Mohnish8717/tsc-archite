import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import sqlite3

# Add project root to sys.path
PROJECT_ROOT = Path("/Users/mohnish/Downloads/tsc architecture")
sys.path.append(str(PROJECT_ROOT))

# Configure Environment
os.environ["TSC_LLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TSC_LLM_MODEL"] = "llama-3.1-8b-instant"

# Import OASIS Components
from tsc.oasis.models import OASISSimulationConfig, OASISAgentProfile
from tsc.oasis.simulation_engine import RunOASISSimulation
from tsc.models.inputs import FeatureProposal, CompanyContext

# Setup logging
logging.basicConfig(level=logging.ERROR) # Minimal logging to see results better
logger = logging.getLogger("sim_test")

async def run_and_report_conversations():
    print("🚀 Running 3-Agent OASIS Simulation for Conversation Extraction...")
    
    # 1. Setup Mock Profiles (Expert Market Personas)
    profiles = []
    personas = [
        ("Dr. Aris Thorne", "Organizational Psychologist", "Expert in workplace productivity and flow states. Focuses on Deep Work disruption.", "INTJ"),
        ("Sarah Jenkins", "Remote Operations Lead", "Manages 1,000+ remote workers. Focuses on Coordination Costs and talent retention.", "ENTJ"),
        ("Marcus Vane", "Employment Law Attorney", "Expert in digital privacy and labor rights. Scrutinizes surveillance for legal compliance.", "ISTJ")
    ]
    
    for i, (name, role, bias, mbti) in enumerate(personas): # 3 agents total
        profiles.append(OASISAgentProfile(
            agent_id=i,
            source_persona_id=f"p_{i}",
            agent_type="external_market_segment",
            user_info_dict={
                "name": name,
                "profile": {
                    "user_profile": f"I am {name}, a {role}. {bias}",
                    "mbti": mbti,
                    "other_info": {"role": role}
                }
            }
        ))
    
    # 2. Setup Config
    sim_id = f"conv_test_{datetime.now().strftime('%H%M%S')}"
    config = OASISSimulationConfig(
        simulation_name=sim_id,
        num_timesteps=3,
        platform_type="reddit",
        population_size=3
    )
    
    # 3. Setup Context
    feature = FeatureProposal(
        title="Mandatory Daily 4-Hour Sync Meetings",
        description="A new policy requiring all developers and stakeholders to completely drop their work and attend a mandatory 4-hour sync meeting every single day to verbally explain every line of code written to upper management. Includes automatic screen capturing to ensure active participation."
    )
    context = CompanyContext(company_name="TSC", mission="Stability First")
    
    # 4. Trigger Mid-Simulation Interview Asynchronously
    async def mid_sim_trigger():
        # Wait until well after timestep 0 completes before firing the interview.
        # Timestep 0 Phase 1 + Phase 2 + sync pauses ≈ 60-90s at 3 agents.
        # 120s guarantees we fire at a clean timestep boundary (start of timestep 1 or 2).
        await asyncio.sleep(120)
        command_file = f"/tmp/oasis_runs/{sim_id}/commands.json"
        
        print("\n📣 [TEST SCRIPT] Firing asynchronous MID-SIMULATION INTERVIEW payload!")
        try:
            with open(command_file, 'w') as f:
                json.dump({
                    "action": "interview", 
                    "questions": [
                        "What is your decisive stance (BULLISH vs BEARISH) on this mandatory 4-hour meeting feature so far?"
                    ]
                }, f)
        except Exception as e:
            print(f"Failed to write mid-sim command: {e}")

    # Launch trigger in background
    trigger_task = asyncio.create_task(mid_sim_trigger())
    
    # 5. Run Simulation
    try:
        series = await RunOASISSimulation(
            config=config,
            agent_profiles=profiles,
            feature=feature,
            context=context,
            base_dir="/tmp/oasis_runs"
        )
        
        trigger_task.cancel()
        
        print("\n✅ Simulation Complete. Extracting internal conversations...")
        print("-" * 60)
        
        # 5. Query Database
        db_path = f"/tmp/oasis_runs/{sim_id}/{sim_id}.sqlite"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get Posts
            cursor.execute("SELECT post_id, content FROM post;")
            posts = cursor.fetchall()
            for pid, pcontent in posts:
                print(f"\n[OP Post]: {pcontent}")
                
                # Get Comments for this post
                cursor.execute("SELECT agent_id, content FROM comment WHERE post_id = ?;", (pid,))
                comments = cursor.fetchall()
                for aid, ccontent in comments:
                    # Map aid back to name
                    agent_name = profiles[int(aid)].user_info_dict['name'] if int(aid) < len(profiles) else "Unknown"
                    print(f"  └─ [{agent_name}]: {ccontent}")
            
            # 6. Show Interview Responses
            print("\n🎤 [DYNAMIC INTERVIEW RESULTS]:")
            print("-" * 60)
            for entry in series.raw_responses:
                agent_id = entry.get("agent_id")
                # Find name
                agent_name = "Unknown"
                for p in profiles:
                    if str(p.agent_id) == str(agent_id):
                        agent_name = p.user_info_dict['name']
                        break
                
                print(f"\nAgent: {agent_name} (ID: {agent_id})")
                for resp in entry.get("responses", []):
                    print(f"  Q: {resp.get('question')}")
                    print(f"  A: {resp.get('content')}")
            
            conn.close()
        else:
            print("❌ Error: DB file not found.")
            
        print("\n\n📊 Full Pipeline Behavioral Analysis & Consensus Results:")
        print("=" * 60)
        from tsc.oasis.clustering import PerformBehavioralClustering, DetectConsensus, CalculateAggregatedMetrics
        clusters = await PerformBehavioralClustering(profiles, series.raw_responses)
        CalculateAggregatedMetrics(clusters, series)
        is_consensus, strength, consensus_type = DetectConsensus(series, config)
        
        print(f"Final Adoption Score: {series.final_adoption_score}/1.0")
        print(f"Consensus Verdict:    {series.consensus_verdict}")
        print(f"Consensus Type:       {consensus_type.upper()} (Strength: {strength})")
        
        print("\nBelief Clusters Identified:")
        for idx, c in enumerate(clusters, 1):
            print(f"  {idx}. {c.cluster_id} ({c.cluster_size} agents)")
            print(f"     Sentiment: {c.sentiment_score} | Dominant Persona: {c.dominant_persona_type}")
            print(f"     Behavior Trace: {c.centroid_behavior}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_and_report_conversations())
