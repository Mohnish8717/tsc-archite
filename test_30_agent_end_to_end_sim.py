import os
import sys

# ── STEP 0: macOS Deadlock Immunity (Absolute Shadowing) ───────────────────────
if sys.platform == "darwin":
    from unittest.mock import MagicMock
    class MockLib:
        def __getattr__(self, name): return MagicMock()
        def __call__(self, *args, **kwargs): return MagicMock()
    
    IMMUNE_TARGETS = [
        "grpc", "grpc.aio", "grpc._cython", "grpc._cython.cygrpc"
    ]
    for m in IMMUNE_TARGETS:
        if m not in sys.modules:
            sys.modules[m] = MockLib()
            
    # Absolute Disable (same as Mock Simulation to trigger safe clustering fallback)
    for m in ["onnxruntime", "tensorflow", "codecarbon", "deepspeed"]:
        sys.modules[m] = None

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

import logging
from datetime import datetime
from pathlib import Path


# Pre-warm removed due to recursive deadlock.
import asyncio
import logging
import re
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import (
    FinalPersona, Stakeholder, PsychologicalProfile
)
from tsc.models.graph import KnowledgeGraph
from tsc.selection.engine import PersonaSelectionEngine
from tsc.selection.tension_vector import FeatureTensionAnalyzer
from tsc.layers.layer3_personas import PersonaGenerator
from tsc.repositories.persona_repository import PersonaRepository
from tsc.db.connection import get_db
from tsc.llm.factory import create_llm_client
from tsc.llm.rate_limiter import reset_groq_bucket
from tsc.oasis.models import OASISSimulationConfig, OASISAgentProfile, UserInfoAdapter
from tsc.oasis.simulation_engine import RunOASISSimulation
from tsc.oasis.clustering import PerformBehavioralClustering, DetectConsensus, CalculateAggregatedMetrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("tsc.market_sim_e2e")

async def main():
    print("\n" + "═" * 80)
    print("🚀  OASIS END-TO-END MARKET SIMULATION (30 AGENTS)")
    print("═" * 80)

    # 0. RESET RATE LIMITER (Pick up env overrides for 70B)
    reset_groq_bucket()

    # 1. SETUP FEATURE & CONTEXT
    feature = FeatureProposal(
        title="AI-Powered Code Review Autopilot",
        description=(
            "An autonomous AI system that performs code reviews on pull requests. "
            "It automatically suggests refactors and can auto-approve PRs that meet "
            "security/style guidelines. Optimized for reducing human toil in boilerplate."
        ),
        effort_weeks_min=4,
        effort_weeks_max=12,
    )
    company = CompanyContext(
        company_name="Antigravity Corp",
        mission="Automate engineering toil to unleash developer creativity.",
        team_size=150,
        tech_stack=["Python", "React", "Groq/FastAPI"],
        current_priorities=["Security", "Developer Velocity"]
    )
    
    llm = create_llm_client()
    repo = PersonaRepository(get_db())
    # Mocking cache and graph store for the generator
    generator = PersonaGenerator(
        llm_client=llm, 
        persona_repo=repo, 
        persona_cache=None, 
        graph_store=None
    )

    # ── PHASE 1: GENERATION PIPELINE ──────────────────────────────────────────
    print("\n[STEP 1] Running Feature Tension Analysis...")
    tension_analyzer = FeatureTensionAnalyzer(llm)
    tension_vector = await tension_analyzer.analyze(feature)

    print("\n[STEP 2] Selecting Seeds from Global Persona DB...")
    # Get candidates from relevant segments
    candidates = (await generator.get_external_personas_for_market("enterprise")) + \
                 (await generator.get_external_personas_for_market("startup"))
    
    # Select optimized seeds for expansion
    seeds = generator._select_seeds_from_candidates(candidates, tension_vector, target_seeds=8)
    print(f"✅ Selected {len(seeds)} optimized basis seeds.")

    print("\n[STEP 3] Grounding Seeds (High-Fidelity Narratives)...")
    grounded_seeds = []
    # Sequential grounding to prevent 429s as requested
    for i, s in enumerate(seeds):
        sh = Stakeholder(name=s.name, role=s.role, persona_type="EXTERNAL", relevance_score=1.0)
        print(f"    - Grounding {sh.name} ({i+1}/{len(seeds)})...", flush=True)
        ctx = await generator._retrieve_context(sh)
        grounded = await generator._generate_profile(sh, ctx, feature, company, [])
        grounded_seeds.append(grounded)

    print("\n[STEP 4] Expanding to 30 Mixed Agents (Synthesizing via JVM)...")
    engine = PersonaSelectionEngine(llm, rng_seed=42)
    # expanded_personas is a list[FinalPersona]
    expanded_personas, result = await engine.select(archetypes=grounded_seeds, feature=feature, target_n=30)
    print(f"✅ Generated {len(expanded_personas)} unique agents via jitter expansion.")

    # ── Detailed Profile Inspection (Pulled from test_selection_engine.py) ────
    print("\n🧐  DETAILED PROFILE INSPECTION (Final Simulation Set):")
    
    # Tension Vector
    print(f"\n🧲  TENSION VECTOR ({len(result.tension_vector.dimensions)} dimensions):")
    for dim, score in result.tension_vector.dimensions.items():
        bar = "▓" * int(min(10, float(abs(score)) * 10))
        f_score = float(score)
        sign = "+" if f_score > 0 else "-" if f_score < 0 else " "
        print(f"   {dim:<30} {sign}{abs(score):.2f}  {bar}")
    print(f"   Required Domains: {', '.join(result.tension_vector.required_domains)}")

    # Pole assignments
    print(f"\n⚡  POLE ASSIGNMENTS ({len(result.poles)} archetypes):")
    pole_icons = {"ATTRACTOR": "✅", "REPELLER": "❌", "SWING": "⚖️ "}
    for p in result.poles:
        icon = pole_icons.get(p.pole.value, "?")
        fragment = " [Minority Voice injected]" if p.minority_voice_fragment else ""
        print(f"   {icon} {p.persona_name:<35} score={p.pole_score:+.3f}{fragment}")

    # Pole Distribution
    print(f"\n📈  POLE DISTRIBUTION (across {result.actual_n} agents):")
    for pole, pct in result.pole_distribution.items():
        bar = "█" * int(pct * 30)
        print(f"   {pole:<12} {pct:.1%}  {bar}")

    # Epistemic Gaps
    print(f"\n🔍  EPISTEMIC COVERAGE ({len(result.epistemic_gaps)} domains):")
    for gap in result.epistemic_gaps:
        status = "✅ Covered" if gap.covered else "⚠️  GAP"
        mvp = " → Minority Voice activated" if gap.minority_voice_activated else ""
        print(f"   {status}: {gap.domain} (count={gap.coverage_count}){mvp}")

    print(f"\n🔢  EXPANSION SUMMARY: {result.expansion_metadata.get('archetype_count')} archetypes "
          f"→ {result.actual_n} agents via '{result.strategy_used}' "
          f"({result.expansion_metadata.get('synthetic_count')} synthetic clones)")
    print("─" * 80)

    print("\n[STEP 5] Mapping FinalPersonas to OASIS Simulation Profiles...")
    oasis_profiles = []
    for i, gp in enumerate(expanded_personas):
        # UserInfoAdapter bridges our internal object (gp) to CAMEL-AI representation
        user_info = UserInfoAdapter.to_oasis_user_info(gp)
        oasis_profiles.append(OASISAgentProfile(
            agent_id=i,
            source_persona_id=gp.name,
            agent_type="external_buyer",
            user_info_dict=user_info
        ))
    print(f"✅ Mapped {len(oasis_profiles)} OASIS Agent Profiles.")

    # ── PHASE 2: SIMULATION ─────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("👥  INITIATING OASIS SIMULATION")
    print("═" * 80)
    
    # Detail Log
    print("\n[DETAILED AGENT REGISTRY]:")
    for i, p in enumerate(oasis_profiles):
        info = p.user_info_dict
        other = info['profile']['other_info']
        role_label = other.get('role', 'Buyer')
        stance = other.get('predicted_stance', 'UNKNOWN')
        mbti_label = info.get('profile', {}).get('mbti', 'UNKNOWN')
        vivid_scene = info.get('profile', {}).get('user_profile', '')[:100].replace('\n', ' ')
        print(f"#{i+1:<2} | {info['name']:<25} | {role_label:<15} | Stance: {stance:<8} | MBTI: {mbti_label}")
        print(f"     SCENE: {vivid_scene}...")

    sim_id = f"e2e_sim_{datetime.now().strftime('%H%M%S')}"
    config = OASISSimulationConfig(
        simulation_name=sim_id,
        num_timesteps=2,
        platform_type="reddit",
        population_size=len(oasis_profiles),
        interview_prompts=[
            "What is your decisive stance (BULLISH vs BEARISH) on this AI-Powered Code Review Autopilot feature?"
        ]
    )

    print(f"\n📊 Starting Simulation Round (Population: 30 | ID: {sim_id} | Timesteps: {config.num_timesteps})")
    print("Waiting for OASIS Agent Debates to resolve asynchronously...")

    try:
        series = await RunOASISSimulation(
            config=config,
            agent_profiles=oasis_profiles,
            feature=feature,
            context=company,
            base_dir="/tmp/oasis_runs"
        )

        
        # ── PHASE 3: ANALYSIS ───────────────────────────────────────────────────
        print("\n" + "═" * 80)
        print("📊  MARKET FIT ANALYSIS REPORT")
        print("═" * 80)
        
        print("\nRunning Behavioral Clustering & Sub-segment Identification...")
        # Pass the LLM client to ensure clustering doesn't fail due to missing model
        clusters = await PerformBehavioralClustering(oasis_profiles, series.raw_responses, llm_client=llm)
        CalculateAggregatedMetrics(clusters, series)
        is_consensus, strength, consensus_type = DetectConsensus(series, config)
        
        print(f"\nFINAL ADOPTION SCORE: {series.final_adoption_score}/1.0")
        print(f"CONSENSUS VERDICT:    {series.consensus_verdict}")
        print(f"CONSENSUS TYPE:       {consensus_type.upper()} (Strength: {strength})")
        
        print("\nBUYER SEGMENT BREAKDOWN (Top Sub-factions):")
        for idx, c in enumerate(clusters, 1):
            print(f"  {idx}. {c.cluster_id:<25} ({c.cluster_size} agents)")
            print(f"     -> Sentiment: {c.sentiment_score:+0.2f}")
            print(f"     -> Behavior:  {c.centroid_behavior}")

        # Full Debate Extraction
        db_path = f"/tmp/oasis_runs/{sim_id}/{sim_id}.sqlite"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            print("\n🔥 [FULL CONVERSATION THREADS LOG]:")
            print("-" * 80)
            
            cursor.execute("SELECT post_id, content FROM post;")
            posts = cursor.fetchall()
            for pid, pcontent in posts:
                cursor.execute("SELECT user_id, content FROM comment WHERE post_id = ?;", (pid,))
                comments = cursor.fetchall()
                for uid, ccontent in comments:
                    uid_int = int(uid)
                    agent_name = oasis_profiles[uid_int].user_info_dict['name'] if uid_int < len(oasis_profiles) else f"Agent_{uid}"
                    print(f"  └─ [{agent_name}]: {ccontent}")
            
            # Show mid-simulation interview responses if any
            print("\n🎤 [DYNAMIC INTERVIEW RESULTS]:")
            print("-" * 60)
            for entry in series.raw_responses:
                agent_id = entry.get("agent_id")
                agent_name = "Unknown"
                for p in oasis_profiles:
                    if str(p.agent_id) == str(agent_id):
                        agent_name = p.user_info_dict['name']
                        break
                
                print(f"\nAgent: {agent_name} (ID: {agent_id})")
                for resp in entry.get("responses", []):
                    print(f"  Q: {resp.get('question')}")
                    print(f"  A: {resp.get('content')}")
                    
            conn.close()

    except Exception as e:
        logger.error(f"Simulation Phase Failed: {e}", exc_info=True)

    print("\n" + "═" * 80)
    print("✅  END-TO-END PIPELINE COMPLETE")
    print("═" * 80 + "\n")

if __name__ == "__main__":
    # ── STEP 3: Apply nest_asyncio AFTER all C++ modules are pre-warmed ──────
    import platform as _plat
    if _plat.system() == "Darwin":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
