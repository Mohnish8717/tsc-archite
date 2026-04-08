"""
Verification script for the PersonaSelectionEngine.

Tests all 5 phases with mock archetypes that mimic real pipeline output.

Run:
    PYTHONPATH=. python3 tsc/scripts/test_selection_engine.py
"""
import asyncio
import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TSC_LLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TSC_LLM_MODEL"] = "llama-3.1-8b-instant"

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import (
    FinalPersona, PsychologicalProfile, PredictedStance,
    EmotionalTriggers, CommunicationStyle, DecisionPattern,
    Stakeholder, StakeholderContextBundle
)
from tsc.models.graph import KnowledgeGraph, GraphEntity
from tsc.selection.engine import PersonaSelectionEngine
from tsc.llm.factory import create_llm_client
from tsc.config import settings


def make_persona(name: str, role: str, bio: str, frustrated_by: list, scared_of: list) -> FinalPersona:
    return FinalPersona(
        name=name,
        role=role,
        profile_word_count=len(bio.split()),
        profile_confidence=0.8,
        influence_strength=0.5,
        receptiveness=0.5,
        psychological_profile=PsychologicalProfile(
            mbti="INTJ",
            full_profile_text=bio,
            key_traits=["analytical", "detail-oriented"],
            emotional_triggers=EmotionalTriggers(
                excited_by=["transparency", "efficiency"],
                scared_of=scared_of,
            ),
            communication_style=CommunicationStyle(default="Direct"),
            decision_pattern=DecisionPattern(risk_tolerance="Medium"),
            predicted_stance=PredictedStance(prediction="BEARISH", feature="AI Code Review"),
        ),
    )


async def main():
    print("\n" + "═" * 70)
    print("🧪  PersonaSelectionEngine — Verification Suite")
    print("═" * 70)

    # ── Mock feature ───────────────────────────────────────────────────
    feature = FeatureProposal(
        title="AI-Powered Code Review Autopilot",
        description=(
            "An autonomous AI system that performs code reviews on all pull requests. "
            "It automatically suggests refactors and can auto-approve PRs that meet "
            "security and style guidelines. Reduced human involvement in reviewing "
            "standard boilerplate or minor changes."
        ),
        effort_weeks_min=4,
        effort_weeks_max=12,
    )



    # ── Database Initialization ────────────────────────────────────────
    from tsc.db.connection import get_db
    from tsc.repositories.persona_repository import PersonaRepository
    from tsc.caching.lru_cache import PersonaCache
    from tsc.layers.layer3_personas import PersonaGenerator
    from unittest.mock import MagicMock, AsyncMock

    llm = create_llm_client(settings=settings)
    db = get_db()
    repo = PersonaRepository(db)
    cache = PersonaCache()
    # Mock GraphStore as it is not needed for DB retrieval
    graph_store = MagicMock()
    graph_store.retrieve_customer_context = AsyncMock(return_value={})
    graph_store.retrieve_stakeholder_context = AsyncMock(return_value={})
    graph_store.get_persona_context = AsyncMock(return_value={})
    
    generator = PersonaGenerator(
        llm_client=llm,
        graph_store=graph_store,
        persona_repo=repo,
        persona_cache=cache
    )

    # ── Phase 1.5: Actual DB Extraction ────────────────────────────────
    print("\n📥  DB Extraction: Searching for candidates in 'enterprise' & 'startup' segments...")
    
    # Retrieve candidates from actual DB
    ent_candidates = await generator.get_external_personas_for_market("enterprise")
    start_candidates = await generator.get_external_personas_for_market("startup")
    candidates = ent_candidates + start_candidates

    if not candidates:
        print("⚠️  No candidates found in DB. Falling back to 'consumer' segment...")
        candidates = await generator.get_external_personas_for_market("consumer")

    print(f"✅  Found {len(candidates)} total candidates in DB.")

    # 1. Get Tension Vector (this usually happens via Engine, but we need it for pre-selection)
    from tsc.selection.tension_vector import FeatureTensionAnalyzer
    
    print("\n[Layer 3] Extracting Tension Vector for pre-selection...")
    analyzer = FeatureTensionAnalyzer(llm)
    tension_vector = await analyzer.analyze(feature)
    
    print(f"\n[Layer 3] Running Tension-Aware Pre-Selection (Target=8 seeds from {len(candidates)} candidates)")
    # Using the actual helper method we added to PersonaGenerator
    seeds = generator._select_seeds_from_candidates(candidates, tension_vector, target_seeds=8)
    
    print("\n🌱  Optimal Seeds Selected from DB:")
    for s in seeds:
        print(f"    - {s.name:<25} | {s.role:<15} | Type: {s.persona_type}")

    # ── Phase 2.1: High-Fidelity Persona Grounding (Step 3.3) ─────────
    # This turns the generic DB records into 2500-word simulation narratives
    print("\n⚡  GROUNDING PHASE: Generating 2500-character grounded profiles for seeds...")
    
    # Mock Company Context for grounding
    company = CompanyContext(
        company_name="Antigravity Corp",
        team_size=150,
        tech_stack=["Python", "React", "Groq"],
        current_priorities=["Security", "Developer Velocity"]
    )
    
    # Mock Knowledge Graph for entity injection
    kg = KnowledgeGraph(nodes={}, edges=[])
    top_entities = [] # Empty for this test or populate with mock GraphEntity if needed
    
    grounded_seeds = []
    for i, s in enumerate(seeds):
        print(f"    [{i+1}/8] Grounding {s.name} ({s.role})...")
        
        # 1. Map FinalPersona to Stakeholder model
        sh = Stakeholder(
            name=s.name,
            role=s.role,
            persona_type=s.persona_type,
            relevance_score=1.0,
            domain_relevance=s.role
        )
        
        # 2. Retrieve context from DB/Graph
        ctx = await generator._retrieve_context(sh)
        
        # 3. Generate high-fidelity 2500-word profile
        try:
            grounded_persona = await generator._generate_profile(
                sh, ctx, feature, company, top_entities
            )
            grounded_seeds.append(grounded_persona)
            print(f"        ✓ Narrative Word Count: {grounded_persona.profile_word_count}")
        except Exception as e:
            print(f"        ⚠️  Grounding failed for {s.name}, using original: {e}")
            grounded_seeds.append(s)

    seeds = grounded_seeds

    # ── Run the engine ─────────────────────────────────────────────────
    engine = PersonaSelectionEngine(llm, rng_seed=42)

    target_n = 30  # Use 30 for fast verification (prod would be 150)
    print(f"\n🚀  Running Selection engine: {len(seeds)} seeds → target {target_n} agents\n")

    expanded: list[FinalPersona]
    expanded, result = await engine.select(
        archetypes=seeds,
        feature=feature,
        target_n=target_n,
    )

    expanded_list: list[FinalPersona] = list(expanded)
    
    # ── Detailed Profile Inspection ───────────────────────────────────
    print("\n🧐  DETAILED PROFILE INSPECTION (Final Simulation Set):")
    
    # 1. Summary of all 30 agents
    print("\nAGENT REGISTRY (30 Agents):")
    print(f"{'#':<3} | {'Name':<45} | {'Role':<15} | {'Pole':<10}")
    print("-" * 80)
    for i, a in enumerate(expanded_list):
        # Determine pole from result
        base_name = a.name.split(" [")[0]
        p_obj = next((p for p in result.poles if p.persona_name == base_name), None)
        pole_str = p_obj.pole.value if p_obj else "SWING"
        print(f"{i+1:<3} | {a.name:<45} | {a.role:<15} | {pole_str:<10}")

    # 2. Detailed sample of an Original Archetype
    archetype = expanded_list[0]
    print(f"\n[SAMPLE: ARCHETYPE] {archetype.name}")
    print(f"Traits: {', '.join(archetype.psychological_profile.key_traits)}")
    print(f"Excited: {', '.join(archetype.psychological_profile.emotional_triggers.excited_by)}")

    # 3. Detailed sample of a Synthetic Clone with Minority Voice
    clones_with_mvp = [a for a in expanded_list if "[" in a.name and "Lived Experience" in a.psychological_profile.full_profile_text]
    if clones_with_mvp:
        clone = clones_with_mvp[0]
        print(f"\n[SAMPLE: SYNTHETIC CLONE + MINORITY VOICE] {clone.name}")
        mv_start = clone.psychological_profile.full_profile_text.find("[Lived Experience")
        mv_text = clone.psychological_profile.full_profile_text[mv_start:mv_start+200] + "..."
        print(f"Injected Context: {mv_text}")
    else:
        # Fallback to a regular clone
        clones = [a for a in expanded_list if "[" in a.name]
        if clones:
            clone = clones[0]
            print(f"\n[SAMPLE: SYNTHETIC CLONE] {clone.name}")
            print(f"Bio Extension: {clone.psychological_profile.full_profile_text[-150:]}")

    # ── Print Results ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("📊  SELECTION RESULT")
    print("═" * 70)

    # Tension Vector
    print(f"\n🧲  TENSION VECTOR ({len(result.tension_vector.dimensions)} dimensions):")
    for dim, score in result.tension_vector.dimensions.items():
        bar = "▓" * int(abs(score) * 10)
        sign = "+" if score > 0 else ""
        print(f"   {dim:<30} {sign}{score:.2f}  {bar}")
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

    # Expansion metadata
    print(f"\n🔢  EXPANSION: {result.expansion_metadata.get('archetype_count')} archetypes "
          f"→ {result.actual_n} agents via '{result.strategy_used}' strategy "
          f"({result.expansion_metadata.get('synthetic_count')} synthetic clones)")

    # ── Log all Personas for Simulation ─────────────────────────────────
    log_path = PROJECT_ROOT / "persona.log"
    print(f"\n📂  LOGGING {len(expanded_list)} PERSONAS TO: {log_path}")
    with open(log_path, "w") as f:
        f.write(f"PERSONA LOG: {len(expanded_list)} Agents for Simulation\n")
        f.write("=" * 80 + "\n\n")
        for i, a in enumerate(expanded_list):
            f.write(f"AGENT #{i+1}\n")
            f.write(f"NAME: {a.name}\n")
            f.write(f"ROLE: {a.role}\n")
            f.write(f"TRAITS: {', '.join(a.psychological_profile.key_traits)}\n")
            f.write("-" * 40 + "\n")
            f.write(a.psychological_profile.full_profile_text + "\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print("\n" + "═" * 70)
    print("✅  Verification complete")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
