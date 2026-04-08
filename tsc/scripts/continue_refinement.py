import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path("/Users/mohnish/Downloads/tsc architecture")
sys.path.append(str(PROJECT_ROOT))

# Configure Environment
os.environ["TSC_LLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TSC_LLM_MODEL"] = "llama-3.1-8b-instant"

# Import Required Components
from tsc.oasis.models import MarketSentimentSeries, OASISAgentProfile, OASISSimulationConfig
from tsc.oasis.clustering import PerformBehavioralClustering, DetectConsensus, CalculateAggregatedMetrics
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.layers.layer5_refinement import RefinementEngine
from tsc.models.gates import GateResult, GatesSummary, GateVerdict
from tsc.models.personas import FinalPersona, PsychologicalProfile, PredictedStance
from tsc.models.graph import KnowledgeGraph
from tsc.models.chunks import ProblemContextBundle
from tsc.llm.factory import create_llm_client
from tsc.config import settings

async def continue_refinement():
    print("🔄 Resuming Refinement from Stored Logs...")
    
    # 1. Setup Context (Corrected with effort estimates)
    feature = FeatureProposal(
        title="Mandatory Daily 4-Hour Sync Meetings",
        description="A new policy requiring all developers and stakeholders to completely drop their work and attend a mandatory 4-hour sync meeting every single day to verbally explain every line of code written to upper management. Includes automatic screen capturing to ensure active participation.",
        effort_weeks_min=4,
        effort_weeks_max=8
    )
    context = CompanyContext(
        company_name="TSC", 
        mission="Stability First",
        team_size=50,
        tech_stack=["Python", "React", "AWS"]
    )
    
    # 2. Reconstruct Profiles (Expert Market Personas)
    profiles = []
    personas = [
        ("Dr. Aris Thorne", "Organizational Psychologist", "Expert in workplace productivity and flow states. Focuses on Deep Work disruption.", "INTJ"),
        ("Sarah Jenkins", "Remote Operations Lead", "Manages 1,000+ remote workers. Focuses on Coordination Costs and talent retention.", "ENTJ"),
        ("Marcus Vane", "Employment Law Attorney", "Expert in digital privacy and labor rights. Scrutinizes surveillance for legal compliance.", "ISTJ")
    ]
    for i, (name, role, bias, mbti) in enumerate(personas):
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

    # 3. Load Stored Interview Responses
    log_dir = Path("/tmp/oasis_runs/conv_test_184702")
    with open(log_dir / "mid_sim_interview_responses.json", 'r') as f:
        raw_responses = json.load(f)

    # 4. Re-calculate Simulation Metrics (The TSC logic)
    print("📊 Re-calculating Behavioral Clusters and Consensus...")
    series = MarketSentimentSeries(
        simulation_id="conv_test_184702",
        feature_proposal_id="mandatory_sync",
        raw_responses=raw_responses
    )
    config = OASISSimulationConfig(simulation_name="offline", population_size=3)
    
    clusters = await PerformBehavioralClustering(profiles, raw_responses)
    CalculateAggregatedMetrics(clusters, series)
    is_consensus, strength, consensus_type = DetectConsensus(series, config)
    
    print(f"Adoption Score: {series.final_adoption_score}/1.0")
    print(f"Consensus:      {series.consensus_verdict} ({consensus_type.upper()})")

    # 5. Wrap Simulation Results into a GatesSummary
    market_fit_result = GateResult(
        gate_id="4.5",
        gate_name="Monte Carlo Market Fit",
        verdict=GateVerdict.FAIL,
        score=0.49,
        details={"consensus": series.consensus_verdict, "clusters": [c.model_dump() for c in clusters]}
    )
    summary = GatesSummary(
        results=[market_fit_result],
        overall_score=0.49,
        failed_gates=["4.5"]

    )
    
    # 6. Map Profiles to FinalPersonas for Layer 5
    persona_objects = []
    for p in profiles:
        persona_objects.append(FinalPersona(
            name=p.user_info_dict['name'],
            role=p.user_info_dict['profile'].get('role', 'Stakeholder'),
            psychological_profile=PsychologicalProfile(
                predicted_stance=PredictedStance(
                    prediction="BEARISH",
                    potential_objections=["burnout", "micromanagement", "privacy"],
                    likely_conditions=["Reduced meeting length", "Asynchronous updates"]
                )
            )
        ))

    # 7. Execute Refinement Engine
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n🛠️  Executing Layer 5: Refinement...")
    llm = create_llm_client(settings=settings)
    refinement_engine = RefinementEngine(llm)
    
    # Mock Graph/Bundle (MANDATORY DATA) - enriched with actual dialogue excerpts (Opt C)
    from tsc.models.graph import GraphEntity, GraphRelationship, RelationshipType
    from tsc.models.chunks import EnrichedChunk
    
    # Extract real agent dialogue snippets from raw_responses for behavioral evidence
    dialogue_chunks = []
    for i, resp in enumerate(raw_responses[:5]):  # top 5 responses
        text = resp.get("content", resp.get("message", str(resp)))[:300]
        if text:
            dialogue_chunks.append(EnrichedChunk(
                chunk_id=f"c_resp_{i:02d}",
                text=text,
                coherence_score=min(0.9, 0.5 + i * 0.05)  # varying coherence
            ))
    
    # Add summary context chunks
    dialogue_chunks.extend([
        EnrichedChunk(
            chunk_id="c_burnout",
            text="Mandatory meetings disrupt Deep Work states. Flow state research shows 4-hour blocks destroy developer productivity by 73% (Newport 2020). Forced synchronous communication is incompatible with remote-first cultures.",
            coherence_score=0.95
        ),
        EnrichedChunk(
            chunk_id="c_legal",
            text="Screen capture mandates raise GDPR/CCPA compliance risks. Mandatory surveillance without consent creates significant labor law exposure. Several EU jurisdictions explicitly prohibit continuous employee monitoring.",
            coherence_score=0.88
        ),
        EnrichedChunk(
            chunk_id="c_retention",
            text="Talent retention risk is extreme. Glassdoor studies show 81% of developers consider meeting overload a dealbreaker. Top performers—who have market leverage—will leave first.",
            coherence_score=0.82
        )
    ])
    
    graph = KnowledgeGraph(
        nodes={
            "prod_01": GraphEntity(id="prod_01", name="Productivity", type="CONCEPT", mentions=10),
            "dev_01": GraphEntity(id="dev_01", name="Developer Happiness", type="METRIC", mentions=8),
            "flow_01": GraphEntity(id="flow_01", name="Deep Work / Flow State", type="CONCEPT", mentions=7),
            "legal_01": GraphEntity(id="legal_01", name="GDPR Compliance", type="RISK", mentions=5),
            "ret_01": GraphEntity(id="ret_01", name="Talent Retention", type="METRIC", mentions=6),
        }, 
        edges=[
            GraphRelationship(
                id="r_01", source_entity="prod_01", target_entity="dev_01", 
                relationship_type=RelationshipType.IMPACTS, confidence=0.9
            ),
            GraphRelationship(
                id="r_02", source_entity="flow_01", target_entity="prod_01", 
                relationship_type=RelationshipType.IMPACTS, confidence=0.85
            ),
        ]
    )
    bundle = ProblemContextBundle(chunks=dialogue_chunks)

    refined_summary = await refinement_engine.process(
        gates_summary=summary,
        feature=feature,
        company=context,
        graph=graph,
        bundle=bundle,
        personas=persona_objects
    )
    
    print("\n" + "═"*70)
    print("✅ REFINEMENT COMPLETE")
    print("═"*70)
    print(f"Original Score:    {summary.overall_score * 100:.1f}%")
    print(f"Refined Score:     {refined_summary.overall_score * 100:.1f}%")
    improvement = refined_summary.overall_score - summary.overall_score
    print(f"Improvement:       {'+' if improvement >= 0 else ''}{improvement * 100:.1f}%")
    
    audit = getattr(refined_summary, "_refinement_audit", [])
    if audit:
        decision = audit[-1]
        accepted = decision['decision']['accepted']
        reason = decision['decision']['reason']
        print(f"\nDecision:          {'✅ ACCEPTED' if accepted else '❌ REJECTED'}")
        print(f"Reason:            {reason}")
        
        refinement = decision.get("refinement", {})
        
        # Print LLM analysis
        analysis = refinement.get("analysis", "N/A")
        print(f"\n📊 LLM ANALYSIS:\n{analysis}")
        
        # Print root causes
        root_causes = refinement.get("root_causes", [])
        if root_causes:
            print("\n🔍 ROOT CAUSES IDENTIFIED:")
            for i, rc in enumerate(root_causes, 1):
                print(f"  {i}. {rc}")
        
        # Print suggestions
        suggestions = refinement.get("suggestions", [])
        if suggestions:
            print(f"\n💡 REFINEMENT SUGGESTIONS ({len(suggestions)} total):")
            for i, s in enumerate(suggestions, 1):
                print(f"  {i}. {s.get('title', 'Untitled')}")
                print(f"     Impact: {s.get('impact', 'N/A')}")
                print(f"     Effort: {s.get('effort_impact', 'N/A')}")
        
        # Print revised scope
        revised_scope = refinement.get("revised_scope", "N/A")
        print(f"\n🎯 REVISED SCOPE FROM LLM:\n{revised_scope}")
        
        # Print confidence
        confidence = refinement.get("confidence", 0)
        print(f"\n🎲 LLM Confidence: {confidence:.0%}")

if __name__ == "__main__":
    asyncio.run(continue_refinement())
