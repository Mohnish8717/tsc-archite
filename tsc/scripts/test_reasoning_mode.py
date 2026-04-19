import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tsc.layers.layer6_ag2_debate import AG2DebateEngine
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona, PsychologicalProfile
from tsc.models.graph import KnowledgeGraph
from tsc.models.gates import GatesSummary

async def test_reasoning_mode():
    print("--- STARTING REASONING-FIRST MODE VERIFICATION ---")
    os.environ["TSC_REASONING_ONLY"] = "true"
    
    # Use a feature that previously triggered RAG-failure escalation
    feature = FeatureProposal(
        title="Ash Social: Synchronous Participation Protocol",
        description="A protocol for real-time synchronous participation in social feeds using neural sync."
    )
    
    context = CompanyContext(
        company_name="Ash Social",
        industry="Social Media",
        current_burn=500000,
        runway_months=12
    )
    
    # Define minimal personas
    personas = [
        FinalPersona(
            name="Sarah_CTO",
            role="Chief Technology Officer",
            domain_expertise=["Architecture", "Scalability", "Sync Protocols"],
            psychological_profile=PsychologicalProfile(full_profile_text="High-vision technical leader.")
        ),
        FinalPersona(
            name="James_CFO",
            role="Chief Financial Officer",
            domain_expertise=["Burn Rate", "Capitalization", "ROI"],
            psychological_profile=PsychologicalProfile(full_profile_text="Conservative financial head.")
        ),
        FinalPersona(
            name="David_CEO",
            role="Chief Executive Officer",
            domain_expertise=["Strategy", "Market Share"],
            psychological_profile=PsychologicalProfile(full_profile_text="Growth-oriented leader.")
        )
    ]
    
    # Create dummy objects for KnowledgeGraph and GatesSummary
    graph = KnowledgeGraph()
    gates_summary = GatesSummary()
    
    engine = AG2DebateEngine(llm_client=None) 
    print(f"Reasoning Mode Active: {engine.reasoning_only}")
    
    # Correct signature for AG2DebateEngine.process()
    result = await engine.process(
        feature=feature,
        company=context,
        graph=graph,
        personas=personas,
        gates_summary=gates_summary
    )
    
    print("\n--- VERIFICATION RESULT ---")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.approval_confidence:.2f}")
    
    # Check if escalation was suppressed
    if result.overall_verdict != "ESCALATED_TO_LAYER_5":
        print("\n✅ VERIFICATION SUCCESS: Simulation completed based on logic without escalation.")
    else:
        print("\n❌ VERIFICATION FAILURE: Simulation escalated despite reasoning_only=true.")

if __name__ == "__main__":
    asyncio.run(test_reasoning_mode())
