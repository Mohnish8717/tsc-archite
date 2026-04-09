import asyncio
import os
import uuid
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Prevent pollution of dev DB
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_debate_storage.db"

from tsc.db.connection import init_db, get_db
from tsc.db.models import Base, Company, Feature, SimulationRun
from tsc.models.debate import ConsensusResult, DebateRound, DebatePosition
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona, PsychologicalProfile
from tsc.models.gates import GatesSummary

async def run_db_persistence_test():
    print("[1] Initializing SQLite Database (test_debate_storage.db)...")
    db = await init_db(Base)
    
    # Generate mock DB records
    company_id = uuid.uuid4()
    feature_id = uuid.uuid4()
    
    # 2. Mocking Deep Models for Real Engine Execution
    feature_prop = FeatureProposal(
        feature_id=str(feature_id),
        title="Quantum Telepathy Communication Module",
        description="A direct-to-brain communication layer utilizing entangled brainwave modulation for instant team-sync across global divisions.",
        target_audience="Enterprise Engineering Teams",
        core_problem="Asynchronous communication delays globally",
        solution_hypothesis="Instant brainwave alignment reduces meeting time by 98%",
        metrics={"KPI": "Time to consensus"},
        constraints={"Budget": "$10m"}
    )
    
    comp_context = CompanyContext(
        company_name="NeuroSync Inc.",
        industry="DeepTech / Biotechnology",
        current_market_position="Series A Startup",
        core_value_proposition="Zero-Latency Humanity",
        key_competitors=["Neuralink", "Synchron"],
        business_model="Enterprise SaaS",
        revenue_stage="Pre-Revenue",
        strategic_goals=["Achieve FDA Phase 1 approval", "Launch MVP in 18 months"]
    )
    
    cto_profile = PsychologicalProfile(
        openness=0.9, conscientiousness=0.7, extraversion=0.5, agreeableness=0.4, neuroticism=0.2,
        dark_triad_traits={}, cognitive_biases=["Optimism Bias"], full_profile_text="Brilliant but reckless visionary CTO pushing the boundaries of physics. Focuses ONLY on technical feasibility and architecture scalability."
    )
    cfo_profile = PsychologicalProfile(
        openness=0.3, conscientiousness=0.9, extraversion=0.5, agreeableness=0.3, neuroticism=0.8,
        dark_triad_traits={}, cognitive_biases=["Loss Aversion"], full_profile_text="Ultra-conservative CFO terrified of FDA fines and unproven R&D sinks. Strict focus on compliance fines, R&D cost, and burn rate."
    )
    ceo_profile = PsychologicalProfile(
        openness=0.8, conscientiousness=0.8, extraversion=0.9, agreeableness=0.7, neuroticism=0.3,
        dark_triad_traits={"narcissism": 0.6}, cognitive_biases=["Overconfidence Effect"], full_profile_text="Charismatic CEO focused on macro-vision, market leadership, and overarching strategic growth."
    )
    ciso_profile = PsychologicalProfile(
        openness=0.4, conscientiousness=0.95, extraversion=0.3, agreeableness=0.2, neuroticism=0.7,
        dark_triad_traits={}, cognitive_biases=["Negativity Bias"], full_profile_text="Paranoid CISO hyper-focused on data privacy, zero-day vulnerabilities, cryptographic security, and risk mitigation."
    )
    cpo_profile = PsychologicalProfile(
        openness=0.85, conscientiousness=0.75, extraversion=0.6, agreeableness=0.8, neuroticism=0.4,
        dark_triad_traits={}, cognitive_biases=["Pro-innovation bias"], full_profile_text="User-centric CPO obsessed with Product-Market fit, user friction, UX/UI excellence, and intuitive feature roadmaps."
    )
    cmo_profile = PsychologicalProfile(
        openness=0.9, conscientiousness=0.6, extraversion=0.95, agreeableness=0.8, neuroticism=0.3,
        dark_triad_traits={}, cognitive_biases=["Halo effect"], full_profile_text="Energetic CMO focused solely on brand perception, go-to-market strategy, PR fallout, and user acquisition."
    )
    legal_profile = PsychologicalProfile(
        openness=0.2, conscientiousness=0.98, extraversion=0.4, agreeableness=0.1, neuroticism=0.6,
        dark_triad_traits={}, cognitive_biases=["Status quo bias"], full_profile_text="Strict Chief Counsel blocking anything that has liability, lawsuits, intellectual property risks, or regulatory compliance issues."
    )
    data_profile = PsychologicalProfile(
        openness=0.7, conscientiousness=0.9, extraversion=0.3, agreeableness=0.5, neuroticism=0.5,
        dark_triad_traits={}, cognitive_biases=["Base rate fallacy"], full_profile_text="Analytical Chief Data Officer focused on telemetry, AI ethics, dataset bias, and measurable KPI tracking."
    )
    sales_profile = PsychologicalProfile(
        openness=0.6, conscientiousness=0.8, extraversion=0.99, agreeableness=0.7, neuroticism=0.2,
        dark_triad_traits={"machiavellianism": 0.5}, cognitive_biases=["Action bias"], full_profile_text="Aggressive Head of Sales demanding enterprise contracts, B2B conversion, overcoming client objections, and hitting revenue targets."
    )
    hr_profile = PsychologicalProfile(
        openness=0.7, conscientiousness=0.8, extraversion=0.8, agreeableness=0.9, neuroticism=0.6,
        dark_triad_traits={}, cognitive_biases=["Empathy gap"], full_profile_text="Empathetic Chief People Officer protecting employee morale, internal culture impact, diversity, and internal training."
    )
    
    personas = [
        FinalPersona(name="Alice_CTO", role="Chief Technology Officer", domain_expertise=["Quantum Physics", "Engineering"], goals=["Ship ASAP"], pain_points=["Regulation delays"], psychological_profile=cto_profile),
        FinalPersona(name="Bob_CFO", role="Chief Financial Officer", domain_expertise=["Finance", "Compliance"], goals=["Conserve cash runway"], pain_points=["Unbounded R&D costs"], psychological_profile=cfo_profile),
        FinalPersona(name="David_CEO", role="Chief Executive Officer", domain_expertise=["Strategy", "Leadership"], goals=["Market domination"], pain_points=["Competitor innovation"], psychological_profile=ceo_profile),
        FinalPersona(name="Sarah_CISO", role="Chief Information Security Officer", domain_expertise=["Cybersecurity", "Risk"], goals=["Zero breaches"], pain_points=["Unvetted features"], psychological_profile=ciso_profile),
        FinalPersona(name="Peter_CPO", role="Chief Product Officer", domain_expertise=["Product Management", "UX/UI"], goals=["Perfect PMF"], pain_points=["User churn"], psychological_profile=cpo_profile),
        FinalPersona(name="Linda_CMO", role="Chief Marketing Officer", domain_expertise=["Marketing", "PR"], goals=["Viral adoption"], pain_points=["Bad press"], psychological_profile=cmo_profile),
        FinalPersona(name="Marcus_Legal", role="Chief Counsel", domain_expertise=["Law", "Compliance"], goals=["Zero lawsuits"], pain_points=["FDA action"], psychological_profile=legal_profile),
        FinalPersona(name="Elena_Data", role="Chief Data Officer", domain_expertise=["Data Science", "Ethics"], goals=["Unbiased models"], pain_points=["Corrupted telemetry"], psychological_profile=data_profile),
        FinalPersona(name="James_Sales", role="Head of Sales", domain_expertise=["Enterprise B2B", "Revenue"], goals=["Hit quota"], pain_points=["Long sales cycles"], psychological_profile=sales_profile),
        FinalPersona(name="Diana_HR", role="Chief People Officer", domain_expertise=["HR", "Culture"], goals=["High morale"], pain_points=["Burnout"], psychological_profile=hr_profile)
    ]
    
    gates = GatesSummary(phase="Debate Setup")
    
    print("[2] Firing up Real AG2 Debate Engine (This may take several minutes)...")
    from tsc.layers.layer6_ag2_debate import AG2DebateEngine
    engine = AG2DebateEngine(llm_client=None)  # Uses environment config via OS variables
    consensus = await engine.process(feature_prop, comp_context, None, personas, gates)
    
    print("\n[3] Saving Live Autonomous Debate mapping into the Relational Store...")
    async with db.get_session() as session:
        # Create Parents
        company = Company(id=company_id, name="Test Co", industry="Tech")
        feature = Feature(id=feature_id, company_id=company_id, title="Telepathic Keyboard")
        session.add(company)
        session.add(feature)
        await session.commit()
    
    async with db.get_session() as session:
        # Save SimulationRun Mapping
        run = SimulationRun(
            feature_id=feature_id,
            company_id=company_id,
            approval_rate=consensus.approval_confidence,
            sentiment_score=0.9,
            risk_assessment={"is_high_risk": False, "tension_shifts": consensus.tension_shifts},
            recommendations=consensus.mitigations,
            simulation_metadata={
                "ag2_transcript": [pos.model_dump() for pos in consensus.debate_rounds[0].positions],
                "transcript_synthesis": consensus.debate_rounds[0].synthesis
            }
        )
        session.add(run)
        await session.commit()
        run_id = run.id
        print(f"    - Inserted correctly! Generated SimulationRun ID: {run_id}")
        
    print("[3] Retrieving Debate Transcript from the Database...")
    from sqlalchemy import select
    async with db.get_session() as session:
        result = await session.execute(select(SimulationRun).where(SimulationRun.id == run_id))
        stored_run = result.scalar_one()
        
        transcript = stored_run.simulation_metadata.get("ag2_transcript", [])
        print("\n--- RETRIEVED AG2 CHAT LOG TRANSCRIPT ---")
        for msg in transcript:
            print(f"[{msg['stakeholder_name']} - {msg['role']}]: {msg['statement']}")
        print("-----------------------------------------")
        print(f"Verification Successful: Got {len(transcript)} transcript nodes natively mapped and persisted.")
        
    await db.close()

if __name__ == "__main__":
    asyncio.run(run_db_persistence_test())
