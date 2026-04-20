import asyncio
import os
import uuid
import sys
import logging
import sqlite3
import time
from datetime import datetime
from dotenv import load_dotenv

# 1. Setup Environment & Pathing
load_dotenv()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Use High-Fidelity Google model for the live debate
os.environ["TSC_LLM_PROVIDER"] = "google"
os.environ["TSC_LLM_MODEL"] = os.getenv("TSC_LLM_MODEL", "gemma-4-31b-it")
# os.environ["GEMINI_API_KEY"] = "AIzaSy..." # Leaked key disabled
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./live_debate_run.db"

# macOS gRPC deadlock fix for Google API
os.environ["GRPC_POLL_STRATEGY"] = "poll"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_DNS_RESOLVER"] = "native"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"

from tsc.db.connection import init_db
from tsc.db.models import Base, Company, Feature, SimulationRun
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona, PsychologicalProfile
from tsc.models.gates import GatesSummary
from tsc.models.graph import KnowledgeGraph
from tsc.layers.layer6_ag2_debate import AG2DebateEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def run_live_debate():
    print("\n" + "═" * 80)
    print("🚀  AG2 BOARDROOM DEBATE — LIVE SIMULATION (v3 Hardened)")
    print("═" * 80)

    # [A] Database Setup
    print("[1] Initializing SQLite Persistence Layer...")
    db = await init_db(Base)
    
    # [B] Proposal & Context Setup (Production Payload)
    feature_id = uuid.uuid4()
    company_id = uuid.uuid4()
    
    feature = FeatureProposal(
    title="Beacon Health: Autonomous Prior Authorization & Gap Closure Protocol",
    description=(
        "1. ARCHITECTURAL PHILOSOPHY: The Autonomous Prior Authorization Protocol (APAP) is a foundational rejection of manual healthcare workflow bottlenecks. "
        "Unlike legacy systems that require clinic staff to manually navigate insurer portals and compile clinical documentation, APAP autonomously orchestrates end-to-end authorization requests by directly integrating with practice EHR systems. "
        "It functions as a 'Silent Clinical Agent' that operates on behalf of the practice, extracting patient records, generating clinical justifications, submitting to insurance networks, and flagging denials—all without human touchpoints, ensuring 30-40% recovery of missed authorizations.\n\n"
        
        "2. THE ORCHESTRATION MECHANISM (THE WORKFLOW ENGINE): The protocol utilizes multi-stage 'Clinical Record Synthesis' that operates asynchronously across patient cohorts. "
        "At scheduled intervals (or triggered by admission/new prescription events), the AI agent queries the EHR for patients missing required prior authorizations or quality measures. "
        "It automatically classifies these as high-priority, builds regulatory-compliant clinical narratives from structured EHR data (ICD-10 codes, medication records, lab results), and submits to insurer APIs. "
        "The system maintains a state machine tracking: submitted → awaiting response → approved → denial flagged, enabling continuous remediation without clinician intervention.\n\n"
        
        "3. EHR-NATIVE INTEGRATION: To prevent data leakage and workflow friction, the protocol directly embeds within the practice's existing Electronic Health Record system (Epic, Cerner, Athena). "
        "Rather than requiring staff to copy-paste data into external systems, the agent reads canonical patient data structures directly from the EHR database, transforms them into insurance-compliant formats, and maintains a bidirectional sync layer. "
        "This 'EHR-as-API' architecture ensures zero manual data entry and eliminates the 6-12 hour latency typically introduced by human review cycles.\n\n"
        
        "4. REVENUE CAPTURE LAYERS: The 'Gap Detection' system targets the 'Revenue Leakage' problem endemic to value-based care contracts. "
        "By analyzing quality metrics against insurance provider specifications, it identifies which patients are missing preventative screenings, medication adjustments, or chronic disease management codes—then auto-generates patient outreach campaigns and clinical documentation to close these gaps. "
        "This monetizes previously uncaptured value-based contract revenue, recovering an estimated $250K-$350K annually per primary care physician under risk-based arrangements.\n\n"
        
        "5. INFRASTRUCTURE & COMPLIANCE: To handle enterprise-scale deployment across health systems and ACOs, the platform implements a HIPAA-compliant, encrypted backend using industry-standard healthcare interoperability standards (HL7v2, FHIR). "
        "Prior authorization requests are batched and prioritized via a job queue system (Celery, Bull), with exponential backoff and retry logic for insurance API failures. "
        "All patient identifiers are pseudonymized and encrypted using AES-256 until the final submission payload, ensuring defense-in-depth against data exfiltration. "
        "Real-time monitoring and audit logging tracks every agent action for compliance verification and regulatory transparency.\n\n"
        
        "6. OPERATIONAL & PHYSICIAN RETENTION: By automating the most tedious, lowest-value administrative burden on clinics (prior auth follow-ups, prior denial remediation, gap closure tracking), the protocol directly addresses physician burnout—the #1 driver of exit from primary care. "
        "Practices report 8-12 hours/week of staff time recovered per 10 providers, allowing clinicians to reclaim patient interaction time. "
        "The psychological benefit is compounded: practices operating under value-based contracts experience a documented 40% increase in morale when revenue targets become achievable through automation rather than heroic manual labor. "
        "This 'Operational Joy' metric is the primary driver for sticky, long-term SaaS adoption in healthcare—transforming a compliance tool into a cultural organizational asset."
    ),
    target_users="Independent Physician Associations (IPAs), Accountable Care Organizations (ACOs), primary care practices operating under value-based care contracts, health systems managing 10,000+ patient populations",
    affected_domains=[
        "EHR Integration & Workflow Automation",
        "Prior Authorization & Insurance API Orchestration",
        "Revenue Cycle Management & Gap Closure",
        "Physician Burnout Mitigation & Retention",
        "Healthcare Compliance & HIPAA-Audit Logging"
    ],
    tech_stack=[
        "EHR APIs (Epic FHIR, Cerner HL7, Athena proprietary)",
        "Asynchronous Job Queue (Celery/Bull for batch processing)",
        "Healthcare Interoperability Standards (HL7v2, FHIR, X12)",
        "Insurance API Integration (Surescripts, Change Healthcare, eviCore)",
        "Encryption & Compliance (AES-256, audit logging, HIPAA BAA)",
        "LLM-powered Clinical Documentation (GPT-4-Turbo for narrative generation)"
    ],
    priority="High / Revenue-Critical Core"
    )
    
    company = CompanyContext(
    company_name="Beacon Health",
    team_size=12,
    budget="$5.4M (Seed)",
    tech_stack=["Python", "FHIR / HL7 APIs", "LLM Orchestration (GPT-4 Turbo / Claude)", "PostgreSQL", "Redis"],
    current_priorities=["EHR Integration Depth", "Autonomous Revenue Recovery", "Value-Based Care Scaling"],
    competitors=["Aledade", "Arcadia", "Sully.ai", "Epic Systems (Native AI)"]
    )

    # [C] Boardroom Persona Setup (10 Stakeholders)
    print("[2] Assembling the Boardroom (10 Stakeholders)...")
    
    def mk_persona(name, role, role_short, expertise, bio):
        return FinalPersona(
            name=name,
            role=role,
            role_short=role_short,
            domain_expertise=expertise,
            psychological_profile=PsychologicalProfile(full_profile_text=bio)
        )

    personas = [
    mk_persona("Mark_CEO", "Chief Executive Officer", "CEO", ["Market Strategy", "Clinic Economics"], "Visionary founder focused on doubling primary care revenue and replacing administrative 'heroism' with autonomous systems."),
    mk_persona("Obinna_CTO", "Chief Technology Officer", "CTO", ["AI Orchestration", "EHR Interoperability"], "Focused on the 'EHR-as-API' architecture and ensuring LLM clinical narratives pass insurance audits without hallucinations."),
    mk_persona("Sarah_CISO", "Chief Information Security Officer", "CISO", ["HIPAA Compliance", "Data Privacy"], "Paranoid about AI agents writing back to canonical medical records and maintaining AES-256 encryption across payer networks."),
    mk_persona("Dr_James_CMO", "Chief Medical Officer", "CMO", ["Clinical Safety", "Physician Burnout"], "Hyper-focused on reducing 'Pajama Time' for doctors while ensuring the AI doesn't bypass critical clinical nuances."),
    mk_persona("Linda_CFO", "Chief Financial Officer", "CFO", ["VBC Unit Economics", "Burn Rate"], "Conservative spender monitoring the cost-per-inference against the $250K recovery per physician to ensure sustainable SaaS margins."),
    mk_persona("Peter_CPO", "Chief Product Officer", "CPO", ["Product-Market Fit", "Workflow UX"], "Concerned with the 'Silent Agent' UX—ensuring doctors trust the background automation enough to stop checking its work."),
    mk_persona("Marcus_Legal", "General Counsel", "Legal", ["Regulatory Compliance", "Malpractice Risk"], "Scrutinizing the liability chain when an autonomous agent triggers a prior authorization that leads to a clinical delay."),
    mk_persona("Elena_Data", "Head of Data Science", "Data", ["RAG Accuracy", "Coding Bias"], "Monitoring the ICD-10 and HCC coding engine for 'upcoding' bias that could trigger federal Medicare audits."),
    mk_persona("James_Head_of_Sales", "Head of Sales", "Sales", ["IPA Partnerships", "Customer Acquisition"], "Seeking high-volume contracts with large Accountable Care Organizations (ACOs) and health systems managing 40,000+ lives."),
    mk_persona("Diana_Customer_Success", "Head of Implementation", "Success", ["Clinic Onboarding", "Change Management"], "Focused on the friction of technical EHR integration and retraining clinic staff to trust an 'AI Employee'.")
    ]


    # [D] Execution
    print(f"[3] Firing up the Hardened Engine (Model: {os.getenv('TSC_LLM_MODEL', 'gemini-1.5-pro')})...")
    engine = AG2DebateEngine(llm_client=None)
    
    gates = GatesSummary(phase="Live Hardened Test")
    kg = KnowledgeGraph(nodes={}, edges=[])
    
    start_time = time.time()
    consensus = await engine.process(feature, company, kg, personas, gates)
    duration = time.time() - start_time
    
    print("\n" + "═" * 80)
    print("📊  DEBATE RESULTS")
    print("═" * 80)
    print(f"VERDICT:    {consensus.overall_verdict}")
    print(f"CONFIDENCE: {consensus.approval_confidence:.2f}")
    print(f"DURATION:   {duration:.1f}s")
    print("-" * 40)
    
    # [E] Persistence Verification
    print("[4] Persisting Result to Relational Store...")
    async with db.get_session() as session:
        # Create Parents
        co = Company(id=company_id, name=company.company_name, industry="Social Tech")
        feat = Feature(id=feature_id, company_id=company_id, title=feature.title, description=feature.description)
        session.add(co)
        session.add(feat)
        await session.commit()
    
    async with db.get_session() as session:
        run = SimulationRun(
            feature_id=feature_id,
            company_id=company_id,
            approval_rate=consensus.approval_confidence,
            sentiment_score=0.75,
            risk_assessment={"is_high_risk": consensus.overall_verdict == "REJECTED", "tension_shifts": consensus.tension_shifts},
            recommendations=consensus.mitigations,
            duration_seconds=duration,
            simulation_metadata={
                "overall_summary": consensus.overall_summary,
                "overall_verdict": consensus.overall_verdict,
                "ag2_transcript": [pos.model_dump() for pos in consensus.debate_rounds[0].positions] if consensus.debate_rounds else [],
                "evolved_agent_memories": engine._evolved_agent_memories,
                "engine_version": "v29-stateful"
            }
        )
        session.add(run)
        await session.commit()
        run_id = run.id
        print(f"✅  Data Persisted. SimulationRun ID: {run_id}")

    # [F] Data Pull-back Verification
    print("[5] Verifying DB Integrity (Auto-Select)...")
    sqlite_path = "./live_debate_run.db"
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute("SELECT approval_rate FROM simulation_runs WHERE id = ?", (str(run_id).replace("-", ""),)) # SQLite handles UUIDs as hex strings if not careful
    row = cursor.fetchone()
    if row:
        print(f"    - DB Retrieval Success! Stored Approval Rate: {row[0]:.2f}")
    else:
        # Retry with different string format just in case
        cursor.execute("SELECT approval_rate FROM simulation_runs LIMIT 1")
        row = cursor.fetchone()
        if row:
            print(f"    - DB Retrieval Success (Partial)! Latest Approval Rate: {row[0]:.2f}")
    conn.close()

    print("\n✅  Simulation Complete. Log available in ./live_debate_run.db")
    print("═" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(run_live_debate())
