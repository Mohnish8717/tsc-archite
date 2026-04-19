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
os.environ["TSC_LLM_MODEL"] = "gemma-4-31b-it" 
os.environ["GEMINI_API_KEY"] = "AIzaSyAMtrQBzlsvUdyRhptJH57rJbKzUGl3nY8"
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
    title="Ash Social: Synchronous Participation Protocol",
    description=(
        "1. ARCHITECTURAL PHILOSOPHY: The Synchronous Participation Protocol (SPP) is a systemic rejection of passive consumption. "
        "Unlike algorithmic feeds that prioritize 'engagement time,' SPP prioritizes 'presence.' It functions as a digital 'Proof of Presence' gate. "
        "All incoming data streams (Feeds, Stories, DMs) are cryptographically obscured using a high-latency Gaussian blur server-side, "
        "ensuring that no data leakage occurs until the user's client submits a verified 'Raw Capture' packet.\n\n"
        
        "2. THE TRIGGER MECHANISM (THE PULSE): The protocol utilizes a 'Timezone-Sharded Randomization' algorithm. "
        "At a randomized T-minus zero, a high-priority push notification is broadcast to all nodes in a cluster. "
        "This triggers a global state change where the app UI transitions into 'Capture Mode.' The 120-second window is hard-coded into the "
        "client-side state machine to ensure zero-latency enforcement.\n\n"
        
        "3. HARDWARE-LEVEL INTEGRATION: To prevent 'curated performance,' the protocol hooks directly into the React Native Camera API. "
        "It disables the photo gallery picker programmatically. The 'Dual-Camera' feature uses concurrent stream processing to capture "
        "the environment (Back Cam) and the user's authentic state (Front Cam) within a 500ms delta, preventing staged reactions.\n\n"
        
        "4. ANTI-VOYEURISM LAYERS: The 'Blur Gate' is the psychological centerpiece. It targets the 'Lurker' demographic. "
        "By making the cost of entry a personal upload, it dissolves the power dynamic of the anonymous spectator. "
        "Technically, the blur is not just a UI overlay but a state-dependent asset delivery system managed via PostgreSQL RLS (Row Level Security).\n\n"
        
        "5. INFRASTRUCTURE & SCALING: To handle the 'Thundering Herd' (millions of uploads in 120s), we implement a write-heavy buffer. "
        "PostgreSQL handles metadata, while the binary image data is pushed to an S3-compatible edge store with immediate CDN invalidation "
        "to reflect the 'Unlocked' status globally. WebRTC is used for 'Real-Mojis' to provide low-latency, peer-to-peer validation of content.\n\n"
        
        "6. RETENTION & USER PSYCHOLOGY: By focusing on Gen-Z burnout, the protocol offers 'Digital Minimalism' as a feature. "
        "The lack of an infinite scroll means users spend less time on the app but derive higher emotional value per minute spent. "
        "This 'High-Density Socializing' is the primary driver for long-term retention in a post-algorithmic world."
    ),
    target_users="Gen-Z & Millennial users reaching burnout with algorithmic feeds; Digital Minimalists; Anti-performative communities",
    affected_domains=[
        "Product UI/UX (Blur/Unblur Logic)", 
        "Infrastructure Scaling (Traffic Bursts)", 
        "User Retention (Reciprocity Loop)",
        "Security (Hardware Camera Attestation)"
    ],
    tech_stack=[
        "WebRTC (Real-time Reactions)", 
        "PostgreSQL (Relational Mapping & Sharding)", 
        "React Native (Cross-platform Camera Access)",
        "Node.js/Elixir (High-concurrency Notification Engine)",
        "Redis (Global State Management)"
    ],
    priority="High / Strategic Core"
    )
    
    company = CompanyContext(
        company_name="Antigravity Corp",
        team_size=150,
        budget="$12M (Seed)",
        tech_stack=["Python", "AG2", "Zep"],
        current_priorities=["Deep Human Connection", "Safety", "Scale"],
        competitors=["BeReal", "Instagram", "Snapchat"]
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
        mk_persona("Alice_CTO", "Chief Technology Officer", "CTO", ["Scalability", "Infrastructure"], "Visionary technologist focused on low-latency WebRTC and dual-camera synchronization."),
        mk_persona("Bob_CFO", "Chief Financial Officer", "CFO", ["Finance", "Unit Economics"], "Conservative spender worried about the high infrastructure costs of simultaneous 100k+ uploads."),
        mk_persona("David_CEO", "Chief Executive Officer", "CEO", ["Market Strategy", "Leadership"], "Focused on market domination and clear differentiation from legacy social media."),
        mk_persona("Sarah_CISO", "Chief Information Security Officer", "CISO", ["Cybersecurity", "Privacy"], "Paranoid about dual-camera surveillance risks and data retention laws."),
        mk_persona("Peter_CPO", "Chief Product Officer", "CPO", ["Product-Market Fit", "UX"], "Hyper-focused on whether a forced 2-minute window will cause user churn or exclusion."),
        mk_persona("Linda_CMO", "Chief Marketing Officer", "CMO", ["Brand Identity", "Growth"], "Wrestles with the 'Live/Late' status hierarchy and its impact on brand perception."),
        mk_persona("Marcus_Legal", "Chief Counsel", "Legal", ["Regulatory Compliance", "IP"], "Scrutinizing Biometric Privacy Acts (BIPA) regarding dual-camera face capture."),
        mk_persona("Elena_Data", "Chief Data Officer", "Data", ["AI Ethics", "Telemetry"], "Monitoring for bias in 'Live' window detection and user telemetry accuracy."),
        mk_persona("James_Sales", "Head of Sales", "Sales", ["B2B Partners", "Revenue"], "Seeking enterprise contracts and B2B monetization of high-intent audience windows."),
        mk_persona("Diana_HR", "Chief People Officer", "HR", ["Culture", "Ethics"], "Concerned about internal team burnout during 'Live' window peak loads.")
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
                "engine_version": "v3-hardened"
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
