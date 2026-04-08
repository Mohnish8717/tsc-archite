import asyncio
import uuid
import logging
import json
import os
from typing import List, Dict, Any
from datetime import datetime

from tsc.db.models import Base, Company, InternalPersona, ExternalPersona
from tsc.db.connection import get_db, init_db
from tsc.repositories.persona_repository import PersonaRepository
from tsc.llm.factory import create_llm_client
from tsc.config import LLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Grounding Data (Simplified for script)
GROUNDING_DATA = """
[INTERVIEWS]
Alice (Power User): Jarring quality drops on commute. Wants selective blurring (background) to keep faces clear.
Bob (engineer): Foveated rendering is a game changer but battery life and on-device privacy are critical.
Charlie (Routine): Wants it to "just work" without constant AI permission requests.
Diana (Privacy Advocate): 100% on-device processing required. No cloud telemetry.
Edward (Executive): CDN cost is the biggest variable. 10% egress saving is millions in profit. Ship before Q4.

[SUPPORT TICKETS]
TKT-101: Battery drain on iPhone 15 Pro.
TKT-102: Bitrate switching causes 2s freeze on rural networks.
TKT-105: Privacy policy lacks biometric eye-tracking coverage.
"""

INTERNAL_ROLES = [
    {"name": "Sarah Chen", "role": "CTO", "title": "Chief Technology Officer"},
    {"name": "Mark Lawson", "role": "Lead Engineer", "title": "Principal Architect"},
    {"name": "Elena Rodriguez", "role": "Product Manager", "title": "Director of Product"},
    {"name": "David Varkey", "role": "Security Lead", "title": "CISO"},
    {"name": "Jim Beam", "role": "DevOps", "title": "Infrastructure Lead"}
]

EXTERNAL_SEGMENTS = [
    {"name": "Alex Riviera", "role": "Enterprise CTO", "title": "CTO of StreamingCorp"},
    {"name": "Priya Sharma", "role": "End User (Power)", "title": "Power User / Traveler"},
    {"name": "Sven Lindholm", "role": "Startup Founder", "title": "CEO of MobileFirst"},
    {"name": "Jordan Lee", "role": "Casual Viewer", "title": "Daily Commuter"},
    {"name": "Fiona Wu", "role": "Compliance Officer", "title": "Privacy Consultant"}
]

async def generate_profile(llm: Any, persona: Dict[str, str], is_internal: bool) -> Dict[str, Any]:
    """Generate a high-fidelity psychological profile using an LLM."""
    prompt = f"""
    Generate a highly realistic and synthetic psychological profile for a persona in the TSC (Temporal Streaming Consensus) system.
    This persona will be used for high-fidelity social simulation and adversarial debate.
    
    PERSONA: {persona['name']}
    ROLE: {persona['role']}
    TITLE: {persona['title']}
    TYPE: {"INTERNAL" if is_internal else "EXTERNAL"}
    
    GROUNDING DATA:
    {GROUNDING_DATA}
    
    Return a JSON object exactly matching this structure:
    {{
      "mbti_type": "...",
      "mbti_description": "...",
      "key_traits": ["trait 1", "trait 2", "trait 3"],
      "emotional_triggers": {{
        "excited_by": ["...", "..."],
        "frustrated_by": ["...", "..."],
        "worried_about": ["...", "..."]
      }},
      "communication_style": {{
        "tone": "Direct/Diplomatic/etc",
        "vocabulary_focus": "technical/financial/user-centric",
        "conflict_approach": "..."
      }},
      "decision_pattern": {{
        "primary_driver": "Efficiency/Security/Cost/etc",
        "data_vs_intuition": 0.0 (0=intuition, 1=data),
        "speed_preference": "fast/slow"
      }},
      "full_profile_text": "A 200-word deep dive into their personality, history, and biases grounded in the provided data."
    }}
    """
    
    try:
        response = await llm.analyze(
            system_prompt="You are an expert organizational psychologist creating synthetic personas for enterprise simulation.",
            user_prompt=prompt,
            temperature=0.7
        )
        return response
    except Exception as e:
        logger.error(f"Failed to generate profile for {persona['name']}: {e}")
        return {}

async def main():
    # 1. Initialize DB
    logger.info("Initializing database...")
    db = await init_db(Base)
    repo = PersonaRepository(db)
    
    # 2. Get LLM Client
    llm = create_llm_client(
        provider=LLMProvider.GROQ,
        model="openai/gpt-oss-120b"
    )
    
    # 3. Create/Get Company
    company_id = uuid.uuid4()
    async with db.get_session() as session:
        # Create a default company
        company = Company(
            id=company_id,
            name="TSC Enterprise",
            industry="Cloud/Video Streaming",
            business_context={"vision": "Cost-effective 4K streaming via AI-driven consensus"}
        )
        session.add(company)
        logger.info(f"Created company: TSC Enterprise (ID: {company_id})")
    
    # 4. Generate & Save Internal Personas
    logger.info("Generating internal personas...")
    for role in INTERNAL_ROLES:
        profile_json = await generate_profile(llm, role, is_internal=True)
        if not profile_json: continue
        
        persona_data = {
            "name": role["name"],
            "role": role["role"],
            "title": role["title"],
            "mbti_type": profile_json.get("mbti_type", "INTJ")[:4],
            "psychological_profile": profile_json,
            "confidence_score": 0.9
        }
        await repo.save_internal_persona(company_id, persona_data)
        
    # 5. Generate & Save External Personas
    logger.info("Generating external personas...")
    for seg in EXTERNAL_SEGMENTS:
        profile_json = await generate_profile(llm, seg, is_internal=False)
        if not profile_json: continue
        
        persona_data = {
            "persona_name": seg["name"], # Note: ExternalPersona model uses persona_name for uniqueness in repo
            "name": seg["name"],
            "role": seg["role"],
            "segment_type": seg["role"], # Map role to segment_type for repo
            "title": seg["title"],
            "mbti_type": profile_json.get("mbti_type", "ESFJ")[:4],
            "psychological_profile": profile_json,
            "use_case": profile_json.get("full_profile_text", "")[:200],
            "confidence_score": 0.85
        }
        await repo.save_external_persona(persona_data)
        
    logger.info("Persona population complete!")
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
