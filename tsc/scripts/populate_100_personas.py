import asyncio
import uuid
import logging
import json
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

from tsc.db.models import Base, Company, ExternalPersona
from tsc.db.connection import get_db, init_db
from tsc.repositories.persona_repository import PersonaRepository
from tsc.llm.factory import create_llm_client
from tsc.config import LLMProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Grounding Data
GROUNDING_DATA = {
    "technical": "Bob (Engineer): Gaze tracking, foveated rendering, battery life, on-device privacy (no cloud). AV1 codec missing (Android).",
    "user_experience": "Alice (Power User): Quality drops on commute. Selective background blurring. Charlie (Regular): No constant AI prompts.",
    "privacy": "Diana (Privacy): 100% on-device, no telemetry, no eye-tracking biometrics policy.",
    "business": "Edward (Exec): CDN cost reduction, 10% egress is millions, ship by Q4."
}

# SWITCHED TO 8B FOR SPEED
PRIMARY_MODEL = "llama-3.1-8b-instant"

# 120 UNIQUE NAMES (Diverse)
DIVERSE_NAMES = [
    "Aarav Sharma", "Zainab Al-Fayed", "Sofia Rossi", "Hiroshi Tanaka", "Elena Kovalev", "Chioma Okoro", "Mateo Garcia", "Amara Khan", "Lukas Schneider", "Yasmine Ben-Zvi",
    "Isabella Conti", "Arjun Patel", "Min-su Kim", "Gabriela Mendez", "Kofi Mensah", "Sasha Ivanova", "Rohan Gupta", "Mei Lin", "Diego Torres", "Fatima Zahra",
    "Liam O'Sullivan", "Nadia Petrova", "Chen Wei", "Aisha Abubakar", "Javier Hernandez", "Sora Nakamura", "Maya Singh", "Omar Farooq", "Yuna Park", "Kaleb Bekele",
    "Emma Watson", "Mohamed Salah", "Katarina Jovic", "Rajesh Kumar", "Li Na", "Carlos Silva", "Anika Das", "Ahmed Mansouri", "Hana Tanaka", "Moussa Traore",
    "Alice Thompson", "Benjamin Lee", "Chloe Dupont", "Daniel Miller", "Eva Garcia", "Felix Weber", "Grace Taylor", "Henry Wilson", "Iris Wang", "Jacob Smith",
    "Kaia Jensen", "Leo Martinez", "Mia Anderson", "Noah Clark", "Olivia Brown", "Pavel Novak", "Quinn Roberts", "Rose Williams", "Samuel Jones", "Tessa Davis",
    "Ulysses Grant", "Vera Chen", "Wyatt Lewis", "Ximena Ruiz", "Yara Sayegh", "Zoe White", "Alexander Fischer", "Bianca Costa", "Caleb Wright", "Daphne Thorne",
    "Elliot Vance", "Frederic Berger", "Gianna Leone", "Hassan Bakir", "Ines Morel", "Julian Pierce", "Kira Volkov", "Leandro Gomez", "Miriam Adler", "Niall Quinn",
    "Oscar Wilde", "Paloma Reyes", "Quintin Shaw", "Raina Kato", "Stefan Draganov", "Tariq Aziz", "Ursula Klein", "Victor Hugo", "Wanda Maximoff", "Xavier Woods",
    "Yoshiro Sato", "Zelda Spellman", "Andres Moreno", "Beatriz Lima", "Callum Macleod", "Dalia Cohen", "Elias Holm", "Fayez Al-Otaibi", "Goran Petrovic", "Hilda Varga",
    "Ilya Smirnov", "Johan Lund", "Kyra Knightley", "Lars Ulrich", "Mona Singh", "Nikos Pappas", "Olga Kurylenko", "Paolo Rossini", "Qian Zhang", "Radek Stepanek",
    "Soren Kierkegaard", "Thalia Grace", "Uta Hagen", "Vito Corleone", "Wen-hsuan Wu", "Xander Harris", "Yana Gupta", "Zuri Hall", "Amira El-Sayed", "Bastian Schweinsteiger"
]

TONES = [
    "Terse & Skeptical", "Enthusiastic Early Adopter", "Hardcore Technical Pedant", "Passive/Resigned", "Aggressive Business-Minded", "Anxious Privacy-Centric", "Wry/Sarcastic"
]

sem = asyncio.Semaphore(15)  # Moderate concurrency for stability

async def generate_and_save(llm: Any, repo: PersonaRepository, archetype: Dict[str, str], name: str, tone: str):
    async with sem:
        relevant_grounding = f"{GROUNDING_DATA['user_experience']}"
        if archetype['segment'] == 'enterprise': relevant_grounding += f" {GROUNDING_DATA['business']}"
        if "Technical" in archetype['name'] or archetype['segment'] == 'startup': relevant_grounding += f" {GROUNDING_DATA['technical']}"
        
        prompt = f"""
        Generate a deep synthetic customer profile for: {name}.
        ARCHETYPE: {archetype['name']}
        TONE: {tone}
        GROUNDING: {relevant_grounding}
        
        JSON STRUCTURE:
        {{
          "name": "{name}",
          "role": "{archetype['segment']}",
          "title": "...",
          "mbti_type": "...",
          "demographics": {{"age": 18-65, "location": "...", "device": "..."}},
          "use_case": "...",
          "psychological_profile": {{
            "traits": ["...", "..."],
            "emotional_triggers": {{...}},
            "decision_bias": "...",
            "fact_bundle": ["Fact 1", "Fact 2", "Fact 3"]
          }},
          "full_profile_text": "Detailed 150-word narrative bio."
        }}
        """
        try:
            res = await llm.analyze(system_prompt=f"You are {name}, a real person.", user_prompt=prompt, temperature=0.8)
            if res and "name" in res:
                persona_data = {
                    "name": res["name"],
                    "role": res["role"],
                    "title": res.get("title", "Customer"),
                    "mbti_type": res.get("mbti_type", "INTJ")[:4],
                    "psychological_profile": res["psychological_profile"],
                    "demographics": res["demographics"],
                    "use_case": f"{res['use_case']}\n\n{res.get('full_profile_text','')}",
                    "confidence_score": 0.95
                }
                await repo.save_external_persona(persona_data)
                logger.info(f"Successfully saved persona: {name}")
                return True
        except Exception as e:
            logger.error(f"Failed for {name}: {e}")
            return False

async def main():
    db = await init_db(Base)
    repo = PersonaRepository(db)
    llm = create_llm_client(provider=LLMProvider.GROQ, model=PRIMARY_MODEL)
    
    # Get Archetypes
    logger.info("Discovering archetypes...")
    archetypes_res = await llm.analyze(
        system_prompt="Analyst.", 
        user_prompt=f"Identify 25 customer archetypes from: {GROUNDING_DATA}. Return JSON like {{'archetypes': [{{'name': '...', 'segment': 'enterprise|startup|consumer'}}]}}"
    )
    archetypes = archetypes_res.get("archetypes", [])[:25]
    if not archetypes: archetypes = [{"name": "Standard Consumer", "segment": "consumer"}]
    
    # Ensure every archetype has a segment field
    for arch in archetypes:
        if 'segment' not in arch:
            arch['segment'] = 'consumer'

    logger.info("Generating 100 personas incrementally...")
    persona_names = random.sample(DIVERSE_NAMES, 100)
    tasks = []
    
    for i in range(100):
        arch = archetypes[i % len(archetypes)]
        name = persona_names[i]
        tone = random.choice(TONES)
        tasks.append(generate_and_save(llm, repo, arch, name, tone))
            
    results = await asyncio.gather(*tasks)
    success_count = sum(1 for r in results if r)
    logger.info(f"Done! {success_count}/100 personas populated.")
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
