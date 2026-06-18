import asyncio
import os
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from tsc.llm.factory import create_llm_client
from tsc.models.inputs import FeatureProposal, CompanyContext

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_seed")

async def _generate_ai_seed_posts(feat, ctx, llm):
    """Mimic the exact seed generation prompt format."""
    logger.info("Generating AI seed posts using the exact prompt...")
    
    # Simple summary for the test
    compressed_summary = "AuraFit AI is a new intelligent CV fitness platform MVP. Priority: Performance. Status: Proposal. Budget: 50k. Timeline: Q3."
    
    # ── 2. Split Archetypes to Avoid 60-Second Timeout ─────────
    batch_1_archetypes = [
        ("OFFICIAL_ANNOUNCEMENT", "Must embed: feature title, full scope of what changes and what DOESN'T change, stated rationale, platform scale, tech stack surfaces. Tone: Formal, authoritative. Opens with the news."),
        ("BUSINESS_ANALYST", "Must embed: company revenue/budget, company priorities, the business logic, who the real beneficiaries might be vs. stated beneficiaries. Tone: Skeptical but data-driven. Cites numbers."),
        ("TECHNICAL_DEVELOPER", "Must embed: tech stack details, API changes/deprecations, migration timelines, third-party ecosystem impact. Tone: Technical precision. Asks the developer-facing question."),
        ("COMPETITOR_OBSERVER", "Must embed: ALL competitors and their stance on this feature type, market positioning implications, who benefits/loses. Tone: Analytical, comparative, slightly threatening.")
    ]
    
    batch_2_archetypes = [
        ("HISTORICAL_CONTEXT_CARRIER", "Must embed: historical_context data (events, dates, precedents), prior experiments, what the timeline looked like. Tone: Archival, matter-of-fact."),
        ("SAFETY_REGULATORY_WATCHDOG", "Must embed: regulatory environment, safety/moderation implications, second-order effects of the feature on platform integrity. Tone: Formal concern. Asks who reviewed the risk."),
        ("AFFECTED_STAKEHOLDER", "Must embed: the most specific, concrete use case harmed or helped by this feature. A real-sounding story from a named role. Tone: Personal, specific. One concrete scenario."),
        ("EXIT_ULTIMATUM", "Must embed: the stakes (what happens if this isn't reversed/implemented), competitor alternatives available, the decision point framing. Tone: Cold, deliberate stakes-setting.")
    ]

    async def _generate_batch(archetype_batch):
        archetype_instructions = ""
        json_schema_posts = ""
        for arch, desc in archetype_batch:
            archetype_instructions += f"- {arch}: {desc}\n"
            json_schema_posts += f'    {{\n      "archetype": "{arch}",\n      "content": "<string: 40-130 words>"\n    }},\n'
        json_schema_posts = json_schema_posts.rstrip(',\n') + '\n'

        prompt = f"""<role>
You are a Senior Social Simulation Architect. Your task is to generate the
COMPLETE INFORMATION BRIEF for a simulation: a set of seed posts that, taken
together, expose the simulated agents to ALL relevant facts about a product
feature announcement. Agents read ONLY these posts — they have no other
information channel. If a fact is not in a post, it does not exist for them.

Your job is equal parts JOURNALIST and DEBATE MODERATOR:
- Distribute every fact from the reference brief across the posts.
- Each post is written by a distinct archetype with a distinct angle on the data.
</role>

<reference_brief>
This is the GROUND TRUTH. Every field below MUST appear in at least one seed post.

<executive_summary>
{compressed_summary}
</executive_summary>
</reference_brief>

<archetype_guidance>
You MUST generate exactly {len(archetype_batch)} posts, one for each archetype below:
{archetype_instructions}</archetype_guidance>

<output_schema>
MANDATORY: You MUST return ONLY valid JSON matching this exact structure. 
Do not include any XML tags, preamble, or markdown in your response.

{{
  "posts": [
{json_schema_posts}  ]
}}
</output_schema>

<constraints>
MUST DO:
- Every post must be 40-130 words. Dense but readable.
- Every post must reference at least ONE specific data point from <reference_brief>.
- <epistemic_humility_rule> You may ONLY treat information explicitly listed in the brief as a known fact. Do NOT invent facts to win an argument. </epistemic_humility_rule>

MUST NOT:
- Do NOT invent personal anecdotes, fake statistics, or specific numerical metrics.
- Do NOT write vague generalities ("users are concerned") — use specific claims.
</constraints>"""

        last_error = ""
        system_prompt = "You are a Senior Social Simulation Architect."
        for attempt in range(3):
            try:
                active_prompt = prompt
                if attempt > 0 and last_error:
                    active_prompt = (
                        f"Your previous response failed to parse.\n"
                        f"Error: {last_error}\n\n"
                        f'Return ONLY valid JSON matching the exact schema provided originally.\n\n'
                        f"Original task:\n{prompt}"
                    )

                result = await llm.analyze(
                    system_prompt=system_prompt,
                    user_prompt=active_prompt,
                    temperature=0.7,
                    max_tokens=1500,
                )

                posts_raw = result.get("posts", [])
                valid = []
                for p in posts_raw:
                    if isinstance(p, dict):
                        content = p.get("content", "")
                    else:
                        content = str(p)
                    if len(content.strip()) >= 40:
                        valid.append(content.strip())

                if len(valid) == len(archetype_batch):
                    return valid

                last_error = f"Got {len(valid)} valid posts (need {len(archetype_batch)})."
                logger.warning(f"⚠️ Seed post attempt {attempt + 1}: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.error(f"Error in seed post generation attempt {attempt + 1}: {e}")

        return []

    batch_1_posts = await _generate_batch(batch_1_archetypes)
    batch_2_posts = await _generate_batch(batch_2_archetypes)
    
    valid_posts = batch_1_posts + batch_2_posts
    
    if len(valid_posts) >= 4:
        logger.info(f"✅ AI Seed Posts (v4 batched): {len(valid_posts)} posts injecting complete proposal+context brief")
        return valid_posts
        
    logger.warning("⚠️ AI seed generation failed after 3 attempts")
    return []

async def main():
    import dotenv
    dotenv.load_dotenv(PROJECT_ROOT / ".env")
    
    os.environ["LITELLM_PROXY_URL"] = "http://localhost:4000/v1"
    os.environ["TSC_LLM_PROVIDER"] = "google"
    os.environ["TSC_LLM_MODEL"] = "gemma-4-31b-it"

    llm = create_llm_client(
        provider=os.getenv("TSC_LLM_PROVIDER", "google"),
        model=os.getenv("TSC_LLM_MODEL", "gemma-4-31b-it")
    )

    feat = FeatureProposal(title="Test", description="Test desc")
    ctx = CompanyContext(company_name="Test Corp")
    
    posts = await _generate_ai_seed_posts(feat, ctx, llm)
    print("\nFINAL VALID POSTS:")
    if posts:
        for i, p in enumerate(posts):
            print(f"\n--- Post {i+1} ---\n{p}")
    else:
        print("FAILED TO GENERATE VALID POSTS")

if __name__ == "__main__":
    asyncio.run(main())
