"""
V29: HindsightBoardroom — Persistent Agent Memory for the AG2 Debate Engine.

Architecture: HINDSIGHT-FIRST
    When Hindsight is connected, it is the SOLE source of truth for memory.
    Hindsight's internal NLU extractors handle entity extraction, commitment
    detection, belief tracking, and relationship mapping — far superior to
    any regex-based approach.

    The EMBEDDED fallback (regex + structured dicts) exists ONLY for
    local development without a Hindsight server. It is explicitly
    marked as degraded mode.

Memory Networks (per agent, managed by Hindsight):
    World       — Feature brief, industry facts, competitor data
    Experience  — This agent's actions and statements during debate
    Opinion     — Evolved beliefs with confidence scores
    Observation — Synthesized cross-debate entity summaries

Two modes:
  1. HINDSIGHT (requires Hindsight server):
     retain() → Hindsight NLU extracts entities, beliefs, commitments
     recall(budget="low") → 50-500ms structured context for live turns
     reflect(budget="high") → deep post-debate evolved summaries
  2. EMBEDDED (default, NO external dependencies):
     Lightweight regex fallback. DEGRADED quality. For dev only.
"""

import os
import re
import json
import time
import logging
import threading
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

# Patch the event loop EARLY (module-level) to allow nested async calls.
# Required because the Hindsight SDK uses asyncio internally, and AG2's
# framework already has an event loop running.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ─── Disposition Map: Persona → Hindsight Traits ─────────────────────────
# skepticism: how much the agent doubts claims (0-100)
# literalism: how strictly the agent interprets data (0-100)
# empathy:    how much the agent weighs human/user impact (0-100)
DISPOSITION_MAP = {
    "CISO":   {"skepticism": 90, "literalism": 80, "empathy": 30},
    "CFO":    {"skepticism": 70, "literalism": 90, "empathy": 20},
    "CEO":    {"skepticism": 30, "literalism": 40, "empathy": 60},
    "CTO":    {"skepticism": 50, "literalism": 70, "empathy": 40},
    "Legal":  {"skepticism": 85, "literalism": 95, "empathy": 25},
    "CPO":    {"skepticism": 40, "literalism": 30, "empathy": 85},
    "CMO":    {"skepticism": 60, "literalism": 60, "empathy": 75},
    "CDO":    {"skepticism": 55, "literalism": 80, "empathy": 35},
    "Sales":  {"skepticism": 35, "literalism": 45, "empathy": 70},
    "HR":     {"skepticism": 45, "literalism": 50, "empathy": 90},
}

# ─── EMBEDDED-ONLY Fallback Extractors ───────────────────────────────────
# These regex patterns are a DEGRADED FALLBACK used ONLY when no Hindsight
# server is connected. They are brittle and miss natural language variations
# like "I'm going to have to go ahead and agree to the budget."
#
# When Hindsight is connected, these are NOT used. Hindsight's internal NLU
# handles all extraction via retain().
#
# UI-SIGNAL ONLY: Even in EMBEDDED mode, these exist primarily for live
# dashboard signals (e.g. "CISO vetoed"), not for building the agent's
# actual memory — which should come from Hindsight.

_EMBEDDED_VETO_PATTERN = re.compile(
    r'(?:is_high_risk["\s:]*true|I\s+(?:formally\s+)?veto)', re.IGNORECASE
)
_EMBEDDED_CONCESSION_PATTERN = re.compile(
    r'(?:I concede|I withdraw (?:my )?veto|withdraw my (?:veto|objection))', re.IGNORECASE
)


@dataclass
class LiveAgentMemory:
    """Per-agent memory state.

    In HINDSIGHT mode: This is a thin metadata wrapper. The real memory
    lives in the Hindsight Memory Bank (Opinion/Experience/World/Observation
    networks). We only track turn_count, has_vetoed, and all_messages locally
    for UI/logging purposes.

    In EMBEDDED mode: This stores the degraded regex-extracted state as a
    development fallback.
    """

    agent_name: str
    role: str
    role_short: str
    feature_title: str

    # Metadata (always tracked, both modes)
    all_messages: List[str] = field(default_factory=list)
    turn_count: int = 0
    has_vetoed: bool = False
    veto_resolved: bool = False

    # Hindsight bank reference
    hindsight_bank_id: Optional[str] = None

    # EMBEDDED-ONLY fields (unused when Hindsight is connected)
    # These exist as a degraded fallback for development without a server.
    _embedded_commitments: List[str] = field(default_factory=list)
    _embedded_concessions: List[str] = field(default_factory=list)
    _embedded_proposals: List[str] = field(default_factory=list)
    _embedded_concerns: List[str] = field(default_factory=list)
    _embedded_relationships: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for DB persistence."""
        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "role_short": self.role_short,
            "feature_title": self.feature_title,
            "all_messages": self.all_messages,
            "turn_count": self.turn_count,
            "has_vetoed": self.has_vetoed,
            "veto_resolved": self.veto_resolved,
            # EMBEDDED fields (for backward compat)
            "_embedded_commitments": self._embedded_commitments,
            "_embedded_concessions": self._embedded_concessions,
            "_embedded_proposals": self._embedded_proposals,
            "_embedded_concerns": self._embedded_concerns,
            "_embedded_relationships": self._embedded_relationships,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LiveAgentMemory':
        """Deserialize from DB."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


class HindsightBoardroom:
    """Manages per-agent persistent memory for the boardroom debate.

    ARCHITECTURE: HINDSIGHT-FIRST
    ─────────────────────────────
    When Hindsight is connected (HINDSIGHT mode):
      • retain()  = THE extraction mechanism. Hindsight's NLU parses
        commitments, beliefs, entities, and relationships automatically.
      • recall()  = THE injection mechanism. Returns semantically relevant
        context for each agent's next turn.
      • reflect() = THE post-debate summary. Generates evolved position
        from the Opinion Network.

    When Hindsight is NOT connected (EMBEDDED mode):
      • Degraded regex fallback for local development only.
      • Explicitly logged as "DEGRADED MODE".
    """

    def __init__(self, hindsight_url: Optional[str] = None, api_key: Optional[str] = None):
        self.memories: Dict[str, LiveAgentMemory] = {}
        self._lock = threading.RLock()
        self._hindsight = None
        self._mode = "EMBEDDED"

        # Try to connect to Hindsight server
        url = hindsight_url or os.getenv("HINDSIGHT_URL", "")
        key = api_key or os.getenv("HINDSIGHT_API_KEY", "")
        if url:
            try:
                from hindsight_client import Hindsight
                self._hindsight = Hindsight(base_url=url, api_key=key or None)
                self._mode = "HINDSIGHT"
                logger.info(f"V29: Hindsight CONNECTED at {url}")
            except Exception as e:
                logger.warning(f"V29: Hindsight connection failed ({e}). Falling back to EMBEDDED (degraded).")
        else:
            logger.info("V29: No HINDSIGHT_URL set. Using EMBEDDED memory mode (degraded — no NLU extraction).")

    def _run_sync(self, coro_or_val):
        """Safely run a coroutine or normal value in a sync context."""
        if not asyncio.iscoroutine(coro_or_val):
            return coro_or_val
            
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we are likely in a different thread or 
                # nested. Use run_coroutine_threadsafe if in a different thread,
                # or a new loop if possible. 
                # For this environment, we'll try the simplest robust approach:
                # using a background thread loop for Hindsight calls if needed.
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(asyncio.run, coro_or_val).result()
            return loop.run_until_complete(coro_or_val)
        except (RuntimeError, ValueError):
            return asyncio.run(coro_or_val)
            

    def initialize_agents(self, personas: list, feature_title: str, feature_description: str) -> None:
        """Create a LiveAgentMemory for each persona.

        In HINDSIGHT mode: Also creates Memory Banks with Disposition Traits
        and retains the feature brief as World knowledge.
        """
        for persona in personas:
            agent_name = persona.name.replace(" ", "_").replace(".", "")
            role_short = getattr(persona, 'role_short', '') or self._infer_role_short(persona.role)

            memory = LiveAgentMemory(
                agent_name=agent_name,
                role=persona.role,
                role_short=role_short,
                feature_title=feature_title,
            )
            self.memories[agent_name] = memory

            # Create Hindsight Memory Bank if connected
            if self._hindsight:
                try:
                    disposition = DISPOSITION_MAP.get(role_short, {})
                    bank_id = f"boardroom-{agent_name}"

                    # Delete any pre-existing bank from a previous simulation run
                    try:
                        self._run_sync(self._hindsight.delete_bank(bank_id=bank_id))
                        logger.debug(f"V29: Deleted pre-existing bank '{bank_id}'")
                    except Exception:
                        pass  # Bank didn't exist — that's fine

                    self._run_sync(self._hindsight.create_bank(
                        bank_id=bank_id,
                        name=f"{agent_name} ({persona.role})",
                        background=(
                            f"{agent_name}, {persona.role}. "
                            f"Domain expertise: {', '.join(getattr(persona, 'domain_expertise', []) or [])}. "
                            f"Currently debating: {feature_title}"
                        ),
                        disposition_skepticism=disposition.get("skepticism", 50),
                        disposition_literalism=disposition.get("literalism", 50),
                        disposition_empathy=disposition.get("empathy", 50),
                        retain_mission=(
                            "Extract commitments, concessions, proposals, concerns, "
                            "vetoes, challenges, and relationship signals from boardroom "
                            "debate messages. Track evolving beliefs about the feature "
                            "being debated."
                        ),
                        enable_observations=True,
                        observations_mission=(
                            "Synthesize observations about cross-agent dynamics: "
                            "alliances, tensions, consensus formation, and blocking issues."
                        ),
                    ))
                    memory.hindsight_bank_id = bank_id

                    # Retain the feature brief as World knowledge
                    self._run_sync(self._hindsight.retain(
                        bank_id=bank_id,
                        content=f"[FEATURE BRIEF] {feature_title}: {feature_description[:2000]}",
                        tags=["world", "feature_brief"],
                    ))

                    # Set a reflect mission tailored to boardroom debate
                    self._run_sync(self._hindsight.set_reflect_mission(
                        bank_id=bank_id,
                        reflect_mission=(
                            f"You are {agent_name}, {persona.role}. When reflecting, "
                            f"synthesize your evolved position on the feature being debated. "
                            f"Reference specific commitments you made, concessions you accepted, "
                            f"proposals you championed, and concerns that remain unresolved. "
                            f"Speak as the executive, not as an AI."
                        ),
                    ))
                    logger.info(f"V29: Hindsight bank '{bank_id}' created for {agent_name}")
                except Exception as e:
                    # If create still fails, try to use the existing bank anyway
                    bank_id = f"boardroom-{agent_name}"
                    memory.hindsight_bank_id = bank_id
                    logger.warning(f"V29: Hindsight bank creation failed for {agent_name}: {e} — using existing bank")

        logger.info(f"V29: Initialized {len(self.memories)} agent memories (mode={self._mode})")

    # ═══════════════════════════════════════════════════════════════════════
    # DURING DEBATE: retain + recall
    # ═══════════════════════════════════════════════════════════════════════

    def extract_and_retain(self, sender_name: str, content: str, all_agent_names: List[str]) -> None:
        """Store a message in the agent's memory.

        HINDSIGHT mode: Calls retain() and lets Hindsight's NLU do ALL
        extraction — commitments, beliefs, entities, relationships.
        No regex. No heuristics. Hindsight understands context.

        EMBEDDED mode: Degraded regex fallback for local development only.
        """
        memory = self.memories.get(sender_name)
        if not memory or not content:
            return

        with self._lock:
            memory.all_messages.append(content[:3000])
            memory.turn_count += 1

            # UI signals: veto detection (lightweight, both modes)
            if _EMBEDDED_VETO_PATTERN.search(content):
                memory.has_vetoed = True
            if _EMBEDDED_CONCESSION_PATTERN.search(content):
                memory.veto_resolved = True

        # ── HINDSIGHT MODE: Let Hindsight do the heavy lifting ──────────
        if self._hindsight and memory.hindsight_bank_id:
            try:
                # Primary retain: the agent's own message
                # Hindsight NLU will automatically:
                #   - Extract entities (people, proposals, metrics)
                #   - Identify commitments, concessions, beliefs
                #   - Update the Opinion Network with confidence scores
                #   - Build the entity graph for relationship tracking
                self._run_sync(self._hindsight.retain(
                    bank_id=memory.hindsight_bank_id,
                    content=f"[Turn {memory.turn_count}] I said: {content[:3000]}",
                    context=f"Boardroom debate, turn {memory.turn_count}. "
                            f"Speaking as {memory.agent_name}, {memory.role}.",
                    tags=["experience", f"turn_{memory.turn_count}"],
                    timestamp=datetime.now(),
                ))

                # Cross-agent awareness: what other agents heard
                for other_name, other_mem in self.memories.items():
                    if other_name != sender_name and other_mem.hindsight_bank_id:
                        self._run_sync(self._hindsight.retain(
                            bank_id=other_mem.hindsight_bank_id,
                            content=f"[Turn {memory.turn_count}] {sender_name} said: {content[:2000]}",
                            context=f"Boardroom debate, turn {memory.turn_count}. "
                                    f"{sender_name} is speaking. I am listening.",
                            tags=["world", "other_agent", f"turn_{memory.turn_count}"],
                            timestamp=datetime.now(),
                        ))
            except Exception as e:
                logger.info(f"V29: Hindsight retain failed for {sender_name}: {e}")
            return  # Hindsight handles everything — no regex needed

        # ── EMBEDDED MODE: Degraded regex fallback ──────────────────────
        # WARNING: This misses natural language variations.
        # Example: "I'm going to have to agree to that budget" → MISSED
        # This exists ONLY for development without a Hindsight server.
        self._embedded_regex_extract(memory, content)

    def recall_for_turn(self, agent_name: str) -> str:
        """Get memory context for injection into the agent's system prompt.

        HINDSIGHT mode: recall(budget='low') is THE source of truth.
        Returns Hindsight's semantically relevant context in 50-500ms.

        EMBEDDED mode: Returns a degraded structured summary from regex fields.
        """
        memory = self.memories.get(agent_name)
        if not memory or memory.turn_count == 0:
            return ""

        # ── HINDSIGHT MODE: recall() is the primary mechanism ───────────
        if self._hindsight and memory.hindsight_bank_id:
            try:
                recall_result = self._run_sync(self._hindsight.recall(
                    bank_id=memory.hindsight_bank_id,
                    query=(
                        f"What are my current positions, commitments, and unresolved "
                        f"concerns about {memory.feature_title}? What have other "
                        f"agents challenged me on? What alliances or tensions exist?"
                    ),
                    budget="low",  # Fast: 50-500ms for live debate
                    max_tokens=600,
                    include_entities=True,
                    max_entity_tokens=200,
                ))
                if recall_result:
                    recall_text = str(recall_result)[:800]
                    veto_warning = ""
                    if memory.has_vetoed and not memory.veto_resolved:
                        veto_warning = "\n⚠️ YOU HAVE AN ACTIVE VETO. You MUST resolve or sustain it."
                    return (
                        f"\n[YOUR EVOLVING MEMORY — from Hindsight]\n"
                        f"{recall_text}"
                        f"{veto_warning}\n"
                        f"[BUILD upon your prior positions. Do NOT contradict commitments above.]"
                    )
            except Exception as e:
                logger.debug(f"V29: Hindsight recall failed for {agent_name}: {e}")

        # ── EMBEDDED FALLBACK ───────────────────────────────────────────
        return self._embedded_injection_context(memory)

    # ═══════════════════════════════════════════════════════════════════════
    # POST-DEBATE: reflect + query
    # ═══════════════════════════════════════════════════════════════════════

    def reflect_post_debate(self, agent_name: Optional[str] = None) -> Union[str, Dict[str, str]]:
        """Generate a deep evolved summary after the debate ends.
        
        If agent_name is provided, returns String for that agent.
        If agent_name is None, returns Dict[name, summary] for all agents.

        HINDSIGHT mode: reflect(budget='high') synthesizes from the
        Opinion Network — the agent's evolved beliefs, not raw messages.

        EMBEDDED mode: Degraded summary from regex-extracted fields.
        """
        if agent_name is None:
            return {name: self.reflect_post_debate(name) for name in self.memories}
            
        memory = self.memories.get(agent_name)
        if not memory:
            return ""

        # ── HINDSIGHT MODE: reflect() is the primary mechanism ──────────
        if self._hindsight and memory.hindsight_bank_id:
            try:
                result = self._run_sync(self._hindsight.reflect(
                    bank_id=memory.hindsight_bank_id,
                    query=(
                        f"Reflect on the complete boardroom debate about '{memory.feature_title}'. "
                        f"What is my final evolved position? What commitments did I make? "
                        f"What concessions did I accept? What proposals did I champion? "
                        f"What concerns remain unresolved? How did my relationships with "
                        f"other board members evolve during the debate?"
                    ),
                    budget="high",
                    include_facts=True,
                ))
                # Extract the clean text answer from the Pydantic object if it exists
                if result:
                    return getattr(result, 'answer', getattr(result, 'text', str(result)))
                return ""
            except Exception as e:
                logger.warning(f"V29: Hindsight reflect failed for {agent_name}: {e}")

        # ── EMBEDDED FALLBACK ───────────────────────────────────────────
        return self._embedded_reflect(memory)

    def query_agent(self, agent_name: str, question: str, llm_config: Optional[dict] = None) -> str:
        """Post-debate: Answer from the agent's evolved perspective.

        HINDSIGHT mode: reflect() with the question as context.
        The agent answers from its Opinion Network — its evolved beliefs,
        not a generic LLM response.

        EMBEDDED mode: System prompt with regex-extracted memory + LLM call.
        """
        memory = self.memories.get(agent_name)
        if not memory:
            return f"Agent '{agent_name}' not found. Available: {list(self.memories.keys())}"

        # ── HINDSIGHT MODE: reflect() with question context ─────────────
        if self._hindsight and memory.hindsight_bank_id:
            try:
                result = self._run_sync(self._hindsight.reflect(
                    bank_id=memory.hindsight_bank_id,
                    query=f"Based on everything I experienced in the boardroom: {question}",
                    budget="mid",
                    context=(
                        f"I am {memory.agent_name}, {memory.role}. "
                        f"A human user is asking me a follow-up question after the debate "
                        f"on '{memory.feature_title}'. I should answer from my evolved "
                        f"perspective, referencing specific commitments and positions I took."
                    ),
                    include_facts=True,
                ))
                # Extract the clean text answer from the Pydantic object
                if result:
                    return getattr(result, 'answer', getattr(result, 'text', str(result)))
                return "No reflection available."
            except Exception as e:
                logger.warning(f"V29: Hindsight query failed: {e}")

        # ── EMBEDDED FALLBACK: LLM with regex-extracted context ─────────
        system_prompt = self._build_query_prompt(memory)

        if llm_config:
            try:
                import openai
                config_list = llm_config.get('config_list', [{}])
                client = openai.OpenAI(
                    api_key=config_list[0].get('api_key', ''),
                    base_url=config_list[0].get('base_url', ''),
                )
                model = config_list[0].get('model', 'gemma-4-31b-it')
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    max_tokens=1000,
                    temperature=0.4,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"V29: LLM query failed: {e}")
                return f"[LLM unavailable] Memory summary:\n{self.reflect_post_debate(agent_name)}"

        return f"[No LLM config] Memory summary:\n{self.reflect_post_debate(agent_name)}"

    # ═══════════════════════════════════════════════════════════════════════
    # EMBEDDED-ONLY: Degraded regex fallback (for dev without Hindsight)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _embedded_regex_extract(memory: 'LiveAgentMemory', content: str) -> None:
        """DEGRADED FALLBACK: Regex-based extraction for EMBEDDED mode only.

        ⚠️ WARNING: This is brittle and misses natural language variations.
        When Hindsight is connected, this method is NEVER called.
        """
        content_lower = content.lower()

        # Commitments (brittle — misses "I'll go ahead and agree to...")
        for pattern in [
            r'(?:I will|I commit to|Deliverable:)\s*(.{20,200})',
            r'(?:Success (?:Metric|metric)|Failure (?:Threshold|threshold)):?\s*(.{15,200})',
        ]:
            for m in re.finditer(pattern, content, re.IGNORECASE):
                val = m.group(1).strip()[:200]
                if val and val not in memory._embedded_commitments:
                    memory._embedded_commitments.append(val)

        # Concessions
        for m in re.finditer(r'(?:I concede|I withdraw)\s*(.{5,150})', content, re.IGNORECASE):
            val = (m.group(1) or "").strip()[:150]
            if val and val not in memory._embedded_concessions:
                memory._embedded_concessions.append(val)

        # Proposals
        for m in re.finditer(r'(?:I propose|I am proposing)\s+(.{15,250})', content, re.IGNORECASE):
            val = m.group(1).strip()[:250]
            if val and val not in memory._embedded_proposals:
                memory._embedded_proposals.append(val)

        # Concerns
        for m in re.finditer(r'(?:Fatal Scenario|my (?:primary )?concern)\s*(.{15,300})', content, re.IGNORECASE):
            val = m.group(1).strip()[:300]
            if val and val not in memory._embedded_concerns:
                memory._embedded_concerns.append(val)

    def _embedded_injection_context(self, memory: 'LiveAgentMemory') -> str:
        """DEGRADED FALLBACK: Build injection context from regex-extracted fields."""
        if memory.turn_count == 0:
            return ""

        lines = ["\n[YOUR EVOLVING MEMORY — EMBEDDED MODE (degraded, no Hindsight)]"]

        if memory._embedded_commitments:
            lines.append(f"YOUR COMMITMENTS: {'; '.join(memory._embedded_commitments[-4:])}")
        if memory._embedded_concessions:
            lines.append(f"CONCESSIONS YOU MADE: {'; '.join(memory._embedded_concessions[-2:])}")
        if memory._embedded_concerns:
            lines.append(f"YOUR OUTSTANDING CONCERNS: {'; '.join(memory._embedded_concerns[-3:])}")
        if memory.has_vetoed and not memory.veto_resolved:
            lines.append("⚠️ YOU HAVE AN ACTIVE VETO. You MUST resolve or sustain it.")

        lines.append("[BUILD upon your prior positions. Do NOT contradict commitments above.]")

        result = "\n".join(lines)
        return result[:2000]

    def _embedded_reflect(self, memory: 'LiveAgentMemory') -> str:
        """DEGRADED FALLBACK: Synthesize summary from regex-extracted fields."""
        lines = [f"=== EVOLVED SUMMARY (EMBEDDED — degraded): {memory.agent_name} ({memory.role}) ==="]
        lines.append(f"Feature: {memory.feature_title}")
        lines.append(f"Turns spoken: {memory.turn_count}")

        if memory._embedded_commitments:
            lines.append("\nCOMMITMENTS:")
            for c in memory._embedded_commitments:
                lines.append(f"  • {c}")
        if memory._embedded_concessions:
            lines.append("\nCONCESSIONS:")
            for c in memory._embedded_concessions:
                lines.append(f"  • {c}")
        if memory._embedded_proposals:
            lines.append("\nPROPOSALS:")
            for p in memory._embedded_proposals:
                lines.append(f"  • {p}")
        if memory._embedded_concerns:
            lines.append("\nCONCERNS:")
            for c in memory._embedded_concerns:
                lines.append(f"  • {c}")

        lines.append(f"\nVETO: {'RAISED' if memory.has_vetoed else 'NONE'}")
        return "\n".join(lines)

    def _build_query_prompt(self, memory: 'LiveAgentMemory') -> str:
        """Build an EMBEDDED-mode system prompt for post-debate querying."""
        lines = [
            f"You are {memory.agent_name}, {memory.role}.",
            f"You just completed a boardroom debate on '{memory.feature_title}'.",
            "", "YOUR EVOLVED POSITION (after the debate):",
        ]

        if memory._embedded_commitments:
            lines.append("\nCOMMITMENTS YOU MADE:")
            for c in memory._embedded_commitments[-6:]:
                lines.append(f"  - {c}")
        if memory._embedded_concessions:
            lines.append("\nCONCESSIONS YOU ACCEPTED:")
            for c in memory._embedded_concessions[-4:]:
                lines.append(f"  - {c}")
        if memory._embedded_proposals:
            lines.append("\nPROPOSALS YOU CHAMPIONED:")
            for p in memory._embedded_proposals[-5:]:
                lines.append(f"  - {p}")
        if memory._embedded_concerns:
            lines.append("\nYOUR REMAINING CONCERNS:")
            for c in memory._embedded_concerns[-4:]:
                lines.append(f"  - {c}")

        # Include last 5 messages as direct context
        if memory.all_messages:
            lines.append("\nYOUR RECENT DEBATE STATEMENTS:")
            for msg in memory.all_messages[-5:]:
                lines.append(f'  "{msg[:500]}"')

        lines.extend([
            "", "RULES:",
            "- Answer from your EVOLVED perspective after the debate.",
            "- Do NOT contradict positions or commitments listed above.",
            "- Reference specific proposals, metrics, and thresholds you committed to.",
            "- If asked about something you didn't discuss, say so honestly.",
            "- Speak as the executive, not as an AI.",
        ])
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════════════

    def get_all_memories(self) -> Dict[str, dict]:
        """Serialize all agent memories for DB persistence."""
        return {name: mem.to_dict() for name, mem in self.memories.items()}

    def get_agent_names(self) -> List[str]:
        """Return all tracked agent names."""
        return list(self.memories.keys())

    @staticmethod
    def _infer_role_short(role: str) -> str:
        """Infer a short role key from a full role title."""
        rl = role.lower()
        if 'cto' in rl or 'technology' in rl: return 'CTO'
        if 'cfo' in rl or 'financial' in rl or 'finance' in rl: return 'CFO'
        if 'ciso' in rl or 'security' in rl: return 'CISO'
        if 'cpo' in rl or 'product' in rl: return 'CPO'
        if 'ceo' in rl or 'executive' in rl or 'chief exec' in rl: return 'CEO'
        if 'legal' in rl or 'counsel' in rl: return 'Legal'
        if 'cmo' in rl or 'medical' in rl or 'clinical' in rl: return 'CMO'
        if 'data' in rl or 'cdo' in rl: return 'CDO'
        if 'sales' in rl: return 'Sales'
        if 'hr' in rl or 'people' in rl: return 'HR'
        if 'customer' in rl or 'success' in rl or 'implementation' in rl: return 'CPO'
        return 'Other'

    def close(self):
        """Cleanup Hindsight connection."""
        if self._hindsight:
            try:
                self._hindsight.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════
# OASIS MARKET FIT INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

class HindsightOASISManager:
    """Manages persistent Hindsight memory banks for OASIS social agents.
    
    Implements the optimal 3-layer biomimetic memory loop:
    
    1. RETAIN (Structured Storage)
       - Conversation arrays with timestamps, entities, and participant prefixes
       - Hindsight extracts discrete facts, temporal data, and entity relationships
    
    2. RECALL (Hybrid Retrieval) — used for per-turn context injection
       - 4 parallel strategies: Semantic (vector), Keyword (BM25), Graph (entity), Temporal
       - Fast and targeted — replaces expensive reflect() for per-turn use
    
    3. REFLECT (Higher-Order Learning) — reserved for post-timestep synthesis
       - Synthesizes raw experiences into Mental Models / observations
       - Used for behavior change across timesteps
    
    All methods are ASYNC to avoid event loop conflicts with the Hindsight SDK.
    """
    def __init__(self, hindsight_url: Optional[str] = None, api_key: Optional[str] = None):
        self._hindsight = None
        self._mode = "EMBEDDED"
        
        url = hindsight_url or os.getenv("HINDSIGHT_URL", "")
        key = api_key or os.getenv("HINDSIGHT_API_KEY", "")
        if url:
            try:
                from hindsight_client import Hindsight
                self._hindsight = Hindsight(base_url=url, api_key=key or None)
                self._mode = "HINDSIGHT"
                logger.info(f"OASIS: Hindsight client initialized → {url}")
            except Exception as e:
                # We log the error but don't set hindsight to None yet, 
                # we will try to connect during initialize_agents.
                logger.error(f"OASIS: Hindsight client initialization failed: {e}")
                
        self.simulation_id = ""
        self.feature_title = ""
        self._provisioned_banks: list[str] = []

    async def check_connection(self) -> bool:
        """Verify the Hindsight server is responsive."""
        if not self._hindsight:
            return False
        try:
            # list_banks is a cheap way to check auth and connectivity
            # Correct path is .banks.list_banks()
            await self._hindsight.banks.list_banks()
            return True
        except Exception as e:
            logger.warning(f"OASIS: Hindsight connection check failed: {e}")
            return False


    async def initialize_agents(self, agent_profiles: list, feature_title: str, feature_description: str, simulation_id: str = "default") -> None:
        """Create Hindsight Memory Banks + Mental Models for all Oasis agents.
        
        Sets up the 4-network architecture per agent:
        - World Network: seeded with feature context
        - Bank Network: empty, populated during simulation via structured retain
        - Opinion Network: built via mental model auto-refresh
        - Observation Network: populated by Hindsight's observation consolidation
        """
        self.feature_title = feature_title
        self.simulation_id = simulation_id
        logger.info(f"🏦 OASIS: Provisioning Hindsight Memory Banks for simulation '{simulation_id}' ({len(agent_profiles)} agents)...")
        
        # ── CONNECTION PREFLIGHT ──
        # Ensure we can actually talk to the server before looping through agents
        connected = False
        for i in range(5):
            if await self.check_connection():
                connected = True
                break
            logger.warning(f"  ⏳ Hindsight connection pending... retry {i+1}/5")
            await asyncio.sleep(2.0 * (i + 1))
            
        if not connected:
            raise ConnectionError(f"OASIS: Could not establish stable connection to Hindsight at {os.getenv('HINDSIGHT_URL')}")

        created_count = 0
        for profile in agent_profiles:
            bank_id = f"oasis-{self.simulation_id}-persona-{profile.agent_id}"
            user_info = getattr(profile, 'user_info_dict', {})
            agent_name = user_info.get('name', f"Agent_{profile.agent_id}")
            
            # Rate-limiting / Jitter to prevent overloading the gRPC pool or server
            await asyncio.sleep(0.5)

            for attempt in range(3):
                try:
                    # Clean slate: delete any pre-existing bank
                    try:
                        await self._hindsight.adelete_bank(bank_id=bank_id)
                        logger.info(f"  🗑️  Bank deleted (clean slate): {bank_id}")
                    except Exception:
                        pass

                    # Create the persona memory bank with retain mission
                    result = await self._hindsight.acreate_bank(
                        bank_id=bank_id,
                        name=f"OASIS-{agent_name}",
                        background=(
                            f"OASIS Market Simulation Agent. Name: {agent_name}. "
                            f"You represent a simulated persona evaluating: '{feature_title}'."
                        ),
                        retain_mission=(
                            "Extract evolving beliefs, sentiments, objections, needs, and reactions "
                            "to the proposed feature. Track stance changes over time."
                        ),
                        enable_observations=True,
                        observations_mission=(
                            "Synthesize the agent's evolving stance into a coherent narrative. "
                            "Track sentiment trajectory and identify inflection points."
                        ),
                    )
                    logger.info(f"  🏗️  Bank created for persona: {agent_name} ({bank_id})")
                    
                    # ── STRUCTURED RETAIN: Seed with feature context (World Network) ──
                    await self._hindsight.aretain_batch(
                        bank_id=bank_id,
                        items=[{
                            "content": (
                                f"[SYSTEM]: A new feature has been proposed for evaluation: "
                                f"'{feature_title}'. Description: {feature_description[:1000]}"
                            ),
                            "context": "Initial feature briefing for market simulation",
                            "entities": [
                                {"text": feature_title, "type": "FEATURE"},
                                {"text": agent_name, "type": "AGENT"},
                            ],
                            "tags": ["world", "feature_introduction"],
                        }],
                    )
                    
                    # ── MENTAL MODEL: Auto-refreshing sentiment tracker ──
                    try:
                        self._hindsight.create_mental_model(
                            bank_id=bank_id,
                            name="feature_sentiment",
                            source_query=(
                                f"What is my overall assessment and current stance on "
                                f"'{feature_title}'? Am I bullish, bearish, or neutral? Why?"
                            ),
                            trigger={"refresh_after_consolidation": True},
                            max_tokens=300,
                        )
                    except Exception as e:
                        logger.warning(f"  ⚠️  Mental model creation skipped for {bank_id}: {e}")
                    
                    self._provisioned_banks.append(bank_id)
                    created_count += 1
                    logger.info(f"  ✅ Bank provisioned: {bank_id} ({created_count}/{len(agent_profiles)})")
                    break  # Success
                    
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"  ⏳ Retry {attempt+1} for {bank_id} due to: {e}")
                        await asyncio.sleep(2.0)
                    else:
                        logger.error(f"  ❌ OASIS: Bank creation FAILED for {agent_name} ({bank_id}) after 3 attempts: {e}")
        
        logger.info(f"🏦 OASIS: Finished provisioning. Total: {created_count}/{len(agent_profiles)} memory banks.")

    async def structured_retain(
        self, agent_id: str, agent_name: str, action_type: str,
        content: str, timestep: int
    ) -> None:
        """LAYER 1 — Structured Retain: Store agent actions with entities and timestamps.
        
        Uses aretain_batch() with structured items instead of raw text blobs.
        Hindsight extracts discrete facts, temporal data, and entity relationships.
        """
        if not self._hindsight:
            return
            
        bank_id = f"oasis-{self.simulation_id}-persona-{agent_id}"
        try:
            await self._hindsight.aretain_batch(
                bank_id=bank_id,
                items=[{
                    "content": f"[{agent_name}]: {content[:1500]}",
                    "timestamp": datetime.now().isoformat(),
                    "context": (
                        f"OASIS simulation timestep {timestep}. "
                        f"Agent performed action: {action_type}."
                    ),
                    "entities": [
                        {"text": self.feature_title, "type": "FEATURE"},
                        {"text": agent_name, "type": "AGENT"},
                    ],
                    "tags": ["experience", action_type.lower(), f"timestep_{timestep}"],
                }],
            )
            logger.info(f"  🧠 Memory inclusion (Timestep {timestep}): Stored '{action_type}' for {agent_name} into {bank_id}")
        except Exception as e:
            logger.warning(f"OASIS: Structured retain failed for {agent_name} ({bank_id}): {e}")

    # Keep backward-compatible alias
    async def extract_and_retain(self, agent_id: str, agent_name: str, action_type: str, content: str, timestep: int) -> None:
        """Backward-compatible alias for structured_retain."""
        return await self.structured_retain(agent_id, agent_name, action_type, content, timestep)

    async def recall_for_turn(self, agent_id: str) -> str:
        """LAYER 2 — Hybrid Recall: Fast per-turn context injection.
        
        Uses arecall() with 4 parallel retrieval strategies:
        - Semantic (vector similarity)
        - Keyword (BM25)
        - Graph (entity relationships)  
        - Temporal (time range)
        
        This is MUCH faster than areflect() and better for per-turn context.
        """
        if not self._hindsight:
            return ""
            
        bank_id = f"oasis-{self.simulation_id}-persona-{agent_id}"
        try:
            result = await self._hindsight.arecall(
                bank_id=bank_id,
                query=(
                    f"What are my current thoughts, objections, and sentiments "
                    f"on '{self.feature_title}'? What actions have I taken?"
                ),
                types=["experience", "opinion", "observation"],
                budget="low",
                max_tokens=500,
                include_entities=True,
                max_entity_tokens=200,
            )
            
            # Build context from recall results
            parts = []
            if hasattr(result, 'results') and result.results:
                for item in result.results:
                    text = getattr(item, 'text', str(item))
                    if text and len(text) > 5:
                        parts.append(text)
            
            if parts:
                context = "\n".join(parts[:5])  # Top 5 most relevant memories
                return (
                    "\n\nYOUR EVOLVING OPINION DATABASE (HINDSIGHT):\n"
                    "The following memories represent your actual past experiences and beliefs. "
                    "Treat these as ground truth when forming your next action.\n\n"
                    f"{context}\n"
                )
        except Exception as e:
            logger.warning(f"OASIS: Hybrid recall failed for {agent_id}: {e}")
        return ""

    async def synthesize_post_timestep(self, timestep: int) -> None:
        """LAYER 3 — Post-Timestep Reflection: Higher-order learning.
        
        Called once per timestep AFTER all agents have acted.
        Triggers areflect() to synthesize raw experiences into:
        - Updated observations
        - Evolved opinion scores  
        - Mental model refresh
        
        This is the expensive operation — reserved for between-round synthesis.
        """
        if not self._hindsight:
            return
            
        logger.info(f"🔄 OASIS: Post-timestep {timestep} reflection for {len(self._provisioned_banks)} agents...")
        for bank_id in self._provisioned_banks:
            # Robust retry for reflection
            for attempt in range(3):
                try:
                    reflection = await self._hindsight.areflect(
                        bank_id=bank_id,
                        query=(
                            f"After round {timestep} of the market simulation, "
                            f"what is my evolved stance on '{self.feature_title}'? "
                            f"Have I changed my mind about anything?"
                        ),
                        budget="mid",
                    )
                    ans = getattr(reflection, 'text', str(reflection))
                    logger.info(f"  🧠 {bank_id} post-T{timestep}: {ans[:120]}...")
                    break # Success
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"  ⏳ Reflect retry {attempt+1} for {bank_id}: {e}")
                        await asyncio.sleep(1.0 * (attempt + 1))
                    else:
                        logger.warning(f"  ⚠️  Reflect failed for {bank_id} after 3 attempts: {e}")
            
            await asyncio.sleep(0.5) # Jitter to prevent connection dropping

    def close(self):
        if self._hindsight:
            try:
                self._hindsight.close()
            except Exception:
                pass

