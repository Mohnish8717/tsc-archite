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
from concurrent.futures import Future
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from tsc.llm.temperatures import MEMORY_QUERY_EMBEDDED

# Patch the event loop EARLY (module-level) to allow nested async calls.
# Required because the Hindsight SDK uses asyncio internally, and AG2's
# framework already has an event loop running.
try:
    import nest_asyncio
    nest_asyncio.apply()
except (ImportError, ValueError):
    pass

# ─── Persistent Hindsight Thread (CRITICAL-2 fix) ─────────────────────────
# A single background thread owns a dedicated asyncio event loop that lives
# for the entire process lifetime. All synchronous Hindsight calls are
# dispatched here via run_coroutine_threadsafe(), eliminating the
# ThreadPoolExecutor-per-call leak that caused thread exhaustion on long debates.
_hindsight_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_hindsight_thread: threading.Thread = threading.Thread(
    target=_hindsight_loop.run_forever,
    name="hindsight-worker",
    daemon=True,  # Dies cleanly when main process exits
)
_hindsight_thread.start()


def _run_in_hindsight_loop(coro) -> Any:
    """Dispatch a coroutine to the persistent Hindsight worker loop.

    Safe to call from ANY thread — including the main asyncio event loop thread.
    Uses run_coroutine_threadsafe() which is the only thread-safe bridge between
    a running loop (main) and another loop (worker).

    Raises:
        TimeoutError: If Hindsight doesn't respond within 30 seconds.
        Exception: Any exception raised inside the coroutine is re-raised here.
    """
    if not asyncio.iscoroutine(coro):
        return coro
    future: Future = asyncio.run_coroutine_threadsafe(coro, _hindsight_loop)
    return future.result(timeout=30)

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

# ─── Runtime LLM Provider Resolution ─────────────────────────────────────
# Reads HINDSIGHT_LLM_PROVIDER from .env and resolves the active config.
# This is for LOGGING/DIAGNOSTICS only — the actual provider config is
# passed to the Hindsight Docker container via start_hindsight_local.sh.

HINDSIGHT_PROVIDER_MAP = {
    "gemini": {
        "env_key": "HINDSIGHT_GEMINI_API_KEY",
        "env_model": "HINDSIGHT_GEMINI_MODEL",
        "default_model": "gemini-2.5-flash",
        "label": "Google Gemini (Cloud)",
    },
    "ollama": {
        "env_key": None,  # No API key needed
        "env_model": "HINDSIGHT_OLLAMA_MODEL",
        "default_model": "gemma3:12b",
        "label": "Ollama (Local, Free)",
    },
    "groq": {
        "env_key": "HINDSIGHT_GROQ_API_KEY",
        "env_model": "HINDSIGHT_GROQ_MODEL",
        "default_model": "llama-3.3-70b-versatile",
        "label": "Groq (Cloud, Fast)",
    },
    "llamacpp": {
        "env_key": None,
        "env_model": None,
        "default_model": "gemma-4-e2b-it",
        "label": "llama.cpp (Local, Built-in)",
    },
}


def get_hindsight_provider_info() -> Dict[str, str]:
    """Resolve the active Hindsight LLM provider from environment.
    
    Returns a dict with keys: provider, model, label, has_key.
    Used for logging and diagnostics — the actual server config is
    set via Docker environment variables.
    """
    provider = os.getenv("HINDSIGHT_LLM_PROVIDER", "gemini").lower()
    config = HINDSIGHT_PROVIDER_MAP.get(provider, HINDSIGHT_PROVIDER_MAP["gemini"])
    
    model = "unknown"
    if config["env_model"]:
        model = os.getenv(config["env_model"], config["default_model"])
    else:
        model = config["default_model"]
    
    has_key = True
    if config["env_key"]:
        has_key = bool(os.getenv(config["env_key"], ""))
    
    return {
        "provider": provider,
        "model": model,
        "label": config["label"],
        "has_key": str(has_key),
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
        self._provider_info = get_hindsight_provider_info()
        self._out_of_credits = False

        # Try to connect to Hindsight server
        url = hindsight_url or os.getenv("HINDSIGHT_URL", "")
        key = api_key or os.getenv("HINDSIGHT_API_KEY", "")
        if url:
            try:
                from hindsight_client import Hindsight
                # Empty API key = local self-hosted mode (no auth required)
                self._hindsight = Hindsight(base_url=url, api_key=key if key else None)
                self._mode = "HINDSIGHT"
                logger.info(
                    f"V29: Hindsight CONNECTED at {url} "
                    f"[LLM: {self._provider_info['label']} → {self._provider_info['model']}]"
                )
            except Exception as e:
                logger.warning(f"V29: Hindsight connection failed ({e}). Falling back to EMBEDDED (degraded).")
        else:
            logger.info("V29: No HINDSIGHT_URL set. Using EMBEDDED memory mode (degraded — no NLU extraction).")

    def _run_sync(self, coro_or_val):
        """Safely run a coroutine from a synchronous context.

        Delegates to the persistent Hindsight worker thread (_hindsight_loop)
        via run_coroutine_threadsafe(). This is safe to call from:
          - The main asyncio loop thread (AG2 / debate engine)
          - Any worker thread
          - A plain synchronous context (no running loop)

        REPLACES the old ThreadPoolExecutor-per-call pattern (CRITICAL-2 fix).
        The old approach created a new executor + asyncio.run() on every retain/
        recall/reflect call, causing thread exhaustion on long debates (50+ calls).
        """
        return _run_in_hindsight_loop(coro_or_val)
            

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
                    err_str = str(e)
                    if "402" in err_str or "Insufficient credits" in err_str:
                        logger.error(f"V29: Hindsight OUT OF CREDITS (402). Disabling API calls globally.")
                        self._hindsight = None
                        self._mode = "EMBEDDED"
                        self._out_of_credits = True
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
                err_str = str(e)
                if "402" in err_str or "Insufficient credits" in err_str:
                    logger.error(f"V29: Hindsight OUT OF CREDITS (402) during retain. Disabling API calls.")
                    self._hindsight = None
                    self._mode = "EMBEDDED"
                    self._out_of_credits = True
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
                err_str = str(e)
                if "402" in err_str or "Insufficient credits" in err_str:
                    logger.error(f"V29: Hindsight OUT OF CREDITS (402) during recall. Disabling API calls.")
                    self._hindsight = None
                    self._mode = "EMBEDDED"
                    self._out_of_credits = True
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
                err_str = str(e)
                if "402" in err_str or "Insufficient credits" in err_str:
                    logger.error(f"V29: Hindsight OUT OF CREDITS (402) during reflect. Disabling API calls.")
                    self._hindsight = None
                    self._mode = "EMBEDDED"
                    self._out_of_credits = True
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
                err_str = str(e)
                if "402" in err_str or "Insufficient credits" in err_str:
                    logger.error(f"V29: Hindsight OUT OF CREDITS (402) during query. Disabling API calls.")
                    self._hindsight = None
                    self._mode = "EMBEDDED"
                    self._out_of_credits = True
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
                    temperature=MEMORY_QUERY_EMBEDDED,
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
        self._provider_info = get_hindsight_provider_info()
        
        url = hindsight_url or os.getenv("HINDSIGHT_URL", "")
        key = api_key or os.getenv("HINDSIGHT_API_KEY", "")
        if url:
            try:
                from hindsight_client import Hindsight
                # Empty API key = local self-hosted mode (no auth required)
                self._hindsight = Hindsight(base_url=url, api_key=key if key else None)
                self._mode = "HINDSIGHT"
                logger.info(
                    f"OASIS: Hindsight client initialized → {url} "
                    f"[LLM: {self._provider_info['label']} → {self._provider_info['model']}]"
                )
            except Exception as e:
                # We log the error but don't set hindsight to None yet, 
                # we will try to connect during initialize_agents.
                logger.error(f"OASIS: Hindsight client initialization failed: {e}")
                
        self.simulation_id = ""
        self.feature_title = ""
        self.feature_description = ""
        self._shared_bank_id: str = ""
        self._provisioned_banks: list[str] = []

        # ── Memory Lifecycle Optimization (Free Tier) ──────────────────
        # Dirty-flag: only reflect agents that produced content this timestep
        self._dirty_agents: set = set() # Now stores agent_id strings like "5", "42"
        # Timestep counter for adaptive reflect frequency
        self._timestep_counter: int = 0
        # Free-tier budget config — minimize token usage
        _tier = os.getenv("HINDSIGHT_TIER", "free").lower()
        self._budget = {
            "free":  {"recall_budget": "low", "reflect_budget": "low",
                      "recall_max_tokens": 300, "reflect_max_tokens": 200,
                      "reflect_every_n": 3, "max_entity_tokens": 100},
            "pro":   {"recall_budget": "low", "reflect_budget": "mid",
                      "recall_max_tokens": 500, "reflect_max_tokens": 400,
                      "reflect_every_n": 2, "max_entity_tokens": 200},
            "scale": {"recall_budget": "mid", "reflect_budget": "high",
                      "recall_max_tokens": 800, "reflect_max_tokens": 600,
                      "reflect_every_n": 1, "max_entity_tokens": 300},
        }.get(_tier, {"recall_budget": "low", "reflect_budget": "low",
                      "recall_max_tokens": 300, "reflect_max_tokens": 200,
                      "reflect_every_n": 3, "max_entity_tokens": 100})
        logger.info(f"OASIS: Memory budget tier='{_tier}' → reflect_every={self._budget['reflect_every_n']}")

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
        """Create a single shared Hindsight Memory Bank for all Oasis agents."""
        self.feature_title = feature_title
        self.feature_description = feature_description
        self.simulation_id = simulation_id
        self._shared_bank_id = f"oasis-{self.simulation_id}"
        
        # ── CONNECTION PREFLIGHT ──
        connected = False
        for i in range(5):
            if await self.check_connection():
                connected = True
                break
            logger.warning(f"  ⏳ Hindsight connection pending... retry {i+1}/5")
            await asyncio.sleep(2.0 * (i + 1))
            
        if not connected:
            raise ConnectionError(f"OASIS: Could not establish stable connection to Hindsight at {os.getenv('HINDSIGHT_URL')}")

        # ─── PROVISIONING SINGLE SHARED BANK ───
        logger.info(f"🏦 OASIS: Provisioning shared Hindsight Memory Bank for {len(agent_profiles)} agents...")
        
        try:
            # Clean slate: delete any pre-existing bank
            try:
                await self._hindsight.adelete_bank(bank_id=self._shared_bank_id)
            except Exception:
                pass

            # Create the shared memory bank
            await self._hindsight.acreate_bank(
                bank_id=self._shared_bank_id,
                name=f"OASIS-Run-{self.simulation_id}",
                background=(
                    f"OASIS Market Simulation. "
                    f"Simulation of multiple personas evaluating: '{self.feature_title}'."
                ),
                retain_mission=(
                    "Extract evolving beliefs, sentiments, objections, and needs for each agent. "
                    "Track stance changes over time."
                ),
                enable_observations=True,
            )
            
            # Seed with feature context for all agents in one batch
            batch_items = []
            for profile in agent_profiles:
                agent_id = str(profile.agent_id)
                agent_name = profile.user_info_dict.get("name", f"Agent_{agent_id}")
                batch_items.append({
                    "content": (
                        f"[AGENT:{agent_id}|{agent_name}] [SYSTEM]: A new feature has been proposed for evaluation: "
                        f"'{self.feature_title}'. Description: {self.feature_description[:1000]}"
                    ),
                    "context": "Initial feature briefing for market simulation",
                    "entities": [
                        {"text": self.feature_title, "type": "FEATURE"},
                        {"text": agent_name, "type": "AGENT"},
                    ],
                    "tags": [f"agent_{agent_id}", "world", "feature_introduction"],
                })
            
            await self._hindsight.aretain_batch(
                bank_id=self._shared_bank_id,
                items=batch_items,
            )
            
            self._provisioned_banks = [self._shared_bank_id]
            logger.info(f"🏦 OASIS: Finished provisioning. 1 shared bank '{self._shared_bank_id}' provisioned for {len(agent_profiles)} agents.")
            
        except Exception as e:
            logger.error(f"  ❌ OASIS: Shared bank creation FAILED: {e}")

    async def structured_retain(
        self, agent_id: str, agent_name: str, action_type: str,
        content: str, timestep: int
    ) -> None:
        """LAYER 1 — Structured Retain: Store agent actions with entities and timestamps.
        
        Uses aretain_batch() with structured items instead of raw text blobs.
        Hindsight extracts discrete facts, temporal data, and entity relationships.
        """
        if not self._hindsight or not self._shared_bank_id:
            return
            
        bank_id = self._shared_bank_id
        try:
            await self._hindsight.aretain_batch(
                bank_id=bank_id,
                items=[{
                    "content": f"[AGENT:{agent_id}|{agent_name}] [{action_type}]: {content[:1500]}",
                    "timestamp": datetime.now().isoformat(),
                    "context": (
                        f"OASIS simulation timestep {timestep}. "
                        f"Agent performed action: {action_type}."
                    ),
                    "entities": [
                        {"text": self.feature_title, "type": "FEATURE"},
                        {"text": agent_name, "type": "AGENT"},
                    ],
                    "tags": [f"agent_{agent_id}", "experience", action_type.lower(), f"timestep_{timestep}"],
                }],
            )
            logger.info(f"  🧠 Memory inclusion (Timestep {timestep}): Stored '{action_type}' for {agent_name} into {bank_id}")
            # Mark agent as dirty — needs reflection this timestep
            self._dirty_agents.add(str(agent_id))
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
        if not self._hindsight or not self._shared_bank_id:
            return ""
            
        bank_id = self._shared_bank_id
        try:
            # Fallback strategy for older SDK versions that don't support filter_tags
            recall_kwargs = {
                "bank_id": bank_id,
                "query": (
                    f"What are agent {agent_id}'s current thoughts, objections, and sentiments "
                    f"on '{self.feature_title}'? What actions has agent {agent_id} taken?"
                ),
                "types": ["experience", "opinion", "observation"],
                "budget": self._budget["recall_budget"],
                "max_tokens": self._budget["recall_max_tokens"],
                "include_entities": True,
                "max_entity_tokens": self._budget["max_entity_tokens"],
            }
            agent_tag = f"agent_{agent_id}"
            agent_prefix = f"AGENT:{agent_id}"

            try:
                # Preferred path: SDK-native tag filtering (O(1) server-side)
                result = await self._hindsight.arecall(**recall_kwargs, filter_tags=[agent_tag])
            except TypeError:
                # SDK version doesn't support filter_tags — fall back to unscoped
                # recall and apply manual post-filtering client-side (CRITICAL-3 fix).
                # Without this filter, Agent X would receive Agent Y's memories in a
                # shared bank, causing false consensus across the agent population.
                logger.debug(
                    f"OASIS: filter_tags unsupported by SDK — applying manual "
                    f"post-filter for agent_{agent_id}"
                )
                result = await self._hindsight.arecall(**recall_kwargs)
                # Manually discard records that don't belong to this agent
                if hasattr(result, 'results') and result.results:
                    result.results = [
                        r for r in result.results
                        if agent_tag in getattr(r, 'content', '')
                        or agent_prefix in getattr(r, 'content', '')
                        or agent_tag in getattr(r, 'tags', [])
                    ]

            # Build context from (now agent-scoped) recall results
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
        
        Optimized for free tier with three cost-saving mechanisms:
        1. DIRTY-FLAG: Only reflect agents that produced content this timestep
        2. ADAPTIVE FREQUENCY: Reflect every N timesteps (configurable by tier)
        3. BELIEF EVOLUTION: Store reflection output back as a new memory
           so subsequent recalls include the evolved stance
        """
        if not self._hindsight or not self._shared_bank_id:
            return
        
        self._timestep_counter += 1
        reflect_every = self._budget["reflect_every_n"]
        
        # Adaptive frequency gate — skip reflection on non-reflect timesteps.
        # IMPORTANT (Path-B drift fix): Do NOT clear _dirty_agents here.
        # Agents that spoke during skipped timesteps must be reflected at the
        # NEXT reflection cycle so their evolved_belief memories are created.
        # Clearing here caused belief starvation: agents whose timestep was
        # skipped never got a reflect call, so their recall() returned stale
        # memories indefinitely, creating false "position stability".
        if self._timestep_counter % reflect_every != 0:
            next_reflect = timestep + (reflect_every - self._timestep_counter % reflect_every)
            logger.info(
                f"⏭️  OASIS: Skipping reflection at timestep {timestep} "
                f"(next reflect at timestep {next_reflect}). "
                f"Accumulating {len(self._dirty_agents)} dirty agents for next cycle."
            )
            return  # _dirty_agents intentionally preserved — accumulates across skipped steps
        
        # Only reflect agents that actually spoke this round
        active_agents = list(self._dirty_agents)
        if not active_agents:
            return

        logger.info(
            f"🔄 OASIS: Post-timestep {timestep} reflection — "
            f"{len(active_agents)} active agents"
        )
        
        bank_id = self._shared_bank_id
        for agent_id in active_agents:
            for attempt in range(2):  # Reduced retries for free tier
                try:
                    reflect_kwargs = {
                        "bank_id": bank_id,
                        "query": (
                            f"After round {timestep} of the market simulation, "
                            f"what is agent {agent_id}'s evolved stance on '{self.feature_title}'? "
                            f"Has agent {agent_id} changed their mind about anything? "
                            f"What specific concerns or support does agent {agent_id} now have?"
                        ),
                        "budget": self._budget["reflect_budget"],
                    }
                    try:
                        reflection = await self._hindsight.areflect(**reflect_kwargs, filter_tags=[f"agent_{agent_id}"])
                    except TypeError:
                        reflection = await self._hindsight.areflect(**reflect_kwargs)
                        
                    ans = getattr(reflection, 'answer',
                                  getattr(reflection, 'text', str(reflection)))
                    logger.info(f"  🧠 Agent {agent_id} post-T{timestep}: {ans[:120]}...")
                    
                    # CRITICAL: Store reflection back as evolved_belief memory
                    # This closes the Act→Retain→Reflect→Store→Recall→Act loop
                    if ans and len(ans) > 10:
                        try:
                            await self._hindsight.aretain_batch(
                                bank_id=bank_id,
                                items=[{
                                    "content": (
                                        f"[AGENT:{agent_id}] [EVOLVED BELIEF after round {timestep}]: "
                                        f"{ans[:600]}"
                                    ),
                                    "tags": [f"agent_{agent_id}", "opinion", "evolved_belief",
                                             f"post_timestep_{timestep}"],
                                    "context": (
                                        "This is my synthesized belief after "
                                        "reflecting on this round of discussion."
                                    ),
                                }],
                            )
                        except Exception:
                            pass  # Non-critical — reflection itself succeeded
                    break  # Success
                except Exception as e:
                    if attempt < 1:
                        logger.warning(f"  ⏳ Reflect retry for Agent {agent_id}: {e}")
                        await asyncio.sleep(1.5)
                    else:
                        logger.warning(f"  ⚠️  Reflect failed for Agent {agent_id}: {e}")
            
            await asyncio.sleep(0.3)  # Jitter
        
        # Reset dirty set for next timestep
        self._dirty_agents.clear()

    async def cleanup_banks(self, simulation_id: str = "") -> int:
        """Delete all Hindsight banks provisioned during this simulation run.
        
        Called when a NEW simulation starts (not after each run) to preserve
        data for post-simulation forensic analysis.
        
        Works for both local Docker and cloud Hindsight.
        """
        if not self._hindsight:
            return 0
        
        deleted = 0
        target_bank = f"oasis-{simulation_id}" if simulation_id else self._shared_bank_id
        
        if target_bank:
            try:
                await self._hindsight.adelete_bank(bank_id=target_bank)
                deleted += 1
                logger.debug(f"  🧹 Deleted bank: {target_bank}")
            except Exception as e:
                logger.debug(f"  ⚠️  Could not delete {target_bank}: {e}")
        
        if target_bank == self._shared_bank_id:
            self._provisioned_banks.clear()
            self._shared_bank_id = ""
            
        logger.info(f"🧹 OASIS: Purged {deleted} Hindsight banks")
        return deleted

    def close(self):
        if self._hindsight:
            try:
                self._hindsight.close()
            except Exception:
                pass


