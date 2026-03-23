from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import numpy as np

from tsc.gates.base import BaseGate
from tsc.llm.base import LLMClient
from tsc.models.chunks import ProblemContextBundle
from tsc.models.gates import GateResult, GateVerdict
from tsc.models.graph import KnowledgeGraph
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona
import mesa

logger = logging.getLogger(__name__)


class AdversaryAgent(mesa.Agent):
    """Mesa agent representing an adversary analyzing a specific risk domain."""
    def __init__(self, unique_id: int, model: mesa.Model, domain: str, context: str, llm_client: LLMClient):
        super().__init__(unique_id, model)
        self.domain = domain
        self.context = context
        self.llm_client = llm_client
        self.risk_data: dict[str, Any] = {}
        
    def step(self):
        """Mesa step function. Simulates cross-domain cascading risk failures."""
        if not getattr(self, "risk_data", None) or "probability" not in self.risk_data:
            return
            
        DOMAIN_WEIGHTS = {
            "Market": {"Technical": 0.06, "Adoption": 0.08},
            "Technical": {"Market": 0.05, "Adoption": 0.07},
            "Adoption": {"Market": 0.09, "Technical": 0.06},
        }
            
        # Observe peer risks in the simulation environment
        peer_influence = 0.0
        for a in self.model.schedule.agents:
            if a != self and getattr(a, "risk_data", None) and "probability" in a.risk_data:
                peer_prob = a.risk_data.get("probability", 0)
                weight = DOMAIN_WEIGHTS.get(self.domain, {}).get(a.domain, 0.05)
                peer_influence += peer_prob * weight
        
        # Escalate risk probability slightly based on weighted peer pressure
        current_prob = float(self.risk_data["probability"])
        noise = np.random.uniform(-0.02, 0.02)
        
        # Cascading failure: if peer risk is high, this domain becomes riskier
        new_prob = np.clip(current_prob + peer_influence + noise, 0, 1)
        self.risk_data["probability"] = float(new_prob)

    async def analyze(self) -> dict[str, Any]:
        """Perform adversarial analysis asynchronously."""
        if self.domain == "Market":
            prompt = self._get_market_prompt()
        elif self.domain == "Technical":
            prompt = self._get_technical_prompt()
        else:
            prompt = self._get_adoption_prompt()
            
        try:
            response = await asyncio.wait_for(
                self.llm_client.analyze(
                    system_prompt=self._get_adversarial_system_prompt(),
                    user_prompt=prompt,
                    temperature=0.8,
                    max_tokens=400,
                ),
                timeout=30.0
            )
            # Early validation
            validated = self.model.gate._validate_risk_dict(response, self.domain.lower())
            self.risk_data = validated
            return validated
        except asyncio.TimeoutError:
            logger.error("%s risk analysis timeout (30s)", self.domain)
            self.risk_data = {"error": "Timeout"}
            return self.risk_data
        except Exception as e:
            logger.error("%s risk analysis failed: %s", self.domain, e, exc_info=True)
            self.risk_data = {"error": str(e)}
            return self.risk_data

    def _get_adversarial_system_prompt(self) -> str:
        return (
            "You are a devil's advocate and harsh critic of the feature proposal. "
            "Your job is to find the FATAL FLAW or WORST CASE SCENARIO. "
            "Be brutally honest. Don't sugarcoat. Find the single biggest risk. "
            "Be specific and actionable."
        )

    def _get_market_prompt(self) -> str:
        return f"""
{self.context}

MARKET RISKS: Find the FATAL FLAW that could kill this commercially.

Consider:
- Competitive pricing pressure
- Market too small for ROI
- Better solutions already exist
- Customer churn for unrelated reasons
- Timing risk (too early/late)

Return JSON:
{{
  "risk": "The single biggest market risk is...",
  "probability": 0.0-1.0,
  "impact": "Financial impact description",
  "severity": "high|medium|low",
  "mitigations": ["action 1", "action 2"]
}}
"""

    def _get_technical_prompt(self) -> str:
        return f"""
{self.context}

TECHNICAL RISKS: Find the worst-case TECHNICAL FAILURE.

Consider:
- Data loss scenarios
- Conflict resolution bugs
- Performance on old devices
- Integration complexity
- Security/privacy issues
- Scalability limits

Return JSON:
{{
  "risk": "The worst technical failure could be...",
  "probability": 0.0-1.0,
  "impact": "System/data impact",
  "severity": "high|medium|low",
  "mitigations": ["testing strategy", "fallback plan"]
}}
"""

    def _get_adoption_prompt(self) -> str:
        return f"""
{self.context}

ADOPTION RISKS: Find the BEHAVIORAL BLOCKER that prevents adoption.

Consider:
- Users don't trust new feature
- Behavior change required but resisted
- Silent failures users don't notice
- Requires training/education
- Conflicts with existing workflows
- Too complex for target audience

Return JSON:
{{
  "risk": "Users won't adopt because...",
  "probability": 0.0-1.0,
  "impact": "Adoption reduction",
  "severity": "high|medium|low",
  "mitigations": ["onboarding", "UX fix", "education"]
}}
"""

class AdversarialModel(mesa.Model):
    """Mesa environment for Adversarial Multi-Agent Analysis"""
    def __init__(self, gate: "RedTeamAdversarialGate", feature: FeatureProposal, company: CompanyContext, personas: list[FinalPersona], graph: KnowledgeGraph, llm_client: LLMClient, contexts: dict[str, str]):
        super().__init__()
        self.gate = gate
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.feature = feature
        self.company = company
        self.personas = personas
        self.graph = graph
        self.llm_client = llm_client
        
        # Initialize an adversary for each domain
        for i, (domain, context) in enumerate(contexts.items()):
            agent = AdversaryAgent(i, self, domain, context, llm_client)
            self.schedule.add(agent)
            
        self.datacollector = mesa.datacollection.DataCollector(
            agent_reporters={
                "RiskData": lambda a: getattr(a, "risk_data", {}),
                "Domain": lambda a: getattr(a, "domain", "Unknown")
            }
        )

    def step(self):
        """Advance the simulation by one step"""
        self.schedule.step()
        self.datacollector.collect(self)

class RedTeamAdversarialGate(BaseGate):
    """Gate 4.6: Red-Team Adversarial Analysis (OASIS Simulation environment)"""
    
    gate_id = "4.6"
    gate_name = "Red-Team Adversarial Analysis"
    gate_domain = "adversarial risk analysis across market, technical, adoption"
    verdict_options = [
        "PASS_WITH_MITIGATION",   # MANAGEABLE risk
        "FEASIBLE_WITH_DEBT",      # ELEVATED risk
        "RISKY",                   # CRITICAL risk
        "FAIL",                    # UNACCEPTABLE risk
    ]
    
    RISK_DOMAINS = ["market", "technical", "adoption"]
    
    def __init__(
        self,
        llm_client: LLMClient,
        graph_store: Optional[Any] = None,
        enable_caching: bool = True,
        cache_ttl_minutes: int = 60,
    ):
        super().__init__(llm_client)
        self._graph_store = graph_store
        self._enable_caching = enable_caching
        self._cache_ttl = cache_ttl_minutes * 60
        self._analysis_cache: dict[str, tuple[GateResult, float]] = {}
        
        logger.info(
            "RedTeamAdversarialGate initialized (caching=%s, ttl=%dm)",
            enable_caching, cache_ttl_minutes
        )

    # ── Cache Management ──────────────────────────────────────────────

    def _get_cache_key(self, feature: FeatureProposal) -> str:
        """Generate cache key for red-team analysis"""
        return f"{feature.title}_red_team".lower()

    def _is_cache_valid(self, cached_time: float) -> bool:
        """Check if cache is still valid"""
        return (time.time() - cached_time) < self._cache_ttl

    # ── Input Validation ──────────────────────────────────────────────

    def _validate_probability(self, prob: Any, context: str = "probability") -> float:
        """Validate and clamp probability to 0-1 range"""
        try:
            p = float(prob)
            original = p
            
            # Clamp to valid range
            p = max(0.0, min(1.0, p))
            
            if p != original:
                logger.warning(
                    "Clamped %s from %.2f to %.2f",
                    context, original, p
                )
            
            return p
        
        except (ValueError, TypeError):
            logger.warning(
                "Invalid %s value: %s (type: %s), using default 0.5",
                context, prob, type(prob).__name__
            )
            return 0.5

    def _validate_severity(self, severity: Any) -> str:
        """Validate severity is one of: high, medium, low"""
        
        valid_severities = {"high", "medium", "low"}
        
        if not severity:
            logger.warning("Missing severity, using default 'medium'")
            return "medium"
        
        severity_str = str(severity).lower().strip()
        
        if severity_str not in valid_severities:
            logger.warning(
                "Invalid severity '%s', valid: %s, using default 'medium'",
                severity_str, valid_severities
            )
            return "medium"
        
        return severity_str

    def _validate_mitigations(self, mitigations: Any) -> list[str]:
        """Validate mitigations is a list of non-empty strings"""
        
        if not mitigations:
            return []
        
        if not isinstance(mitigations, list):
            logger.warning(
                "Mitigations not a list (type: %s), skipping",
                type(mitigations).__name__
            )
            return []
        
        validated = []
        
        for idx, mitigation in enumerate(mitigations):
            if not mitigation:
                logger.debug("Skipping empty mitigation %d", idx)
                continue
            
            mitigation_str = str(mitigation).strip()
            
            if len(mitigation_str) < 5:
                logger.debug("Mitigation %d too short, skipping", idx)
                continue
            
            if len(mitigation_str) > 200:
                mitigation_str = mitigation_str[:197] + "..."
                logger.debug("Truncated mitigation %d to 200 chars", idx)
            
            validated.append(mitigation_str)
        
        return validated[:5]  # Max 5 mitigations

    def _validate_risk_text(self, risk: Any) -> str:
        """Validate risk description text"""
        
        if not risk:
            return "Unknown risk"
        
        risk_str = str(risk).strip()
        
        if len(risk_str) < 10:
            logger.warning("Risk text too short: %s", risk_str)
            return "Unknown risk"
        
        if len(risk_str) > 300:
            risk_str = risk_str[:297] + "..."
        
        return risk_str

    def _validate_risk_dict(self, risk_data: Any, domain: str) -> dict[str, Any]:
        """Validate entire risk analysis response"""
        
        if not risk_data or not isinstance(risk_data, dict):
            logger.warning(
                "%s risk analysis returned invalid data type: %s",
                domain, type(risk_data).__name__
            )
            return self._create_default_risk(domain, "Analysis returned invalid format")
        
        # Validate required fields
        required = ["risk", "probability", "impact", "severity"]
        missing = [f for f in required if f not in risk_data]
        
        if missing:
            logger.warning(
                "%s risk analysis missing fields: %s",
                domain, missing
            )
        
        # Validate each field
        validated = {
            "risk": self._validate_risk_text(risk_data.get("risk")),
            "probability": self._validate_probability(
                risk_data.get("probability"),
                f"{domain}_probability"
            ),
            "impact": str(risk_data.get("impact", ""))[:200],
            "severity": self._validate_severity(risk_data.get("severity")),
            "mitigations": self._validate_mitigations(risk_data.get("mitigations")),
        }
        
        return validated

    def _create_default_risk(self, domain: str, reason: str = "Analysis failed") -> dict[str, Any]:
        """Create a safe default risk when analysis fails"""
        
        defaults = {
            "market": {
                "risk": "Potential market-related complications",
                "probability": 0.3,
                "impact": "Uncertain market impact",
                "severity": "medium",
                "mitigations": ["Validate market assumptions", "Monitor adoption closely"],
            },
            "technical": {
                "risk": "Potential technical implementation challenges",
                "probability": 0.3,
                "impact": "Uncertain technical impact",
                "severity": "medium",
                "mitigations": ["Prototype high-risk components", "Plan for technical debt"],
            },
            "adoption": {
                "risk": "Potential user adoption barriers",
                "probability": 0.3,
                "impact": "Uncertain adoption impact",
                "severity": "medium",
                "mitigations": ["User testing", "Clear onboarding"],
            },
        }
        
        default = defaults.get(domain, defaults["technical"])
        logger.warning(
            "%s risk analysis failed (%s), using default risk",
            domain, reason
        )
        
        return default

    # ── Context Building ──────────────────────────────────────────────

    def _format_company_context(self, company: CompanyContext) -> str:
        """Format company context for prompts"""
        
        return f"""
COMPANY CONTEXT:
- Name: {company.company_name or 'Unspecified'}
- Team Size: {company.team_size or 'Unknown'}
- Budget: {company.budget or 'Unspecified'}
- Current Priorities: {', '.join(company.current_priorities or ['Unspecified'])}
- Tech Stack: {', '.join(company.tech_stack or ['Unspecified'])}
- Constraints: {', '.join(company.constraints or ['None'])}
- Competitors: {', '.join(company.competitors or ['Unspecified'])}
"""

    def _summarize_persona_stances(self, personas: list[FinalPersona]) -> str:
        """Extract and summarize stakeholder positions"""
        
        if not personas:
            return "No stakeholder feedback available"
        
        summaries = []
        
        for persona in personas:
            try:
                stance = persona.psychological_profile.predicted_stance
                concerns = ", ".join(stance.potential_objections[:2]) if stance.potential_objections else "None noted"
                conditions = ", ".join(stance.likely_conditions[:2]) if stance.likely_conditions else "None specified"
                
                summaries.append(
                    f"- {persona.name} ({persona.role}): {stance.prediction} "
                    f"| Concerns: {concerns} | Conditions: {conditions}"
                )
            
            except Exception as e:
                logger.warning("Failed to extract stance for %s: %s", persona.name, e)
                summaries.append(f"- {persona.name} ({persona.role}): Unable to extract stance")
        
        return "\n".join(summaries) if summaries else "No stakeholder data available"

    def _extract_technical_entities(self, graph: KnowledgeGraph) -> str:
        """Extract technical entities from knowledge graph"""
        
        if not graph or not graph.nodes:
            return "No technical entities identified"
        
        technical_entities = [
            e for e in graph.nodes.values()
            if e.type in ["CONSTRAINT", "PAIN_POINT", "METRIC"]
        ]
        
        if not technical_entities:
            return "No specific technical constraints identified"
        
        entities_str = "\n".join(
            f"- {e.name} ({e.type}): mentioned {e.mentions}x"
            for e in sorted(
                technical_entities,
                key=lambda x: x.mentions,
                reverse=True
            )[:5]
        )
        
        return entities_str

    def _extract_adoption_signals(self, personas: list[FinalPersona]) -> str:
        """Extract adoption signals from personas"""
        
        if not personas:
            return "No adoption signals available"
        
        signals = []
        
        for persona in personas:
            try:
                stance = persona.psychological_profile.predicted_stance
                
                if stance.prediction == "APPROVED":
                    signals.append(f"✓ {persona.name} is motivated")
                elif stance.prediction == "CONDITIONAL_APPROVE":
                    signals.append(f"△ {persona.name} needs conditions met")
                else:
                    signals.append(f"✗ {persona.name} has reservations")
            
            except Exception as e:
                logger.debug("Failed to extract signal for %s: %s", persona.name, e)
        
        return "\n".join(signals) if signals else "No adoption signals"

    def _build_market_context(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        personas: list[FinalPersona],
    ) -> str:
        """Build comprehensive market analysis context"""
        
        company_context = self._format_company_context(company)
        persona_summary = self._summarize_persona_stances(personas)
        
        return f"""
FEATURE DETAILS:
- Title: {feature.title}
- Description: {feature.description[:300]}
- Target Users: {feature.target_users or 'Unspecified'}
- Target User Count: {feature.target_user_count or 'Unknown'}
- Effort: {feature.effort_weeks_min or '?'}-{feature.effort_weeks_max or '?'} weeks

{company_context}

STAKEHOLDER FEEDBACK:
{persona_summary}
"""

    def _build_technical_context(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
    ) -> str:
        """Build comprehensive technical analysis context"""
        
        company_context = self._format_company_context(company)
        technical_risks = self._extract_technical_entities(graph)
        
        return f"""
FEATURE: {feature.title}
EFFORT: {feature.effort_weeks_min or '?'}-{feature.effort_weeks_max or '?'} weeks

{company_context}

KNOWN TECHNICAL ENTITIES:
{technical_risks}
"""

    def _build_adoption_context(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        personas: list[FinalPersona],
    ) -> str:
        """Build comprehensive adoption analysis context"""
        
        persona_summary = self._summarize_persona_stances(personas)
        adoption_signals = self._extract_adoption_signals(personas)
        
        return f"""
FEATURE: {feature.title}
TARGET USERS: {feature.target_users or 'Unspecified'}
TARGET USER COUNT: {feature.target_user_count or 'Unknown'}

STAKEHOLDER PERSPECTIVES:
{persona_summary}

ADOPTION SIGNALS:
{adoption_signals}

COMPANY CONTEXT:
- Team Size: {company.team_size or 'Unknown'}
- Current Priorities: {', '.join(company.current_priorities or ['Unknown'])}
- Constraints: {', '.join(company.constraints or ['None'])}
"""

    # ── Risk Analysis ─────────────────────────────────────────────────

    # ── Risk Scoring ──────────────────────────────────────────────────

    def _calculate_overall_risk(
        self,
        market: dict[str, Any],
        technical: dict[str, Any],
        adoption: dict[str, Any],
    ) -> float:
        """Calculate weighted overall risk with missing data handling"""
        
        risks = {}
        
        for domain, data in [
            ("market", market),
            ("technical", technical),
            ("adoption", adoption),
        ]:
            # Handle missing/invalid data
            if not data or not isinstance(data, dict):
                logger.warning(
                    "%s risk data missing/invalid, using neutral default",
                    domain
                )
                risks[domain] = 0.5  # Neutral default
                continue
            
            try:
                prob = self._validate_probability(
                    data.get("probability"), f"{domain}_probability"
                )
                
                severity = self._validate_severity(data.get("severity"))
                
                severity_weight = {
                    "high": 1.0,
                    "medium": 0.6,
                    "low": 0.3,
                }.get(severity, 0.5)
                
                risk_score = prob * severity_weight
                risks[domain] = float(np.clip(risk_score, 0, 1))
                
                logger.debug(
                    "%s risk: prob=%.2f, severity=%s, score=%.2f",
                    domain, prob, severity, risk_score
                )
            
            except Exception as e:
                logger.warning(
                    "Error calculating %s risk: %s, using default 0.5",
                    domain, e
                )
                risks[domain] = 0.5
        
        # Weighted combination with explicit weights
        overall = (
            (risks["market"] * 0.4) +      # Market risk: 40%
            (risks["technical"] * 0.4) +   # Technical risk: 40%
            (risks["adoption"] * 0.2)      # Adoption risk: 20%
        )
        
        # Final clamp
        overall = float(np.clip(overall, 0, 1))
        
        logger.info(
            "Overall risk: market=%.2f, technical=%.2f, adoption=%.2f → %.2f",
            risks["market"], risks["technical"], risks["adoption"], overall
        )
        
        return overall

    def _map_risk_to_gate_verdict(self, risk_score: float) -> GateVerdict:
        """Map risk score to GateVerdict enum (single, correct mapping)"""
        
        # Validate risk score
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Single mapping logic - no duplication
        if risk_score < 0.33:
            verdict = GateVerdict.PASS_WITH_MITIGATION
            risk_level = "MANAGEABLE"
        elif risk_score < 0.66:
            verdict = GateVerdict.FEASIBLE_WITH_DEBT
            risk_level = "ELEVATED"
        elif risk_score < 0.85:
            verdict = GateVerdict.RISKY
            risk_level = "CRITICAL"
        else:
            verdict = GateVerdict.FAIL
            risk_level = "UNACCEPTABLE"
        
        logger.debug(
            "Mapped risk score %.2f → %s (%s)",
            risk_score, verdict.value, risk_level
        )
        
        return verdict

    # ── Reasoning Building ────────────────────────────────────────────

    def _build_reasoning(
        self,
        overall_risk: float,
        market: dict[str, Any],
        technical: dict[str, Any],
        adoption: dict[str, Any],
    ) -> str:
        """Build detailed reasoning with safety checks for missing data"""
        
        try:
            # Build risk summary with safety
            risk_items = []
            
            for domain, data in [
                ("Market", market),
                ("Technical", technical),
                ("Adoption", adoption),
            ]:
                if not data or not isinstance(data, dict):
                    logger.debug("%s risk data missing", domain)
                    continue
                
                risk_text = data.get("risk", "Unknown risk")[:60]
                prob = data.get("probability", 0)
                
                if risk_text and prob:
                    risk_items.append((domain, risk_text, prob))
            
            # Sort by probability
            risk_items.sort(key=lambda x: x[2], reverse=True)
            
            # Build risk string
            if risk_items:
                risk_str = " | ".join(
                    f"{domain}: {text}... ({p:.0%})"
                    for domain, text, p in risk_items[:2]
                )
            else:
                risk_str = "Unable to extract specific risks"
            
            # Collect mitigations safely
            all_mitigations = []
            
            for data in [market, technical, adoption]:
                if data and isinstance(data, dict):
                    mitigations = data.get("mitigations", [])
                    if isinstance(mitigations, list):
                        all_mitigations.extend(mitigations[:2])
            
            # Build mitigation string
            if all_mitigations:
                mitigation_str = ", ".join(all_mitigations[:3])
            else:
                mitigation_str = "Address identified risks systematically"
            
            return (
                f"Risk assessment: {overall_risk:.2f}/1.0. "
                f"Key risks: {risk_str}. "
                f"Recommended: {mitigation_str}"
            )
        
        except Exception as e:
            logger.warning("Error building reasoning: %s", e)
            return f"Risk assessment: {overall_risk:.2f}/1.0. Risk analysis complete."


    # ── Diagnostics ───────────────────────────────────────────────────

    def get_diagnostics(self, result: GateResult) -> dict[str, Any]:
        """Get diagnostics for red-team gate execution"""
        
        if not result or not result.details:
            return {"error": "No result details"}
        
        details = result.details
        
        return {
            "gate_id": self.gate_id,
            "verdict": result.verdict.value,
            "overall_risk": details.get("overall_risk", 0),
            "market_risk": details.get("market_risks", {}).get("probability", 0),
            "technical_risk": details.get("technical_risks", {}).get("probability", 0),
            "adoption_risk": details.get("adoption_risks", {}).get("probability", 0),
            "top_risks": self._extract_top_risks(details),
            "mitigations": self._extract_top_mitigations(details),
            "cache_size": len(self._analysis_cache),
        }

    def _extract_top_risks(self, details: dict) -> list[str]:
        """Extract top risks from details"""
        risks = []
        
        for domain in ["market_risks", "technical_risks", "adoption_risks"]:
            data = details.get(domain, {})
            risk = data.get("risk")
            if risk:
                risks.append(f"{domain.split('_')[0]}: {risk[:50]}")
        
        return risks

    def _extract_top_mitigations(self, details: dict) -> list[str]:
        """Extract top mitigations from details"""
        mitigations = []
        
        for domain in ["market_risks", "technical_risks", "adoption_risks"]:
            data = details.get(domain, {})
            mits = data.get("mitigations", [])
            if mits:
                mitigations.extend(mits[:1])
        
        return mitigations[:5]

    def _filter_internal_personas(
        self,
        personas: list[FinalPersona],
    ) -> list[FinalPersona]:
        """Filter for internal stakeholder personas (Gate 4.6 uses internal pushback)"""
        internal = [p for p in personas if p.persona_type == "INTERNAL"]
        if not internal:
            logger.warning("No internal personas found for Red-Team Gate, using all available")
            return personas
        
        logger.debug("Filtered down to %d internal personas for Red-Team", len(internal))
        return internal

    # ── Main Evaluation ───────────────────────────────────────────────

    async def evaluate(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
    ) -> GateResult:
        """Execute adversarial red-team analysis"""
        
        t0 = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(feature)
        
        if self._enable_caching and cache_key in self._analysis_cache:
            cached_result, cached_time = self._analysis_cache[cache_key]
            
            if self._is_cache_valid(cached_time):
                logger.info("Using cached red-team analysis for %s", feature.title)
                return cached_result
            else:
                logger.debug("Red-team cache expired, running fresh analysis")
                del self._analysis_cache[cache_key]
        
        logger.info("Red-Team Gate 4.6: Starting adversarial analysis for %s", feature.title)

        # Filter for internal personas (Fix: red team focuses on internal risk)
        internal_personas = self._filter_internal_personas(personas)

        # Build Contexts
        contexts = {
            "Market": self._build_market_context(feature, company, internal_personas),
            "Technical": self._build_technical_context(feature, company, graph),
            "Adoption": self._build_adoption_context(feature, company, internal_personas),
        }

        # Initialize Mesa Model
        model = AdversarialModel(self, feature, company, internal_personas, graph, self._llm, contexts)
        
        # 1. Pre-compute LLM initial adversarial attacks for all agents asynchronously
        tasks = [agent.analyze() for agent in model.schedule.agents]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 2. Truly simulate adversarial cascading risk debate using Mesa step loop
        num_steps = 8
        for _ in range(num_steps):
            model.step()
        
        # 3. Idiomatically extract simulation results from Mesa DataCollector
        df = model.datacollector.get_agent_vars_dataframe()
        
        def process_result(domain_name: str, default_key: str):
            if df.empty:
                return self._create_default_risk(default_key, "Mesa Datacollector empty")
                
            try:
                # Retrieve the final state (the last simulation step)
                last_step = df.index.get_level_values("Step").max()
                final_step_df = df.xs(last_step, level="Step")
                domain_rows = final_step_df[final_step_df["Domain"] == domain_name]
                
                if not domain_rows.empty:
                    res = domain_rows.iloc[-1]["RiskData"]
                    if isinstance(res, dict) and not res.get("error"):
                        return self._validate_risk_dict(res, domain_name)
                        
            except Exception as e:
                logger.warning("Error reading Mesa Datacollector for %s: %s", domain_name, e)
                
            return self._create_default_risk(default_key, "Analysis extraction failed")

        market_risks = process_result("Market", "market")
        technical_risks = process_result("Technical", "technical")
        adoption_risks = process_result("Adoption", "adoption")
        
        logger.info("✓ Completed 3-domain red-team analysis")
        
        # Aggregate risk scores
        overall_risk = self._calculate_overall_risk(
            market_risks, technical_risks, adoption_risks
        )
        
        # Map to verdict
        verdict = self._map_risk_to_gate_verdict(overall_risk)
        
        elapsed = time.time() - t0
        logger.info(
            "Red-Team Gate 4.6: overall_risk=%.2f, verdict=%s (%.1fs)",
            overall_risk, verdict.value, elapsed
        )
        
        result = GateResult(
            gate_id=self.gate_id,
            gate_name=self.gate_name,
            verdict=verdict,
            score=round(1 - overall_risk, 2),
            reasoning=self._build_reasoning(
                overall_risk, market_risks, technical_risks, adoption_risks
            ),
            details={
                "overall_risk": float(overall_risk),
                "market_risks": market_risks,
                "technical_risks": technical_risks,
                "adoption_risks": adoption_risks,
            },
        )
        
        # Cache result
        if self._enable_caching:
            self._analysis_cache[cache_key] = (result, time.time())
        
        return result
