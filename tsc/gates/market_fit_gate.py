from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import mesa
import networkx as nx
from pyDOE2 import lhs  # Latin Hypercube Sampling
from scipy.stats import norm, uniform

from tsc.gates.base import BaseGate
from tsc.llm.base import LLMClient
from tsc.models.chunks import ProblemContextBundle
from tsc.models.gates import GateResult, GateVerdict
from tsc.models.graph import KnowledgeGraph
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona

logger = logging.getLogger(__name__)

@dataclass
class AgentWorldState:
    """World state for a single agent"""
    agent_id: int
    feature_affinity: float  # 0-1, how much agent cares about this feature
    pain_points: list[str]  # Relevant pain points for this agent
    urgency_level: float     # 0-1, how urgent is this for them
    adoption_probability: float  # Base adoption probability
    device_capability: float # 0-1, can their device run this
    tech_comfort: float      # 0-1, tech savviness
    segment: str = "General" # Persona role mapping

class MarketAdoptionAgent(mesa.Agent):
    """
    Idiomatic Mesa Agent representing a stakeholder mapping to a user segment.
    Evolves over time influenced by its own baseline LLM assessment and network neighbors.
    """
    
    def __init__(
        self,
        unique_id: int,
        model: MarketAdoptionModel,
        world_state: AgentWorldState,
    ):
        super().__init__(unique_id, model)
        self.world_state = world_state
        self.segment = self._classify_segment()
        
        # State evolved during simulation
        self.base_llm_probability: float = world_state.adoption_probability
        self.adopts: bool = False
        self.confidence: float = 0.5
        self.reasoning: str = ""
        self.influence_factor = np.random.uniform(0.1, 0.4) # How much neighbors matter

    def step(self):
        """
        Mesa Step: Evaluate probability of adoption based on inherent base probability
        and the influence of networked neighbors.
        """
        
        # If already adopted, generally stays adopted in this simple model
        if self.adopts:
            return
            
        # React to neighbors
        neighbors = []
        if self.model.grid is not None:
            # NetworkGrid uses get_neighbors
            neighbor_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
            neighbors = [self.model.schedule.agents[n] for n in neighbor_nodes]
            
        adopted_neighbors = sum(1 for n in neighbors if n.adopts)
        total_neighbors = len(neighbors)
        
        neighbor_pressure = 0.0
        if total_neighbors > 0:
            neighbor_pressure = (adopted_neighbors / total_neighbors) * self.influence_factor
            
        # Current effective probability
        effective_prob = self.base_llm_probability + neighbor_pressure
        
        # Make the stochastic decision this step
        if np.random.random() < effective_prob:
            self.adopts = True
            self.confidence = min(1.0, self.confidence + 0.2)
            if not self.reasoning:
                self.reasoning = "Adopted through combined inherent fit and network influence."

    def set_llm_decision(self, adopts: bool, confidence: float, reasoning: str):
        """Called statically before the simulation steps to inject LLM baseline base."""
        if adopts:
            self.base_llm_probability = 0.7 + (confidence * 0.3)
            # If the LLM strongly said yes, they might adopt instantly on step 0
            if np.random.random() < self.base_llm_probability:
                self.adopts = True
        else:
            self.base_llm_probability = 0.0 + (confidence * 0.3)
            
        self.confidence = confidence
        self.reasoning = reasoning

    def _classify_segment(self) -> str:
        """Classify agent into user segment"""
        if self.world_state.urgency_level > 0.7:
            return "power_user"
        elif self.world_state.tech_comfort > 0.7:
            return "tech_savvy"
        elif self.world_state.feature_affinity > 0.7:
            return "motivated_user"
        else:
            return "routine_user"

class MarketAdoptionModel(mesa.Model):
    """
    Idiomatic Mesa environment for Market Adoption simulation.
    Features a NetworkX graph structure to simulate viral/neighbor adoption.
    """
    def __init__(
        self,
        num_agents: int,
        feature: FeatureProposal,
        company: CompanyContext,
        agent_states: list[AgentWorldState],
    ):
        super().__init__()
        self.num_agents = num_agents
        self.feature = feature
        self.schedule = mesa.time.RandomActivation(self)
        
        # Create a small-world network for realistic human-like clustering
        k = min(10, num_agents - 1)
        if k < 2: k = 2
        p = 0.2
        self.G = nx.watts_strogatz_graph(n=num_agents, k=k, p=p)
        
        # Validate node count (Fix 8)
        if len(self.G.nodes) != num_agents:
            logger.warning(
                "Network nodes (%d) mismatch with num_agents (%d). Scaling grid.",
                len(self.G.nodes), num_agents
            )
        
        self.grid = mesa.space.NetworkGrid(self.G)
        
        # Initialize agents and place on network
        for i, state in enumerate(agent_states):
            if i >= len(self.G.nodes):
                logger.warning("Agent %d outside network bounds, skipping placement", i)
                continue
            agent = MarketAdoptionAgent(i, self, state)
            self.schedule.add(agent)
            self.grid.place_agent(agent, i)
            
        # DataCollector for collecting MESA standard output
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "Total_Adopted": lambda m: sum(1 for a in m.schedule.agents if a.adopts),
                "Adoption_Rate": lambda m: sum(1 for a in m.schedule.agents if a.adopts) / m.num_agents if m.num_agents else 0
            },
            agent_reporters={
                "Adopts": "adopts",
                "Confidence": "confidence",
                "Segment": "segment"
            }
        )
        
        self.running = True

    def step(self):
        """Advance the simulation by one step"""
        self.schedule.step()
        self.datacollector.collect(self)
        
        # Early stopping if adoption stabilizes
        # Early stopping if adoption stabilizes
        if sum(1 for a in self.schedule.agents if a.adopts) == self.num_agents:
            self.running = False

class MonteCarloMarketFitGate(BaseGate):
    """Gate 4.5: Market Fit (MESA-powered Monte Carlo Simulation)"""
    
    gate_id = "4.5"
    gate_name = "Market Fit (Monte Carlo Simulation)"
    gate_domain = "market analysis, adoption forecasting"
    verdict_options = [
        "STRONG_FIT",
        "MODERATE_FIT",
        "WEAK_FIT",
        "NO_FIT"
    ]
    
    # Class-level mapping (Fix 2)
    _verdict_mapping = {
        "STRONG_FIT": GateVerdict.PASS,
        "MODERATE_FIT": GateVerdict.PASS_WITH_MITIGATION,
        "WEAK_FIT": GateVerdict.FEASIBLE_WITH_DEBT,
        "NO_FIT": GateVerdict.FAIL,
    }
    
    def __init__(
        self,
        llm_client: LLMClient,
        graph_store: Optional[Any] = None,
        num_agents: int = 3000,
        enable_parallel: bool = True,
    ):
        super().__init__(llm_client)
        self._graph_store = graph_store
        self._num_agents = num_agents
        self._parallel = enable_parallel
    
    async def evaluate(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
    ) -> GateResult:
        """Execute Monte Carlo market fit simulation"""
        
        t0 = time.time()
        logger.info(
            "Monte Carlo Gate 4.5: Starting %d agent simulation",
            self._num_agents
        )
        
        # Comprehensive Input Validation (Fix 5)
        self._validate_inputs(feature, company, graph, bundle, personas)
        
        external_personas = self._validate_personas_are_external(personas)
        
        # Step 1: Generate agent world states (LHS)
        agent_states = self._generate_agent_states(
            external_personas, feature, company
        )
        logger.info("✓ Generated %d agent world states", len(agent_states))
        
        # Step 2: Setup the MESA Model & Network
        model = MarketAdoptionModel(self._num_agents, feature, company, agent_states)
        self._assign_agents_to_personas(list(model.schedule.agents), external_personas)
        
        # Step 3: Pre-compute LLM decisions to inject base probabilities
        await self._inject_llm_baselines(model, feature, graph)
        logger.info("✓ Injected LLM baseline probabilities")
        
        # Step 4: Run the simulation for T steps (E.g. representing 12 months)
        SIM_STEPS = 12
        for step in range(SIM_STEPS):
            if not model.running:
                logger.info("Simulation stabilized early at step %d", step)
                break
            model.step()
            
        logger.info("✓ Completed %d MESA simulation steps", SIM_STEPS)
        
        # Step 5: Aggregate results
        results = self._aggregate_results(model, feature, company)
        
        # Step 6: Map to verdict (Fix 2 Simplification)
        verdict_str = self._map_adoption_to_verdict(results["adoption_rate"])
        verdict = self._verdict_mapping.get(verdict_str, GateVerdict.PASS)
        score = results["adoption_rate"] * 10
        
        elapsed = time.time() - t0
        logger.info(
            "Monte Carlo Gate 4.5: adoption=%.2f%%, verdict=%s (%.1fs)",
            results["adoption_rate"] * 100,
            verdict_str,
            elapsed
        )
            
        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.gate_name,
            verdict=verdict,
            score=round(score, 1),
            reasoning=self._build_reasoning(results),
            details=results,
        )

    def _validate_inputs(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
    ) -> None:
        """Validate all inputs before simulation (Fix 5)"""
        
        if not feature or not feature.title:
            raise ValueError("Feature title is required")
        
        if not feature.description or len(feature.description) < 20:
            raise ValueError(
                f"Feature description too short "
                f"({len(feature.description) if feature.description else 0} chars, "
                f"minimum 20 required)"
            )
        
        if not company or not company.company_name:
            raise ValueError("Company name is required")
        
        if not company.team_size or company.team_size < 1:
            raise ValueError(
                f"Company team_size invalid ({company.team_size}, must be >= 1)"
            )
        
        if not graph or not graph.nodes or len(graph.nodes) < 3:
            raise ValueError(
                f"Knowledge graph requires at least 3 entities "
                f"(found {len(graph.nodes) if graph and graph.nodes else 0})"
            )
        
        if not bundle or not bundle.chunks or len(bundle.chunks) < 5:
            raise ValueError(
                f"Problem context bundle requires at least 5 chunks "
                f"(found {len(bundle.chunks) if bundle and bundle.chunks else 0})"
            )
        
        if not personas:
            raise ValueError("At least 1 external persona required")
        
        logger.info("✓ All inputs validated successfully")

    def _get_high_priority_pain_points(self, feature: FeatureProposal) -> list[str]:
        return ["Major workflow blocker", "Critical security gap"]

    def _get_medium_priority_pain_points(self, feature: FeatureProposal) -> list[str]:
        return ["Inefficient process", "Slow performance"]

    def _get_low_priority_pain_points(self, feature: FeatureProposal) -> list[str]:
        return ["Minor UI annoyance"]
        
    def _get_top_pain_points(self, graph: KnowledgeGraph) -> list[str]:
        if not graph:
            return ["General pain point"]
        return [n for n, d in graph.nodes(data=True) if d.get("type") == "PAIN_POINT"][:5]

    def _validate_personas_are_external(
        self,
        personas: list[FinalPersona],
    ) -> list[FinalPersona]:
        """Ensure personas are external customer types"""
        
        external = [
            p for p in personas
            if getattr(p, "persona_type", "INTERNAL") == "EXTERNAL"
        ]
        
        if not external:
            logger.error(
                "Monte Carlo gate received %d personas, "
                "but 0 are external customer personas. "
                "Expected external personas for market fit analysis.",
                len(personas)
            )
            # Use default external persona fallback if none found
            from tsc.models.personas import PsychologicalProfile, PredictedStance, DecisionPattern
            external = [
                FinalPersona(
                    name="Default External User",
                    role="End User",
                    psychological_profile=PsychologicalProfile(
                        mbti="INTJ",
                        predicted_stance=PredictedStance(prediction="APPROVE", confidence=0.8),
                        decision_pattern=DecisionPattern(speed="Fast", risk_tolerance="High", key_drivers=[])
                    ),
                    evidence_sources=[],
                    profile_word_count=50,
                    profile_confidence=0.8,
                    persona_type="EXTERNAL"
                )
            ]
            logger.info("Using default external persona fallback for Monte Carlo.")
            
        logger.info(
            "Validated %d external personas for market analysis",
            len(external)
        )
        
        return external
    
    def _generate_agent_states(
        self,
        personas: list[FinalPersona],
        feature: FeatureProposal,
        company: CompanyContext,
    ) -> list[AgentWorldState]:
        """Generate realistic agent world states with equal persona distribution"""
        
        if not personas:
            logger.warning("No personas providing for agent generation. Using default distribution.")
            return self._generate_fallback_agent_states(feature)

        num_personas = len(personas)
        agents_per_persona = self._num_agents // num_personas
        remainder = self._num_agents % num_personas
        
        logger.info(
            "Distributing %d agents across %d personas (~%d each)",
            self._num_agents, num_personas, agents_per_persona
        )
        
        agent_states = []
        agent_id_counter = 1
        
        # Default device distribution
        device_dist = norm(loc=0.7, scale=0.15)
        
        for i, persona in enumerate(personas):
            # Calculate agents for this persona (handle remainder)
            count = agents_per_persona + (1 if i < remainder else 0)
            if count <= 0:
                continue
            
            # Build distributions specific to THIS persona
            char = self._extract_persona_characteristics([persona])
            dist = self._build_distributions_from_personas(char)
            
            affinity_dist = dist["affinity"]
            urgency_dist = dist["urgency"]
            tech_dist = dist["tech_comfort"]
            
            # Latin Hypercube Sampling for THIS persona's sub-segment
            lhs_samples = lhs(
                n=4,  # 4 dimensions
                samples=count,
                criterion='center'
            )
            
            for lhs_row in lhs_samples:
                # Convert LHS uniform samples to actual distributions
                affinity = np.clip(affinity_dist.ppf(lhs_row[0]), 0, 1)
                urgency = np.clip(urgency_dist.ppf(lhs_row[1]), 0, 1)
                device = np.clip(device_dist.ppf(lhs_row[2]), 0, 1)
                tech = np.clip(tech_dist.ppf(lhs_row[3]), 0, 1)
                
                # Determine pain points based on affinity
                if affinity > 0.7:
                    pain_points = self._get_high_priority_pain_points(feature)
                elif affinity > 0.4:
                    pain_points = self._get_medium_priority_pain_points(feature)
                else:
                    pain_points = self._get_low_priority_pain_points(feature)
                
                # Base adoption probability
                base_prob = (affinity + urgency + device) / 3
                
                agent_states.append(
                    AgentWorldState(
                        agent_id=agent_id_counter,
                        feature_affinity=float(affinity),
                        pain_points=pain_points,
                        urgency_level=float(urgency),
                        adoption_probability=float(base_prob),
                        device_capability=float(device),
                        tech_comfort=float(tech),
                        segment=persona.role, # Tag with persona role
                    )
                )
                agent_id_counter += 1
        
        logger.info(
            "Generated %d agent states across %d persona segments",
            len(agent_states), num_personas
        )
        
        return agent_states

    def _generate_fallback_agent_states(self, feature: FeatureProposal) -> list[AgentWorldState]:
        """Fallback when no personas are available"""
        # (This is just for completeness, similar to original global LHS)
        lhs_samples = lhs(n=4, samples=self._num_agents, criterion='center')
        aff_dist = norm(0.5, 0.2)
        urg_dist = norm(0.5, 0.2)
        dev_dist = norm(0.7, 0.15)
        tech_dist = norm(0.6, 0.2)
        
        states = []
        for i, row in enumerate(lhs_samples):
            states.append(AgentWorldState(
                agent_id=i + 1,
                feature_affinity=float(np.clip(aff_dist.ppf(row[0]), 0, 1)),
                pain_points=self._get_medium_priority_pain_points(feature),
                urgency_level=float(np.clip(urg_dist.ppf(row[1]), 0, 1)),
                adoption_probability=0.5,
                device_capability=float(np.clip(dev_dist.ppf(row[2]), 0, 1)),
                tech_comfort=float(np.clip(tech_dist.ppf(row[3]), 0, 1)),
            ))
        return states
    
    def _extract_persona_characteristics(
        self,
        personas: list[FinalPersona],
    ) -> dict[str, Any]:
        """Extract deterministic distribution characteristics from personas (Fix 3)"""
        
        characteristics = {
            "affinity": [],
            "urgency": [],
            "tech_comfort": [],
            "segment_names": [],
        }
        
        for persona in personas:
            try:
                profile = persona.psychological_profile
                
                # Extract affinity from predicted stance
                stance = profile.predicted_stance.prediction
                stance_confidence = profile.predicted_stance.confidence or 0.7
                
                if "APPROVE" in stance:
                    affinity = min(1.0, stance_confidence + 0.15)
                elif "CONDITIONAL" in stance:
                    affinity = stance_confidence
                else:  # REJECT
                    affinity = max(0.0, 1.0 - stance_confidence)
                
                characteristics["affinity"].append(float(affinity))
                
                # Extract urgency from decision speed
                speed = profile.decision_pattern.speed or "Balanced"
                persona_confidence = persona.profile_confidence or 0.5
                
                if "Decisive" in speed or "Fast" in speed:
                    # Scale urgency based on persona confidence
                    urgency = 0.7 + (persona_confidence * 0.25)
                else:
                    urgency = 0.3 + (persona_confidence * 0.4)
                
                characteristics["urgency"].append(float(urgency))
                
                # Extract tech comfort from MBTI
                mbti = profile.mbti or "ENTJ"
                
                if len(mbti) > 1 and mbti[1] == "N":  # Intuitive (Intuition implies novelty seeking)
                    tech_base = 0.8
                else:  # Sensing
                    tech_base = 0.55
                
                # Scale by persona confidence
                tech_comfort = tech_base + ((persona.profile_confidence or 0.5) * 0.15)
                tech_comfort = max(0.0, min(1.0, tech_comfort))
                
                characteristics["tech_comfort"].append(float(tech_comfort))
                characteristics["segment_names"].append(persona.role)
            
            except Exception as e:
                logger.warning(
                    "Failed to extract characteristics from %s: %s",
                    persona.name, e
                )
                continue
        
        return characteristics


    def _build_distributions_from_personas(
        self,
        characteristics: dict[str, Any],
    ) -> dict[str, Any]:
        """Build statistical distributions from persona characteristics"""
        
        distributions = {}
        
        # Affinity distribution
        affinity_scores = characteristics["affinity"]
        if affinity_scores:
            affinity_mean = np.mean(affinity_scores)
            affinity_std = np.std(affinity_scores) or 0.15
        else:
            affinity_mean = 0.6
            affinity_std = 0.2
        
        distributions["affinity"] = norm(
            loc=affinity_mean,
            scale=affinity_std
        )
        
        # Urgency distribution
        urgency_scores = characteristics["urgency"]
        if urgency_scores:
            urgency_mean = np.mean(urgency_scores)
            urgency_std = np.std(urgency_scores) or 0.2
        else:
            urgency_mean = 0.5
            urgency_std = 0.25
        
        distributions["urgency"] = norm(
            loc=urgency_mean,
            scale=urgency_std
        )
        
        # Tech comfort distribution
        tech_scores = characteristics["tech_comfort"]
        if tech_scores:
            tech_mean = np.mean(tech_scores)
            tech_std = np.std(tech_scores) or 0.2
        else:
            tech_mean = 0.65
            tech_std = 0.2
        
        distributions["tech_comfort"] = norm(
            loc=tech_mean,
            scale=tech_std
        )
        
        logger.info(
            "Built distributions from personas: "
            "affinity(μ=%.2f, σ=%.2f), "
            "urgency(μ=%.2f, σ=%.2f), "
            "tech(μ=%.2f, σ=%.2f)",
            affinity_mean, affinity_std,
            urgency_mean, urgency_std,
            tech_mean, tech_std,
        )
        
        return distributions

    async def _inject_llm_baselines(
        self,
        model: MarketAdoptionModel,
        feature: FeatureProposal,
        graph: KnowledgeGraph,
    ) -> None:
        """Inject LLM decisions into agents as baseline probabilities with timeout support"""
        
        agents = model.schedule.agents
        pain_points = self._get_top_pain_points(graph)
        
        # Batch LLM decisions
        batch_size = 50
        
        for i in range(0, len(agents), batch_size):
            batch = agents[i:i+batch_size]
            
            logger.info(
                "Injecting LLM baselines for agent batch %d/%d",
                i // batch_size + 1,
                (len(agents) + batch_size - 1) // batch_size,
            )
            
            tasks = [
                self._get_agent_llm_decision(
                    agent, feature, pain_points
                )
                for agent in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for agent, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "LLM decision failed for agent %d: %s",
                        agent.unique_id, result
                    )
                    # Use world state probability as fallback
                    agent.set_llm_decision(
                        adopts=False,
                        confidence=0.5,
                        reasoning="LLM failed, using fallback"
                    )
                else:
                    adopts, confidence, reasoning = result
                    agent.set_llm_decision(adopts, confidence, reasoning)

    async def _get_agent_llm_decision(
        self,
        agent: MarketAdoptionAgent,
        feature: FeatureProposal,
        pain_points: list[str],
    ) -> tuple[bool, float, str]:
        """Get LLM decision for single agent with JSON parsing (Fix 1)"""
        
        prompt = self._build_agent_decision_prompt(
            agent, feature, pain_points
        )
        
        try:
            # Use asyncio.wait_for for timeout protection (Fix 1)
            response_text = await asyncio.wait_for(
                self._llm.analyze(
                    system_prompt="You are a realistic potential user evaluating a new software feature. "
                    "Consider your specific demographic parameters to decide if you adopt the feature. "
                    "Base your decision strongly on the pain points and context provided.",
                    user_prompt=prompt,
                    temperature=0.7,
                    max_tokens=200,
                ),
                timeout=5.0
            )
            
            return self._parse_agent_llm_response(response_text, agent.unique_id)
            
        except asyncio.TimeoutError:
            logger.warning("LLM timeout for agent %d", agent.unique_id)
            raise
        except Exception as e:
            logger.warning("Agent decision LLM call failed: %s", e)
            raise

    def _parse_agent_llm_response(
        self,
        response_text: str,
        agent_id: int,
    ) -> tuple[bool, float, str]:
        """Parse and validate LLM JSON response (Fix 1)"""
        
        # If the analyze() actually returns a dict already (depends on provider implementation)
        # we check for that, otherwise we parse JSON.
        if isinstance(response_text, dict):
            response_dict = response_text
        else:
            try:
                response_dict = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning(
                    "Agent %d: Invalid JSON: %s...",
                    agent_id, str(response_text)[:100]
                )
                return (False, 0.5, "JSON parse failed")
        
        # Parse adopts (bool or string)
        adopts_raw = response_dict.get("adopts")
        if isinstance(adopts_raw, bool):
            adopts = adopts_raw
        elif isinstance(adopts_raw, str):
            adopts = adopts_raw.lower() in ("true", "yes", "1")
        else:
            adopts = False
        
        # Parse confidence (float, clipped to [0, 1])
        try:
            confidence = float(response_dict.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            logger.debug("Agent %d: Invalid confidence, using default", agent_id)
            confidence = 0.5
        
        # Parse reasoning (string, truncated to 500 chars)
        reasoning = str(response_dict.get("reasoning", ""))
        if not reasoning or len(reasoning) < 3:
            reasoning = "No reasoning provided"
        if len(reasoning) > 500:
            reasoning = reasoning[:500]
        
        logger.debug(
            "Agent %d decision: adopts=%s, confidence=%.2f",
            agent_id, adopts, confidence
        )
        
        return (adopts, confidence, reasoning)

    def _build_agent_decision_prompt(
        self,
        agent: MarketAdoptionAgent,
        feature: FeatureProposal,
        pain_points: list[str],
    ) -> str:
        """Build contextual prompt for this agent"""
        
        return f"""
AGENT CONTEXT:
- Feature Affinity: {agent.world_state.feature_affinity:.2f}/1.0
- Urgency Level: {agent.world_state.urgency_level:.2f}/1.0
- Device Capability: {agent.world_state.device_capability:.2f}/1.0
- Tech Comfort: {agent.world_state.tech_comfort:.2f}/1.0
- Segment: {agent.world_state.segment}

FEATURE: {feature.title}
Description: {feature.description[:200]}

YOUR PAIN POINTS:
{chr(10).join(f'- {p}' for p in pain_points[:3])}

DECISION:
Would you adopt {feature.title}? 
Consider your specific context and pain points.

Return JSON:
{{
  "adopts": true|false,
  "confidence": 0.0-1.0,
  "reasoning": "Why or why not?"
}}
"""

    def _assign_agents_to_personas(
        self,
        agents: list[MarketAdoptionAgent],
        personas: list[FinalPersona],
    ) -> dict[str, list[int]]:
        """Map actual simulation agents back to persona segments based on their tags"""
        
        segment_map = {}
        
        # Initialize map with persona roles
        for p in personas:
            segment_map[p.role] = []
            
        if not personas:
            segment_map["General"] = [a.unique_id for a in agents]
            return segment_map
            
        # Use the segment tag stored in the agent (set during generation)
        for agent in agents:
            segment = getattr(agent, "segment", "General")
            if segment not in segment_map:
                segment_map[segment] = []
            segment_map[segment].append(agent.unique_id)
            
        logger.info(
            "Mapped %d agents to %d segments: %s",
            len(agents),
            len(segment_map),
            {k: len(v) for k, v in segment_map.items()},
        )
        
        return segment_map

    def _aggregate_results(
        self,
        model: MarketAdoptionModel,
        feature: FeatureProposal,
        company: CompanyContext,
    ) -> dict[str, Any]:
        """Aggregate adoption decisions into statistics with real percentiles (Fix 4)"""
        
        try:
            df = model.datacollector.get_agent_vars_dataframe()
        except Exception as e:
            logger.error("Failed to get DataCollector: %s", e)
            return self._empty_results()
            
        if df.empty or self._num_agents == 0:
            return self._empty_results()
        
        # Get final state (max step)
        last_step_idx = df.index.get_level_values('Step').max()
        if pd.isna(last_step_idx):
            return self._empty_results()
            
        final_df = df.xs(last_step_idx, level="Step")
        
        # Extract binary adoption decisions
        adoptions = final_df["Adopts"].astype(int).tolist()
        adoption_rate = np.mean(adoptions)
        
        # Real percentiles from population distribution (Fix 4)
        adoptions_sorted = sorted(adoptions)
        adoption_p10 = np.percentile(adoptions_sorted, 10)
        adoption_p50 = np.percentile(adoptions_sorted, 50)
        adoption_p90 = np.percentile(adoptions_sorted, 90)
        
        # Segment breakdown
        segment_rates = final_df.groupby("Segment")["Adopts"].mean().to_dict()
        segment_breakdown = {str(k): float(v) for k, v in segment_rates.items()}
        
        # Business impact (Fix 11)
        business_impact = self._calculate_business_impact(
            adoption_rate, feature, company
        )
        
        return {
            "adoption_rate": float(adoption_rate),
            "adoption_p10": float(adoption_p10),
            "adoption_p50": float(adoption_p50),
            "adoption_p90": float(adoption_p90),
            "num_agents": self._num_agents,
            "segment_breakdown": segment_breakdown,
            "business_impact": business_impact,
            "variance": float(np.var(adoptions)),
        }

    def _calculate_business_impact(
        self,
        adoption_rate: float,
        feature: FeatureProposal,
        company: CompanyContext,
    ) -> dict[str, Any]:
        """Calculate ROI based on real business inputs (Fix 11)"""
        
        user_count = getattr(feature, 'target_user_count', 10000) or 10000
        
        # ARPU from feature context if available
        arpu_increase = 500  # Default
        if hasattr(feature, 'business_context') and feature.business_context:
            arpu_increase = feature.business_context.get("target_arpu", 500)
            
        # Implementation cost from effort weeks
        effort_weeks = feature.effort_weeks_max or 4
        hourly_rate = 150
        implementation_cost = effort_weeks * 40 * hourly_rate
        
        expected_revenue = adoption_rate * user_count * arpu_increase
        expected_value = expected_revenue - implementation_cost
        roi = expected_value / implementation_cost if implementation_cost > 0 else 0
        
        return {
            "expected_revenue": float(expected_revenue),
            "implementation_cost": float(implementation_cost),
            "expected_value": float(expected_value),
            "roi": float(roi),
        }

    def _empty_results(self) -> dict[str, Any]:
        """Return empty result set"""
        return {
            "adoption_rate": 0.0,
            "adoption_p10": 0.0,
            "adoption_p50": 0.0,
            "adoption_p90": 0.0,
            "num_agents": self._num_agents,
            "segment_breakdown": {},
            "business_impact": {
                "expected_revenue": 0.0,
                "implementation_cost": 0.0,
                "expected_value": 0.0,
                "roi": 0.0,
            },
            "variance": 0.0,
        }
    
    def _map_adoption_to_verdict(self, adoption_rate: float) -> str:
        """Map adoption rate to verdict"""
        
        if adoption_rate >= 0.70:
            return "STRONG_FIT"
        elif adoption_rate >= 0.50:
            return "MODERATE_FIT"
        elif adoption_rate >= 0.30:
            return "WEAK_FIT"
        else:
            return "NO_FIT"
    
    def _build_reasoning(self, results: dict[str, Any]) -> str:
        """Build detailed reasoning"""
        
        if results['num_agents'] == 0:
            return "No agents simulated."
            
        return (
            f"Monte Carlo simulation with {results['num_agents']} agents: "
            f"{results['adoption_rate']:.1%} adoption rate "
            f"(p10={results['adoption_p10']:.1%}, p90={results['adoption_p90']:.1%}). "
            f"Expected ROI: {results['business_impact']['roi']:.2f}x. "
            f"Strong adoption in: {', '.join(k for k, v in results['segment_breakdown'].items() if v > 0.60)}"
        )
    