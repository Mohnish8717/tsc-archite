import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import random
from tsc.models.personas import FinalPersona
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.graph import KnowledgeGraph
from tsc.oasis.models import OpinionVector, OASISAgentProfile, OASISSimulationConfig, UserInfoAdapter

logger = logging.getLogger(__name__)

async def BuildBeliefVector(
    persona: FinalPersona,
    feature: FeatureProposal,
    company_context: CompanyContext,
    kg: Optional[KnowledgeGraph] = None,
    handle_edge_cases: bool = True
) -> Tuple[OpinionVector, str]:
    """
    Build belief vector from persona evidence (TSC Legacy support).
    
    Returns:
        (OpinionVector, edge_case_label)
    """
    logger.debug(f"Building belief vector for persona {persona.name}")
    
    # Check for edge cases first
    edge_case_label = ""
    if handle_edge_cases:
        edge_case = _detect_edge_cases(persona, feature)
        if edge_case:
            logger.warning(f"Edge case detected for {persona.name}: {edge_case}")
            edge_case_label = edge_case
    
    print(f"  Extracting tech feasibility...")
    tech_feasibility = _extract_technical_feasibility(persona, feature, company_context)
    print(f"  Extracting market demand...")
    market_demand = _extract_market_demand(persona, feature)
    resource_alignment = _extract_resource_alignment(persona, company_context)
    risk_tolerance = _extract_risk_tolerance(persona)
    adoption_velocity = _extract_adoption_velocity(persona, feature, market_demand)
    
    # Calibrate confidence
    evidence_count = len(persona.evidence_sources) if hasattr(persona, "evidence_sources") else 0
    confidence = _calibrate_confidence(
        persona=persona,
        evidence_count=evidence_count,
        grounding_score=persona.profile_confidence 
    )
    
    belief = OpinionVector(
        technical_feasibility=float(np.clip(tech_feasibility, -1.0, 1.0)),
        market_demand=float(np.clip(market_demand, -1.0, 1.0)),
        resource_alignment=float(np.clip(resource_alignment, -1.0, 1.0)),
        risk_tolerance=float(np.clip(risk_tolerance, -1.0, 1.0)),
        adoption_velocity=float(np.clip(adoption_velocity, -1.0, 1.0)),
        confidence=confidence,
        source_persona_id=persona.name,
        evidence_count=evidence_count,
    )
    
    return belief, edge_case_label

def _extract_technical_feasibility(
    persona: FinalPersona,
    feature: FeatureProposal,
    company_context: CompanyContext
) -> float:
    """Extract technical feasibility dimension from evidence."""
    score = 0.0
    tech_keywords_positive = ["aligns", "compatible", "simple", "integrates", "straightforward", "easy"]
    tech_keywords_negative = ["requires rewrite", "incompatible", "complex", "debt", "unstable", "legacy"]
    
    source_text = persona.psychological_profile.full_profile_text
    evidence_lower = source_text.lower()
    
    for keyword in tech_keywords_positive:
        if keyword in evidence_lower:
            score += 0.2
    for keyword in tech_keywords_negative:
        if keyword in evidence_lower:
            score -= 0.3
    
    # MBTI boost/penalty
    mbti = persona.psychological_profile.mbti
    if mbti and len(mbti) >= 3 and mbti[2] == 'T':  # Thinking
        score += 0.1
    
    return score

def _extract_market_demand(persona: FinalPersona, feature: FeatureProposal) -> float:
    """Extract market demand dimension."""
    drivers = persona.psychological_profile.emotional_triggers.excited_by
    pain_points = persona.psychological_profile.emotional_triggers.frustrated_by
    
    driver_count = len(drivers)
    pain_point_count = len(pain_points)
    
    if driver_count + pain_point_count == 0:
        demand_score = 0.1
    else:
        demand_score = (driver_count - pain_point_count) / (driver_count + pain_point_count + 1)
    
    if persona.persona_type == "EXTERNAL":
        demand_score += 0.2
    
    return demand_score

def _extract_resource_alignment(
    persona: FinalPersona,
    company_context: CompanyContext
) -> float:
    """Extract resource alignment dimension."""
    score = 0.0
    if company_context.team_size > 50:
        score += 0.2
    if "Python" in str(company_context.tech_stack):
        score += 0.1
    return score

def _extract_risk_tolerance(persona: FinalPersona) -> float:
    """Extract risk tolerance from MBTI and evidence."""
    score = 0.0
    mbti = persona.psychological_profile.mbti
    if mbti and len(mbti) >= 4:
        if mbti[3] == 'P':  # Perceiving
            score += 0.15
        else:  # Judging
            score -= 0.15
    return score

def _extract_adoption_velocity(
    persona: FinalPersona,
    feature: FeatureProposal,
    market_demand: float
) -> float:
    """Extract adoption velocity dimension."""
    score = market_demand * 0.4
    if feature.effort_weeks_max and feature.effort_weeks_max < 4:
        score += 0.2
    return score

def _calibrate_confidence(
    persona: FinalPersona,
    evidence_count: int,
    grounding_score: float
) -> float:
    """Calibrate confidence based on evidence quality and quantity."""
    confidence = 0.4
    if evidence_count > 0:
        confidence += min(0.3, evidence_count * 0.05)
    if grounding_score > 0.7:
        confidence += 0.2
    return float(np.clip(confidence, 0.0, 1.0))

def _detect_edge_cases(persona: FinalPersona, feature: FeatureProposal) -> Optional[str]:
    """Detect and label edge cases."""
    if persona.profile_word_count < 100:
        return "sparse_persona_data"
    mbti = persona.psychological_profile.mbti
    if not mbti or mbti == "XXXX" or len(mbti) < 4:
        return "missing_mbti"
    return None

async def InitializeOASISAgents(
    personas: List[FinalPersona],
    feature: FeatureProposal,
    context: CompanyContext,
    config: OASISSimulationConfig,
    handle_edge_cases: bool = True,
    kg: Optional[KnowledgeGraph] = None,
    market_context: Optional[Dict[str, Any]] = None
) -> Tuple[List[OASISAgentProfile], List[str]]:
    """Initialize Actual OASIS agents from personas with resampling."""
    agents = []
    edge_cases_triggered = []
    
    for i, persona in enumerate(personas):
        belief, edge_case = await BuildBeliefVector(
            persona=persona,
            feature=feature,
            company_context=context,
            kg=kg,
            handle_edge_cases=handle_edge_cases
        )
        if edge_case:
            edge_cases_triggered.append(f"{persona.name}:{edge_case}")
            
        # MiroFish Optimization: Enrich bio with market context before mapping
        original_bio = persona.psychological_profile.full_profile_text
        grounding_context = ""
        if market_context:
            grounding_context = "\n### Market Environment Context\n"
            if market_context.get("competitors"):
                grounding_context += f"- Competitors: {', '.join(market_context['competitors'])}\n"
            if market_context.get("geography"):
                grounding_context += f"- Target Geography: {', '.join(market_context['geography'])}\n"
            if market_context.get("pricing_tiers"):
                grounding_context += f"- Current Pricing Tiers: {', '.join(market_context['pricing_tiers'])}\n"
            
            # MiroFish Integration: Add company priorities
            if context and context.current_priorities:
                grounding_context += f"- Company Priorities: {', '.join(context.current_priorities)}\n"
            
            # Temporarily prepend to bio for UserInfoAdapter
            persona.psychological_profile.full_profile_text = grounding_context + "\n" + original_bio

        # Map to CAMEL-AI UserInfo
        print(f"  Mapping to OASIS user info...")
        user_info = UserInfoAdapter.to_oasis_user_info(persona, config.platform_type)
        
        # Restore original bio
        persona.psychological_profile.full_profile_text = original_bio
        
        agent = OASISAgentProfile(
            agent_id=i,
            source_persona_id=persona.name,
            agent_type=persona.persona_type,
            user_info_dict=user_info if isinstance(user_info, dict) else (user_info.dict() if hasattr(user_info, "dict") else user_info.__dict__),
            initial_belief=belief,
            influence_strength=persona.influence_strength,
            receptiveness=persona.receptiveness,
        )
        agents.append(agent)
    
    # Resample to target count (MiroFish style large-scale simulation)
    if len(agents) < config.num_agents and len(agents) > 0:
        agents = _resample_agents(agents, config.num_agents)
    
    logger.info(f"Initialized {len(agents)} agents for actual OASIS simulation")
    return agents, edge_cases_triggered

def _resample_agents(agents: List[OASISAgentProfile], target_count: int) -> List[OASISAgentProfile]:
    """Resample agents to reach target count (preserving persona distribution)."""
    original_count = len(agents)
    missing = target_count - original_count
    for i in range(missing):
        source = random.choice(agents[:original_count])
        # Clone with new ID
        new_agent = OASISAgentProfile(
            agent_id=original_count + i,
            source_persona_id=source.source_persona_id,
            agent_type=source.agent_type,
            user_info_dict=source.user_info_dict.copy(),
            initial_belief=source.initial_belief,
            influence_strength=source.influence_strength,
            receptiveness=source.receptiveness,
        )
        # Update username to ensure uniqueness in DB
        new_agent.user_info_dict['user_name'] += f"_{i}"
        agents.append(new_agent)
    return agents
