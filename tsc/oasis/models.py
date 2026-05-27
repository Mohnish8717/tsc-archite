from __future__ import annotations
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tsc.models.personas import FinalPersona
    from oasis.social_platform.config.user import UserInfo

class OpinionVector(BaseModel):
    """Multi-dimensional opinion of agent on feature proposal (TSC Legacy)."""
    technical_feasibility: float = Field(..., ge=-1.0, le=1.0)
    market_demand: float = Field(..., ge=-1.0, le=1.0)
    resource_alignment: float = Field(..., ge=-1.0, le=1.0)
    risk_tolerance: float = Field(..., ge=-1.0, le=1.0)
    adoption_velocity: float = Field(..., ge=-1.0, le=1.0)
    
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_persona_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    evidence_count: int = 0
    
    def magnitude(self) -> float:
        """Euclidean norm of opinion vector."""
        dims = [
            self.technical_feasibility,
            self.market_demand,
            self.resource_alignment,
            self.risk_tolerance,
            self.adoption_velocity
        ]
        return math.sqrt(sum(d**2 for d in dims)) / math.sqrt(5)

class UserInfoAdapter:
    """Adapts TSC FinalPersona to CAMEL-AI OASIS UserInfo."""
    
    @staticmethod
    def to_oasis_user_info(persona: FinalPersona, recsys_type: str = "reddit") -> Dict[str, Any]:
        """Constructs the dictionary expected by OASIS SocialAgent.
        
        For EXTERNAL personas, market_context and buyer_journey are injected
        into other_info so the CAMEL-AI agent reasons from a buyer's economic
        perspective rather than only from psychological friction signals.
        """
        # Note: We return a dict instead of constructor UserInfo object to avoid
        # deadlocks caused by heavy library imports (torch, grpc) during the preparation phase.

        pp = persona.psychological_profile

        # ── Structured Emotional Triggers ─────────────────────────────────────
        et = pp.emotional_triggers
        emotional_triggers_dict: Dict[str, Any] = {
            "excited_by": et.excited_by if et else [],
            "frustrated_by": et.frustrated_by if et else [],
            "scared_of": et.scared_of if et else [],
        }

        # ── Communication Style ───────────────────────────────────────────────
        cs = pp.communication_style
        communication_style_dict: Dict[str, Any] = {
            "default": cs.default if cs else "Direct",
            "formality": cs.formality if cs else "Semi-formal",
            "conflict_handling": cs.conflict_handling if cs else "Pragmatic",
            "preferred_channels": cs.preferred_channels if cs else [],
        }

        # ── Decision Pattern ──────────────────────────────────────────────────
        dp = pp.decision_pattern
        decision_pattern_dict: Dict[str, Any] = {
            "speed": dp.speed if dp else "Moderate",
            "preference": dp.preference if dp else "Data-driven",
            "influencers": dp.influencers if dp else [],
            "justification": dp.justification if dp else "",
            "risk_tolerance": dp.risk_tolerance if dp else "Medium",
        }

        # ── Predicted Stance ──────────────────────────────────────────────────
        ps = pp.predicted_stance
        predicted_stance_dict: Dict[str, Any] = {
            "feature": ps.feature if ps else "",
            "prediction": ps.prediction if ps else "",
            "confidence": ps.confidence if ps else 0.0,
            "likely_conditions": ps.likely_conditions if ps else [],
            "potential_objections": ps.potential_objections if ps else [],
        }

        other_info: Dict[str, Any] = {
            "role": persona.role,
            "traits": pp.key_traits,
            "user_profile": pp.full_profile_text[:2000],
            "gender": getattr(persona, 'gender', 'unknown'),
            "age": getattr(persona, 'age', 30),
            "mbti": pp.mbti,
            "mbti_description": pp.mbti_description,
            "ocean_scores": pp.ocean_scores,
            "country": getattr(persona, 'country', 'US'),
            # Structured psychological fields
            "emotional_triggers": emotional_triggers_dict,
            "communication_style": communication_style_dict,
            "decision_pattern": decision_pattern_dict,
            "predicted_stance": predicted_stance_dict,
            "questions_they_will_ask": pp.questions_they_will_ask,
            # FinalPersona metadata
            "domain_expertise": persona.domain_expertise,
            "profile_confidence": persona.profile_confidence,
            "grounding_quality": persona.grounding_quality,
            "persona_type": persona.persona_type,
            "network_position_hint": persona.network_position_hint,
            "influence_strength": persona.influence_strength,
            "receptiveness": persona.receptiveness,
            "evidence_sources": persona.evidence_sources,
        }

        # Inject market/buyer context for EXTERNAL personas
        mc = getattr(persona, "market_context", None)
        bj = getattr(persona, "buyer_journey", None)
        if mc is not None:
            other_info["market_context"] = {
                "company_size_band": mc.company_size_band,
                "buyer_role": mc.buyer_role,
                "annual_solution_budget_usd": mc.annual_solution_budget_usd,
                "pricing_sensitivity": mc.pricing_sensitivity,
                "sales_cycle_weeks": mc.sales_cycle_weeks,
                "deployment_preference": mc.deployment_preference,
                "industry_vertical": mc.industry_vertical,
                "regulatory_burden": mc.regulatory_burden,
            }
        if bj is not None:
            other_info["buyer_journey"] = {
                "awareness_channel": bj.awareness_channel,
                "evaluation_trigger": bj.evaluation_trigger,
                "key_proof_points": bj.key_proof_points,
                "deal_breakers": bj.deal_breakers,
                "success_metric": bj.success_metric,
                "roi_threshold_months": bj.roi_threshold_months,
                "willingness_to_pay_band": bj.willingness_to_pay_band,
            }

        profile_data = {
            "user_profile": pp.full_profile_text[:2000],
            "gender": getattr(persona, 'gender', 'unknown'),
            "age": getattr(persona, 'age', 30),
            "mbti": pp.mbti,
            "country": getattr(persona, 'country', 'US'),
            "other_info": other_info,
        }

        return {
            "user_name": persona.name.lower().replace(" ", "_"),
            "name": persona.name,
            "description": persona.psychological_profile.mbti_description,
            "profile": profile_data,
            "recsys_type": recsys_type
        }

class OASISAgentProfile(BaseModel):
    """Bridge between TSC Persona and CAMEL-AI SocialAgent."""
    agent_id: int
    source_persona_id: str
    agent_type: str  # "internal_stakeholder" | "customer_segment"
    
    # Required for CAMEL-AI SocialAgent
    user_info_dict: Dict[str, Any] 
    
    # Legacy metrics tracking (Optional)
    initial_belief: Optional[OpinionVector] = None
    current_belief: Optional[OpinionVector] = None
    
    # Metadata for prediction reporting
    influence_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    receptiveness: float = Field(default=0.5, ge=0.0, le=1.0)

class BeliefCluster(BaseModel):
    """Grouping of agents with similar behavioral patterns in OASIS."""
    cluster_id: str
    centroid_behavior: str
    members: List[str]  # Agent IDs
    cluster_size: int
    dominant_persona_type: str
    sentiment_score: float = 0.5 # Derived from traces/interviews

class OASISSimulationConfig(BaseModel):
    """Configuration for Actual CAMEL-AI OASIS simulation."""
    simulation_name: str
    platform_type: str = "reddit" # "twitter" | "reddit"
    num_agents: int = 150
    num_timesteps: int = 10 # 1 timestep = 1 hour usually
    simulation_speed: int = 60 # Clock magnification
    
    db_path: str = ":memory:"
    enable_graph_memory: bool = True
    
    # Discovery/Interview config
    enable_interview_phase: bool = True
    interview_sample_size: int = 30
    interview_prompts: List[str] = [
        # 1. Behavioral anchoring — forces specific, not abstract, recall
        "Walk me through the LAST TIME you encountered the exact problem this feature is supposed to solve. "
        "What did you actually do? What did it cost you in time, money, or reputation?",
        # 2. Adoption ladder — extracts realistic commitment level with price threshold
        "On a scale of 1-10, how likely are you to use this in your first week after launch? "
        "What would need to be true for that number to be a 9 or 10? What would make it a 1?",
        # 3. WTP — structured pricing probe extracting three numbers
        "If this feature required an additional monthly charge, what price would make you say "
        "'definitely yes', 'maybe', and 'definitely no'? Give me three specific numbers.",
        # 4. Risk surfacing — open-ended objection probe
        "What is the ONE thing about this feature that, if it went wrong, would make you actively "
        "recommend against it to colleagues? How likely do you think that failure is?",
        # 5. Competitive exit vector — identifies displacement risk
        "If this feature ships as described and does not work for your workflow, what is your next move? "
        "Which alternative would you look at first, and why that one specifically?",
        # 6. Advocacy signal — NPS behavioral proxy (not a rating)
        "Is there someone on your team you would forward this announcement to right now? "
        "What would you say to them in your own words — not a summary, your actual message?",
    ]

class MarketSentimentSeries(BaseModel):
    """Time-series output derived from OASIS traces and interviews."""
    simulation_id: str
    feature_proposal_id: str
    
    timesteps: List[int] = []
    adoption_rate_cumulative: List[float] = []
    sentiment_volatility: List[float] = []
    
    # Aggregated Insights
    final_adoption_score: float = 0.0 # 0.0 to 1.0
    consensus_verdict: str = "NEUTRAL"
    key_objections: List[str] = Field(default_factory=list)
    segment_breakdown: List[BeliefCluster] = Field(default_factory=list)
    
    # Behavioral Clustering & Consensus Extensions
    belief_clusters: List[BeliefCluster] = Field(default_factory=list)
    consensus_strength: float = 0.0
    consensus_type: str = "fragmented"
    convergence_reached: bool = False
    raw_responses: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Agent-Specific Drilldown
    agent_interactions: Dict[str, List[str]] = Field(default_factory=dict)
    agent_alignment: Dict[str, float] = Field(default_factory=dict)
    
    # LLM Aggregate Analysis
    aggregate_analysis: Optional[str] = None
    population_size: int = 0
    focus_group_insights: Dict[str, Any] = Field(default_factory=dict)
    
    # Raw Data pointers
    db_snapshot_path: Optional[str] = None
    trace_log_path: Optional[str] = None

class SimulationStatus(str, Enum):
    """Broad lifecycle states for a simulation, including preparation."""
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"

class SimulationMetadata(BaseModel):
    """Metadata for simulation preparation and project context."""
    simulation_id: str
    project_id: str
    graph_id: str
    
    # Preparation Stats
    entities_count: int = 0
    profiles_count: int = 0
    entity_types: List[str] = Field(default_factory=list)
    
    # Platform Config
    enable_twitter: bool = True
    enable_reddit: bool = True
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Preparation reasoning from LLM
    config_reasoning: Optional[str] = None

class SimulationParameters(BaseModel):
    """Result of LLM config generation for OASIS."""
    num_agents: int
    num_timesteps: int
    platform_type: str
    interview_prompts: List[str]
    generation_reasoning: Optional[str] = None

class RunnerStatus(str, Enum):
    """Lifecycle states for the SimulationRunner."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentAction(BaseModel):
    """Detailed record of a single agent action in the simulation."""
    timestep: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    agent_name: str
    action_type: str
    content: Any
    platform: str = "reddit" # twitter | reddit
    success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RoundSummary(BaseModel):
    """Summary of activity within a single simulation timestep."""
    timestep: int
    start_time: datetime
    end_time: Optional[datetime] = None
    actions_count: int = 0
    active_agents: List[str] = Field(default_factory=list)

class SimulationRunState(BaseModel):
    """Persistent snapshot of a simulation's progress and health."""
    simulation_id: str
    status: RunnerStatus = RunnerStatus.IDLE
    
    # Progress
    current_timestep: int = 0
    total_timesteps: int = 0
    percent_complete: float = 0.0
    
    # Platform Tracking
    platforms_active: List[str] = Field(default_factory=list)
    platform_completion: Dict[str, bool] = Field(default_factory=dict)
    
    # Action Buffer
    recent_actions: List[AgentAction] = Field(default_factory=list, max_length=50)
    total_actions_count: int = 0
    
    # Timestamps
    started_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Failure diagnostics
    error: Optional[str] = None
    process_pid: Optional[int] = None

    def add_action(self, action: AgentAction):
        """Add action to recent buffer and update totals."""
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > 50:
            self.recent_actions.pop()
        self.total_actions_count += 1
        self.updated_at = datetime.utcnow()
        if self.total_timesteps > 0:
            self.percent_complete = round((action.timestep + 1) / self.total_timesteps * 100, 2)


# ═══════════════════════════════════════════════════════════════════════
# PREDICTIVE REALITY ENGINE: Decision Journal + Prediction Report
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DecisionJournal:
    """Per-agent behavioral state machine that evolves across timesteps.
    
    Accumulates Game Master signals into a running state vector.
    When thresholds are crossed, decision events are triggered.
    All labels are DYNAMIC — discovered from simulation data, not hardcoded.
    """
    agent_id: str
    agent_name: str
    segment_source: str = ""  # From persona profile (e.g. occupation)
    
    # ── Running State Vector (updated by GM signals) ──
    # Generic dimensions that work for ANY product/feature:
    satisfaction: float = 0.5     # Product satisfaction (0=hate, 1=love)
    frustration: float = 0.0     # Accumulated friction
    trust: float = 0.5           # Trust in the company/product
    urgency: float = 0.0         # How urgently they want to act
    advocacy: float = 0.0        # Would they recommend? (NPS proxy)
    
    # ── Signal History (append-only log) ──
    signals: list = field(default_factory=list)
    # Each: {"timestep": 2, "type": "exit_intent", "intensity": -0.8,
    #         "quote": "I'm switching to Teams", "factors": ["privacy"]}
    
    # ── Decision Events (threshold-triggered) ──
    decisions: list = field(default_factory=list)
    # Each: {"timestep": 3, "decision": "HIGH_RISK", "confidence": 0.85,
    #         "trigger": "frustration > 0.75", "factors": [...]}
    
    # ── Telemetry (from persona profile — grounding) ──
    tenure_months: int = 0
    team_size: int = 1
    monthly_spend: float = 0.0
    
    def update_from_signal(self, signal: dict):
        """Apply a GM signal to the state vector."""
        self.signals.append(signal)
        
        if "satisfaction_delta" in signal:
            # SOTA structured LLM classifier directly provides the deltas!
            self.satisfaction = max(0.0, min(1.0, self.satisfaction + signal["satisfaction_delta"]))
            self.frustration = max(0.0, min(1.0, self.frustration + signal["frustration_delta"]))
            self.trust = max(0.0, min(1.0, self.trust + signal["trust_delta"]))
            
            adv_state = signal.get("primary_advocacy_state", "").lower()
            if "promoter" in adv_state:
                self.advocacy = min(1.0, self.advocacy + 0.15)
            elif "detractor" in adv_state:
                self.advocacy = max(0.0, self.advocacy - 0.15)
        else:
            intensity = signal.get("intensity", 0.0)
            
            # State vector update rules (domain-agnostic)
            if intensity < -0.3:
                self.frustration = min(1.0, self.frustration + abs(intensity) * 0.3)
                self.satisfaction = max(0.0, self.satisfaction + intensity * 0.2)
                self.trust = max(0.0, self.trust + intensity * 0.15)
            elif intensity > 0.3:
                self.satisfaction = min(1.0, self.satisfaction + intensity * 0.2)
                self.advocacy = min(1.0, self.advocacy + intensity * 0.15)
                self.frustration = max(0.0, self.frustration - intensity * 0.1)
        
        # Urgency tracks magnitude of recent signals
        recent = self.signals[-3:] if len(self.signals) >= 3 else self.signals
        self.urgency = min(1.0, sum(abs(s.get("intensity", 0)) for s in recent) / 3.0)
        
        # Threshold-based decision events
        ts = signal.get("timestep", 0)
        if self.frustration > 0.75 and not self._has_decision("HIGH_RISK"):
            self.decisions.append({
                "timestep": ts, "decision": "HIGH_RISK",
                "confidence": round(self.frustration, 2),
                "trigger": f"frustration={self.frustration:.2f}",
                "factors": list({s["type"] for s in self.signals[-5:]}),
            })
        if self.satisfaction > 0.75 and self.advocacy > 0.5 and not self._has_decision("CHAMPION"):
            self.decisions.append({
                "timestep": ts, "decision": "CHAMPION",
                "confidence": round(self.satisfaction, 2),
                "trigger": f"satisfaction={self.satisfaction:.2f}, advocacy={self.advocacy:.2f}",
                "factors": list({s["type"] for s in self.signals[-5:]}),
            })
    
    def _has_decision(self, decision_type: str) -> bool:
        return any(d["decision"] == decision_type for d in self.decisions)
    
    def state_vector(self) -> list:
        """Return numerical state for clustering."""
        return [self.satisfaction, self.frustration, self.trust, self.urgency, self.advocacy]
    
    def prompt_summary(self) -> str:
        """Generate a compact summary with state-driven action directive for agent prompt."""
        trend_sat = "↓" if len(self.signals) > 2 and self.satisfaction < 0.4 else "↑" if self.satisfaction > 0.6 else "→"
        trend_fru = "↑" if self.frustration > 0.5 else "→"

        lines = [
            f"\nYOUR DECISION JOURNAL (internal state):",
            f"  Satisfaction: {self.satisfaction:.2f} ({trend_sat})",
            f"  Frustration:  {self.frustration:.2f} ({trend_fru})",
            f"  Trust:        {self.trust:.2f}",
        ]
        if self.signals:
            recent_types = [s["type"] for s in self.signals[-3:]]
            lines.append(f"  Recent signals: {', '.join(recent_types)}")

        # State-driven action directive — tells the agent WHAT to do, not just what it feels
        if self.frustration > 0.75:
            lines.append(
                f"  🚨 ACTION: Your frustration is CRITICAL. Express this frustration explicitly, "
                f"naming the specific pain point driving it. Do not soften your language."
            )
        elif self.frustration > 0.5:
            lines.append(
                f"  ⚠️ ACTION: You are frustrated. Ask a pointed question or demand a concrete "
                f"answer — timeline, fix, or roadmap date — before you accept any reassurance."
            )
        elif self.trust < 0.3:
            lines.append(
                f"  🔍 ACTION: You do not trust the company right now. Challenge any claim "
                f"without evidence. Ask for proof, data, or a public commitment."
            )
        elif self.advocacy > 0.6:
            lines.append(
                f"  ✅ ACTION: You are an advocate. Proactively recommend this to someone "
                f"in the thread using concrete reasons from your own workflow — not generic praise."
            )
        elif self.satisfaction > 0.7:
            lines.append(
                f"  💬 ACTION: You are satisfied. Add a constructive perspective that builds "
                f"on the discussion — what could make this even better for your specific use case?"
            )

        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "segment_source": self.segment_source,
            "satisfaction": round(self.satisfaction, 3),
            "frustration": round(self.frustration, 3),
            "trust": round(self.trust, 3),
            "urgency": round(self.urgency, 3),
            "advocacy": round(self.advocacy, 3),
            "state": {"satisfaction": round(self.satisfaction, 3),
                      "frustration": round(self.frustration, 3),
                      "trust": round(self.trust, 3),
                      "urgency": round(self.urgency, 3),
                      "advocacy": round(self.advocacy, 3)},
            "signal_count": len(self.signals),
            "signals": self.signals[-10:],  # Last 10 for forensics
            "decisions": self.decisions,
            "telemetry": {"tenure_months": self.tenure_months,
                          "team_size": self.team_size,
                          "monthly_spend": self.monthly_spend},
        }


class PredictionReport(BaseModel):
    """Quantitative output from the Predictive Reality Engine.
    
    All metrics are COMPUTED from simulation data — no hardcoded labels.
    Segments are discovered dynamically via clustering.
    """
    simulation_id: str
    feature_title: str = ""
    population_size: int = 0
    timesteps_completed: int = 0
    
    # ── Dynamic Segments (from clustering) ──
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    # Each: {"name": "LLM-generated", "size": 12, "pct": 0.24,
    #         "avg_satisfaction": 0.3, "avg_frustration": 0.7, ...}
    
    # ── Risk Distribution ──
    risk_distribution: Dict[str, float] = Field(default_factory=dict)
    # {"HIGH_RISK": 0.23, "MODERATE": 0.45, "LOW_RISK": 0.32}
    
    # ── Time-Series Curves ──
    satisfaction_curve: List[float] = Field(default_factory=list)
    frustration_curve: List[float] = Field(default_factory=list)
    trust_curve: List[float] = Field(default_factory=list)
    
    # ── Derived Business Metrics ──
    net_promoter_score: float = 0.0
    churn_velocity: float = 0.0       # Rate of frustration increase
    adoption_momentum: float = 0.0    # Rate of satisfaction increase
    
    # ── Decision Events ──
    decision_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # ── Top Risk Factors ──
    top_risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    # [{"factor": "privacy", "frequency": 0.34}, ...]
    
    # ── LLM Executive Summary ──
    executive_summary: str = ""
    
    # ── Focus Group Insights (Phase 2) ──
    focus_group_insights: Dict[str, Any] = Field(default_factory=dict)
    
    # ── Per-Agent Journals ──
    agent_journals: List[Dict[str, Any]] = Field(default_factory=list)
