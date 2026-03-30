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
        """Constructs the dictionary expected by OASIS SocialAgent."""
        # Note: We return a dict instead of constructor UserInfo object to avoid
        # deadlocks caused by heavy library imports (torch, grpc) during the preparation phase.
        
        profile_data = {
            "user_profile": persona.psychological_profile.full_profile_text[:1000],
            "gender": getattr(persona, 'gender', 'unknown'),
            "age": getattr(persona, 'age', 30),
            "mbti": persona.psychological_profile.mbti,
            "country": getattr(persona, 'country', 'US'),
            "other_info": {
                "role": persona.role,
                "traits": persona.psychological_profile.key_traits,
                "user_profile": persona.psychological_profile.full_profile_text[:2000],
                "gender": getattr(persona, 'gender', 'unknown'),
                "age": getattr(persona, 'age', 30),
                "mbti": persona.psychological_profile.mbti,
                "country": getattr(persona, 'country', 'US'),
            }
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
    num_timesteps: int = 24 # 1 timestep = 1 hour usually in MiroFish
    simulation_speed: int = 60 # Clock magnification
    
    db_path: str = ":memory:"
    enable_graph_memory: bool = True
    
    # Discovery/Interview config
    interview_prompts: List[str] = [
        "What is your overall opinion on the proposed feature?",
        "What are the main risks you see in this implementation?",
        "Would you use this feature in your daily workflow?"
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
