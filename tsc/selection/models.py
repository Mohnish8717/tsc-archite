"""
Data models for the Persona Selection Engine.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PoleType(str, Enum):
    ATTRACTOR = "ATTRACTOR"  # score > +0.3  (adopter)
    REPELLER  = "REPELLER"   # score < -0.3  (critic)
    SWING     = "SWING"      # [-0.3, +0.3]  (persuadable / most valuable)


class PersonaCategory(str, Enum):
    """
    Priority-matrix category for a persona.

    PRIMARY     — ~70%  Direct daily users (ICP). Their friction score matters most.
    STAKEHOLDER — ~20%  Decision-makers who don't use the feature but control its fate.
                        (Upper Management, HR, Legal)
    ANTI        — ~10%  Intentional critics / anti-personas. Reveal feature boundaries.
    """
    PRIMARY     = "PRIMARY"
    STAKEHOLDER = "STAKEHOLDER"
    ANTI        = "ANTI"


class PriorityMatrix(BaseModel):
    """
    Target allocation weights for the 70/20/10 priority split.
    Values should sum to 1.0.
    """
    primary_weight:     float = 0.70
    stakeholder_weight: float = 0.20
    anti_weight:        float = 0.10


class TensionVector(BaseModel):
    """
    Named tension dimensions extracted from a feature proposal.

    dimensions: {name: float}
        Positive  → feature *benefits* agents on this axis
        Negative  → feature *threatens* agents on this axis

    required_domains: list of domains the simulation MUST cover
        (used by EpistemicCoverageChecker)
    """
    dimensions: Dict[str, float] = Field(default_factory=dict)
    required_domains: List[str]  = Field(default_factory=list)
    raw_llm_output: Optional[str] = None


class PersonaPole(BaseModel):
    """Attractor/Repeller/Swing assignment + Priority category for one FinalPersona."""
    persona_name: str
    pole: PoleType
    pole_score: float                     # projection onto principal tension axis [-1, 1]
    category: PersonaCategory = PersonaCategory.PRIMARY  # Priority matrix slot
    influence_weight: float = 1.0         # weighting multiplier for adoption score
    domain_expertise: List[str] = Field(default_factory=list)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    minority_voice_fragment: Optional[str] = None   # injected by coverage checker


class EpistemicGap(BaseModel):
    """A domain required by TensionVector but absent from selected personas."""
    domain: str
    covered: bool = False
    coverage_count: int = 0
    minority_voice_activated: bool = False
    fragment_injected: Optional[str] = None


class SelectionResult(BaseModel):
    """Full output of PersonaSelectionEngine.select()."""
    tension_vector: TensionVector
    poles: List[PersonaPole]
    epistemic_gaps: List[EpistemicGap]
    pole_distribution: Dict[str, float]      # {"ATTRACTOR": 0.60, ...}
    category_distribution: Dict[str, float]  # {"PRIMARY": 0.70, ...}
    priority_matrix: PriorityMatrix
    target_n: int
    actual_n: int
    strategy_used: str                       # "expert_jury" | "proportional" | "gmm"
    # Expanded agent metadata (added by synthetic expander)
    expansion_metadata: Optional[Dict[str, Any]] = None

