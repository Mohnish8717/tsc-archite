"""
Temporal Analysis — Opinion Evolution & Volatility Tracking
Gracefully handles OASISAgentProfile objects that may lack belief vectors.
"""
import logging
import numpy as np
from typing import List, Dict, Any

from tsc.oasis.models import MarketSentimentSeries

logger = logging.getLogger(__name__)


def AnalyzeOpinionEvolution(series: MarketSentimentSeries) -> Dict[str, Any]:
    """Extract trends and velocity from the opinion trajectory."""
    if not getattr(series, "global_opinion_centroid", None):
        return {}

    try:
        start_v = series.global_opinion_centroid[0].to_vector()
        end_v = series.global_opinion_centroid[-1].to_vector()
        shift = np.array(end_v) - np.array(start_v)

        dims = ["feasibility", "demand", "resources", "risk", "velocity"]
        move_idx = int(np.argmax(np.abs(shift)))
        primary_trend = f"{dims[move_idx]} shifted by {shift[move_idx]:.2f}"

        return {
            "overall_shift": shift.tolist(),
            "primary_trend": primary_trend,
            "momentum": float(np.linalg.norm(shift)),
            "divergence": float(np.max(series.volatility_index)) if getattr(series, "volatility_index", None) else 0.0,
        }
    except Exception as e:
        logger.warning(f"Opinion evolution analysis failed: {e}")
        return {}


def CalculateVolatility(agents: List[Any]) -> float:
    """
    Measure instantaneous opinion divergence across the agent population.
    Gracefully returns 0.0 when agents lack belief vectors (e.g. OASISAgentProfile).
    """
    if not agents:
        return 0.0

    vecs = []
    for a in agents:
        belief = getattr(a, "current_belief", None) or getattr(a, "initial_belief", None)
        if belief is None:
            continue
        try:
            if hasattr(belief, "to_vector"):
                vecs.append(belief.to_vector())
            elif hasattr(belief, "technical_feasibility"):
                vecs.append([
                    belief.technical_feasibility,
                    belief.market_demand,
                    belief.resource_alignment,
                    belief.risk_tolerance,
                    belief.adoption_velocity,
                ])
        except Exception:
            continue

    if len(vecs) < 2:
        return 0.0

    return float(np.std(vecs, axis=0).mean())
