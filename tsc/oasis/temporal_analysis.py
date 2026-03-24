import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from tsc.oasis.models import MarketSentimentSeries, OpinionVector

logger = logging.getLogger(__name__)

def AnalyzeOpinionEvolution(series: MarketSentimentSeries) -> Dict[str, Any]:
    """
    Extract trends and velocity from the opinion trajectory.
    """
    if not series.global_opinion_centroid:
        return {}

    # Calculate overall shift
    start_v = series.global_opinion_centroid[0].to_vector()
    end_v = series.global_opinion_centroid[-1].to_vector()
    shift = np.array(end_v) - np.array(start_v)
    
    # Identify dimensions with most movement
    dims = ["feasibility", "demand", "resources", "risk", "velocity"]
    move_idx = np.argmax(np.abs(shift))
    primary_trend = f"{dims[move_idx]} shifted by {shift[move_idx]:.2f}"

    return {
        "overall_shift": shift.tolist(),
        "primary_trend": primary_trend,
        "momentum": np.linalg.norm(shift),
        "divergence": np.max(series.volatility_index) if series.volatility_index else 0.0,
    }

def CalculateVolatility(agents: List[Any]) -> float:
    """
    Measure the instantaneous divergence of opinions in the population.
    """
    if not agents:
        return 0.0
    
    vecs = []
    for a in agents:
        belief = getattr(a, "current_belief", None) or getattr(a, "initial_belief", None)
        if belief:
            try:
                vecs.append(belief.to_vector())
            except AttributeError:
                # Some models use .magnitude() instead, or dump directly
                dims = [belief.technical_feasibility, belief.market_demand, belief.resource_alignment, belief.risk_tolerance, belief.adoption_velocity]
                vecs.append(dims)
                
    if len(vecs) < 2:
        return 0.0
        
    # Standard deviation across population
    return float(np.std(vecs, axis=0).mean())
