import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter

from tsc.oasis.models import (
    OASISAgentProfile, 
    BeliefCluster, 
    MarketSentimentSeries,
    OASISSimulationConfig
)

logger = logging.getLogger(__name__)

import re

async def PerformBehavioralClustering(
    agents: List[OASISAgentProfile],
    simulation_results: Any # List[Dict[str, Any]] containing final responses
) -> List[BeliefCluster]:
    """
    Cluster agents based on their behavioral traces and interview sentiment.
    (MiroFish style behavioral segmentation)
    """
    if not agents or not simulation_results:
        return []

    clusters = []
    
    # Stratified clustering based on interview sentiments
    segments = {
        "BULLISH": [],
        "BEARISH": [],
        "NEUTRAL": []
    }
    
    pos_keywords = re.compile(r"(love|great|good|excellent|needed|helpful|yes|awesome)", re.I)
    neg_keywords = re.compile(r"(expensive|hate|bad|confusing|redundant|no|useless|flaw)", re.I)
    
    # Map agent ID to responses
    response_map = {str(r["agent_id"]): " ".join([resp.get("content", "") for resp in r["responses"]]) 
                    for r in simulation_results}

    for agent in agents:
        agent_id_str = str(agent.agent_id)
        content_blob = response_map.get(agent_id_str, "")
        
        if pos_keywords.search(content_blob):
            segments["BULLISH"].append(agent)
        elif neg_keywords.search(content_blob):
            segments["BEARISH"].append(agent)
        else:
            segments["NEUTRAL"].append(agent)
            
    for seg_name, members in segments.items():
        if not members:
            continue
            
        # Determine dominant persona type in this cluster
        type_counts = Counter([m.agent_type for m in members])
        dominant_type = type_counts.most_common(1)[0][0] if type_counts else "UNKNOWN"
            
        cluster = BeliefCluster(
            cluster_id=f"segment_{seg_name.lower()}",
            centroid_behavior=f"Agents exhibiting {seg_name} sentiment towards the proposal",
            members=[str(m.agent_id) for m in members],
            cluster_size=len(members),
            dominant_persona_type=dominant_type,
            sentiment_score=0.9 if seg_name == "BULLISH" else (0.2 if seg_name == "BEARISH" else 0.5)
        )
        clusters.append(cluster)
        
    return clusters

def DetectConsensus(
    results: MarketSentimentSeries,
    config: OASISSimulationConfig
) -> Tuple[bool, float, str]:
    """
    Determine consensus type from aggregated adoption scores.
    """
    score = results.final_adoption_score
    
    if score > 0.8:
        return True, score, "full"
    elif score > 0.5:
        return True, score, "majority"
    elif score > 0.3:
        return False, score, "polarized"
        
    return False, score, "fragmented"

async def PerformBeliefClustering(agents: List[OASISAgentProfile]) -> List[BeliefCluster]:
    """Wrapper for behavioral clustering for backward compatibility."""
    return await PerformBehavioralClustering(agents, None)
