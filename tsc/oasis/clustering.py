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
    
    # Map agent ID to responses
    response_map = {str(r["agent_id"]): " ".join([resp.get("content", "") for resp in r["responses"]]).lower() 
                    for r in simulation_results}
                    
    pos_keywords = re.compile(r"\b(love|great|good|excellent|needed|helpful|yes|awesome|support|agree|beneficial|valuable|efficient|innovative|game-changer)\b", re.I)
    neg_keywords = re.compile(r"\b(expensive|hate|bad|confusing|redundant|no|useless|flaw|concerned|concern|burnout|skeptical|overhead|disruptive|disrupt|inefficient|unnecessary|impact)\b", re.I)

    for agent in agents:
        agent_id_str = str(agent.agent_id)
        content_blob = response_map.get(agent_id_str, "")
        
        # Count sentiment matches
        pos_hits = len(pos_keywords.findall(content_blob))
        neg_hits = len(neg_keywords.findall(content_blob))
        
        if pos_hits > neg_hits and pos_hits > 0:
            segments["BULLISH"].append(agent)
        elif neg_hits > pos_hits and neg_hits > 0:
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

def CalculateAggregatedMetrics(clusters: List[BeliefCluster], series: MarketSentimentSeries):
    """
    Deterministically computes final adoption score and verdict from NLP-clustered segments,
    replacing the need for a secondary LLM audit step.
    """
    if not clusters:
        series.final_adoption_score = 0.5
        series.consensus_verdict = "NEUTRAL"
        return
        
    total_agents = sum(c.cluster_size for c in clusters)
    if total_agents == 0:
        return
        
    bullish_agents = sum(c.cluster_size for c in clusters if c.sentiment_score > 0.6)
    bearish_agents = sum(c.cluster_size for c in clusters if c.sentiment_score < 0.4)
    
    avg_score = sum(c.sentiment_score * (c.cluster_size / total_agents) for c in clusters)
    series.final_adoption_score = round(avg_score, 2)
    
    if bullish_agents > bearish_agents and avg_score > 0.55:
        series.consensus_verdict = "BULLISH"
    elif bearish_agents > bullish_agents and avg_score < 0.45:
        series.consensus_verdict = "BEARISH"
    else:
        series.consensus_verdict = "NEUTRAL"

async def PerformBeliefClustering(agents: List[OASISAgentProfile]) -> List[BeliefCluster]:
    """Wrapper for behavioral clustering for backward compatibility."""
    return await PerformBehavioralClustering(agents, None)
