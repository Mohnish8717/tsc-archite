import logging
import os
import numpy as np
import json
from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.cluster import KMeans

# Force single-threading for macOS stability before any heavy ML imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from tsc.oasis.models import (
    OASISAgentProfile, 
    BeliefCluster, 
    MarketSentimentSeries,
    OASISSimulationConfig
)

logger = logging.getLogger(__name__)

# Lazy-loaded embedder
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            # SOTA Upgrade: Move to BGE-Large (1024-dim) for high-fidelity clustering
            # Note: BGE-M3 not currently supported in this fastembed version, using BGE-Large-v1.5
            logger.info("Loading SOTA FastEmbed model (BAAI/bge-large-en-v1.5) for high-fidelity clustering...")
            _embedder = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
            logger.info("✓ BGE-Large model loaded.")
        except Exception as e:
            logger.error(f"Failed to load SOTA embedder (BGE-Large): {e}. Falling back to small model.")
            try:
                from fastembed import TextEmbedding
                _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                logger.info("✓ Fallback BGE-Small model loaded.")
            except:
                return None
    return _embedder

async def PerformBehavioralClustering(
    agents: List[OASISAgentProfile],
    simulation_results: Any,
    llm_client: Any = None
) -> List[BeliefCluster]:
    """
    Cluster agents using K-Means on semantic embeddings, followed by LLM aggregation.
    """
    if not agents or not simulation_results:
        return []

    # 1. Prepare agent dialogue blobs
    agent_texts = []
    agent_id_to_idx = {}
    
    # Map agent ID to responses
    response_map = {str(r["agent_id"]): " ".join([resp.get("content", "") for resp in r["responses"]]) 
                    for r in simulation_results}

    valid_agents = []
    for agent in agents:
        content = response_map.get(str(agent.agent_id), "").strip()
        if content:
            valid_agents.append(agent)
            agent_texts.append(content)
            agent_id_to_idx[str(agent.agent_id)] = len(valid_agents) - 1

    if not agent_texts:
        return []

    # 2. Semantic Vectorization (MacOS Optimized)
    embedder = _get_embedder()
    if embedder is None:
        logger.warning("Embedder unavailable, falling back to basic clustering")
        return await _fallback_clustering(valid_agents, response_map)

    embeddings = np.array(list(embedder.embed(agent_texts)))
    
    # 3. K-Means Clustering (SOTA Fix: Dynamic Cluster Cap)
    # Scale clusters with population: 1 cluster per ~40-80 agents for large pools.
    # For small tests (3 agents), it remains 3. For 1,000 agents, it moves to ~12.
    num_clusters = max(min(len(valid_agents), 3), min(len(valid_agents) // 40, 12))
    
    logger.info(f"Clustering {len(valid_agents)} agents into {num_clusters} semantic buckets...")
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # 4. LLM Aggregation
    clusters = []
    from tsc.llm.factory import create_llm_client
    from tsc.config import LLMProvider
    
    # Ensure we have an LLM client
    if llm_client is None:
        llm_client = create_llm_client(provider=LLMProvider.GROQ, model="llama-3.1-8b-instant")

    for k in range(num_clusters):
        member_indices = [i for i, label in enumerate(cluster_labels) if label == k]
        if not member_indices:
            continue
            
        members = [valid_agents[i] for i in member_indices]
        
        # Sample representative content (closest to centroid)
        dists = np.linalg.norm(embeddings[member_indices] - centroids[k], axis=1)
        # Get indices of top 2-3 closest
        closest_internal_indices = np.argsort(dists)[:3]
        sampled_content = [agent_texts[member_indices[idx]] for idx in closest_internal_indices]
        
        # Aggregation Prompt
        agg_prompt = f"""Analyze these representative messages from a group of market participants reacting to a feature proposal.
Messages:
{chr(10).join([f'- {c}' for c in sampled_content])}

Assign a descriptive Name for this group and summarize their shared core stance (Centroid Behavior).
Also, provide a collective sentiment score from 0.0 (completely bearish) to 1.0 (completely bullish).

Output ONLY JSON in this format:
{{
  "name": "Cluster Name",
  "behavior": "Detailed summary of collective belief",
  "sentiment": 0.5
}}"""

        try:
            resp = await llm_client.generate("You are a market analyst.", agg_prompt)
            # Clean JSON
            resp = resp.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(resp)
            name = data.get("name", f"Segment {k+1}")
            behavior = data.get("behavior", "Semantic cluster of similar views.")
            sentiment = float(data.get("sentiment", 0.5))
        except Exception as e:
            logger.error(f"LLM Aggregation failed for cluster {k}: {e}")
            name = f"Segment {k+1}"
            behavior = "Semantic cluster of participants."
            sentiment = 0.5

        # Determine dominant persona type
        type_counts = Counter([m.agent_type for m in members])
        dominant_type = type_counts.most_common(1)[0][0] if type_counts else "UNKNOWN"

        clusters.append(BeliefCluster(
            cluster_id=f"cluster_{k}",
            centroid_behavior=behavior,
            members=[str(m.agent_id) for m in members],
            cluster_size=len(members),
            dominant_persona_type=dominant_type,
            sentiment_score=sentiment,
            description=name # Custom field or repurpose existing
        ))

    return clusters

async def _fallback_clustering(agents, response_map):
    # Simple keyword fallback if ML tools fail
    clusters = []
    # (Simplified version of old logic or just one big cluster)
    segments = {"MARKET_FEEDBACK": agents}
    for seg_name, members in segments.items():
        cluster = BeliefCluster(
            cluster_id="segment_fallback",
            centroid_behavior="Aggregated feedback (ML fallback mode)",
            members=[str(m.agent_id) for m in members],
            cluster_size=len(members),
            dominant_persona_type=Counter([m.agent_type for m in members]).most_common(1)[0][0],
            sentiment_score=0.5
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
    elif score > 0.6:
        return True, score, "majority"
    elif score > 0.4:
        return False, score, "polarized"
    return False, score, "fragmented"

def CalculateAggregatedMetrics(clusters: List[BeliefCluster], series: MarketSentimentSeries):
    """
    Computes final adoption score and verdict from semantic clusters.
    SOTA Fix: Implements Weighted Verdict based on Consensus Strength.
    """
    if not clusters:
        series.final_adoption_score = 0.5
        series.consensus_verdict = "NEUTRAL"
        return
        
    total_agents = sum(c.cluster_size for c in clusters)
    if total_agents == 0: return

    # Calculate weighted average sentiment
    avg_score = sum(c.sentiment_score * (c.cluster_size / total_agents) for c in clusters)
    series.final_adoption_score = round(avg_score, 2)
    
    # SOTA Fix: Calculate Consensus Strength (inverse of variance)
    # High strength = Cohesive (agreement), Low strength = Polarized (disagreement)
    sentiments = [c.sentiment_score for c in clusters]
    std_dev = np.std(sentiments) if len(sentiments) > 1 else 0.0
    strength = max(0.0, 1.0 - (float(std_dev) * 2.0)) # 1.0 is perfect agreement
    series.consensus_strength = round(strength, 2)

    # Weighted Verdict Logic
    # If consensus is tight (Strength > 0.7), we lower the "Bullish" bar to 0.55
    # If consensus is polarized (Strength < 0.4), we raise the "Bullish" bar to 0.65
    bullish_threshold = 0.6
    bearish_threshold = 0.4

    if strength > 0.7:
        bullish_threshold = 0.55
        bearish_threshold = 0.45
    elif strength < 0.4:
        bullish_threshold = 0.65
        bearish_threshold = 0.35

    if avg_score >= bullish_threshold:
        series.consensus_verdict = "BULLISH"
    elif avg_score <= bearish_threshold:
        series.consensus_verdict = "BEARISH"
    else:
        series.consensus_verdict = "NEUTRAL"

async def PerformBeliefClustering(agents: List[OASISAgentProfile]) -> List[BeliefCluster]:
    """Wrapper for behavioral clustering for backward compatibility."""
    return await PerformBehavioralClustering(agents, [])
