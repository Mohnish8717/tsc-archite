"""
Phase 1: Feature Tension Analyzer.

Uses an LLM to decompose a FeatureProposal into a named TensionVector —
a map of {dimension_name: float} where:
  -1.0 → feature strongly threatens agents on this axis
  +1.0 → feature strongly benefits agents on this axis

Also extracts the set of domains the simulation MUST cover
(used later by EpistemicCoverageChecker).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from tsc.models.inputs import FeatureProposal
from tsc.selection.models import TensionVector

logger = logging.getLogger(__name__)




_SYSTEM = """UNIVERSAL TENSION VECTOR DECOMPOSITION PROMPT
Feature-Agnostic, High-Accuracy Framework for All Product Features

You are a Senior Product Strategist and Organizational Psychologist specializing 
in technology adoption and organizational change. Your job is to decompose a 
product feature into its REAL tension axes — the specific, measurable points 
of value and friction that determine how stakeholders will react.

You produce a TensionVector: a set of named dimensions each scored from -1.0 
(strong threat/friction) to +1.0 (strong benefit/value).

CRITICAL RULES (Non-Negotiable)
1. FEATURE-SPECIFIC, NOT GENERIC
   ✓ "Architectural Judgment Removal" (specific decision type being lost)
   ✓ "Standard Boilerplate Velocity" (specific efficiency gain)
   ✗ "Complexity" (generic, meaningless)
   ✗ "Value" (too vague)
   
   Test: Could this dimension apply to 3+ other features?
   If yes, it's too generic. Narrow it down.

2. BALANCED MARKET REALISM
   Modern markets are driven by both FOMO (value/gains) and FUD (friction/threats).
   You MUST identify both why someone would CHAMPION this feature and why 
   someone would BLOCK it. A realistic feature has both + and - axes.

3. CALIBRATION PRECISION (No Generic +/- 0.70 Scoring)
   IMPACT LEVELS:
   5 = Existential (Role destruction OR Game-changing utility)      → |0.88| to |1.0|
   4 = High impact (Autonomy loss OR Major productivity leap)       → |0.75| to |0.87|
   3 = Moderate impact (Noticeable friction OR Consistent time save) → |0.55| to |0.74|
   2 = Mild (Minor overhead OR Slight convenience)                  → |0.25| to |0.54|
   1 = Negligible (Almost no impact)                                → 0.0 to |0.24|
   
   FREQUENCY LEVELS:
   A = Daily (>80% of work affected)
   B = Weekly (50-80% of work weeks)
   C = Monthly (20-50% of time)
   D = Quarterly (5-20% of time)
   E = Rare (<5% of time)
   
   Scores must be differentiated and realistically calibrated.

4. COMPLETE COVERAGE OF 5 UNIVERSAL CATEGORIES
   You MUST produce dimensions across all relevant categories:
   a) DECISION AUTHORITY & AUTONOMY
   b) OBSERVATION & PRIVACY SURFACE
   c) EXPERTISE & PROFESSIONAL IDENTITY
   d) OPERATIONAL FRICTION & COGNITIVE LOAD
   e) ECONOMIC & STRATEGIC VALUE
   
   Ideal range: 6-12 dimensions across all categories.

5. REQUIRED DOMAINS (Expertise Needed to Model Dynamics)
   Include specific domains like "Privacy Law", "Labor Relations", "Software Engineering",
   "Data Science", "Security", or "Business Strategy".
   Minimum: 3 required domains.
"""

_USER_TEMPLATE = """Feature: {title}
Description: {description}

Decompose this feature into its real tension axes following the 5 Universal Categories.

Return ONLY valid JSON with this exact structure:
{{
  "feature": "{title}",
  "description": "...",
  "dimensions": {{
    "<Dimension Name With Spaces>": {{
      "category": "<One of the 5 categories>",
      "impact_level": "<1-5>",
      "frequency": "<A-E>",
      "score": <float between -1.0 and 1.0>,
      "reasoning": "<Why this impact? Why this frequency?>",
      "required_domains": ["<Domain 1>", "<Domain 2>"]
    }},
    ...
  }},
  "required_domains": ["<Domain 1>", "<Domain 2>", ...],
  "net_tension": <float>,
  "category_breakdown": {{...}},
  "adoption_prediction": {{...}}
}}

Rules:
- 6 to 12 dimensions required
- Every dimension name MUST use SPACES (e.g., "Developer Autonomy", NOT "DeveloperAutonomy")
- Scores must be floats between -1.0 and 1.0
- No explanation, no markdown outside the JSON block.
"""




class FeatureTensionAnalyzer:
    """
    Phase 1 of PersonaSelectionEngine.
    Extracts a TensionVector from a FeatureProposal using one LLM call.
    """

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    async def analyze(self, feature: FeatureProposal) -> TensionVector:
        """Run Phase 1 — returns a TensionVector."""
        logger.info("Phase 1: Extracting tension vector for '%s'", feature.title)
        try:
            prompt = _USER_TEMPLATE.format(
                title=feature.title,
                description=feature.description[:1000],
            )
            result = await self._llm.analyze(
                system_prompt=_SYSTEM,
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=2000, # Increased to prevent JSON truncation
            )
            tv = self._parse(result)
            logger.info(
                "Phase 1 complete: %d dimensions, %d required domains",
                len(tv.dimensions), len(tv.required_domains)
            )
            return tv
        except Exception as e:
            logger.warning("LLM tension analysis failed (%s), using heuristic fallback", e)
            return self._heuristic_fallback(feature)

    # ──────────────────────────────────────────────────────────────────
    # Parsing
    # ──────────────────────────────────────────────────────────────────

    def _parse(self, result: Any) -> TensionVector:
        """Parse LLM JSON output into TensionVector."""
        raw: Optional[str] = None

        # result can be dict (from analyze()) or raw string
        if isinstance(result, dict):
            # Try common keys
            raw = result.get("text") or result.get("content") or result.get("response")
            if not raw:
                # Maybe the dict IS the data
                if "dimensions" in result:
                    return self._build_from_dict(result)
                raw = str(result)
        else:
            raw = str(result)

        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        data = json.loads(raw)
        return self._build_from_dict(data)

    def _build_from_dict(self, data: Dict[str, Any]) -> TensionVector:
        dims: Dict[str, float] = {}
        for k, v in data.get("dimensions", {}).items():
            try:
                # Handle the new nested dictionary format
                if isinstance(v, dict):
                    # In the Universal Prompt, the score is inside `v["score"]`
                    # but if it's missing, maybe it's in the string
                    score_val = v.get("score")
                    if score_val is not None:
                        # Extract just the float in case the LLM returned a string like "-0.85"
                        if isinstance(score_val, str):
                            match = re.search(r"[-+]?\d*\.\d+|\d+", score_val)
                            if match:
                                dims[str(k)] = max(-1.0, min(1.0, float(match.group())))
                        else:
                            dims[str(k)] = max(-1.0, min(1.0, float(score_val)))
                else:
                    # Fallback for simple structure { "dim": -0.5 }
                    dims[str(k)] = max(-1.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                pass
        
        domains: List[str] = []
        # Support extracting required_domains from the root
        if isinstance(data.get("required_domains"), list):
            domains.extend([str(d) for d in data.get("required_domains", [])])
            
        # Also support extracting from each dimension if the root list is missing
        if not domains:
            for k, v in data.get("dimensions", {}).items():
                if isinstance(v, dict) and isinstance(v.get("required_domains"), list):
                    for d in v["required_domains"]:
                        if str(d) not in domains:
                            domains.append(str(d))

        return TensionVector(dimensions=dims, required_domains=domains)

    # ──────────────────────────────────────────────────────────────────
    # Heuristic Fallback (no LLM call)
    # ──────────────────────────────────────────────────────────────────

    def _heuristic_fallback(self, feature: FeatureProposal) -> TensionVector:
        """
        Keyword-based fallback when LLM call fails.
        Produces a usable (if coarse) TensionVector.
        """
        text = (feature.description + " " + feature.title).lower()
        dims: Dict[str, float] = {}
        domains: List[str] = []

        _rules = [
            # (keyword_triggers, dimension_name, base_score, domain)
            (["meeting", "sync", "standup", "call"], "Time Autonomy", -0.6, "Productivity"),
            (["screen capture", "record", "monitor", "surveillance"], "Privacy Surface", -0.8, "Privacy"),
            (["gdpr", "compliance", "legal", "policy"], "Regulatory Risk", -0.5, "Legal"),
            (["transparency", "visibility", "dashboard", "report"], "Information Flow", +0.6, None),
            (["collaboration", "team", "coordination", "align"], "Team Coherence", +0.4, None),
            (["automation", "ai", "ml", "algorithm"], "Technical Complexity", -0.3, "DevOps"),
            (["cost", "budget", "revenue", "price"], "Budget Alignment", -0.2, None),
        ]

        for triggers, dim_name, score, domain in _rules:
            if any(t in text for t in triggers):
                dims[dim_name] = score
                if domain and domain not in domains:
                    domains.append(domain)

        if not dims:
            dims = {"Feature Impact": 0.0}
            domains = ["General"]

        logger.info("Heuristic fallback: %d dimensions", len(dims))
        return TensionVector(dimensions=dims, required_domains=domains)
