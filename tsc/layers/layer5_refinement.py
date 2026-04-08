"""Layer 5: Iterative Refinement.

If gates flag major issues, refine the proposal and re-run failed gates.
Maximum 1 refinement iteration to prevent infinite loops.

Critical fixes:
  1. Comprehensive input validation
  2. Context-rich refinement prompts
  3. Quality validation of refinements
  4. Intelligent feature modification
  5. Gate-by-gate comparison 
  6. Smart decision-making on acceptance
  7. Refinement tracking and audit trail

Optimizations:
  1. Refinement caching
  2. Progressive refinement strategies
  3. Specialized system prompts
  4. Refinement tracking (max 1 try)
  5. Diagnostics & Metrics
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from tsc.layers.layer4_gates import GateExecutor
from tsc.llm.base import LLMClient
from tsc.models.chunks import ProblemContextBundle
from tsc.models.gates import GatesSummary
from tsc.models.graph import KnowledgeGraph
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona

logger = logging.getLogger(__name__)


class RefinementEngine:
    """Layer 5: Conditionally refine and re-run failed gates."""

    def __init__(
        self,
        llm_client: LLMClient,
        gate_fail_threshold: float = 0.5,
        enable_caching: bool = True,
    ) -> None:
        self._llm = llm_client
        self._threshold = gate_fail_threshold
        self._enable_caching = enable_caching
        self._refinement_cache: dict[str, dict[str, Any]] = {}
        self._refinement_attempts: dict[str, int] = {}
        self._max_attempts = 1  # Max 1 refinement iteration
        logger.info("RefinementEngine initialized (cache=%s)", enable_caching)

    # ═════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════

    async def process(
        self,
        gates_summary: GatesSummary,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
    ) -> GatesSummary:
        """Execute Layer 5 with full validation and intelligence."""
        t0 = time.time()

        logger.info("Layer 5: Iterative Refinement")

        # Step 0: Validate inputs
        try:
            self._validate_inputs(
                gates_summary, feature, company, graph, bundle, personas
            )
            logger.info("✓ Validated all inputs")
        except ValueError as e:
            logger.error("Input validation failed: %s", e)
            raise

        # Early exit: all passed
        if gates_summary.all_passed:
            logger.info("✓ All gates passed, skipping refinement")
            return gates_summary

        # Early exit: no critical failures
        if not gates_summary.failed_gates:
            logger.info("✓ No critical failures, skipping refinement")
            return gates_summary

        # Track attempts (OPT-4)
        feature_key = feature.title.lower()
        attempts = self._refinement_attempts.get(feature_key, 0)

        if attempts >= self._max_attempts:
            logger.info(
                "Max refinement attempts (%d) reached for '%s'",
                self._max_attempts,
                feature.title,
            )
            return gates_summary

        logger.info(
            "Layer 5: %d gates failed (score: %.2f), attempting refinement",
            len(gates_summary.failed_gates),
            gates_summary.overall_score,
        )

        # Step 1: Check cache (OPT-1)
        cache_key = self._get_cache_key(feature, gates_summary)

        if self._enable_caching and cache_key in self._refinement_cache:
            cached = self._refinement_cache[cache_key]
            logger.info(
                "✓ Using cached refinement result (improvement: %.2f → %.2f)",
                gates_summary.overall_score,
                cached["gates_summary"].overall_score,
            )
            self._refinement_attempts[feature_key] = attempts + 1
            return cached["gates_summary"]

        # Step 2: Select refinement strategy (OPT-2)
        strategy = self._select_refinement_strategy(
            gates_summary, company, personas
        )
        logger.info("✓ Selected refinement strategy: %s", strategy)

        # Step 3: Build refinement prompt with context
        refinement_prompt = self._build_refinement_prompt(
            feature, gates_summary, company, graph, personas, bundle
        )
        logger.info("✓ Built refinement prompt")

        # Step 4: Get LLM suggestions
        try:
            system_prompt = self._get_refinement_system_prompt(strategy)

            refinement_response = await self._llm.analyze(
                system_prompt=system_prompt,
                user_prompt=refinement_prompt,
                temperature=0.5,
                max_tokens=3000,
            )

            if not refinement_response:
                raise ValueError("LLM returned empty response")

            logger.info("✓ Received refinement suggestions from LLM")
        except Exception as e:
            logger.warning("LLM refinement failed: %s", e)
            self._refinement_attempts[feature_key] = attempts + 1
            return gates_summary

        # Step 5: Validate refinement quality
        is_valid, issues = self._validate_refinement(refinement_response)

        if not is_valid:
            logger.warning("Refinement validation failed: %s", issues)
            self._refinement_attempts[feature_key] = attempts + 1
            return gates_summary

        logger.info("✓ Refinement validation passed")

        # Step 6: Apply refinement to feature
        refined_feature = self._apply_refinement(feature, refinement_response)
        logger.info("✓ Applied refinement to feature")

        # Step 7: Re-run ONLY failed gates with refined feature (selective re-evaluation)
        try:
            executor = GateExecutor(self._llm)
            try:
                refined_summary = await executor.process_failed_only(
                    refined_feature, company, graph, bundle, personas,
                    previous_summary=gates_summary,
                )
                logger.info(
                    "✓ Selectively re-ran %d failed gates (score: %.2f → %.2f)",
                    len(gates_summary.failed_gates),
                    gates_summary.overall_score,
                    refined_summary.overall_score,
                )
            except (AttributeError, TypeError):
                # Fallback to full re-run if process_failed_only not available
                logger.info("Falling back to full gate re-evaluation")
                refined_summary = await executor.process(
                    refined_feature, company, graph, bundle, personas
                )
                logger.info(
                    "✓ Re-ran all gates with refined feature (score: %.2f → %.2f)",
                    gates_summary.overall_score,
                    refined_summary.overall_score,
                )
        except Exception as e:
            logger.warning("Failed to re-run gates: %s", e)
            self._refinement_attempts[feature_key] = attempts + 1
            return gates_summary

        # Step 8: Compare results gate-by-gate
        comparison = self._compare_gate_results(gates_summary, refined_summary)
        logger.info("✓ Completed gate-by-gate comparison")

        # Step 9: Decide whether to accept refinement
        should_accept, decision_reason = self._should_accept_refinement(
            gates_summary, refined_summary, comparison, refinement_response
        )

        # Step 10: Store audit trail
        final_summary = self._store_refinement_decision(
            refined_summary if should_accept else gates_summary,
            refinement_response,
            comparison,
            should_accept,
            decision_reason,
        )

        # Step 11: Cache if accepted
        if self._enable_caching and should_accept:
            self._refinement_cache[cache_key] = {
                "gates_summary": final_summary,
                "improvement": refined_summary.overall_score
                - gates_summary.overall_score,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Track attempt
        self._refinement_attempts[feature_key] = attempts + 1

        # Log completion
        elapsed = time.time() - t0
        logger.info(
            "Layer 5 complete: %s, final score=%.2f (improvement: %+.2f), time=%.1fs",
            "✓ REFINED" if should_accept else "✗ ORIGINAL",
            final_summary.overall_score,
            final_summary.overall_score - gates_summary.overall_score,
            elapsed,
        )

        return final_summary

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #1: Input Validation
    # ═════════════════════════════════════════════════════════════════

    def _validate_inputs(
        self,
        gates_summary: GatesSummary,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
    ) -> None:
        """Comprehensive input validation."""
        # GatesSummary validation
        if not gates_summary:
            raise ValueError("gates_summary is required")

        if not hasattr(gates_summary, "results") or not gates_summary.results:
            raise ValueError("gates_summary.results is empty")

        if not hasattr(gates_summary, "overall_score"):
            raise ValueError("gates_summary missing overall_score")

        # Feature validation
        if not feature or not feature.title:
            raise ValueError("Feature proposal missing title")

        if not feature.description or len(feature.description) < 10:
            raise ValueError("Feature description too short")

        if (
            feature.effort_weeks_min is None
            or feature.effort_weeks_max is None
        ):
            raise ValueError("Feature missing effort estimate")

        if (
            feature.effort_weeks_min < 1
            or feature.effort_weeks_max < feature.effort_weeks_min
        ):
            raise ValueError("Invalid effort estimate")

        # Company validation
        if not company:
            raise ValueError("Company context required")

        if not company.company_name:
            logger.warning("Company missing company_name")

        if not company.team_size or company.team_size < 1:
            raise ValueError("Company team_size invalid")

        # Graph validation
        if not graph or not graph.nodes:
            raise ValueError("Knowledge graph has no entities")

        if not graph.edges:
            logger.warning("Knowledge graph has no relationships")

        # Bundle validation
        if not bundle or not bundle.chunks:
            raise ValueError("Problem context bundle is empty")

        # Personas validation
        if personas is None:
            personas = []

        if not isinstance(personas, list):
            raise ValueError("personas must be a list")

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #2: Context-Rich Prompt
    # ═════════════════════════════════════════════════════════════════

    def _build_refinement_prompt(
        self,
        feature: FeatureProposal,
        gates_summary: GatesSummary,
        company: CompanyContext,
        graph: KnowledgeGraph,
        personas: list[FinalPersona],
        bundle: Any = None,
    ) -> str:
        """Build detailed refinement prompt with comprehensive context."""
        failed_gates_detail = self._format_failed_gates(gates_summary)
        stakeholder_summary = self._summarize_stakeholder_concerns(personas)
        company_summary = self._summarize_company_context(company)
        key_entities = self._extract_key_entities(graph)
        behavioral_evidence = self._extract_behavioral_evidence(bundle)

        passed_count = len(gates_summary.results) - len(
            gates_summary.failed_gates
        )

        return f"""
═══════════════════════════════════════════════════════════════════════════
FEATURE PROPOSAL REFINEMENT REQUEST
═══════════════════════════════════════════════════════════════════════════

Feature Title: {feature.title}
# Scores are on a 0.0–1.0 scale. Showing as % for clarity.
Current Market Adoption Score: {gates_summary.overall_score * 100:.1f}% (PASS threshold: ≥50%)
Status: {len(gates_summary.failed_gates)} gates FAILED, {passed_count} gates passed

───────────────────────────────────────────────────────────────────────────
CURRENT FEATURE DESCRIPTION:
───────────────────────────────────────────────────────────────────────────
{feature.description}

Target Users: {feature.target_users}
Current Effort Estimate: {feature.effort_weeks_min}-{feature.effort_weeks_max} weeks

───────────────────────────────────────────────────────────────────────────
FAILED GATES ANALYSIS (Priority Order):
───────────────────────────────────────────────────────────────────────────
{failed_gates_detail}

───────────────────────────────────────────────────────────────────────────
STAKEHOLDER FEEDBACK & CONCERNS:
───────────────────────────────────────────────────────────────────────────
{stakeholder_summary}

───────────────────────────────────────────────────────────────────────────
COMPANY CONSTRAINTS & CONTEXT:
───────────────────────────────────────────────────────────────────────────
{company_summary}

───────────────────────────────────────────────────────────────────────────
KEY ENTITIES & DEPENDENCIES:
───────────────────────────────────────────────────────────────────────────
{key_entities}

───────────────────────────────────────────────────────────────────────────
BEHAVIORAL EVIDENCE FROM SIMULATION (Actual agent dialogue):
───────────────────────────────────────────────────────────────────────────
{behavioral_evidence}

───────────────────────────────────────────────────────────────────────────
YOUR TASK:
───────────────────────────────────────────────────────────────────────────

Analyze the failed gates and suggest SPECIFIC, ACTIONABLE modifications to 
the feature proposal that would address the root causes.

Focus on:
1. SCOPE REDUCTION: Remove low-value, high-effort components
2. PHASING: Split into MVP + follow-up phases
3. CONSTRAINT RESOLUTION: Address technical/resource blockers
4. TIMELINE: Reduce effort estimate through simplification

Your suggestions should:
- Address the root cause of failures (not just symptoms)
- Be realistic given company constraints
- Respect stakeholder concerns
- Maintain customer value
- Include effort impact estimates

───────────────────────────────────────────────────────────────────────────
RESPONSE FORMAT (VALID JSON):
───────────────────────────────────────────────────────────────────────────
{{
  "analysis": "Your analysis of why gates failed (150-250 words)",
  "root_causes": [
    "root cause 1",
    "root cause 2",
    "root cause 3"
  ],
  "refinement_strategy": "Which approach (scope_reduction | phasing | constraint_resolution | timeline_reduction | combination)",
  "suggestions": [
    {{
      "title": "Suggestion 1 title",
      "description": "Detailed description of what to change and why",
      "impact": "How this addresses failed gates",
      "effort_impact": "-2 weeks"
    }},
    {{
      "title": "Suggestion 2 title",
      "description": "...",
      "impact": "...",
      "effort_impact": "-1 week"
    }}
  ],
  "revised_scope": "Concise summary of what the MVP would include after refinement (100-150 words)",
  "phasing_plan": {{
    "mvp": {{
      "scope": "What goes in MVP",
      "effort_weeks": 2
    }},
    "phase_2": {{
      "scope": "What goes in Phase 2",
      "effort_weeks": 3
    }}
  }},
  "revised_effort_weeks": {{
    "min": 2,
    "max": 4
  }},
  "confidence": 0.85,
  "reasoning": "Why you believe these changes will help pass the failed gates"
}}

Do not include markdown formatting, only valid JSON.
"""

    def _format_failed_gates(self, gates_summary: GatesSummary) -> str:
        """Format failed gates with reasoning."""
        # Safety for missing `passed` boolean attribute
        def _did_gate_fail(g: Any) -> bool:
            return getattr(
                g, "passed", g.gate_id in gates_summary.failed_gates
            )

        failed = [g for g in gates_summary.results if _did_gate_fail(g)]

        if not failed:
            return "No failed gates"

        formatted = []
        for idx, gate in enumerate(failed, 1):
            reasoning = getattr(gate, "reasoning", str(gate.details))[:200]
            formatted.append(
                f"{idx}. {gate.gate_name} (Score: {gate.score:.2f}/10)\n"
                f"   Verdict: {gate.verdict.value}\n"
                f"   Issue: {reasoning}"
            )

        return "\n".join(formatted)

    def _summarize_stakeholder_concerns(
        self, personas: list[FinalPersona]
    ) -> str:
        """Summarize stakeholder concerns and objections."""
        if not personas:
            return "No stakeholder feedback available"

        summary_lines = []

        for persona in personas:
            stance = persona.psychological_profile.predicted_stance

            summary_lines.append(
                f"- {persona.name} ({persona.role}): {stance.prediction}"
            )

            if stance.potential_objections:
                objections = ", ".join(stance.potential_objections[:3])
                summary_lines.append(f"  Concerns: {objections}")

            if stance.likely_conditions:
                conditions = ", ".join(stance.likely_conditions[:2])
                summary_lines.append(f"  Would approve if: {conditions}")

        return "\n".join(summary_lines)

    def _summarize_company_context(self, company: CompanyContext) -> str:
        """Summarize company constraints."""
        lines = [
            f"Company: {company.company_name}",
            f"Team Size: {company.team_size} people",
            f"Budget: {company.budget}" if company.budget else "Budget: Not specified",
            f"Current Priorities: {', '.join(company.current_priorities)}",
            f"Tech Stack: {', '.join(company.tech_stack)}",
        ]

        if company.constraints:
            lines.append(f"Constraints: {', '.join(company.constraints)}")

        return "\n".join(lines)

    def _extract_key_entities(self, graph: KnowledgeGraph) -> str:
        """Extract key entities and relationships."""
        top_entities = sorted(
            graph.nodes.values(), key=lambda e: e.mentions, reverse=True
        )[:10]

        entity_lines = [
            f"- {e.name} ({e.type}): mentioned {e.mentions}x"
            for e in top_entities
        ]

        relationships = graph.edges[:10] if graph.edges else []
        if relationships:
            rel_lines = [
                f"- {rel.relationship_type.value}: "
                f"{graph.nodes[rel.source_entity].name} ↔ "
                f"{graph.nodes[rel.target_entity].name}"
                for rel in relationships
                if rel.source_entity in graph.nodes
                and rel.target_entity in graph.nodes
            ]
            entity_lines.extend(rel_lines)

        return "\n".join(entity_lines[:15])

    def _extract_behavioral_evidence(self, bundle: ProblemContextBundle) -> str:
        """Opt A: Extract top behavioral insights from simulation chunks."""
        if not bundle or not bundle.chunks:
            return "No behavioral evidence available."
        # Sort by coherence (highest = most signal-rich) and take top 5
        top_chunks = sorted(
            bundle.chunks,
            key=lambda c: getattr(c, 'coherence_score', 0.0),
            reverse=True
        )[:5]
        lines = []
        for i, chunk in enumerate(top_chunks, 1):
            text = getattr(chunk, 'text', str(chunk))[:300]
            score = getattr(chunk, 'coherence_score', 0.0)
            lines.append(f"{i}. [coherence={score:.2f}] {text}")
        return "\n".join(lines) if lines else "No behavioral evidence available."


    def _validate_refinement(
        self, refinement: Any
    ) -> tuple[bool, list[str]]:
        """Validate refinement response quality."""
        issues = []

        if not isinstance(refinement, dict):
            return False, ["Refinement response is not a dict"]

        required_fields = [
            "analysis",
            "suggestions",
            "revised_scope",
            "revised_effort_weeks",
            "confidence",
        ]

        for field in required_fields:
            if field not in refinement:
                issues.append(f"Missing required field: {field}")

        if issues:
            return False, issues

        # Validate analysis
        analysis = refinement.get("analysis", "")
        if not analysis or len(analysis) < 50:
            issues.append("Analysis too short (< 50 chars)")
        if len(analysis) > 2000:
            # PA-4 fix: truncate instead of rejecting
            refinement["analysis"] = analysis[:2000] + "..."
            logger.info("Truncated analysis from %d to 2000 chars", len(analysis))

        # Validate suggestions
        suggestions = refinement.get("suggestions", [])
        if not isinstance(suggestions, list):
            issues.append("Suggestions not a list")
        elif len(suggestions) < 1:
            issues.append("No suggestions provided")
        elif len(suggestions) > 10:
            issues.append("Too many suggestions (> 10)")
        else:
            for idx, sugg in enumerate(suggestions):
                if not isinstance(sugg, dict):
                    issues.append(f"Suggestion {idx} not a dict")
                elif "title" not in sugg or "description" not in sugg:
                    issues.append(f"Suggestion {idx} missing title/description")
                elif len(sugg.get("description", "")) < 20:
                    issues.append(f"Suggestion {idx} description too short")

        # Validate revised scope
        scope = refinement.get("revised_scope", "")
        if not scope or len(scope) < 30:
            issues.append("Revised scope too short (< 30 chars)")
        if len(scope) > 2000:
            # PA-4 fix: truncate instead of rejecting
            refinement["revised_scope"] = scope[:2000] + "..."
            logger.info("Truncated revised_scope from %d to 2000 chars", len(scope))

        # Validate effort estimate
        effort = refinement.get("revised_effort_weeks", {})
        if not isinstance(effort, dict):
            issues.append("revised_effort_weeks not a dict")
        else:
            try:
                min_weeks = float(effort.get("min", 0))
                max_weeks = float(effort.get("max", 0))

                if min_weeks < 1:
                    issues.append("revised_effort_weeks.min < 1")
                if max_weeks < min_weeks:
                    issues.append("revised_effort_weeks.max < min")
                if max_weeks > 50:
                    issues.append("revised_effort_weeks.max > 50 (unrealistic)")
            except (TypeError, ValueError):
                issues.append("Can't parse revised_effort_weeks as numbers")

        # Validate confidence
        confidence = refinement.get("confidence", 0)
        try:
            confidence = float(confidence)
            if confidence < 0.3:
                issues.append("Confidence too low (< 0.3)")
            if confidence > 1.0:
                issues.append("Confidence > 1.0")
        except (TypeError, ValueError):
            issues.append("Confidence not a number")

        is_valid = len(issues) == 0
        if issues:
            logger.warning("Refinement validation issues: %s", issues)

        return is_valid, issues

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #4: Intelligent Modification
    # ═════════════════════════════════════════════════════════════════

    def _apply_refinement(
        self,
        feature: FeatureProposal,
        refinement: dict[str, Any],
    ) -> FeatureProposal:
        """Apply refinement suggestions to feature."""
        # Create refined copy (pydantic API depending on version)
        if hasattr(feature, "model_copy"):
            refined = feature.model_copy(deep=True)
        else:
            refined = feature.copy(deep=True)

        revised_scope = refinement.get("revised_scope", "")
        if revised_scope:
            refined.description = (
                f"{refined.description}\n\n"
                f"### REFINED SCOPE ###\n{revised_scope}"
            )

        effort = refinement.get("revised_effort_weeks", {})
        if effort:
            try:
                refined.effort_weeks_min = int(
                    max(1, float(effort.get("min", refined.effort_weeks_min)))
                )
                refined.effort_weeks_max = int(
                    max(
                        refined.effort_weeks_min,
                        float(effort.get("max", refined.effort_weeks_max)),
                    )
                )
            except (TypeError, ValueError):
                logger.warning("Could not parse effort estimate modifications")

        # Store refinement metadata directly as dynamic attribute
        if not hasattr(refined, "_refinement_history"):
            refined._refinement_history = []

        refined._refinement_history.append(
            {
                "iteration": len(refined._refinement_history) + 1,
                "timestamp": datetime.utcnow().isoformat(),
                "suggestions_count": len(refinement.get("suggestions", [])),
                "strategy": refinement.get("refinement_strategy", ""),
                "confidence": refinement.get("confidence", 0),
                "revised_scope": revised_scope,
            }
        )

        logger.info(
            "Applied refinement to '%s' (effort: %d-%d weeks, confidence: %.2f)",
            refined.title,
            refined.effort_weeks_min,
            refined.effort_weeks_max,
            refinement.get("confidence", 0),
        )

        return refined

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #5: Gate-by-Gate Comparison
    # ═════════════════════════════════════════════════════════════════

    def _compare_gate_results(
        self,
        original: GatesSummary,
        refined: GatesSummary,
    ) -> dict[str, Any]:
        """Detailed gate-by-gate comparison."""
        comparison = {
            "overall_improvement": refined.overall_score - original.overall_score,
            "gates_improved": 0,
            "gates_worsened": 0,
            "gates_status_changed": 0,
            "failed_gates_fixed": 0,
            "newly_failed_gates": 0,
            "gate_details": {},
        }

        original_by_id = {g.gate_id: g for g in original.results}
        refined_by_id = {g.gate_id: g for g in refined.results}

        for gate_id, refined_gate in refined_by_id.items():
            original_gate = original_by_id.get(gate_id)
            if not original_gate:
                continue

            score_change = refined_gate.score - original_gate.score

            if score_change > 0.1:
                comparison["gates_improved"] += 1
            elif score_change < -0.1:
                comparison["gates_worsened"] += 1

            orig_passed = gate_id not in original.failed_gates
            refn_passed = gate_id not in refined.failed_gates

            if orig_passed != refn_passed:
                comparison["gates_status_changed"] += 1

                if not orig_passed and refn_passed:
                    comparison["failed_gates_fixed"] += 1
                elif orig_passed and not refn_passed:
                    comparison["newly_failed_gates"] += 1

            comparison["gate_details"][gate_id] = {
                "name": refined_gate.gate_name,
                "original_score": round(original_gate.score, 2),
                "refined_score": round(refined_gate.score, 2),
                "change": round(score_change, 2),
                "original_passed": orig_passed,
                "refined_passed": refn_passed,
                "refined_verdict": refined_gate.verdict.value,
            }

        logger.info(
            "Gate comparison: improvement=%.2f, fixed=%d, broke=%d, "
            "improved=%d, worsened=%d",
            comparison["overall_improvement"],
            comparison["failed_gates_fixed"],
            comparison["newly_failed_gates"],
            comparison["gates_improved"],
            comparison["gates_worsened"],
        )

        return comparison

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #6: Smart Acceptance Decision
    # ═════════════════════════════════════════════════════════════════

    def _should_accept_refinement(
        self,
        original: GatesSummary,
        refined: GatesSummary,
        comparison: dict[str, Any],
        refinement_data: dict[str, Any],
    ) -> tuple[bool, str]:
        """Intelligent decision on whether to accept refinement."""
        reasons = []

        overall_improvement = comparison["overall_improvement"]
        if overall_improvement > 0.05:  # OPT-B: lowered from 0.10 to 0.05
            reasons.append(f"improvement:{overall_improvement:.2f}")

        fixed = comparison["failed_gates_fixed"]
        broken = comparison["newly_failed_gates"]

        if fixed > 0 and broken == 0:
            reasons.append(f"fixed_gates:{fixed}_no_regressions")

        if refined.overall_score >= 0.50 and original.overall_score < 0.50:
            reasons.append("reached_pass_threshold")

        confidence = refinement_data.get("confidence", 0.5)
        if overall_improvement > 0.05 and confidence >= 0.60:  # OPT-B: lowered threshold
            reasons.append(
                f"good_improvement_high_confidence:{overall_improvement:.2f}"
            )

        if comparison["gates_improved"] > comparison["gates_worsened"]:
            reasons.append(
                f"net_positive:{comparison['gates_improved']}-"
                f"{comparison['gates_worsened']}"
            )

        should_accept = len(reasons) > 0
        decision_reason = "; ".join(reasons) if reasons else "no_improvement"

        if should_accept:
            logger.info(
                "✓ Accepting refinement - reasons: %s", decision_reason
            )
        else:
            logger.info(
                "✗ Rejecting refinement - overall: %.2f → %.2f, "
                "fixed: %d, broken: %d",
                original.overall_score,
                refined.overall_score,
                fixed,
                broken,
            )

        return should_accept, decision_reason

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #7: Protocol Audit Trail
    # ═════════════════════════════════════════════════════════════════

    def _store_refinement_decision(
        self,
        gates_summary: GatesSummary,
        refinement_data: dict[str, Any],
        comparison: dict[str, Any],
        accepted: bool,
        reason: str,
    ) -> GatesSummary:
        """Store refinement decision in audit trail."""
        if not hasattr(gates_summary, "_refinement_audit"):
            gates_summary._refinement_audit = []

        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": len(gates_summary._refinement_audit) + 1,
            "original_score": gates_summary.overall_score - comparison["overall_improvement"] if accepted else gates_summary.overall_score,
            "refinement": {
                "strategy": refinement_data.get("refinement_strategy", ""),
                "suggestions_count": len(refinement_data.get("suggestions", [])),
                "confidence": refinement_data.get("confidence", 0),
            },
            "comparison": {
                "overall_improvement": comparison["overall_improvement"],
                "fixed_gates": comparison["failed_gates_fixed"],
                "broken_gates": comparison["newly_failed_gates"],
            },
            "decision": {
                "accepted": accepted,
                "reason": reason,
            },
        }

        gates_summary._refinement_audit.append(audit_entry)

        logger.info(
            "Stored refinement audit: iteration=%d, accepted=%s",
            audit_entry["iteration"],
            accepted,
        )

        return gates_summary

    # ═════════════════════════════════════════════════════════════════
    # OPT-1: Refinement Caching / Key Generation
    # ═════════════════════════════════════════════════════════════════

    def _get_cache_key(
        self, feature: FeatureProposal, gates_summary: GatesSummary
    ) -> str:
        """Generate cache key for refinement."""
        return f"{feature.title}_{gates_summary.overall_score:.2f}"

    # ═════════════════════════════════════════════════════════════════
    # OPT-2: Progressive Strategies
    # ═════════════════════════════════════════════════════════════════

    def _select_refinement_strategy(
        self,
        gates_summary: GatesSummary,
        company: CompanyContext,
        personas: list[FinalPersona],
    ) -> str:
        """Select refinement strategy based on failure pattern."""
        failed_gates = [
            g
            for g in gates_summary.results
            if g.gate_id in gates_summary.failed_gates
        ]

        categories = {
            "technical": 0,
            "resource": 0,
            "market": 0,
            "timeline": 0,
            "risk": 0,
        }

        for gate in failed_gates:
            gate_name_lower = gate.gate_name.lower()

            if "technical" in gate_name_lower or "feasibility" in gate_name_lower:
                categories["technical"] += 1
            elif "resource" in gate_name_lower or "effort" in gate_name_lower:
                categories["resource"] += 1
            elif "market" in gate_name_lower or "adoption" in gate_name_lower:
                categories["market"] += 1
            elif "timeline" in gate_name_lower or "constraint" in gate_name_lower:
                categories["timeline"] += 1
            elif "risk" in gate_name_lower:
                categories["risk"] += 1

        if categories["technical"] > 0:
            return "technical_simplification"
        elif categories["resource"] > 0:
            return "phasing_approach"
        elif categories["timeline"] > 0:
            return "effort_reduction"
        elif categories["market"] > 0:
            return "market_adjustment"

        return "general_refinement"

    # ═════════════════════════════════════════════════════════════════
    # OPT-3: Specialized Prompts
    # ═════════════════════════════════════════════════════════════════

    def _get_refinement_system_prompt(self, strategy: str) -> str:
        """Get specialized system prompt based on strategy."""
        base = (
            "You are a product strategy expert specializing in feature evaluation.\n"
            "Your task is to suggest refinements to feature proposals that failed evaluation gates."
        )

        strategies = {
            "technical_simplification": base
            + """

Focus on TECHNICAL refinements:
- Simplify technical approach (reduce complexity)
- Break into smaller technical components
- Reduce dependencies on other systems
- Use proven technologies over novel approaches
- Suggest architectural improvements""",
            "phasing_approach": base
            + """

Focus on PHASING refinements:
- Define MVP with core features only
- Move nice-to-have features to Phase 2+
- Reduce initial scope to minimize effort
- Plan incremental rollout strategy
- Suggest feature priorities""",
            "effort_reduction": base
            + """

Focus on EFFORT refinements:
- Identify high-effort, low-value components to remove
- Suggest reuse of existing code/components
- Propose automation opportunities
- Consider build-vs-buy decisions
- Estimate effort savings for each change""",
            "market_adjustment": base
            + """

Focus on MARKET refinements:
- Address customer pain points directly
- Adjust target user segment if needed
- Emphasize unique value proposition
- Consider competitive landscape
- Refine go-to-market approach""",
            "general_refinement": base
            + """

Focus on HOLISTIC refinements:
- Balance all competing concerns
- Suggest systematic improvements
- Address root causes of failures
- Consider organizational capacity
- Propose realistic, achievable changes""",
        }

        return strategies.get(strategy, strategies["general_refinement"])

    # ═════════════════════════════════════════════════════════════════
    # OPT-5: Diagnostics
    # ═════════════════════════════════════════════════════════════════

    def get_layer5_diagnostics(
        self, gates_summary: GatesSummary
    ) -> dict[str, Any]:
        """Get refinement diagnostics."""
        audit = getattr(gates_summary, "_refinement_audit", [])

        if not audit:
            return {
                "refinements_applied": 0,
                "caching_enabled": self._enable_caching,
            }

        accepted = sum(1 for a in audit if a["decision"]["accepted"])
        overall_imp = sum(
            a["comparison"]["overall_improvement"] for a in audit
        )
        avg_imp = overall_imp / len(audit) if audit else 0
        gates_fixed = sum(a["comparison"]["fixed_gates"] for a in audit)
        cache_hits = sum(1 for a in audit if a.get("from_cache", False))

        return {
            "refinements_attempted": len(audit),
            "refinements_accepted": accepted,
            "overall_improvement": round(overall_imp, 3),
            "avg_improvement_per_attempt": round(avg_imp, 3),
            "gates_fixed_total": gates_fixed,
            "cache_size": len(self._refinement_cache),
            "cache_hits": cache_hits,
        }
