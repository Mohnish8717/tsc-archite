"""Layer 4: Sequential Gate Execution.

Orchestrates all 8 gates in sequence and produces a GatesSummary.

Critical fixes:
  1. Comprehensive input validation
  2. Gate result validation and error handling
  3. Intelligent weighted scoring
  4. Rich error context and diagnostics
  5. Gate execution tracking and timing
  6. Smart recommendations
  7. Result deduplication

Optimizations:
  1. Gate result caching
  2. Parallel execution mode
  3. Execution mode selection
  4. Diagnostics & Metrics
  5. Performance monitoring
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from tsc.gates.base import BaseGate
from tsc.gates.implementations import ALL_GATES
from tsc.llm.base import LLMClient
from tsc.models.chunks import ProblemContextBundle
from tsc.models.gates import GateResult, GatesSummary, GateVerdict
from tsc.models.graph import KnowledgeGraph
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona

logger = logging.getLogger(__name__)

# Verdicts that count as "passed"
PASSING_VERDICTS = {
    GateVerdict.PASS,
    GateVerdict.PASS_WITH_MITIGATION,
    GateVerdict.FEASIBLE_WITH_DEBT,
    GateVerdict.EXISTS_NEEDS_ADAPTATION,
    GateVerdict.STRONG_FIT,
    GateVerdict.MANAGEABLE,
}




class GateFactory:
    """Factory for creating gates with proper parameter passing"""

    def __init__(
        self,
        llm_client: LLMClient,
        graph_store: Optional[Any] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        self._llm = llm_client
        self._graph_store = graph_store
        self._config = config or {}

    def create_gate(self, gate_cls: type[BaseGate], num_agents_override: Optional[int] = None) -> BaseGate:
        """Create a gate instance with proper dependencies and configuration"""
        gate_id = gate_cls.gate_id if hasattr(gate_cls, "gate_id") else "unknown"
        gate_type = self._detect_gate_type(gate_id)
        
        # Override num_agents if provided
        num_agents = num_agents_override or self._config.get("num_agents", 150)

        try:
            if gate_type == "monte_carlo":
                return gate_cls(
                    llm_client=self._llm,
                    graph_store=self._graph_store,
                    num_agents=num_agents,
                    enable_parallel=self._config.get("enable_parallel_agents", True),
                )
            elif gate_type == "red_team":
                return gate_cls(
                    llm_client=self._llm,
                    graph_store=self._graph_store,
                    enable_caching=self._config.get("enable_caching", True),
                    cache_ttl_minutes=self._config.get("cache_ttl_minutes", 60),
                )
            else:
                return gate_cls(self._llm)
        except Exception as e:
            logger.error("Failed to create gate %s: %s", gate_id, e, exc_info=True)
            raise

    def _detect_gate_type(self, gate_id: str) -> str:
        """Detect gate type from gate_id"""
        # Match by ID or generic name
        if gate_id == "4.5" or "monte_carlo" in gate_id.lower():
            return "monte_carlo"
        elif gate_id == "4.6" or "red_team" in gate_id.lower():
            return "red_team"
        return "standard"


class GateExecutor:
    """Layer 4: Execute all gates with validation and intelligence"""

    def __init__(
        self,
        llm_client: LLMClient,
        graph_store: Optional[Any] = None,
        enable_caching: bool = True,
        parallel_execution: bool = False,
        cache_ttl_minutes: int = 60,
        gate_config: Optional[dict[str, Any]] = None,
    ):
        self._llm = llm_client
        self._graph_store = graph_store
        self._enable_caching = enable_caching
        self._parallel = parallel_execution
        self._cache_ttl = cache_ttl_minutes * 60
        self._gate_cache: dict[str, tuple[GateResult, float]] = {}
        self.PASSING_VERDICTS = PASSING_VERDICTS

        self._gate_factory = GateFactory(
            llm_client=llm_client,
            graph_store=graph_store,
            config=gate_config or {},
        )

        logger.info(
            "GateExecutor initialized (caching=%s, parallel=%s)",
            enable_caching,
            parallel_execution,
        )

    # ── Input Validation ─────────────────────────────────────────────

    def _validate_inputs(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: Optional[list[FinalPersona]],
    ) -> None:
        """Comprehensive input validation"""
        # Feature validation
        if not feature:
            raise ValueError("feature is required")

        if not feature.title or len(feature.title) < 3:
            raise ValueError("feature.title must be at least 3 characters")

        if not feature.description or len(feature.description) < 10:
            raise ValueError("feature.description must be at least 10 characters")

        if feature.effort_weeks_min is None or feature.effort_weeks_max is None:
            raise ValueError("feature effort estimates are required")

        if not isinstance(feature.effort_weeks_min, (int, float)):
            raise ValueError("effort_weeks_min must be numeric")

        if not isinstance(feature.effort_weeks_max, (int, float)):
            raise ValueError("effort_weeks_max must be numeric")

        if feature.effort_weeks_min < 1:
            raise ValueError("effort_weeks_min must be >= 1")

        if feature.effort_weeks_max < feature.effort_weeks_min:
            raise ValueError("effort_weeks_max must be >= effort_weeks_min")

        # Company validation
        if not company:
            raise ValueError("company context is required")

        if not company.company_name:
            logger.warning("company.company_name is empty")

        if not company.team_size or company.team_size < 1:
            raise ValueError("company.team_size must be >= 1")

        # Graph validation
        if not graph:
            raise ValueError("knowledge graph is required")

        if not graph.nodes:
            raise ValueError("knowledge graph has no entities")

        if len(graph.nodes) < 5:
            logger.warning("knowledge graph has few entities: %d", len(graph.nodes))

        # Bundle validation
        if not bundle:
            raise ValueError("problem context bundle is required")

        if not bundle.chunks:
            raise ValueError("problem context bundle has no chunks")

        if len(bundle.chunks) < 10:
            logger.warning("problem context bundle has few chunks: %d", len(bundle.chunks))

        # Personas validation
        if personas is None:
            personas = []

        if not isinstance(personas, list):
            raise ValueError("personas must be a list or None")

    # ── Result Validation ────────────────────────────────────────────

    def _validate_gate_result(self, result: Any) -> bool:
        """Validate gate result structure and values"""
        if result is None:
            logger.warning("Gate returned None result")
            return False

        if not hasattr(result, "gate_id") or not result.gate_id:
            logger.warning("Gate result missing gate_id")
            return False

        if not hasattr(result, "gate_name") or not result.gate_name:
            logger.warning("Gate result missing gate_name")
            return False

        if not hasattr(result, "verdict") or result.verdict is None:
            logger.warning("Gate result missing verdict")
            return False

        if not self._is_valid_verdict(result.verdict):
            logger.warning("Gate result has unknown verdict: %s", result.verdict)
            return False

        if not hasattr(result, "score"):
            logger.warning("Gate result missing score")
            return False

        try:
            score = float(result.score)
            if not (0.0 <= score <= 10.0):
                logger.warning("Gate score out of range [0, 10]: %.2f", score)
                return False
        except (TypeError, ValueError):
            logger.warning("Gate score not numeric: %s", result.score)
            return False

        if hasattr(result, "reasoning"):
            if result.reasoning and not isinstance(result.reasoning, str):
                logger.warning("Gate reasoning not a string")
                return False

        if hasattr(result, "details"):
            if result.details and not isinstance(result.details, dict):
                logger.warning("Gate details not a dict")
                return False

        return True

    def _is_valid_verdict(self, verdict: Any) -> bool:
        """Check if verdict is a known GateVerdict"""
        valid_verdicts = {
            GateVerdict.PASS,
            GateVerdict.PASS_WITH_MITIGATION,
            GateVerdict.FEASIBLE_WITH_DEBT,
            GateVerdict.EXISTS_NEEDS_ADAPTATION,
            GateVerdict.STRONG_FIT,
            GateVerdict.MANAGEABLE,
            GateVerdict.RISKY,
            GateVerdict.FAIL,
        }
        return verdict in valid_verdicts

    def _is_passing_verdict(self, verdict: GateVerdict) -> bool:
        """Check if verdict counts as passing"""
        return verdict in self.PASSING_VERDICTS

    def _create_error_result(
        self,
        gate_id: str,
        gate_name: str,
        error_msg: str,
        error_type: str = "unknown",
    ) -> GateResult:
        """Create standardized error result"""
        return GateResult(
            gate_id=gate_id,
            gate_name=gate_name,
            verdict=GateVerdict.RISKY,
            score=0.2,  # Very low score for errors
            details={
                "error_type": error_type,
                "error_message": error_msg[:500],
                "is_error_result": True,
                "reasoning": f"Gate evaluation failed: {error_msg[:100]}", # mapped from generic reasoning
            },
        )

    # ── Scoring ──────────────────────────────────────────────────────

    def _calculate_overall_score(
        self,
        results: list[GateResult],
    ) -> tuple[float, dict[str, Any]]:
        """Calculate weighted overall score with quality metrics"""
        if not results:
            return 0.0, {"error": "No gate results"}

        # Gate weights
        weights = {
            "technical_feasibility": 1.2,
            "sota_alignment": 1.0,
            "battery_network_constraints": 1.1,
            "market_demand_alignment": 1.2,
            "monte_carlo_simulation": 0.9,
            "red_team_analysis": 1.0,
            "feature_interaction_analysis": 0.9,
            "execution_readiness": 1.1,
        }

        total_weight = 0.0
        weighted_sum = 0.0
        error_results = []
        valid_results = []

        for result in results:
            if self._is_error_result(result):
                error_results.append(result)
                continue

            gate_id = result.gate_id or "unknown"
            weight = weights.get(gate_id, 1.0)

            try:
                score = float(result.score)
            except (TypeError, ValueError):
                logger.warning("Skipping result with non-numeric score: %s", gate_id)
                error_results.append(result)
                continue

            weighted_sum += score * weight
            total_weight += weight
            valid_results.append(result)

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        error_count = len(error_results)
        if error_count > 0:
            penalty = min(overall_score, error_count * 0.15)
            overall_score = max(0.0, overall_score - penalty)
            logger.warning(
                "Applied %.2f penalty for %d gate errors", penalty, error_count
            )

        overall_score = max(0.0, min(10.0, overall_score))

        return round(overall_score, 2), {
            "total_gates": len(results),
            "valid_gates": len(valid_results),
            "error_gates": error_count,
            "weighted_sum": round(weighted_sum, 2),
            "total_weight": round(total_weight, 2),
            "error_penalty": round(error_count * 0.15, 2),
        }

    def _is_error_result(self, result: GateResult) -> bool:
        """Check if result is an error result"""
        if result.score <= 0.2 and result.verdict == GateVerdict.RISKY:
            return True

        if result.details and result.details.get("is_error_result"):
            return True

        return False

    # ── Gate Execution ───────────────────────────────────────────────

    async def _execute_gates_sequential(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
        num_simulations: Optional[int] = None,
    ) -> tuple[list[GateResult], dict[str, float]]:
        """Execute gates sequentially"""
        results: list[GateResult] = []
        gate_timings: dict[str, float] = {}
        execution_order = []

        for idx, gate_cls in enumerate(ALL_GATES, 1):
            gate_id = gate_cls.gate_id if hasattr(gate_cls, "gate_id") else f"gate_{idx}"
            gate_name = gate_cls.gate_name if hasattr(gate_cls, "gate_name") else f"Gate {idx}"

            execution_order.append(gate_id)
            gate_start = time.time()

            try:
                logger.debug("Executing gate %d/%d: %s", idx, len(ALL_GATES), gate_id)

                result = await self._execute_gate_with_cache(
                    gate_cls, feature, company, graph, bundle, personas, num_simulations
                )

                gate_duration = time.time() - gate_start
                gate_timings[gate_id] = gate_duration

                if not self._validate_gate_result(result):
                    logger.warning(
                        "Gate %s returned invalid result, creating error result",
                        gate_id,
                    )
                    result = self._create_error_result(
                        gate_id, gate_name, "Invalid result structure"
                    )

                results.append(result)

                logger.info(
                    "Gate %s: %s (score: %.2f, time: %.2fs)",
                    gate_id,
                    result.verdict.value,
                    result.score,
                    gate_duration,
                )

            except asyncio.TimeoutError as e:
                gate_duration = time.time() - gate_start
                gate_timings[gate_id] = gate_duration

                logger.error(
                    "Gate %s timeout after %.2fs: %s",
                    gate_id,
                    gate_duration,
                    str(e),
                    exc_info=True,
                )

                results.append(
                    self._create_error_result(
                        gate_id,
                        gate_name,
                        f"Gate timeout after {gate_duration:.1f}s",
                        error_type="timeout",
                    )
                )

            except Exception as e:
                gate_duration = time.time() - gate_start
                gate_timings[gate_id] = gate_duration

                logger.error(
                    "Gate %s failed after %.2fs: %s (%s)",
                    gate_id,
                    gate_duration,
                    str(e),
                    type(e).__name__,
                    exc_info=True,
                )

                results.append(
                    self._create_error_result(
                        gate_id,
                        gate_name,
                        str(e)[:200],
                        error_type=type(e).__name__,
                    )
                )

        if execution_order != [
            g.gate_id if hasattr(g, "gate_id") else f"gate_{i+1}"
            for i, g in enumerate(ALL_GATES)
        ]:
            logger.warning("Gate execution order mismatch")

        return results, gate_timings

    async def _execute_gates_parallel(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
        num_simulations: Optional[int] = None,
    ) -> tuple[list[GateResult], dict[str, float]]:
        """Execute gates in parallel for speed"""
        gate_timings: dict[str, float] = {}

        async def execute_single_gate(gate_cls, index):
            gate_id = gate_cls.gate_id if hasattr(gate_cls, "gate_id") else f"gate_{index}"
            gate_start = time.time()
            try:
                result = await self._execute_gate_with_cache(
                    gate_cls, feature, company, graph, bundle, personas
                )
                
                gate_duration = time.time() - gate_start
                gate_timings[gate_id] = gate_duration

                if not self._validate_gate_result(result):
                    result = self._create_error_result(
                        gate_id,
                        getattr(gate_cls, "gate_name", f"Gate {index}"),
                        "Invalid result",
                    )
                return result
            except Exception as e:
                gate_duration = time.time() - gate_start
                gate_timings[gate_id] = gate_duration
                
                logger.error("Parallel gate execution failed for %s: %s", gate_id, e)
                return self._create_error_result(
                    gate_id,
                    getattr(gate_cls, "gate_name", f"Gate {index}"),
                    str(e),
                )

        tasks = [
            execute_single_gate(gate_cls, idx + 1)
            for idx, gate_cls in enumerate(ALL_GATES)
        ]

        results = await asyncio.gather(*tasks)
        valid_results = [r for r in results if r is not None]

        return valid_results, gate_timings

    async def _execute_gate_with_cache(
        self,
        gate_cls,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
        num_simulations: Optional[int] = None,
    ) -> Optional[GateResult]:
        """Execute gate with caching"""
        gate_id = gate_cls.gate_id if hasattr(gate_cls, "gate_id") else "unknown"
        cache_key = self._get_cache_key(feature, gate_id)

        if self._enable_caching and cache_key in self._gate_cache:
            result, cached_time = self._gate_cache[cache_key]

            if self._is_cache_valid(cached_time):
                logger.debug("Using cached result for %s", gate_id)
                return result
            else:
                logger.debug("Cache expired for %s", gate_id)
                del self._gate_cache[cache_key]

        # Use factory for proper instantiation with parameters (Fix CRITICAL)
        gate = self._gate_factory.create_gate(gate_cls, num_agents_override=num_simulations)
            
        result = await gate.evaluate(feature, company, graph, bundle, personas)

        if self._enable_caching and result:
            self._gate_cache[cache_key] = (result, time.time())
            logger.debug("Cached result for %s", gate_id)

        return result

    def _get_cache_key(self, feature: FeatureProposal, gate_id: str) -> str:
        """Generate cache key for gate result"""
        return f"{feature.title}_{gate_id}".lower()

    def _is_cache_valid(self, cached_time: float) -> bool:
        """Check if cached result is still valid"""
        return (time.time() - cached_time) < self._cache_ttl

    # ── Result Processing ────────────────────────────────────────────

    def _deduplicate_results(
        self,
        results: list[GateResult],
    ) -> tuple[list[GateResult], int]:
        """Remove duplicate gate results, keeping first occurrence"""
        seen_ids = {}
        duplicates = 0

        for result in results:
            gate_id = result.gate_id

            if gate_id in seen_ids:
                logger.warning(
                    "Duplicate gate result: %s (keeping first, skipping duplicate)",
                    gate_id,
                )
                duplicates += 1
                continue

            seen_ids[gate_id] = result

        deduplicated = list(seen_ids.values())

        if duplicates > 0:
            logger.warning("Removed %d duplicate results", duplicates)

        return deduplicated, duplicates

    def _generate_recommendation(
        self,
        results: list[GateResult],
        all_passed: bool,
        overall_score: float,
    ) -> tuple[str, str]:
        """Generate detailed recommendation with reasoning"""
        if all_passed:
            return (
                "PROCEED_TO_DEBATE",
                "All gates passed. Feature is ready for stakeholder discussion and debate.",
            )

        failed_results = [
            r for r in results if not self._is_passing_verdict(r.verdict)
        ]
        failed_ids = [r.gate_id for r in failed_results]

        critical_gates = {
            "technical_feasibility",
            "execution_readiness",
            "battery_network_constraints",
        }

        critical_failures = [f for f in failed_ids if f in critical_gates]

        if len(critical_failures) >= 2:
            return (
                "NEEDS_MAJOR_REFINEMENT",
                f"Multiple critical gates failed: {', '.join(critical_failures)}. "
                f"Feature needs substantial rework.",
            )

        if overall_score < 0.20:
            return (
                "RECONSIDER_FEATURE",
                f"Overall score very low ({overall_score:.2f}/1.0). "
                f"Consider major redesign or shelving feature.",
            )

        if overall_score < 0.40:
            return (
                "NEEDS_MAJOR_REFINEMENT",
                f"Overall score low ({overall_score:.2f}/1.0). "
                f"Significant refinement needed before proceeding.",
            )

        if len(critical_failures) == 1:
            return (
                "NEEDS_REFINEMENT",
                f"Critical gate failed: {critical_failures[0]}. "
                f"Target this gate for refinement.",
            )

        if len(failed_ids) > len(results) / 2:
            return (
                "NEEDS_MAJOR_REFINEMENT",
                f"More than half of gates failed ({len(failed_ids)}/{len(results)}). "
                f"Major refinement required.",
            )

        if overall_score >= 0.60:
            return (
                "NEEDS_MINOR_REFINEMENT",
                f"Score acceptable ({overall_score:.2f}/1.0) with minor issues. "
                f"Address {len(failed_ids)} failing gate(s) and proceed.",
            )

        return (
            "NEEDS_REFINEMENT",
            f"{len(failed_ids)} gate(s) failed. "
            f"Targeted refinement needed (score: {overall_score:.2f}/1.0).",
        )

    def _get_slowest_gates(
        self,
        gate_timings: dict[str, float],
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """Get slowest gates for performance analysis"""
        if not gate_timings:
            return []

        sorted_gates = sorted(gate_timings.items(), key=lambda x: x[1], reverse=True)

        result = []
        for gate_id, duration in sorted_gates[:top_n]:
            result.append(
                {"gate_id": gate_id, "duration_seconds": round(duration, 2)}
            )

        return result

    def _log_performance_metrics(
        self,
        results: list[GateResult],
        gate_timings: dict[str, float],
        total_time: float,
    ) -> None:
        """Log detailed performance metrics"""
        if not gate_timings:
            return

        times = list(gate_timings.values())
        avg_time = sum(times) / len(times) if times else 0
        max_time = max(times) if times else 0
        min_time = min(times) if times else 0

        logger.info(
            "Gate execution performance: total=%.2fs, avg=%.2fs, min=%.2fs, max=%.2fs",
            total_time,
            avg_time,
            min_time,
            max_time,
        )

        slowest = sorted(gate_timings.items(), key=lambda x: x[1], reverse=True)[:3]

        for gate_id, duration in slowest:
            logger.debug("Slow gate: %s (%.2fs)", gate_id, duration)

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_layer4_diagnostics(self, summary: GatesSummary) -> dict[str, Any]:
        """Get comprehensive gate execution diagnostics"""
        if not hasattr(summary, "_diagnostics"):
            return {"error": "No diagnostics available"}

        diags = summary._diagnostics

        verdict_counts = {}
        for result in summary.results:
            verdict = result.verdict.value
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        scores = [r.score for r in summary.results]

        return {
            "total_gates": len(summary.results),
            "passed_gates": len(summary.passed_gates),
            "failed_gates": len(summary.failed_gates),
            "all_passed": summary.all_passed,
            "overall_score": summary.overall_score,
            "score_statistics": {
                "mean": round(sum(scores) / len(scores), 2) if scores else 0,
                "min": round(min(scores), 2) if scores else 0,
                "max": round(max(scores), 2) if scores else 0,
            },
            "verdict_distribution": verdict_counts,
            "scoring_metrics": diags.get("scoring_metrics", {}),
            "gate_timings": diags.get("gate_timings", {}),
            "slowest_gates": diags.get("slowest_gates", []),
            "total_execution_time": diags.get("total_time", 0),
            "recommendation": summary.recommendation,
        }

    # ── Main Process ─────────────────────────────────────────────────

    async def process(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
        num_simulations: Optional[int] = None,
    ) -> GatesSummary:
        """Execute all gates with full validation"""
        t0 = time.time()

        mode = "parallel" if self._parallel else "sequentially"
        logger.info("Layer 4: Executing %d gates %s", len(ALL_GATES), mode)

        try:
            self._validate_inputs(feature, company, graph, bundle, personas)
            logger.info("✓ Validated all inputs")
        except ValueError as e:
            logger.error("Input validation failed: %s", e)
            raise

        if self._parallel:
            results, gate_timings = await self._execute_gates_parallel(
                feature, company, graph, bundle, personas, num_simulations
            )
        else:
            results, gate_timings = await self._execute_gates_sequential(
                feature, company, graph, bundle, personas, num_simulations
            )

        logger.info("✓ All %d gates executed", len(results))

        results, duplicates = self._deduplicate_results(results)

        if duplicates > 0:
            logger.warning("Removed %d duplicate gate results", duplicates)

        overall_score, scoring_metrics = self._calculate_overall_score(results)

        passed_gates = [
            r.gate_id for r in results if self._is_passing_verdict(r.verdict)
        ]
        failed_gates = [
            r.gate_id for r in results if not self._is_passing_verdict(r.verdict)
        ]

        all_passed = len(failed_gates) == 0

        recommendation, recommendation_reason = self._generate_recommendation(
            results, all_passed, overall_score
        )

        summary = GatesSummary(
            results=results,
            overall_score=overall_score,
            all_passed=all_passed,
            failed_gates=failed_gates,
            passed_gates=passed_gates,
            needs_refinement=not all_passed,
            recommendation=recommendation,
            recommendation_reason=recommendation_reason,
        )

        # Inject diagnostics (allowed by ConfigDict extra="allow")
        summary._diagnostics = {
            "scoring_metrics": scoring_metrics,
            "gate_timings": gate_timings,
            "total_time": round(time.time() - t0, 2),
            "slowest_gates": self._get_slowest_gates(gate_timings),
        }

        self._log_performance_metrics(results, gate_timings, time.time() - t0)

        elapsed = time.time() - t0
        logger.info(
            "Layer 4: %d/%d gates passed, "
            "overall score: %.2f/1.0, recommendation: %s (%.1fs)",
            len(passed_gates),
            len(results),
            overall_score,
            recommendation,
            elapsed,
        )

        return summary

    async def process_failed_only(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
        previous_summary: GatesSummary,
        num_simulations: Optional[int] = None,
    ) -> GatesSummary:
        """Re-run only previously failed gates with cache isolation.

        Phase 15 Bug Fixes:
          - Bug #1: Uses gate_id (lowercase) not GATE_ID — fixes zero-match issue
          - Bug #3: Evicts stale cache entries before re-run — forces fresh LLM evaluation
          - Bug #4: Uses GateFactory.create_gate() — fixes TypeError for Monte Carlo gate
        """
        if not previous_summary.failed_gates:
            logger.info("No failed gates to re-evaluate — returning original summary")
            return previous_summary

        failed_ids = set(previous_summary.failed_gates)
        logger.info("Selective re-evaluation: re-running %d failed gates: %s",
                    len(failed_ids), failed_ids)

        t0 = time.time()
        re_run_results: list[GateResult] = []
        re_run_timings: dict[str, float] = {}

        for gate_class in ALL_GATES:
            # BUG #1 FIX: use lowercase gate_id (was GATE_ID — never existed on any class)
            gate_id = getattr(gate_class, 'gate_id', None) or getattr(gate_class, 'GATE_ID', gate_class.__name__)
            if gate_id not in failed_ids:
                continue

            # BUG #3 FIX: Evict stale cache so the refined feature gets a fresh LLM call
            stale_cache_key = self._get_cache_key(feature, gate_id)
            if stale_cache_key in self._gate_cache:
                logger.debug("Evicting stale cache for refined re-run: %s", gate_id)
                del self._gate_cache[stale_cache_key]

            gt0 = time.time()
            try:
                # BUG #4 FIX: Use GateFactory instead of bare gate_class(self._llm)
                gate = self._gate_factory.create_gate(gate_class, num_agents_override=num_simulations)
                result = await gate.evaluate(feature, company, graph, bundle, personas)
                re_run_results.append(result)
                re_run_timings[gate_id] = time.time() - gt0
                logger.info("Re-ran gate %s: score=%.2f, verdict=%s",
                            gate_id, result.score, result.verdict.value)
            except Exception as e:
                logger.error("Failed to re-run gate %s: %s", gate_id, e, exc_info=True)
                # Keep original failed result rather than swallowing silently
                for orig in previous_summary.results:
                    if orig.gate_id == gate_id:
                        re_run_results.append(orig)
                        break
                re_run_timings[gate_id] = time.time() - gt0

        # Merge: keep passed results from previous run + re-run results
        merged: list[GateResult] = []
        re_run_ids = {r.gate_id for r in re_run_results}
        for orig in previous_summary.results:
            if orig.gate_id in re_run_ids:
                # Use re-run result
                for rr in re_run_results:
                    if rr.gate_id == orig.gate_id:
                        merged.append(rr)
                        break
            else:
                merged.append(orig)

        # Safety: include any re-run results not found in original list
        merged_ids = {r.gate_id for r in merged}
        for rr in re_run_results:
            if rr.gate_id not in merged_ids:
                merged.append(rr)

        # Recalculate summary
        overall_score, scoring_metrics = self._calculate_overall_score(merged)
        passed_gates = [r.gate_id for r in merged if self._is_passing_verdict(r.verdict)]
        new_failed = [r.gate_id for r in merged if not self._is_passing_verdict(r.verdict)]
        all_passed = len(new_failed) == 0
        recommendation, reason = self._generate_recommendation(merged, all_passed, overall_score)

        summary = GatesSummary(
            results=merged,
            overall_score=overall_score,
            all_passed=all_passed,
            failed_gates=new_failed,
            passed_gates=passed_gates,
            needs_refinement=not all_passed,
            recommendation=recommendation,
            recommendation_reason=reason,
        )

        summary._diagnostics = {
            "scoring_metrics": scoring_metrics,
            "gate_timings": re_run_timings,
            "total_time": round(time.time() - t0, 2),
            "slowest_gates": self._get_slowest_gates(re_run_timings),
            "selective_reeval": True,
            "original_failed": list(failed_ids),
            "still_failing": new_failed,
        }

        logger.info(
            "Selective re-eval complete: %d/%d gates now pass "
            "(was %d/%d), score: %.2f/1.0 (was %.2f/1.0)",
            len(passed_gates), len(merged),
            len(previous_summary.passed_gates), len(previous_summary.results),
            overall_score, previous_summary.overall_score,
        )

        return summary
