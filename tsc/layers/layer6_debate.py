"""Layer 6: Debate & Consensus.

Stakeholders debate the feature through 3 rounds:
1. Initial positions
2. Negotiation
3. Final consensus

Critical fixes:
  1. Comprehensive input validation
  2. LLM response validation with fallbacks
  3. Robust verdict extraction with confidence
  4. Consensus quality validation
  5. Intelligent concern/condition extraction
  6. Nuanced confidence calculation
  7. Debate coherence verification
  8. Smart phase planning based on consensus
  9. Debate summary and synthesis

Optimizations:
  1. Debate result caching
  2. Parallel stakeholder contributions
  3. Execution mode selection
  4. Diagnostics & Metrics
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Optional, Tuple

from tsc.llm.base import LLMClient
from tsc.llm.prompts import (
    DEBATE_ROUND1_USER,
    DEBATE_ROUND2_USER,
    DEBATE_ROUND3_USER,
    DEBATE_SYSTEM,
)
from tsc.models.debate import (
    ConsensusResult,
    DebatePosition,
    DebateRound,
    PhaseGate,
    PhaseSpecification,
    StakeholderApproval,
    SuccessCriteria,
)
from tsc.models.gates import GatesSummary
from tsc.models.graph import KnowledgeGraph
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona

logger = logging.getLogger(__name__)


class DebateEngine:
    """Layer 6: Orchestrate stakeholder debate and consensus"""

    def __init__(
        self,
        llm_client: LLMClient,
        enable_caching: bool = True,
        parallel_rounds: bool = True,
        cache_ttl_minutes: int = 60,
    ):
        self._llm = llm_client
        self._enable_caching = enable_caching
        self._parallel = parallel_rounds
        self._cache_ttl = cache_ttl_minutes * 60
        self._debate_cache: dict[str, Tuple[ConsensusResult, float]] = {}

        logger.info(
            "DebateEngine initialized (caching=%s, parallel=%s)",
            enable_caching,
            parallel_rounds,
        )

    # ── Input Validation ─────────────────────────────────────────────

    def _validate_inputs(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        personas: list[FinalPersona],
        gates_summary: GatesSummary,
    ) -> None:
        """Comprehensive input validation"""
        # Feature validation
        if not feature or not feature.title:
            raise ValueError("Feature proposal required with title")

        if not feature.description or len(feature.description) < 10:
            raise ValueError("Feature description required (min 10 chars)")

        # Company validation
        if not company:
            raise ValueError("Company context required")

        if not company.company_name:
            logger.warning("Company missing company_name")

        if not company.team_size or company.team_size < 1:
            raise ValueError("Company team_size invalid")

        # Graph validation
        if not graph:
            raise ValueError("Knowledge graph required")

        if not graph.nodes:
            raise ValueError("Knowledge graph has no entities")

        # Personas validation
        if not personas:
            raise ValueError("At least 1 stakeholder required")

        if not isinstance(personas, list):
            raise ValueError("Personas must be a list")

        for idx, persona in enumerate(personas):
            if not persona.name:
                raise ValueError(f"Persona {idx} missing name")
            if not persona.role:
                raise ValueError(f"Persona {idx} missing role")

        # Gates summary validation
        if not gates_summary:
            raise ValueError("Gates summary required")

        if not gates_summary.results:
            raise ValueError("Gates summary has no results")

        logger.info("✓ Input validation passed")

    # ── Statement Generation ─────────────────────────────────────────

    async def _get_stakeholder_statement(
        self,
        persona: FinalPersona,
        context_prompt: str,
        system_prompt: str,
        round_name: str,
    ) -> str:
        """Get LLM statement with fallback"""
        try:
            statement = await self._llm.generate(
                system_prompt=system_prompt,
                user_prompt=context_prompt,
                temperature=0.6,
                max_tokens=600,
            )

            # Validate response
            if not statement or len(statement.strip()) < 20:
                logger.warning(
                    "LLM returned empty/short response for %s, using default",
                    persona.name,
                )
                statement = self._generate_default_statement(persona, round_name)

            return statement.strip()

        except asyncio.TimeoutError:
            logger.warning(
                "LLM timeout for %s (%s), using default statement",
                persona.name,
                round_name,
            )
            return self._generate_default_statement(persona, round_name)

        except Exception as e:
            logger.warning(
                "LLM generation failed for %s: %s, using default",
                persona.name,
                e,
            )
            return self._generate_default_statement(persona, round_name)

    def _generate_default_statement(
        self,
        persona: FinalPersona,
        round_name: str,
    ) -> str:
        """Generate default statement when LLM fails"""
        stance = persona.psychological_profile.predicted_stance
        stance_text = stance.prediction.lower()

        if round_name == "Initial Positions":
            objections = (
                ", ".join(stance.potential_objections[:2])
                if stance.potential_objections
                else "timing"
            )
            conditions = (
                ", ".join(stance.likely_conditions[:2])
                if stance.likely_conditions
                else "resources"
            )

            return (
                f"I am {persona.name}, {persona.role}. "
                f"Based on my analysis, I believe this feature is {stance_text}. "
                f"Key concerns include: {objections}. "
                f"Important conditions: {conditions}."
            )

        elif round_name == "Negotiation":
            return (
                f"After considering colleagues' perspectives, "
                f"my position remains {stance_text}. "
                f"I acknowledge the concerns raised, particularly around "
                f"technical feasibility and resource constraints. "
                f"However, I believe the market opportunity outweighs these risks."
            )

        else:  # Final Consensus
            return (
                f"Given the full discussion, I believe the consensus should be "
                f"to proceed with {stance_text}. "
                f"We should address key risks through phased rollout and "
                f"continuous monitoring of success metrics."
            )

    # ── Verdict Extraction ───────────────────────────────────────────

    def _extract_verdict(self, text: str) -> Tuple[str, float]:
        """Extract verdict with confidence score"""
        if not text:
            return "CONDITIONAL_APPROVE", 0.3

        text_lower = text.lower()

        # Check for rejection (strongest signal)
        reject_patterns = [
            r"\b(reject|reject|dismiss|decline|decline|veto|against|cannot|not|no)\b",
            r"\b(would not|should not|must not|should decline)\b",
        ]

        for pattern in reject_patterns:
            if re.search(pattern, text_lower):
                confidence = 0.85
                logger.debug(
                    "Extracted verdict: REJECTED (confidence: %.2f)", confidence
                )
                return "REJECTED", confidence

        # Check for conditional approval
        conditional_patterns = [
            r"\b(conditional|depends|if|pending|contingent|only if|provided that)\b",
            r"\b(need to|must|should.*first|subject to)\b",
        ]

        conditional_found = False
        for pattern in conditional_patterns:
            if re.search(pattern, text_lower):
                conditional_found = True
                break

        if conditional_found:
            confidence = 0.75
            logger.debug(
                "Extracted verdict: CONDITIONAL_APPROVE (confidence: %.2f)", confidence
            )
            return "CONDITIONAL_APPROVE", confidence

        # Check for strong approval
        approve_patterns = [
            r"\b(approve|support|favor|endorse|agree|yes|positive)\b",
            r"\b(should proceed|we should|let\'s do|i recommend|i support)\b",
        ]

        for pattern in approve_patterns:
            if re.search(pattern, text_lower):
                confidence = 0.80
                logger.debug(
                    "Extracted verdict: APPROVED (confidence: %.2f)", confidence
                )
                return "APPROVED", confidence

        # Default: uncertain/conditional
        logger.debug(
            "Could not extract clear verdict, defaulting to CONDITIONAL_APPROVE"
        )
        return "CONDITIONAL_APPROVE", 0.50

    # ── Consensus Validation ────────────────────────────────────────

    def _validate_consensus(
        self,
        consensus_verdict: str,
        consensus_confidence: float,
        positions: list[DebatePosition],
    ) -> Tuple[bool, str]:
        """Validate consensus makes sense given positions"""
        if not positions:
            return True, "No positions to validate"

        approvals = sum(1 for p in positions if p.verdict == "APPROVED")
        rejections = sum(1 for p in positions if p.verdict == "REJECTED")
        conditionals = sum(
            1 for p in positions if p.verdict == "CONDITIONAL_APPROVE"
        )
        total = len(positions)

        issues = []

        if rejections == total and consensus_verdict == "APPROVED":
            issues.append(f"All {total} stakeholders rejected but consensus is APPROVED")

        if approvals == total and consensus_verdict == "REJECTED":
            issues.append(f"All {total} stakeholders approved but consensus is REJECTED")

        if approvals > total / 2 and consensus_verdict == "REJECTED":
            issues.append(
                f"Majority ({approvals}/{total}) approved but consensus is REJECTED"
            )

        if rejections > total / 2 and consensus_verdict == "APPROVED":
            issues.append(
                f"Majority ({rejections}/{total}) rejected but consensus is APPROVED "
                f"(should be CONDITIONAL at best)"
            )

        if approvals == total and consensus_confidence < 0.80:
            issues.append(
                f"All stakeholders approve but confidence is only {consensus_confidence}"
            )

        if rejections > 0 and consensus_confidence > 0.90:
            issues.append(
                f"Some stakeholders reject ({rejections}) but confidence is {consensus_confidence}"
            )

        if issues:
            logger.warning("Consensus validation issues: %s", issues)
            return False, "; ".join(issues)

        logger.info("✓ Consensus validation passed")
        return True, "Consensus logically consistent with positions"

    # ── Concern/Condition Extraction ─────────────────────────────────

    def _extract_concerns(self, text: str) -> list[str]:
        """Extract concerns with robust parsing"""
        if not text:
            return []

        concerns = []
        keywords = [
            "concern",
            "risk",
            "worry",
            "issue",
            "problem",
            "challenge",
            "blocker",
            "obstacle",
            "difficulty",
            "threat",
            "danger",
        ]

        sentences = re.split(r"[.!?]\s+", text)
        seen = set()

        for sentence in sentences:
            sentence_lower = sentence.lower()

            has_keyword = any(
                re.search(rf"\b{kw}\b", sentence_lower) for kw in keywords
            )

            if not has_keyword:
                continue

            clean = sentence.strip()
            clean = re.sub(r"^[-•*\s]+", "", clean).strip()
            clean = clean[:200]

            if len(clean) < 10:
                continue

            if clean.lower() in [
                "concerns include",
                "concerns are",
                "there are concerns",
            ]:
                continue

            if clean in seen:
                continue

            seen.add(clean)
            concerns.append(clean)

        logger.info("Extracted %d concerns", len(concerns))
        return concerns[:15]

    def _extract_conditions(self, text: str) -> list[str]:
        """Extract conditions/requirements with robust parsing"""
        if not text:
            return []

        conditions = []
        keywords = [
            "condition",
            "require",
            "must",
            "should",
            "need",
            "gate",
            "criteria",
            "prerequisite",
            "dependent",
            "provided that",
            "if",
            "contingent",
        ]

        sentences = re.split(r"[.!?]\s+", text)
        seen = set()

        for sentence in sentences:
            sentence_lower = sentence.lower()

            has_keyword = any(
                re.search(rf"\b{kw}\b", sentence_lower) for kw in keywords
            )

            if not has_keyword:
                continue

            clean = sentence.strip()
            clean = re.sub(r"^[-•*\s]+", "", clean).strip()
            clean = clean[:200]

            if len(clean) < 10:
                continue

            if len(clean) < 20 or "condition" not in clean.lower():
                if not any(
                    x in clean.lower()
                    for x in ["must", "should", "need", "require"]
                ):
                    continue

            if clean in seen:
                continue

            seen.add(clean)
            conditions.append(clean)

        logger.info("Extracted %d conditions", len(conditions))
        return conditions[:15]

    # ── Confidence Calculation ───────────────────────────────────────

    def _calculate_confidence(
        self,
        positions: list[DebatePosition],
        gates_summary: GatesSummary,
    ) -> float:
        """Calculate nuanced confidence score"""
        if not positions or not gates_summary:
            return 0.5

        gate_score = gates_summary.overall_score / 10.0

        verdicts = [p.verdict for p in positions]
        approval_count = sum(1 for v in verdicts if v == "APPROVED")
        conditional_count = sum(1 for v in verdicts if v == "CONDITIONAL_APPROVE")
        rejection_count = sum(1 for v in verdicts if v == "REJECTED")

        total = len(verdicts)

        alignment_score = (approval_count * 1.0 + conditional_count * 0.5) / total

        dissent_ratio = rejection_count / total
        dissent_penalty = 1.0 - (dissent_ratio * 0.5)

        confidence = (
            (gate_score * 0.40)
            + (alignment_score * 0.40)
            + (dissent_penalty * 0.20)
        )

        confidence = max(0.0, min(1.0, confidence))

        if rejection_count > 0 and gate_score > 0.8:
            confidence = min(confidence, 0.75)

        if approval_count == total and gate_score < 0.5:
            confidence = min(confidence, 0.70)

        logger.info(
            "Confidence calculation: gates=%.2f, alignment=%.2f, "
            "dissent=%.2f, verdicts=%d:%d:%d → %.2f",
            gate_score,
            alignment_score,
            dissent_penalty,
            approval_count,
            conditional_count,
            rejection_count,
            confidence,
        )

        return round(confidence, 2)

    # ── Debate Coherence ────────────────────────────────────────────

    def _verify_debate_coherence(
        self,
        round1: DebateRound,
        round2: DebateRound,
        round3: DebateRound,
    ) -> dict[str, Any]:
        """Verify debate evolved logically across rounds"""
        coherence_issues = []

        round1_verdicts = {
            p.stakeholder_name: p.verdict for p in round1.positions
        }

        round2_verdicts = {
            p.stakeholder_name: p.verdict for p in (round2.positions or [])
        }

        round3_verdict = (
            round3.positions[0].verdict if round3.positions else "UNKNOWN"
        )

        if round2_verdicts:
            position_shifts = sum(
                1
                for name in round1_verdicts
                if round1_verdicts[name] != round2_verdicts.get(name)
            )

            if position_shifts == 0:
                coherence_issues.append(
                    "No position shifts Round 1→2 (static debate)"
                )

        round2_approvals = sum(
            1 for v in round2_verdicts.values() if v == "APPROVED"
        )
        round2_rejections = sum(
            1 for v in round2_verdicts.values() if v == "REJECTED"
        )
        round2_total = len(round2_verdicts) if round2_verdicts else len(round1_verdicts)

        if round2_total > 0:
            approval_ratio = round2_approvals / round2_total
            rejection_ratio = round2_rejections / round2_total

            if rejection_ratio > 0.5 and round3_verdict == "APPROVED":
                coherence_issues.append(
                    f"Round 3 contradicts Round 2: "
                    f"{round2_rejections}/{round2_total} rejected but "
                    f"consensus is APPROVED"
                )

            if approval_ratio > 0.5 and round3_verdict == "REJECTED":
                coherence_issues.append(
                    f"Round 3 contradicts Round 2: "
                    f"{round2_approvals}/{round2_total} approved but "
                    f"consensus is REJECTED"
                )

        for round_obj in [round1, round2, round3]:
            for pos in round_obj.positions:
                if not pos.statement or len(pos.statement) < 20:
                    coherence_issues.append(
                        f"Empty/short statement from {pos.stakeholder_name} "
                        f"in Round {round_obj.round_number}"
                    )

        if coherence_issues:
            logger.warning("Debate coherence issues: %s", coherence_issues)
        else:
            logger.info("✓ Debate coherence validation passed")

        return {
            "coherent": len(coherence_issues) == 0,
            "issues": coherence_issues,
            "position_shifts": position_shifts if round2_verdicts else 0,
        }

    # ── Phase Planning ──────────────────────────────────────────────

    def _generate_phase_plan(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        consensus_verdict: str,
        positions: list[DebatePosition],
    ) -> list[PhaseSpecification]:
        """Generate phased rollout plan based on consensus"""
        phases = []
        base_weeks = feature.effort_weeks_min or 4

        if consensus_verdict == "APPROVED":
            phases.append(
                PhaseSpecification(
                    name=f"{feature.title}",
                    timeline=f"{base_weeks}-{feature.effort_weeks_max or base_weeks + 2} weeks",
                    scope=["Complete feature set with all planned functionality"],
                    cost_estimate=str(company.budget) if company.budget else "TBD",
                )
            )

        elif consensus_verdict == "CONDITIONAL_APPROVE":
            phases.append(
                PhaseSpecification(
                    name=f"{feature.title} - MVP",
                    timeline=f"{base_weeks} weeks",
                    scope=[self._get_mvp_scope(feature)],
                    cost_estimate=str(company.budget) if company.budget else "TBD",
                )
            )

            phases.append(
                PhaseSpecification(
                    name=f"{feature.title} - Phase 2",
                    timeline=f"{base_weeks + 2}-{base_weeks + 4} weeks",
                    scope=[self._get_phase2_scope(feature)],
                    cost_estimate="TBD",
                )
            )

        else:  # REJECTED
            phases.append(
                PhaseSpecification(
                    name=f"{feature.title} - Pilot",
                    timeline=f"{base_weeks // 2} weeks",
                    scope=[self._get_pilot_scope(feature)],
                    cost_estimate="Minimal",
                )
            )

        logger.info("Generated %d phases for %s", len(phases), feature.title)
        return phases

    def _get_mvp_scope(self, feature: FeatureProposal) -> str:
        """Define MVP scope (remove nice-to-haves)"""
        return (
            f"{feature.title} - Core Features Only: "
            f"Focus on {feature.target_users or 'core users'}. "
            f"Exclude: reporting, integrations, advanced options."
        )

    def _get_phase2_scope(self, feature: FeatureProposal) -> str:
        """Define Phase 2 scope"""
        return (
            f"{feature.title} - Enhancement Features: "
            f"Based on Phase 1 feedback. "
            f"Includes: reporting, integrations, user requested features."
        )

    def _get_pilot_scope(self, feature: FeatureProposal) -> str:
        """Define pilot scope (minimal viable)"""
        return (
            f"{feature.title} - Pilot Program: "
            f"Limited to {getattr(feature, 'target_user_count', 50)} selected users. "
            f"Minimal scope to validate core hypothesis."
        )

    # ── Summary Generation ──────────────────────────────────────────

    def _generate_debate_summary(
        self,
        rounds: list[DebateRound],
        final_verdict: str,
    ) -> dict[str, Any]:
        """Generate executive summary of debate"""
        summary = {
            "final_verdict": final_verdict,
            "round_summaries": [],
            "key_agreements": [],
            "key_disagreements": [],
            "consensus_reasoning": "",
            "next_steps": [],
        }

        all_positions = [p for r in rounds for p in r.positions]

        for round_obj in rounds:
            verdicts = [p.verdict for p in round_obj.positions]
            verdict_counts = {}

            for v in verdicts:
                verdict_counts[v] = verdict_counts.get(v, 0) + 1

            summary["round_summaries"].append(
                {
                    "round_number": round_obj.round_number,
                    "round_name": round_obj.round_name,
                    "verdict_breakdown": verdict_counts,
                    "stakeholder_count": len(round_obj.positions),
                }
            )

        approval_count = sum(1 for p in all_positions if p.verdict == "APPROVED")
        if approval_count > 0:
            approvers = [
                p.stakeholder_name
                for p in all_positions
                if p.verdict == "APPROVED"
            ]
            summary["key_agreements"].append(
                f"{approval_count} stakeholder(s) support: {', '.join(set(approvers))}"
            )

        rejection_count = sum(1 for p in all_positions if p.verdict == "REJECTED")
        if rejection_count > 0:
            rejectors = [
                p.stakeholder_name
                for p in all_positions
                if p.verdict == "REJECTED"
            ]
            summary["key_disagreements"].append(
                f"{rejection_count} stakeholder(s) have concerns: {', '.join(set(rejectors))}"
            )

        all_concerns = []
        for pos in all_positions:
            all_concerns.extend(pos.key_concerns or [])

        if all_concerns:
            unique_concerns = list(set(all_concerns))[:5]
            summary["key_concerns"] = unique_concerns

        unique_stakeholders_count = len(set(p.stakeholder_name for p in all_positions))

        if approval_count >= unique_stakeholders_count and unique_stakeholders_count > 0:
            summary["consensus_reasoning"] = (
                f"Full consensus: all {unique_stakeholders_count} stakeholders approve"
            )
        elif rejection_count >= unique_stakeholders_count and unique_stakeholders_count > 0:
            summary["consensus_reasoning"] = (
                f"Full rejection: all {unique_stakeholders_count} stakeholders have reservations"
            )
        else:
            summary["consensus_reasoning"] = (
                f"Mixed consensus: {approval_count} approvals, {rejection_count} rejections. "
                "Conditional approval prudent."
            )

        if final_verdict == "APPROVED":
            summary["next_steps"] = [
                "Engineering kickoff meeting",
                "Design finalization",
                "Resource allocation",
                "Sprint planning",
            ]
        elif final_verdict == "CONDITIONAL_APPROVE":
            summary["next_steps"] = [
                "Address key concerns",
                "MVP scope definition",
                "Pilot user selection",
                "Success metrics definition",
            ]
        else:
            summary["next_steps"] = [
                "Redesign review",
                "Stakeholder discussion",
                "Market validation",
                "Reconsideration timeline",
            ]

        return summary

    # ── Helpers ──────────────────────────────────────────────────────

    def _build_system_prompt(self, persona: FinalPersona) -> str:
        """Build system prompt"""
        return DEBATE_SYSTEM.render(
            name=persona.name,
            role=persona.role,
            title=persona.role,
        )

    def _get_top_entities(self, graph: KnowledgeGraph) -> list[dict]:
        """Get top entities from graph"""
        top_entities = sorted(
            graph.nodes.values(), key=lambda e: e.mentions, reverse=True
        )[:10]

        return [{"name": e.name, "mentions": e.mentions} for e in top_entities]

    def _build_approvals(
        self, positions: list[DebatePosition]
    ) -> list[StakeholderApproval]:
        """Build approval objects"""
        return [
            StakeholderApproval(
                stakeholder=p.stakeholder_name,
                role=p.role,
                verdict=p.verdict,
                confidence=p.confidence,
                conditions=p.conditions,
            )
            for p in positions
        ]

    # ── Caching ─────────────────────────────────────────────────────

    def _get_cache_key(self, feature: FeatureProposal, persona_count: int) -> str:
        """Generate cache key for debate result"""
        return f"{feature.title}_{persona_count}".lower()

    def _is_cache_valid(self, cached_time: float) -> bool:
        """Check if cached debate is still valid"""
        return (time.time() - cached_time) < self._cache_ttl

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_layer6_diagnostics(self, consensus: ConsensusResult) -> dict[str, Any]:
        """Get comprehensive debate diagnostics"""
        if not consensus:
            return {"error": "No consensus result"}

        all_positions = []
        for round_obj in consensus.debate_rounds:
            all_positions.extend(round_obj.positions)

        verdicts = [p.verdict for p in all_positions]
        verdict_counts = {}
        for v in verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        confidences = [p.confidence for p in all_positions if p.confidence]

        phases_list = []
        if isinstance(consensus.phase_1, dict):
            phases_list = consensus.phase_1.get("phases", [])
        elif hasattr(consensus.phase_1, "dict"):
            p1_dict = consensus.phase_1.dict()
            phases_list = p1_dict.get("phases", [])
        
        return {
            "final_verdict": consensus.overall_verdict,
            "confidence": consensus.approval_confidence,
            "total_positions": len(all_positions),
            "verdict_breakdown": verdict_counts,
            "confidence_statistics": {
                "mean": round(sum(confidences) / len(confidences), 2)
                if confidences
                else 0,
                "min": round(min(confidences), 2) if confidences else 0,
                "max": round(max(confidences), 2) if confidences else 0,
            },
            "phases": len(phases_list) if phases_list else 1,
            "success_criteria": len(
                consensus.success_criteria.criteria
                if consensus.success_criteria
                else {}
            ),
            "mitigations": len(consensus.mitigations or []),
        }

    # ── Round Execution ────────────────────────────────────────────

    async def _round1_initial_positions(
        self,
        feature: FeatureProposal,
        personas: list[FinalPersona],
        gates_summary: GatesSummary,
        top_entities: list[dict],
    ) -> DebateRound:
        """Round 1: Sequential execution"""
        positions: list[DebatePosition] = []

        for persona in personas:
            system = self._build_system_prompt(persona)
            prompt = DEBATE_ROUND1_USER.render(
                name=persona.name,
                role=persona.role,
                profile_summary=persona.psychological_profile.full_profile_text[:500],
                feature=feature,
                gate_results=[
                    {
                        "gate_name": g.gate_name,
                        "verdict": g.verdict.value,
                        "score": g.score,
                    }
                    for g in gates_summary.results
                ],
                top_entities=top_entities,
            )

            statement = await self._get_stakeholder_statement(
                persona, prompt, system, "Initial Positions"
            )

            verdict, confidence = self._extract_verdict(statement)

            positions.append(
                DebatePosition(
                    stakeholder_name=persona.name,
                    role=persona.role,
                    statement=statement,
                    verdict=verdict,
                    confidence=confidence,
                    key_concerns=self._extract_concerns(statement),
                    conditions=self._extract_conditions(statement),
                )
            )

        return DebateRound(
            round_number=1,
            round_name="Initial Positions",
            positions=positions,
        )

    async def _round1_initial_positions_parallel(
        self,
        feature: FeatureProposal,
        personas: list[FinalPersona],
        gates_summary: GatesSummary,
        top_entities: list[dict],
    ) -> DebateRound:
        """Execute Round 1 in parallel for all stakeholders"""

        async def get_position(persona):
            system = self._build_system_prompt(persona)
            prompt = DEBATE_ROUND1_USER.render(
                name=persona.name,
                role=persona.role,
                profile_summary=persona.psychological_profile.full_profile_text[:500],
                feature=feature,
                gate_results=[
                    {
                        "gate_name": g.gate_name,
                        "verdict": g.verdict.value,
                        "score": g.score,
                    }
                    for g in gates_summary.results
                ],
                top_entities=top_entities,
            )

            statement = await self._get_stakeholder_statement(
                persona, prompt, system, "Initial Positions"
            )

            verdict, confidence = self._extract_verdict(statement)

            return DebatePosition(
                stakeholder_name=persona.name,
                role=persona.role,
                statement=statement,
                verdict=verdict,
                confidence=confidence,
                key_concerns=self._extract_concerns(statement),
                conditions=self._extract_conditions(statement),
            )

        tasks = [get_position(persona) for persona in personas]
        positions = await asyncio.gather(*tasks, return_exceptions=True)

        valid_positions = [p for p in positions if not isinstance(p, Exception)]

        if len(valid_positions) < len(positions):
            failed = len(positions) - len(valid_positions)
            logger.warning("Round 1: %d/%d positions failed", failed, len(positions))

        return DebateRound(
            round_number=1,
            round_name="Initial Positions",
            positions=valid_positions,
        )

    async def _round2_negotiation(
        self,
        feature: FeatureProposal,
        personas: list[FinalPersona],
        round1: DebateRound,
    ) -> DebateRound:
        """Round 2: Negotiation"""
        positions: list[DebatePosition] = []

        for persona in personas:
            others = [
                p for p in round1.positions if p.stakeholder_name != persona.name
            ]

            system = self._build_system_prompt(persona)
            prompt = DEBATE_ROUND2_USER.render(
                name=persona.name,
                role=persona.role,
                other_positions=others,
            )

            statement = await self._get_stakeholder_statement(
                persona, prompt, system, "Negotiation"
            )

            verdict, confidence = self._extract_verdict(statement)

            positions.append(
                DebatePosition(
                    stakeholder_name=persona.name,
                    role=persona.role,
                    statement=statement,
                    verdict=verdict,
                    confidence=confidence,
                    key_concerns=self._extract_concerns(statement),
                    conditions=self._extract_conditions(statement),
                )
            )

        return DebateRound(
            round_number=2,
            round_name="Negotiation",
            positions=positions,
        )

    async def _round3_consensus(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        personas: list[FinalPersona],
        round1: DebateRound,
        round2: DebateRound,
        gates_summary: GatesSummary,
    ) -> tuple[DebateRound, ConsensusResult]:
        """Round 3: Consensus"""
        leader = personas[0]
        all_positions = round2.positions or round1.positions

        system = self._build_system_prompt(leader)
        prompt = DEBATE_ROUND3_USER.render(
            name=leader.name,
            role=leader.role,
            all_positions=all_positions,
        )

        consensus_text = await self._get_stakeholder_statement(
            leader, prompt, system, "Final Consensus"
        )

        overall_verdict, ext_conf = self._extract_verdict(consensus_text)
        approvals = self._build_approvals(all_positions)
        confidence = self._calculate_confidence(all_positions, gates_summary)

        round3 = DebateRound(
            round_number=3,
            round_name="Final Consensus",
            positions=[
                DebatePosition(
                    stakeholder_name=leader.name,
                    role=leader.role,
                    statement=consensus_text,
                    verdict=overall_verdict,
                    confidence=confidence,
                )
            ],
            synthesis=consensus_text,
        )

        consensus = ConsensusResult(
            feature_name=feature.title,
            overall_verdict=overall_verdict,
            approval_confidence=confidence,
            stakeholder_verdicts={p.stakeholder_name: p.verdict for p in all_positions},
            approvals=approvals,
            phase_1=PhaseSpecification(
                name=f"{feature.title} - Phase 1",
                timeline=f"{feature.effort_weeks_min or 4} weeks",
                cost_estimate=str(company.budget) if company.budget else "TBD",
            ),
            success_criteria=SuccessCriteria(
                criteria={
                    "adoption_target": "40% within 6 weeks",
                    "quality_gate": "Zero critical incidents",
                }
            ),
            mitigations=[c for p in all_positions for c in p.conditions],
            next_steps=["Engineering kickoff", "Design review", "Development sprint"],
            debate_rounds=[round1, round2, round3],
        )

        return round3, consensus

    # ── Main Process ─────────────────────────────────────────────────

    async def process(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        personas: list[FinalPersona],
        gates_summary: GatesSummary,
    ) -> ConsensusResult:
        """Run debate with full validation integration."""
        t0 = time.time()

        logger.info("Layer 6: Starting debate with %d stakeholders", len(personas))

        try:
            self._validate_inputs(feature, company, graph, personas, gates_summary)
            logger.info("✓ Validated all inputs")
        except ValueError as e:
            logger.error("Input validation failed: %s", e)
            raise

        cache_key = self._get_cache_key(feature, len(personas))

        if self._enable_caching and cache_key in self._debate_cache:
            cached_result, cached_time = self._debate_cache[cache_key]
            if self._is_cache_valid(cached_time):
                logger.info("Using cached debate result")
                return cached_result
            else:
                del self._debate_cache[cache_key]

        top_entities = self._get_top_entities(graph)

        if self._parallel:
            logger.info("Running debate rounds in parallel mode")
            round1 = await self._round1_initial_positions_parallel(
                feature, personas, gates_summary, top_entities
            )
        else:
            logger.info("Running debate rounds in sequential mode")
            round1 = await self._round1_initial_positions(
                feature, personas, gates_summary, top_entities
            )

        logger.info("Round 1 complete: %d positions", len(round1.positions))

        round2 = await self._round2_negotiation(feature, personas, round1)
        logger.info("Round 2 complete: negotiation done")

        round3, consensus = await self._round3_consensus(
            feature, company, personas, round1, round2, gates_summary
        )

        # Apply intelligence validations
        is_consistent, issue_msg = self._validate_consensus(
            consensus.overall_verdict,
            consensus.approval_confidence,
            round2.positions or round1.positions,
        )

        coherence = self._verify_debate_coherence(round1, round2, round3)

        phases = self._generate_phase_plan(
            feature, company, consensus.overall_verdict, round2.positions or round1.positions
        )
        if phases:
            consensus.phase_1 = phases[0]
            # Extra phases can be stored on _diagnostics or a dynamic attr if you choose

        summary = self._generate_debate_summary(
            [round1, round2, round3], consensus.overall_verdict
        )
        consensus.next_steps = summary.get("next_steps", [])

        # Assign back extra data safely
        if not hasattr(consensus, "_diagnostics"):
            consensus._diagnostics = {}
        consensus._diagnostics["coherence"] = coherence
        consensus._diagnostics["summary"] = summary
        consensus._diagnostics["validation_consistent"] = is_consistent
        consensus._diagnostics["validation_message"] = issue_msg

        if self._enable_caching:
            self._debate_cache[cache_key] = (consensus, time.time())

        logger.info(
            "Layer 6 complete: %s (confidence: %.2f, %.1fs)",
            consensus.overall_verdict,
            consensus.approval_confidence,
            time.time() - t0,
        )

        return consensus
