"""
Phase 5: Epistemic Coverage Checker + Minority Voice Protocol.

Checks that every domain required by the TensionVector has at least
minimal coverage in the selected agent pool.

If a domain has ZERO coverage:
  → Activates the "Minority Voice Protocol" on 3 swing voters.
  → Injects a narrative "Lived Experience Fragment" instead of a bare fact.
  → This causes the concern to emerge NATURALLY during OASIS simulation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from tsc.models.personas import FinalPersona
from tsc.selection.models import EpistemicGap, PersonaPole, PoleType, TensionVector

logger = logging.getLogger(__name__)

# ─── Narrative fragments per domain ──────────────────────────────────
# These are injected as background memory into swing voters.

_DOMAIN_THREAT_FRAGMENTS: Dict[str, str] = {
    "Privacy": (
        "During your time at a data-sensitive startup, you witnessed the fallout from a tech company being fined "
        "€4.3 million by the Dutch DPA. The issue was the lack of explicit consent for continuous monitoring. "
        "The resulting fine destroyed team morale and triggered a mass resignation. You now instinctively "
        "look for surveillance 'mission creep' in every new tool."
    ),
    "Legal": (
        "You were once a key witness in an employment tribunal where an 'always-on' AI culture was "
        "judicially scrutinized. The court determined that mandatory tracking constituted constructive dismissal. "
        "This experience taught you that policies which appear 'lightweight' can create massive legal exposure. "
        "You now raise the alarm whenever a mandatory process is introduced without a legal impact assessment."
    ),
    "Labor Relations": (
        "In a previous role, you saw a team's velocity drop by 60% after the introduction of an automated 'review bot' "
        "that enforced rigid rules without understanding intent. The engineers began gaming the system, "
        "leading to a degradation in architectural quality. You now believe engineering automation should "
        "empower developers, not act as a gated supervisor."
    ),
    "Software Engineering": (
        "You've seen 'Agile transformation' fail when metrics became targets rather than tools. In one company, "
        "automated tracking led to 'metric hacking' where developers prioritized easy tasks over critical architecture. "
        "You are now highly skeptical of any tool that attempts to quantify developer performance without "
        "deep human context."
    ),
    "General": (
        "You have a habit of thinking through second-order consequences of any major policy change. You have seen "
        "well-intentioned systems fail because they didn't account for 'unintended' human behaviors. "
        "You often ask 'who else does this affect?' and 'what's the worst-case abuse?' before endorsing technology."
    ),
}

_DOMAIN_VALUE_FRAGMENTS: Dict[str, str] = {
    "Software Engineering": (
        "You were part of a legendary team that scaled a platform 10x in a year by ruthlessly automating "
        "boilerplate reviews and deployment pipelines. This 'Autopilot' approach freed you to focus on "
        "distributed systems architecture and high-level design. You are an evangelist for tools that "
        "remove the 'drudge work' and let engineers be creative again."
    ),
    "Data Science": (
        "You remember the 'Before Times' when data quality took up 80% of your week. Since adopting "
        "automated semantic validation, your team's innovative output has tripled. You've seen firsthand "
        "how shifting the burden of consistency to the system allows for massive leaps in model accuracy "
        "and personal career growth."
    ),
    "Business Strategy": (
        "In your last executive role, you saw a startup disrupt an entire industry by using AI to "
        "compress a three-week review cycle into ten minutes. The competitive advantage was insurmountable. "
        "You believe that in a saturated market, speed and automated precision aren't just features — they "
        "are the only way to survive and thrive."
    ),
    "General": (
        "You are an early adopter who has consistently profited from being ahead of the curve. You've seen "
        "how initial resistance to automation usually fades as the massive productivity gains become "
        "apparent. You enjoy being the 'Champion' for tools that represent the next logical leap in "
        "organizational efficiency."
    ),
}


class EpistemicCoverageChecker:
    """
    Phase 5 of PersonaSelectionEngine.

    API:
        check(personas, poles, tension_vector)
          → (updated_personas, List[EpistemicGap])
    """

    def check(
        self,
        personas: List[FinalPersona],
        poles: List[PersonaPole],
        tension_vector: TensionVector,
    ) -> tuple[List[FinalPersona], List[EpistemicGap]]:
        """
        Run coverage check. Updates swing voters in-place with
        narrative fragments for uncovered domains.
        """
        gaps: List[EpistemicGap] = []

        if not tension_vector.required_domains:
            logger.info("Phase 5: No required domains — skipping coverage check")
            return personas, gaps

        # Build coverage map: domain → count of personas that mention it
        coverage = self._measure_coverage(personas, tension_vector.required_domains)

        # Collect target personas for injection
        if poles:
            # Phase 5 (Expansion Phase): Target swing voters
            swing_voters = sorted(
                [p for p in poles if p.pole == PoleType.SWING],
                key=lambda p: abs(p.pole_score),
            )
            swing_names = {p.persona_name for p in swing_voters}
            target_personas = [p for p in personas if p.name.split(" [")[0] in swing_names
                            or p.name in swing_names]
            target_poles = swing_voters
        else:
            # Phase 1.7 (Seed Phase): Target all available seeds
            target_personas = personas
            target_poles = []

        for domain in tension_vector.required_domains:
            count = coverage.get(domain, 0)
            gap = EpistemicGap(domain=domain, covered=(count > 0), coverage_count=count)

            if count == 0:
                # Determine polarity: is this domain a "Value" or a "Threat" for this feature?
                polarity = self._get_domain_polarity(domain, tension_vector)
                
                logger.info(
                    "Phase 5: Epistemic gap for '%s' (polarity=%.2f) — activating Minority Voice", 
                    domain, polarity
                )
                
                activated, fragment = self._activate_minority_voice(
                    domain=domain,
                    polarity=polarity,
                    target_personas=target_personas,
                    target_poles=target_poles,
                )
                gap.minority_voice_activated = activated > 0
                gap.fragment_injected = fragment
                logger.info(
                    "Minority Voice Protocol: injected '%s' fragment into %d swing voters",
                    "Success Story" if polarity > 0 else "Lived Experience", activated
                )
            else:
                logger.info("Phase 5: Domain '%s' covered by %d agents ✓", domain, count)

            gaps.append(gap)

        return personas, gaps

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_domain_polarity(self, domain: str, tension_vector: TensionVector) -> float:
        """
        Calculate if a domain represents a 'Value' (>0) or 'Threat' (<0) 
        for this specific feature based on tension dimensions.
        """
        relevant_scores = []
        d_lower = domain.lower()
        
        for dim, score in tension_vector.dimensions.items():
            if d_lower in dim.lower():
                relevant_scores.append(score)
        
        if not relevant_scores:
            # Fallback: check if the domain is generally negative (Privacy/Legal) 
            # or positive (Strategy/Data Science)
            if d_lower in ["privacy", "legal", "labor relations"]:
                return -0.5
            return 0.5
            
        return sum(relevant_scores) / len(relevant_scores)

    def _measure_coverage(
        self,
        personas: List[FinalPersona],
        required_domains: List[str],
    ) -> Dict[str, int]:
        """Count how many personas mention each required domain."""
        coverage: Dict[str, int] = {d: 0 for d in required_domains}

        for persona in personas:
            text = self._get_all_text(persona).lower()
            for domain in required_domains:
                if domain.lower() in text:
                    coverage[domain] += 1

        return coverage

    def _activate_minority_voice(
        self,
        domain: str,
        polarity: float,
        target_personas: List[FinalPersona],
        target_poles: List[PersonaPole],
    ) -> tuple[int, str]:
        """
        Inject a narrative fragment into up to 3 swing voters.
        Returns (number of personas updated, the fragment used).
        """
        if polarity > 0:
            fragment = _DOMAIN_VALUE_FRAGMENTS.get(domain, _DOMAIN_VALUE_FRAGMENTS["General"])
            label = "Success Story"
            trigger_list_attr = "excited_by"
        else:
            fragment = _DOMAIN_THREAT_FRAGMENTS.get(domain, _DOMAIN_THREAT_FRAGMENTS["General"])
            label = "Lived Experience"
            trigger_list_attr = "scared_of"

        target_count = min(3, len(target_personas))
        targets = target_personas[:target_count]

        for persona in targets:
            original_bio = persona.psychological_profile.full_profile_text
            narrative = f"\n\n[{label} — {domain}]\n{fragment}"
            persona.psychological_profile.full_profile_text = original_bio + narrative

            # Add to emotional triggers
            existing = getattr(persona.psychological_profile.emotional_triggers, trigger_list_attr)
            trigger_label = f"{domain} impact"
            if trigger_label not in existing:
                existing.append(trigger_label)

        # Update pole metadata to record injection
        for target_pole in target_poles:
            if target_pole.persona_name in {p.name for p in targets}:
                target_pole.minority_voice_fragment = fragment

        return len(targets), fragment

    def _get_all_text(self, persona: FinalPersona) -> str:
        pp = persona.psychological_profile
        parts = [
            persona.role,
            pp.full_profile_text,
            " ".join(pp.key_traits),
            " ".join(pp.emotional_triggers.excited_by),
            " ".join(pp.emotional_triggers.frustrated_by),
            " ".join(pp.emotional_triggers.scared_of),
        ]
        return " ".join(p for p in parts if p)
