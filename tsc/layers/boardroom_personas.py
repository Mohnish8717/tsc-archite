"""
Boardroom Persona Factory
===========================

Generates 10 high-quality, opinionated executive personas for the AG2
Stakeholder Debate Engine. These are static C-suite personas dynamically
adapted to the specific company context (tech stack, competitors, priorities)
and feature under debate.

Design Philosophy:
  - Quality over quantity: 10 handcrafted executive archetypes, not LLM-generated generics.
  - Company-context injection: Each persona's bio is grounded in the actual company's
    domain, tech stack, competitors, and strategic priorities.
  - Feature-aware tension: Personas have natural stance biases based on how the feature
    intersects their domain — ensuring productive conflict in debate.
  - Compatible with run_live_debate.py quality bar (mk_persona pattern).
"""

from __future__ import annotations

import logging
from typing import Optional

from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona, PsychologicalProfile

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Boardroom Seat Definitions — 10 Fixed Executive Archetypes
# ═════════════════════════════════════════════════════════════════════════════

BOARDROOM_SEATS = [
    {
        "seat": "CEO",
        "name_template": "{prefix}_CEO",
        "role": "Chief Executive Officer",
        "role_short": "CEO",
        "expertise_base": ["Market Strategy", "Growth", "Fundraising"],
        "bio_template": (
            "Visionary founder obsessed with {priority_1} and convinced that the company's "
            "survival depends on outmaneuvering {competitor_1}. Thinks in 18-month windows. "
            "Will approve bold bets if the TAM math works, but has zero patience for "
            "engineering projects that can't show pipeline impact within one quarter. "
            "Privately worried about burn rate but will never admit it in front of the board."
        ),
        "stance_logic": "supportive_if_growth",  # Supports features that drive revenue/growth
        "influence": 0.95,
        "receptiveness": 0.4,
    },
    {
        "seat": "CTO",
        "name_template": "{prefix}_CTO",
        "role": "Chief Technology Officer",
        "role_short": "CTO",
        "expertise_base": ["Architecture", "Tech Debt", "Scalability"],
        "bio_template": (
            "Architect-turned-executive who built the original {tech_1} stack and now guards "
            "it like a cathedral. Deeply skeptical of any feature that introduces a new runtime "
            "dependency or breaks the {tech_2} contract. Has been burned by 'quick wins' that "
            "created 6 months of tech debt. Will demand an RFC with load-test projections before "
            "any green light. Secretly respects bold architectural moves but needs data to override "
            "his instinct to say no."
        ),
        "stance_logic": "skeptical_on_complexity",
        "influence": 0.90,
        "receptiveness": 0.35,
    },
    {
        "seat": "CISO",
        "name_template": "{prefix}_CISO",
        "role": "Chief Information Security Officer",
        "role_short": "CISO",
        "expertise_base": ["Data Privacy", "Compliance", "Threat Modeling"],
        "bio_template": (
            "Paranoid by profession and proud of it. Has blocked three feature launches in the "
            "last year for insufficient access controls. Views every new API surface as an attack "
            "vector. Insists on encryption-at-rest, audit logging, and pen-test signoff before "
            "any customer-facing change. Currently laser-focused on {compliance_concern} compliance "
            "and will torpedo anything that jeopardizes the company's certification posture. "
            "The only executive who reads CVE bulletins for fun."
        ),
        "stance_logic": "skeptical_on_security",
        "influence": 0.80,
        "receptiveness": 0.25,
    },
    {
        "seat": "CMO",
        "name_template": "{prefix}_CMO",
        "role": "Chief Medical/Marketing Officer",
        "role_short": "CMO",
        "expertise_base": ["Domain Expertise", "Customer Outcomes", "Quality"],
        "bio_template": (
            "The domain authority in the room. Hyper-focused on whether this feature actually "
            "solves the problem users described in the last 30 support escalations. Has seen too "
            "many 'innovative' features that users never adopted because product didn't talk to "
            "customers first. Will demand user testing evidence and refuse to endorse anything "
            "that adds cognitive load to the core workflow. Carries a mental database of every "
            "customer complaint from the last two quarters."
        ),
        "stance_logic": "conditional_on_user_evidence",
        "influence": 0.85,
        "receptiveness": 0.55,
    },
    {
        "seat": "CFO",
        "name_template": "{prefix}_CFO",
        "role": "Chief Financial Officer",
        "role_short": "CFO",
        "expertise_base": ["Unit Economics", "Burn Rate", "ROI Modeling"],
        "bio_template": (
            "Conservative spender who measures every initiative in cost-per-unit and payback "
            "period. Currently managing a {budget} budget with {runway_concern}. Will ask "
            "'what's the cost of NOT doing this?' before 'what does it cost?' Has killed "
            "projects that couldn't demonstrate 3x ROI within 12 months. Respects the CTO's "
            "technical judgment but will override it if the unit economics are compelling enough. "
            "Keeps a private spreadsheet comparing the company's margins to {competitor_1}'s."
        ),
        "stance_logic": "skeptical_on_cost",
        "influence": 0.85,
        "receptiveness": 0.40,
    },
    {
        "seat": "CPO",
        "name_template": "{prefix}_CPO",
        "role": "Chief Product Officer",
        "role_short": "CPO",
        "expertise_base": ["Product-Market Fit", "UX Strategy", "Roadmap Prioritization"],
        "bio_template": (
            "Obsessed with the 'Silent UX' — the experience users have when nobody is watching. "
            "Pushes back on features that add surface area without reducing time-to-value. "
            "Has strong opinions about feature discoverability and will reject any proposal that "
            "requires a training video to explain. Currently balancing {priority_1} against "
            "three other roadmap bets and will force-rank ruthlessly. Believes the best features "
            "are the ones users never notice because they just work."
        ),
        "stance_logic": "conditional_on_ux",
        "influence": 0.85,
        "receptiveness": 0.60,
    },
    {
        "seat": "Legal",
        "name_template": "{prefix}_Legal",
        "role": "General Counsel",
        "role_short": "Legal",
        "expertise_base": ["Regulatory Compliance", "Liability", "Contract Risk"],
        "bio_template": (
            "Scrutinizes every feature through the lens of 'what happens when this goes wrong "
            "and we get sued?' Has a mental model of every regulatory framework that touches "
            "the product ({compliance_concern}, data residency, third-party liability). Will "
            "demand a risk matrix and indemnification strategy before any customer-facing "
            "automation. Not anti-innovation — just anti-liability. Has saved the company from "
            "two potential lawsuits by catching issues that engineering dismissed as 'edge cases.'"
        ),
        "stance_logic": "skeptical_on_liability",
        "influence": 0.75,
        "receptiveness": 0.30,
    },
    {
        "seat": "Data",
        "name_template": "{prefix}_Data",
        "role": "Head of Data & ML",
        "role_short": "Data",
        "expertise_base": ["ML Accuracy", "Data Quality", "Model Bias"],
        "bio_template": (
            "Monitors every model output for bias, drift, and silent failures. Has seen "
            "'AI-powered' features ship with 60%% accuracy because nobody invested in evaluation "
            "harnesses. Will demand precision/recall benchmarks, edge-case coverage, and a "
            "rollback plan before any ML feature goes live. Currently concerned about the "
            "{tech_1} pipeline's data quality and whether the training distribution matches "
            "production. The person most likely to say 'your demo worked, but will it work "
            "at scale with dirty data?'"
        ),
        "stance_logic": "conditional_on_data_quality",
        "influence": 0.75,
        "receptiveness": 0.50,
    },
    {
        "seat": "Sales",
        "name_template": "{prefix}_Sales",
        "role": "Head of Sales",
        "role_short": "Sales",
        "expertise_base": ["Revenue Pipeline", "Customer Acquisition", "Competitive Win Rate"],
        "bio_template": (
            "Lives and dies by the quarterly number. Wants features that close deals — specifically "
            "the three enterprise prospects currently stuck in evaluation because the product "
            "lacks {priority_2}. Has zero interest in 'platform plays' that don't appear on a "
            "feature comparison matrix against {competitor_1}. Will champion anything that "
            "shortens the sales cycle or increases ACV, and will personally demo it to prospects "
            "the week it ships. Currently frustrated that engineering is building 'infrastructure' "
            "instead of 'features customers asked for.'"
        ),
        "stance_logic": "supportive_if_revenue",
        "influence": 0.70,
        "receptiveness": 0.70,
    },
    {
        "seat": "CustomerSuccess",
        "name_template": "{prefix}_CS",
        "role": "Head of Customer Success & Implementation",
        "role_short": "CS",
        "expertise_base": ["Onboarding", "Retention", "Change Management"],
        "bio_template": (
            "The voice of the customer's pain. Manages every implementation from kickoff to "
            "steady-state and knows exactly where users abandon workflows. Currently dealing "
            "with 3 at-risk accounts that cite '{priority_1}' as their primary frustration. "
            "Will support any feature that reduces churn, but will fight hard against anything "
            "that adds onboarding complexity or requires retraining existing users. Carries a "
            "mental NPS score for every customer and can predict churn 60 days out. "
            "The person most likely to say 'have you actually talked to a customer about this?'"
        ),
        "stance_logic": "conditional_on_adoption",
        "influence": 0.70,
        "receptiveness": 0.65,
    },
]


class BoardroomPersonaFactory:
    """Generates 10 company-context-aware executive personas for AG2 debate.

    These are NOT LLM-generated — they are handcrafted archetypes with
    company-specific context injection (tech stack, competitors, priorities)
    to produce opinionated, conflict-rich boardroom dynamics.
    """

    @staticmethod
    def create_boardroom(
        company: CompanyContext,
        feature: Optional[FeatureProposal] = None,
    ) -> list[FinalPersona]:
        """Create 10 executive personas adapted to the company context.

        Args:
            company: Company context for domain-specific grounding
            feature: Optional feature under debate for stance calibration

        Returns:
            List of 10 FinalPersona objects ready for AG2 debate
        """
        # Extract company context for template injection
        ctx = BoardroomPersonaFactory._extract_context(company, feature)

        personas: list[FinalPersona] = []
        for seat in BOARDROOM_SEATS:
            persona = BoardroomPersonaFactory._build_persona(seat, ctx, feature)
            personas.append(persona)

        logger.info(
            "BoardroomPersonaFactory: Created %d executive personas for %s",
            len(personas), company.company_name,
        )
        return personas

    @staticmethod
    def _extract_context(
        company: CompanyContext,
        feature: Optional[FeatureProposal] = None,
    ) -> dict:
        """Extract template variables from company context."""
        tech_stack = company.tech_stack or []
        competitors = company.competitors or []
        priorities = company.current_priorities or []

        # Derive compliance concern from domain signals
        desc_lower = (feature.description if feature else "").lower()
        company_lower = company.company_name.lower()
        all_text = desc_lower + " " + company_lower + " " + " ".join(tech_stack).lower()

        if any(k in all_text for k in ["hipaa", "health", "medical", "clinical", "patient", "ehr"]):
            compliance = "HIPAA/HITECH"
        elif any(k in all_text for k in ["pci", "payment", "fintech", "banking", "financial"]):
            compliance = "PCI-DSS/SOX"
        elif any(k in all_text for k in ["gdpr", "europe", "privacy"]):
            compliance = "GDPR/CCPA"
        elif any(k in all_text for k in ["government", "fedramp", "federal"]):
            compliance = "FedRAMP/FISMA"
        else:
            compliance = "SOC 2 Type II"

        # Derive runway concern from budget
        budget = company.budget or "undisclosed"
        budget_lower = budget.lower()
        if any(k in budget_lower for k in ["seed", "pre-", "angel"]):
            runway_concern = "18 months of runway remaining"
        elif any(k in budget_lower for k in ["series a", "a round"]):
            runway_concern = "comfortable runway but board pressure on ARR growth"
        elif any(k in budget_lower for k in ["series b", "series c", "growth"]):
            runway_concern = "strong runway but efficiency expectations from late-stage investors"
        else:
            runway_concern = "tight budget discipline"

        # Company name prefix for persona names
        name_parts = company.company_name.split()
        prefix = name_parts[0] if name_parts else "Exec"

        return {
            "prefix": prefix,
            "tech_1": tech_stack[0] if len(tech_stack) > 0 else "the core platform",
            "tech_2": tech_stack[1] if len(tech_stack) > 1 else "the API layer",
            "competitor_1": competitors[0] if len(competitors) > 0 else "the market leader",
            "competitor_2": competitors[1] if len(competitors) > 1 else "emerging competitors",
            "priority_1": priorities[0] if len(priorities) > 0 else "core product quality",
            "priority_2": priorities[1] if len(priorities) > 1 else "customer retention",
            "priority_3": priorities[2] if len(priorities) > 2 else "operational efficiency",
            "compliance_concern": compliance,
            "budget": budget,
            "runway_concern": runway_concern,
            "company_name": company.company_name,
        }

    @staticmethod
    def _build_persona(
        seat: dict,
        ctx: dict,
        feature: Optional[FeatureProposal],
    ) -> FinalPersona:
        """Build a single FinalPersona from seat definition + context."""
        # Render name
        name = seat["name_template"].format(**ctx)

        # Render bio
        bio = seat["bio_template"].format(**ctx)

        # Adapt expertise to company domain
        expertise = list(seat["expertise_base"])  # copy
        if feature and feature.affected_domains:
            # Add the most relevant domain from the feature
            for domain in feature.affected_domains[:2]:
                if domain not in expertise:
                    expertise.append(domain)

        # Calculate stance bias for this feature
        stance_text = ""
        if feature:
            stance_text = BoardroomPersonaFactory._compute_stance(
                seat["stance_logic"], feature, ctx
            )

        full_profile = f"{bio}\n\nSTANCE ON CURRENT FEATURE: {stance_text}" if stance_text else bio

        return FinalPersona(
            name=name,
            role=seat["role"],
            role_short=seat["role_short"],
            domain_expertise=expertise,
            psychological_profile=PsychologicalProfile(
                full_profile_text=full_profile,
            ),
            influence_strength=seat["influence"],
            receptiveness=seat["receptiveness"],
            persona_type="INTERNAL",
        )

    @staticmethod
    def _compute_stance(
        stance_logic: str,
        feature: FeatureProposal,
        ctx: dict,
    ) -> str:
        """Compute natural stance bias based on persona archetype + feature."""
        desc_lower = feature.description.lower()
        title_lower = feature.title.lower()
        all_text = desc_lower + " " + title_lower

        if stance_logic == "supportive_if_growth":
            if any(k in all_text for k in ["revenue", "growth", "acquisition", "market", "scale"]):
                return "Leans SUPPORTIVE — sees direct revenue/growth impact. Will push for aggressive timeline."
            return "NEUTRAL — needs convincing that this moves the revenue needle, not just a tech project."

        elif stance_logic == "skeptical_on_complexity":
            if any(k in all_text for k in ["rewrite", "migration", "new framework", "replace"]):
                return "Leans SKEPTICAL — sees architectural risk and potential for scope creep."
            if any(k in all_text for k in ["autonomous", "ai", "automated", "agent"]):
                return "CAUTIOUSLY INTERESTED — excited by the tech but worried about reliability at scale."
            return "NEUTRAL — will evaluate based on technical merit and implementation plan quality."

        elif stance_logic == "skeptical_on_security":
            if any(k in all_text for k in ["api", "external", "third-party", "patient", "user data"]):
                return "Leans SKEPTICAL — new attack surface. Will demand threat model and pen-test plan."
            return "WATCHFUL — no obvious red flags but will probe for hidden data exposure risks."

        elif stance_logic == "conditional_on_user_evidence":
            if any(k in all_text for k in ["customer", "user", "feedback", "support"]):
                return "Leans SUPPORTIVE — directly addresses customer-reported pain. Wants pilot data."
            return "SKEPTICAL — where's the user evidence? Has this been validated with actual customers?"

        elif stance_logic == "skeptical_on_cost":
            if any(k in all_text for k in ["cost reduction", "efficiency", "savings", "recovery"]):
                return "Leans SUPPORTIVE — clear cost-positive ROI. Will model the payback period."
            return "SKEPTICAL — needs a 12-month ROI projection before committing budget."

        elif stance_logic == "conditional_on_ux":
            if any(k in all_text for k in ["simplif", "automat", "seamless", "invisible"]):
                return "Leans SUPPORTIVE — reduces user friction. Needs to see the UX flow before endorsing."
            return "CONCERNED — does this add complexity to the core workflow? Wants user testing evidence."

        elif stance_logic == "skeptical_on_liability":
            if any(k in all_text for k in ["autonomous", "automated decision", "without human"]):
                return "Leans SKEPTICAL — autonomous systems create liability exposure. Needs human-in-the-loop safeguards."
            return "NEUTRAL — standard regulatory review will surface any issues."

        elif stance_logic == "conditional_on_data_quality":
            if any(k in all_text for k in ["ml", "ai", "model", "prediction", "classification"]):
                return "CAUTIOUS — will demand accuracy benchmarks, bias audits, and a monitoring plan."
            return "NEUTRAL — no ML concerns, but will probe data pipeline reliability."

        elif stance_logic == "supportive_if_revenue":
            if any(k in all_text for k in ["customer", "enterprise", "deal", "contract", "sales"]):
                return "STRONGLY SUPPORTIVE — this closes deals. Will personally demo it to prospects."
            return "LUKEWARM — nice to have but doesn't appear on competitive feature matrices."

        elif stance_logic == "conditional_on_adoption":
            if any(k in all_text for k in ["onboard", "churn", "retention", "adoption"]):
                return "SUPPORTIVE — directly addresses adoption friction. Needs migration/rollout plan."
            return "CAUTIOUS — will this break existing workflows? Needs change management assessment."

        return "UNDECIDED — needs more information before taking a position."
