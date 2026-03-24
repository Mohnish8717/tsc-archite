"""Gates 4.1-4.8: All gate implementations.

Each gate is a subclass of BaseGate with specific context and questions.
All gates are generalized — no hardcoded domain references.
"""

from __future__ import annotations

import numpy as np

from tsc.gates.base import BaseGate
from tsc.models.chunks import ProblemContextBundle
from tsc.models.gates import GateResult, GateVerdict, MonteCarloResults
from tsc.models.graph import KnowledgeGraph
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona


# ─── Gate 4.1: Technical Code Viability ──────────────────────────────


class TechnicalViabilityGate(BaseGate):
    gate_id = "4.1"
    gate_name = "Technical Code Viability"
    gate_domain = "technical architecture and engineering"
    verdict_options = "FEASIBLE, FEASIBLE_WITH_DEBT, RISKY, NOT_FEASIBLE"

    def _build_context(self, feature, company, graph, bundle, personas):
        return f"""TECHNICAL CONTEXT:
- Current tech stack: {', '.join(company.tech_stack) or 'Not specified'}
- Team size: {company.team_size or 'Unknown'}
- Known constraints: {', '.join(company.constraints) or 'None specified'}
- Existing features: {', '.join(feature.existing_features) or 'None listed'}"""

    def _build_questions(self):
        return """QUESTIONS TO ANSWER:
1. Can we build this with the existing tech stack?
2. Are there proven patterns we can reuse?
3. What technical debt exists in related areas?
4. Estimated technical effort (weeks)?
5. What are the main technical risks?"""


# ─── Gate 4.2: SOTA Probe ───────────────────────────────────────────


class SOTAProbeGate(BaseGate):
    gate_id = "4.2"
    gate_name = "SOTA Probe"
    gate_domain = "technology research and build-vs-buy analysis"
    verdict_options = "BUILD, ADAPT_EXISTING, BUY, EXISTS_NEEDS_ADAPTATION"

    def _build_context(self, feature, company, graph, bundle, personas):
        return f"""RESEARCH CONTEXT:
Research existing solutions and approaches for: {feature.title}

Consider:
1. Open source solutions and frameworks
2. Commercial/SaaS solutions
3. Custom implementations by similar companies
4. Academic research and emerging patterns

Current stack context: {', '.join(company.tech_stack) or 'Not specified'}"""

    def _build_questions(self):
        return """FOR EACH EXISTING SOLUTION:
- How mature is it?
- Can we adapt it to our use case?
- Build vs Buy analysis
- Cost implications
- Integration effort

FINAL QUESTION: Should we BUILD from scratch, ADAPT existing pattern, or BUY external service?"""


# ─── Gate 4.3: Resource Impact Assessment ────────────────────────────


class ResourceImpactGate(BaseGate):
    gate_id = "4.3"
    gate_name = "Resource Impact Assessment"
    gate_domain = "system resources, performance, and infrastructure impact"
    verdict_options = "PASS, PASS_WITH_MITIGATION, RISKY, FAIL"

    def _build_context(self, feature, company, graph, bundle, personas):
        return f"""RESOURCE CONTEXT:
Analyze the resource impact of {feature.title} on:
1. CPU / compute requirements
2. Memory / storage requirements
3. Network bandwidth
4. Battery / power (if applicable for mobile/IoT)
5. Infrastructure scaling needs

Target platform: Based on {', '.join(company.tech_stack) or 'general web/mobile'}
Target users: {feature.target_users}"""

    def _build_questions(self):
        return """ESTIMATE:
1. Additional compute requirements
2. Storage impact (data size, growth rate)
3. Performance impact on existing system
4. Which deployment environments are most impacted?
5. What mitigations are necessary?
6. How do we monitor resource impact in production?"""


# ─── Gate 4.4: Infrastructure Requirements ──────────────────────────


class InfrastructureGate(BaseGate):
    gate_id = "4.4"
    gate_name = "Infrastructure Requirements"
    gate_domain = "deployment, scaling, and infrastructure planning"
    verdict_options = "PASS, PASS_WITH_MITIGATION, RISKY, FAIL"

    def _build_context(self, feature, company, graph, bundle, personas):
        return f"""INFRASTRUCTURE CONTEXT:
Analyze deployment and infrastructure needs for {feature.title}:
1. New services or components needed
2. Database changes or new data stores
3. API changes (new endpoints, protocol changes)
4. Network topology changes
5. Third-party service dependencies
6. Deployment pipeline changes

Current infrastructure: {', '.join(company.tech_stack) or 'Not specified'}"""

    def _build_questions(self):
        return """QUESTIONS:
1. What new infrastructure is required?
2. Can existing infra handle the additional load?
3. What are the dependencies?
4. Deployment complexity assessment
5. Rollback plan if deployment fails
6. Cost of new infrastructure"""


# ─── New Gates (4.5 & 4.6) ──────────────────────────────────────────
# Using hybrid OASIS + Mesa implementations

from tsc.layers.layer4_gates.gate_4_5_market_fit import MarketFitGate
from tsc.layers.layer4_gates.gate_4_6_red_team import RedTeamGate

# Re-register for sequential execution
MonteCarloMarketFitGate = MarketFitGate
RedTeamAdversarialGate = RedTeamGate

# ─── Gate 4.7: Feature Interactions ─────────────────────────────────


class FeatureInteractionsGate(BaseGate):
    gate_id = "4.7"
    gate_name = "Feature Interactions"
    gate_domain = "feature interaction analysis and product architecture"
    verdict_options = "PASS, PASS_WITH_ADJUSTMENTS, RISKY, FAIL"

    def _build_context(self, feature, company, graph, bundle, personas):
        existing = feature.existing_features or company.current_priorities
        features_list = "\n".join(f"- {f}" for f in existing) if existing else "- No existing features listed"

        return f"""INTERACTION ANALYSIS:
New feature: {feature.title}

Existing features/systems:
{features_list}

For each existing feature, analyze:
1. Does the new feature CONFLICT with it?
2. Does it COMPLEMENT it?
3. Is there NO IMPACT?
4. Are there shared dependencies?"""

    def _build_questions(self):
        return """For each interaction:
- Interaction type (CONFLICT_HIGH, CONFLICT_LOW, COMPLEMENT, NEUTRAL)
- Description of the interaction
- Resolution strategy if conflict
- Impact on implementation approach

Overall: Are conflicts resolvable? What adjustments are needed?"""


# ─── Gate 4.8: Execution Feasibility ────────────────────────────────


class ExecutionGate(BaseGate):
    gate_id = "4.8"
    gate_name = "Execution Feasibility"
    gate_domain = "project management, resource planning, and execution"
    verdict_options = "FEASIBLE, FEASIBLE_TIGHT, RISKY, NOT_FEASIBLE"

    def _build_context(self, feature, company, graph, bundle, personas):
        return f"""EXECUTION ANALYSIS:
Feature: {feature.title}
Estimated effort: {feature.effort_weeks_min}-{feature.effort_weeks_max} weeks
Team size: {company.team_size or 'Unknown'}
Budget: {company.budget or 'Unknown'}
Current priorities: {', '.join(company.current_priorities) or 'None listed'}
Constraints: {', '.join(company.constraints) or 'None listed'}"""

    def _build_questions(self):
        return """ASSESS:
1. TIMELINE: Is the effort estimate realistic? Buffer needed?
2. RESOURCES: Do we have the right people? Can we allocate them?
3. DEPENDENCIES: External deps, approvals, third-party services?
4. RISKS TO EXECUTION: Scope creep, attrition, integration complexity?
5. GO/NO-GO: Can we ship on time with acceptable quality?"""


# ─── Gate Registry ──────────────────────────────────────────────────

ALL_GATES = [
    TechnicalViabilityGate,
    SOTAProbeGate,
    ResourceImpactGate,
    InfrastructureGate,
    MonteCarloMarketFitGate,
    RedTeamAdversarialGate,
    FeatureInteractionsGate,
    ExecutionGate,
]
