"""Jinja2 prompt templates for all LLM calls.

All templates are provider-agnostic — no Anthropic/OpenAI-specific patterns.
Templates are parameterized with feature context, graph data, and persona data.
"""

from __future__ import annotations

from jinja2 import Environment, BaseLoader

_env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)

# ─────────────────────────────────────────────────────────────────────
# Layer 1: NLP Enrichment
# ─────────────────────────────────────────────────────────────────────

ENRICHMENT_SYSTEM = """You are an expert NLP analyst. Analyze the given text chunk and extract structured information. Be precise and evidence-based."""

ENRICHMENT_USER = _env.from_string("""Analyze this text chunk and extract:

1. **Entities** — each entity with type (PERSON, ORG, PRODUCT, CONSTRAINT, PAIN_POINT, METRIC), text, confidence (0-1), and value/unit if numeric
2. **Sentiment** — label (POSITIVE, NEUTRAL, NEGATIVE) with confidence score (0-1)
3. **Urgency** — level 1-5 (5=critical/blocking, 4=important/soon, 3=would like, 2=maybe, 1=if possible)
4. **Topic** — category (feature_request, bug_report, question, feedback, constraint) with confidence (0-1)
5. **Metrics** — any numeric facts with value, unit, and context

Text chunk:
---
{{ text }}
---

Source: {{ source_file }} ({{ source_type }})

Return JSON:
{
  "entities": [{"text": "...", "type": "...", "confidence": 0.0, "value": null, "unit": null}],
  "sentiment": {"label": "...", "score": 0.0},
  "urgency": 3,
  "topic_category": "...",
  "topic_confidence": 0.0,
  "metrics": [{"value": 0, "unit": "...", "context": "..."}]
}""")

# ─────────────────────────────────────────────────────────────────────
# Layer 2: Relationship Extraction
# ─────────────────────────────────────────────────────────────────────

RELATIONSHIP_SYSTEM = """You are an expert at extracting relationships between entities from text. Identify how entities relate to each other using these relationship types: REQUESTS, CAUSES, IMPACTS, DEPENDS_ON, CONFLICTS_WITH, MENTIONED_WITH."""

RELATIONSHIP_USER = _env.from_string("""Given these entities found in the text:
{% for entity in entities %}
- {{ entity.name }} ({{ entity.type }})
{% endfor %}

And this text context:
---
{{ text }}
---

Extract relationships between the entities.

Return JSON:
{
  "relationships": [
    {
      "source": "entity_name",
      "target": "entity_name",
      "type": "REQUESTS|CAUSES|IMPACTS|DEPENDS_ON|CONFLICTS_WITH|MENTIONED_WITH",
      "confidence": 0.0,
      "weight": 0.0,
      "evidence": "brief quote or description"
    }
  ]
}""")

# ─────────────────────────────────────────────────────────────────────
# Layer 3: Stakeholder Selection
# ─────────────────────────────────────────────────────────────────────

STAKEHOLDER_SELECTION_SYSTEM = """You are an expert organizational analyst. Given a feature proposal and company context, identify the 3-5 most relevant internal stakeholders who should evaluate this feature."""

STAKEHOLDER_SELECTION_USER = _env.from_string("""Feature being evaluated:
- Title: {{ feature.title }}
- Description: {{ feature.description }}
- Target Users: {{ feature.target_users }}
- Affected Domains: {{ feature.affected_domains | join(', ') }}
- Effort: {{ feature.effort_weeks_min }}-{{ feature.effort_weeks_max }} weeks

Company Context:
- Team Size: {{ company.team_size }}
- Tech Stack: {{ company.tech_stack | join(', ') }}
- Current Priorities: {{ company.current_priorities | join(', ') }}
{% if company.stakeholders %}
Known Stakeholders:
{% for s in company.stakeholders %}
- {{ s.name }} ({{ s.role }}): {{ s.title | default('') }}
{% endfor %}
{% endif %}

Key entities from analysis:
{% for entity in top_entities[:15] %}
- {{ entity.name }} ({{ entity.type }}, {{ entity.mentions }} mentions, urgency: {{ entity.average_urgency | round(1) }})
{% endfor %}

Identify 3-5 stakeholders with:
- Name (use known stakeholders if provided, otherwise generate realistic names)
- Role and title
- Relevance score (0-1)
- Which domains they cover
- Their decision authority level

Return JSON:
{
  "stakeholders": [
    {
      "name": "...",
      "role": "...",
      "title": "...",
      "relevance_score": 0.0,
      "domain_relevance": "...",
      "decision_authority": "high|medium|low"
    }
  ]
}""")

EXTERNAL_STAKEHOLDER_SELECTION_SYSTEM = """You are an expert market analyst. Given a feature proposal and company context, identify the 3-5 most relevant EXTERNAL customer segments and personas who would use or buy this feature."""

EXTERNAL_STAKEHOLDER_SELECTION_USER = _env.from_string("""Feature being evaluated:
- Title: {{ feature.title }}
- Description: {{ feature.description }}
- Target Users: {{ target_users }}
- Target User Count: {{ target_user_count | default('Unknown') }}

Company Context:
- Tech Stack: {{ company.tech_stack | join(', ') }}
- Current Priorities: {{ company.current_priorities | join(', ') }}

Key market entities / pain points:
{% for entity in top_entities[:15] %}
- {{ entity.name }} ({{ entity.type }}, {{ entity.mentions }} mentions, urgency: {{ entity.average_urgency | round(1) }})
{% endfor %}

Identify 3-5 external customer personas with:
- Name (e.g., "Enterprise CTO", "End User")
- Persona Type (segment type, e.g., "enterprise", "startup")
- Title / Role
- Relevance score (0-1)
- Use case / why they care (domain_relevance)
- Decision authority (high=buyer, medium=influencer, low=end-user)

Return JSON:
{
  "external_customers": [
    {
      "name": "...",
      "persona_type": "...",
      "title": "...",
      "relevance_score": 0.0,
      "use_case": "...",
      "decision_authority": "high|medium|low"
    }
  ]
}""")

# ─────────────────────────────────────────────────────────────────────
# Layer 3: Psychological Profiling
# ─────────────────────────────────────────────────────────────────────

PERSONA_SYSTEM = """You are an expert organizational psychologist and decision analyst. Generate a detailed psychological profile of a stakeholder based on documented facts, past decisions, and organizational context. Focus on how they THINK, DECIDE, and COMMUNICATE. Be specific to the provided facts. Avoid generic statements. Ground everything in the evidence provided."""

PERSONA_SYSTEM_GROUNDED = """You are an expert organizational psychologist. Your goal is to generate a persona profile that is 100% GROUNDED in document evidence.

RULES:
1. Every trait (MBTI, emotions, drivers) MUST be derived from the provided FACTS.
2. If facts show high urgency and negative sentiment about a topic, that is a PAIN POINT.
3. If facts show positive sentiment, that is an EXCITEMENT DRIVER.
4. CITE your evidence for every major claim using [Fact: <text snippet>] notation.
5. If evidence is insufficient for a trait (e.g. MBTI), state "NOT ENOUGH EVIDENCE" rather than guessing.
6. Focus on specific behavioral patterns (e.g. "Direct communication" vs "Diplomatic") based on their documented quotes and actions.
"""

PERSONA_USER = _env.from_string("""Based on these facts about {{ name }}, a {{ role }} ({{ title }}):

ORGANIZATIONAL CONTEXT:
{% for fact in org_context %}
- {{ fact }}
{% endfor %}

{% if personal_facts %}
DOCUMENTED FACTS:
{% for fact in personal_facts %}
- {{ fact }}
{% endfor %}
{% endif %}

{% if constraint_context %}
CONSTRAINTS:
{% for fact in constraint_context %}
- {{ fact }}
{% endfor %}
{% endif %}

FEATURE BEING EVALUATED:
- Title: {{ feature.title }}
- Description: {{ feature.description }}
- Target Users: {{ feature.target_users }}
- Effort: {{ feature.effort_weeks_min }}-{{ feature.effort_weeks_max }} weeks

KEY EVIDENCE:
{% for entity in top_entities[:10] %}
- {{ entity.name }}: {{ entity.mentions }} mentions, avg urgency {{ entity.average_urgency | round(1) }}
{% endfor %}

---

Generate a comprehensive ~2000-word psychological profile covering:

1. PERSONALITY TYPE & COGNITIVE STYLE (400 words)
   - MBTI type and explanation
   - How they process information and make decisions
   - Key strengths and blindspots

2. EMOTIONAL TRIGGERS & MOTIVATION (400 words)
   - What excites, frustrates, and scares them
   - How they react under pressure

3. COMMUNICATION & COLLABORATION (300 words)
   - Preferred communication style
   - How they handle disagreement

4. DECISION PATTERNS (300 words)
   - Speed, data vs intuition, solo vs collaborative
   - Track record

5. VALUES & WHAT WOULD SWAY THEM (300 words)
   - What they fundamentally care about
   - What would change their mind

6. PREDICTED STANCE ON THIS FEATURE (300 words)
   - Will they approve, reject, or condition?
   - What conditions would they require?
   - What questions will they ask?
   - Confidence in this prediction""")

# ─────────────────────────────────────────────────────────────────────
# Layer 4: Gate Analysis (Generic template)
# ─────────────────────────────────────────────────────────────────────

GATE_SYSTEM = """You are an expert {{ gate_domain }} analyst evaluating a feature proposal. Provide a rigorous, evidence-based assessment. Be specific about risks, mitigations, and recommendations."""

GATE_USER = _env.from_string("""FEATURE: {{ feature.title }}
DESCRIPTION: {{ feature.description }}
TARGET USERS: {{ feature.target_users }}{% if feature.target_user_count %} ({{ feature.target_user_count }}){% endif %}

COMPANY CONTEXT:
- Tech Stack: {{ company.tech_stack | join(', ') }}
- Team Size: {{ company.team_size }}
- Budget: {{ company.budget }}
- Current Priorities: {{ company.current_priorities | join(', ') }}

{% if gate_specific_context %}
{{ gate_specific_context }}
{% endif %}

KEY EVIDENCE FROM DATA:
{% for entity in top_entities[:10] %}
- {{ entity.name }} ({{ entity.type }}): {{ entity.mentions }} mentions, urgency {{ entity.average_urgency | round(1) }}
{% endfor %}

{{ gate_questions }}

Provide verdict: {{ verdict_options }}

Return JSON:
{
  "gate_id": "{{ gate_id }}",
  "gate_name": "{{ gate_name }}",
  "verdict": "...",
  "score": 0.0,
  "details": { ... },
  "risks": [
    {"risk_category": "...", "description": "...", "probability": 0.0, "impact": "...", "mitigation": "..."}
  ],
  "recommendations": ["..."]
}""")

# ─────────────────────────────────────────────────────────────────────
# Layer 6: Debate Rounds
# ─────────────────────────────────────────────────────────────────────

DEBATE_SYSTEM = """You are {{ name }}, a {{ role }} ({{ title }}). You are participating in a feature evaluation debate. Stay in character based on your psychological profile. Be specific, cite evidence from the data, and clearly state your position."""

DEBATE_ROUND1_USER = _env.from_string("""You are {{ name }}, {{ role }}.

Your psychological profile summary:
{{ profile_summary }}

The feature being debated: {{ feature.title }}
{{ feature.description }}

Gate results summary:
{% for gate in gate_results %}
- {{ gate.gate_name }}: {{ gate.verdict }} (score: {{ gate.score }})
{% endfor %}

Key data points:
{% for entity in top_entities[:8] %}
- {{ entity.name }}: {{ entity.mentions }} mentions
{% endfor %}

Present your INITIAL POSITION on this feature (200-300 words):
- Do you APPROVE, REJECT, or CONDITIONALLY APPROVE?
- What are your key concerns?
- What conditions do you require?
- What questions do you have for others?""")

DEBATE_ROUND2_USER = _env.from_string("""You are {{ name }}, {{ role }}.

Other stakeholders have stated their positions:
{% for position in other_positions %}
**{{ position.stakeholder_name }} ({{ position.role }}):** {{ position.statement }}
{% endfor %}

Respond to their points (200-300 words):
- Address concerns raised by others
- Propose trade-offs or compromises
- Update your position if warranted
- Identify areas of agreement""")

DEBATE_ROUND3_USER = _env.from_string("""You are {{ name }}, {{ role }}, leading the consensus synthesis.

All stakeholder positions after negotiation:
{% for position in all_positions %}
**{{ position.stakeholder_name }} ({{ position.role }}):** {{ position.verdict }}
Conditions: {{ position.conditions | join(', ') }}
{% endfor %}

Create the FINAL CONSENSUS statement (300-400 words):
- Overall verdict (APPROVED / REJECTED / CONDITIONAL)
- Phase 1 scope and timeline
- Success criteria (quantifiable)
- Phase 2 gate conditions (if applicable)
- Agreed mitigations
- Next steps with owners""")

# ─────────────────────────────────────────────────────────────────────
# Layer 7: Specification Generation
# ─────────────────────────────────────────────────────────────────────

SPEC_SYSTEM = """You are a senior technical writer and product manager. Generate a detailed, actionable implementation specification. Be specific to the team context. Cite evidence from the evaluation."""

SPEC_USER = _env.from_string("""Generate a detailed technical specification for:

FEATURE: {{ feature.title }}
DESCRIPTION: {{ feature.description }}

CONSENSUS:
- Verdict: {{ consensus.overall_verdict }}
- Confidence: {{ consensus.approval_confidence }}
- Phase 1 Scope: {{ consensus.phase_1.scope | join(', ') }}
- Timeline: {{ consensus.phase_1.timeline }}
- Cost: {{ consensus.phase_1.cost_estimate }}

GATE RESULTS:
{% for gate in gate_results %}
- {{ gate.gate_name }}: {{ gate.verdict }} ({{ gate.score }})
{% endfor %}

COMPANY CONTEXT:
- Tech Stack: {{ company.tech_stack | join(', ') }}
- Team Size: {{ company.team_size }}

STAKEHOLDER CONDITIONS:
{% for approval in consensus.approvals %}
- {{ approval.stakeholder }} ({{ approval.role }}): {{ approval.conditions | join(', ') }}
{% endfor %}

Generate a comprehensive specification covering:
1. FEATURE OVERVIEW (400 words)
2. REQUIREMENTS — functional and non-functional (300 words)
3. TECHNICAL DESIGN — architecture, storage, protocol (500 words)
4. USER FLOWS — key scenarios (400 words)
5. SUCCESS CRITERIA — quantitative and qualitative (300 words)
6. RISK MITIGATIONS — with evidence (300 words)
7. DEVELOPMENT TASKS — granular, assignable, with effort and priority (400 words)

Format as professional Markdown.
Cite gate results as evidence.
Include tables where appropriate.""")

# ─────────────────────────────────────────────────────────────────────
# Layer 8: Leadership Summary
# ─────────────────────────────────────────────────────────────────────

SUMMARY_SYSTEM = """You are writing an executive summary for senior leadership. Be concise, data-driven, and action-oriented."""

SUMMARY_USER = _env.from_string("""Summarize this feature evaluation in 2-3 sentences for leadership:

Feature: {{ feature.title }}
Verdict: {{ verdict }}
Confidence: {{ confidence }}
Key Data: {{ key_metrics }}
ROI: {{ roi }}
Timeline: {{ timeline }}
Top Risk: {{ top_risk }}""")
