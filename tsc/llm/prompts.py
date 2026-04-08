"""Jinja2 prompt templates for all LLM calls.

All templates are provider-agnostic — no Anthropic/OpenAI-specific patterns.
Templates are parameterized with feature context, graph data, and persona data.
"""

from __future__ import annotations

from jinja2 import Environment, BaseLoader

_env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)

# ─────────────────────────────────────────────────────────────────────
# Layer 1: Semantic Chunking (SOTA)
# ─────────────────────────────────────────────────────────────────────

SEMANTIC_CHUNKING_SYSTEM = """You are a semantic chunking engine optimized for qualitative research data.
Your task is to split raw interview/document text into coherent chunks that:
1. Preserve speaker identity and direct quotes
2. Respect semantic boundaries (topic shifts, emotional weight)
3. Maintain sufficient context for downstream entity extraction
4. Never split critical constraints or concerns mid-thought"""

SEMANTIC_CHUNKING_USER = _env.from_string("""STRICT RULES:
- Every chunk must include speaker attribution line at top: "[SPEAKER: {name}] {timestamp}" (if applicable)
- Direct quotes must ALWAYS be complete (quote_start → quote_end with no splits)
- Chunks: 600-1200 words (preserves sentence coherence)
- Overlap: 2 sentences at boundary between chunks
- Constraint flags: Automatically tag chunks containing "must not", "requires", "impossible", "won't"

CHUNK METADATA FORMAT (required for every chunk):
{
  "chunk_id": "...",
  "speaker": "...",
  "timestamp": "...",
  "primary_topic": "...",
  "secondary_topics": [...],
  "constraint_flags": [...],
  "critical_quote": null,
  "semantic_coherence_score": 0.0,
  "next_chunk_preview": "..."
}

---INPUT DOCUMENT---
{{ document_content }}

---PROCESSING---
Task:
1. Read the entire document
2. Identify all speaker changes, topic boundaries, and quoted sections
3. Create logical segments that preserve speaker voice and critical context
4. For each segment, output the chunk text plus metadata
5. Ensure overlap: last 2 sentences of chunk_N appear as first 2 sentences of chunk_N+1

Output as valid JSON array:
[
  {
    "id": "chunk_001",
    "speaker": "...",
    "timestamp": "...",
    "text": "...",
    "metadata": {...},
    "constraints": [...],
    "quotes": [{"text": "...", "timestamp": "..."}],
    "semantic_score": 0.0
  }
]

Return ONLY valid JSON. No markdown, no explanations outside JSON.""")

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
# Layer 2: Grounded NER (SOTA)
# ─────────────────────────────────────────────────────────────────────

GROUNDED_NER_SYSTEM = """You are a precision entity extraction engine. Your goal is to identify critical entities and link them to EXACT quotes from the source text.
You must filter out generalities and focus on specific personas, organizations, technical constraints, and metrics."""

GROUNDED_NER_USER = _env.from_string("""Extract entities from this text. 
For every entity, you MUST provide:
1. The exact text of the entity.
2. Its type (PERSON, ORG, PRODUCT, CONSTRAINT, PAIN_POINT, METRIC).
3. A direct quote from the source that justifies this entity.
4. A confidence score (0.0 to 1.0).

STRICT FILTERING:
- Do not extract trivial entities (e.g., "today", "the team", "something").
- CONSTRAINTs must be specific technical or business requirements.
- PAIN_POINTs must describe a specific user frustration.

---TEXT---
{{ text }}

Return ONLY valid JSON array:
[
  {
    "text": "...",
    "type": "...",
    "evidence_quote": "...",
    "confidence": 0.0
  }
]""")

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
# Layer 2: Grounded Relationships (SOTA)
# ─────────────────────────────────────────────────────────────────────

GROUNDED_RELATIONSHIP_SYSTEM = """You are a precision relationship extraction engine. Your goal is to identify how entities in a knowledge graph interact and link every relationship to an EXACT quote from the source text."""

GROUNDED_RELATIONSHIP_USER = _env.from_string("""Analyze the interaction between these entities in the provided text.
For every relationship found, you MUST provide:
1. The source entity name.
2. The target entity name.
3. The type of relationship (REQUESTS, CAUSES, IMPACTS, DEPENDS_ON, CONFLICTS_WITH, MENTIONED_WITH).
4. An exact quote from the text that proves this relationship exists.
5. A confidence score (0.0 to 1.0).

---ENTITIES---
{% for ent in entities %}- {{ ent }}{% endfor %}

---TEXT---
{{ text }}

Return ONLY valid JSON array:
[
  {
    "source": "...",
    "target": "...",
    "type": "...",
    "evidence_quote": "...",
    "confidence": 0.0
  }
]""")

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

PERSONA_SYSTEM_GROUNDED = """You are an expert organizational psychologist and character writer. Your goal is to generate personas that feel like REAL, SPECIFIC PEOPLE — not archetypes or job descriptions.

CORE RULES:
1. Every trait (MBTI, emotions, drivers) MUST be derived from the provided FACTS.
2. If facts show high urgency and negative sentiment about a topic, that is a PAIN POINT.
3. If facts show positive sentiment, that is an EXCITEMENT DRIVER.
4. CITE your evidence for every major claim using [Fact: <text snippet>] notation.
5. If evidence is insufficient for a trait (e.g. MBTI), state "NOT ENOUGH EVIDENCE" rather than guessing.
6. DO NOT write generic corporate-speak. Write like a documentary filmmaker who spent a week shadowing this person.
7. The persona must feel SPECIFIC — their concerns, vocabulary, and reactions should be impossible to swap with another role.
8. You MUST write the VIVID SCENE (Section 0), SIGNATURE QUOTE (Section 7), PROFESSIONAL BACKSTORY (Section 8), and ROLE VOCABULARY (Section 9). These are mandatory.
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

Generate a comprehensive ~2500-word psychological profile covering ALL sections below:

0. VIVID SCENE — WHO IS THIS PERSON RIGHT NOW? (100 words)
   Write a single, vivid third-person paragraph placing {{ name }} in their natural work environment on a typical Tuesday at 10am.
   - What is on their screen?
   - What is the one thing worrying them today?
   - What meeting are they about to walk into and how do they feel about it?
   Make every detail match their specific role as {{ role }}. NO generic statements.

1. PERSONALITY TYPE & COGNITIVE STYLE (400 words)
   - MBTI type and explanation grounded in ONE specific past professional experience that reveals this tendency
   - How they process information and make decisions
   - Key strengths and blindspots with concrete behavioural examples

2. EMOTIONAL TRIGGERS & MOTIVATION (400 words)
   - What excites them — include ONE real-world example tied to their role (e.g. when a specific outcome made them feel validated)
   - What frustrates them — include ONE real-world example of a situation that genuinely annoys someone in this role
   - What scares them — the professional fear that keeps them up at night
   - How they react under pressure

3. COMMUNICATION & COLLABORATION (300 words)
   - Preferred communication style and preferred channel (Slack vs meeting vs doc)
   - How they handle disagreement — do they go direct, get political, or disengage?
   - What kind of colleague they are and what kind drives them crazy

4. DECISION PATTERNS (300 words)
   - Speed, data vs intuition, solo vs collaborative
   - One past decision they made that exemplifies their style
   - What information they demand before committing

5. VALUES & WHAT WOULD SWAY THEM (300 words)
   - What they fundamentally care about in their work
   - The single argument or data point that could change their mind
   - Their non-negotiables

6. PREDICTED STANCE ON THIS FEATURE (300 words)
   - Will they approve, reject, or condition?
   - What conditions would they require?
   - What specific questions will they ask in the review meeting?
   - Confidence in this prediction and why

7. SIGNATURE QUOTE (60 words)
   Write a verbatim first-person quote that {{ name }} would actually say in a feature review meeting about {{ feature.title }}.
   The quote must:
   - Sound natural and match their role's vocabulary
   - Reveal their deepest bias or concern about this feature
   - Not be generic — it must be impossible to attribute to someone with a different role
   Format: "<quote>"

8. PROFESSIONAL BACKSTORY (150 words)
   Describe 1-2 formative professional experiences that created {{ name }}'s current worldview, biases, and communication style.
   - What did they do before this role?
   - What success or failure fundamentally shaped how they think?
   - Cite any available evidence; if none, extrapolate realistically from their role context.

9. ROLE VOCABULARY (list only)
   List exactly 6 domain-specific terms or phrases that {{ name }} naturally uses in conversations — the jargon of their specific job.
   These must be authentic to {{ role }} and NOT generic business buzzwords.
   Format as a simple bulleted list.""")

# ─────────────────────────────────────────────────────────────────────
# Layer 4: Gate Analysis (Generic template)
# ─────────────────────────────────────────────────────────────────────

GATE_SYSTEM = _env.from_string("""You are an expert {{ gate_domain }} analyst evaluating a feature proposal. Provide a rigorous, evidence-based assessment. Be specific about risks, mitigations, and recommendations.""")

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

DEBATE_SYSTEM = _env.from_string("""You are {{ name }}, a {{ role }} ({{ title }}). You are participating in a high-stakes, adversarial feature evaluation debate. 

STRICT RULES:
1. STAY IN CHARACTER: Use the tone and priorities defined in your psychological profile.
2. EVIDENCE GROUNDING: Every claim or critique you make MUST be backed by a specific data point from the Knowledge Graph or your direct quotes.
3. BE ADVERSARIAL: Do not just agree. Actively hunt for risks, hidden costs, or downstream impacts your colleagues might be ignoring.
4. CITATION: Use [Evidence: <snippet>] for every supporting fact.""")

DEBATE_ROUND1_USER = _env.from_string("""You are {{ name }}, {{ role }}. 

Your psychological profile summary:
{{ profile_summary }}

The feature being debated: {{ feature.title }}
{{ feature.description }}

Gate results summary:
{% for gate in gate_results %}
- {{ gate.gate_name }}: {{ gate.verdict }} (score: {{ gate.score }})
{% endfor %}

--- MARKET SIMULATION INSIGHTS (OASIS) ---
{% if market_fit_insights %}
- Predicted Adoption Score: {{ market_fit_insights.adoption_score | round(2) }}
- Consensus Type: {{ market_fit_insights.consensus_type }} (Strength: {{ market_fit_insights.consensus_strength | round(2) }})
- Market Segments: {{ market_fit_insights.segment_count }} clusters identified
- Key Market Objections: 
  {% for obj in market_fit_insights.objections %}
  * {{ obj }}
  {% endfor %}
{% else %}
- Actual social simulation data not available for this round.
{% endif %}

Key data points / Evidence:
{% for entity in top_entities[:10] %}
- {{ entity.name }}: {{ entity.mentions }} mentions, urgency: {{ entity.average_urgency | round(1) if entity.average_urgency else 'N/A' }}
{% endfor %}

State your INITIAL POSITION (200-300 words).
You MUST:
1. Cite at least 1 specific insight from the OASIS market simulation above.
2. Cite at least 2 specific data points from the Knowledge Graph evidence.
3. Identify the #1 biggest risk from YOUR perspective.
4. Clearly state: APPROVE, REJECT, or CONDITIONALLY APPROVE.""")

DEBATE_ROUND2_USER = _env.from_string("""You are {{ name }}, {{ role }}. 

OTHER STAKEHOLDERS HAVE STATED THEIR POSITIONS:
{% for position in other_positions %}
**{{ position.stakeholder_name }} ({{ position.role }}):**
Verdict: {{ position.verdict }}
Statement: {{ position.statement }}
{% endfor %}

ADVERSARIAL CRITIQUE ROUND (250-350 words):
Your goal is to pressure-test the positions of your colleagues.
1. Identify at least TWO flaws or overlooked risks in your colleagues' statements.
2. Use EVDIENCE from your own data/quotes to counter their arguments.
3. If they approve, find reasons why they might be too optimistic. If they reject, find reasons why they might be too conservative.
4. Maintain your character's unique bias and priorities.""")

DEBATE_ROUND3_USER = _env.from_string("""You are {{ name }}, {{ role }}. This is the FINAL ROUND: REBUTTAL & RESOLUTION.

FULL DEBATE HISTORY:
{% for position in all_positions %}
**{{ position.stakeholder_name }} ({{ position.role }}):**
Verdict: {{ position.verdict }}
Key Points: {{ position.statement[:500] }}...
{% endfor %}

FINAL ACTION (300-400 words):
1. REBUT any unfair critiques leveled against your position in the previous round.
2. Provide your FINAL IRREVOCABLE VERDICT (APPROVED / REJECTED / CONDITIONAL).
3. If CONDITIONAL, list exactly 3 "Deal-breaker" conditions that must be met.
4. Synthesize the most grounded path forward that balances the adversarial tensions raised.""")

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

# ─────────────────────────────────────────────────────────────────────
# Layer 9: Sub-Query Generation (InsightForge)
# ─────────────────────────────────────────────────────────────────────

SUB_QUERY_GEN_SYSTEM = """You are a precision research analyst. Decompose a complex product feature proposal into 3-5 specific, distinct search queries for a knowledge graph. 
Focus on identifying:
1. Technical constraints and dependencies
2. Potential user pain points or objections
3. Business/Market risks
4. Historical context or comparable features
"""

SUB_QUERY_GEN_USER = _env.from_string("""Feature Proposal:
- Title: {{ feature.title }}
- Description: {{ feature.description }}
- Target Users: {{ feature.target_users }}

Generate exactly 3-5 search queries that will help uncover the most relevant facts from a knowledge graph to evaluate this feature.
Queries should be concise (max 10 words each).

Return ONLY a JSON array of strings:
[
  "query 1",
  "query 2",
  "query 3"
]""")

# ─────────────────────────────────────────────────────────────────────
# Layer 3: External Persona Psychological Profiling
#
# Domain-agnostic: applies to SaaS tools, physical products, service
# launches, pricing changes, policy rollouts, B2B/B2C market entries,
# healthcare, finance, retail — any business context.
# ─────────────────────────────────────────────────────────────────────

EXTERNAL_PERSONA_SYSTEM = """You are an expert market researcher, buyer psychologist, and go-to-market strategist.

Your task is to generate a deep, realistic buyer profile for an EXTERNAL market persona — a customer, buyer segment, or market archetype who will evaluate, purchase, or adopt (or reject) a business offering.

CRITICAL RULES:
1. These are EXTERNAL personas — customers or buyers, NOT internal employees or stakeholders.
2. The offering can be anything: a software tool, physical product, new service, pricing change, policy shift, business initiative, or strategic capability. Do NOT assume it is a technical or software feature.
3. Treat this as market research: think like a buyer anthropologist who has interviewed 50 people in this segment.
4. Every claim MUST be grounded in the evidence and context provided. Cite with [Evidence: <snippet>] notation.
5. Balance friction (objections, fears) with VALUE (what problems this solves, what success looks like).
6. The persona must feel specific — their vocabulary, concerns, and reasoning must be impossible to attribute to a different segment.
7. Write five NEW mandatory sections (10–14) covering the buyer's economic world, purchase journey, and market positioning. These are REQUIRED and must be substantive."""

EXTERNAL_PERSONA_USER = _env.from_string("""You are profiling {{ name }}, an external market persona representing: {{ role }}

ORGANIZATIONAL CONTEXT (about their company/environment):
{% for fact in org_context %}
- {{ fact }}
{% endfor %}

{% if personal_facts %}
DOCUMENTED EVIDENCE (from customer interviews, support tickets, market research):
{% for fact in personal_facts %}
- {{ fact }}
{% endfor %}
{% endif %}

{% if constraint_context %}
CONSTRAINTS & SIGNALS:
{% for fact in constraint_context %}
- {{ fact }}
{% endfor %}
{% endif %}

OFFERING BEING EVALUATED:
- Title: {{ feature.title }}
- Description: {{ feature.description }}
- Target Audience: {{ feature.target_users }}
- Effort / Timeline: {{ feature.effort_weeks_min }}-{{ feature.effort_weeks_max }} weeks

KEY MARKET SIGNALS:
{% for entity in top_entities[:10] %}
- {{ entity.name }}: {{ entity.mentions }} mentions, avg urgency {{ entity.average_urgency | round(1) }}
{% endfor %}

---

Generate a comprehensive ~2500-word buyer profile covering ALL sections below.
Sections 0–9 cover psychology. Sections 10–14 cover the BUYER'S MARKET WORLD (mandatory).

0. VIVID SCENE — WHO IS THIS BUYER RIGHT NOW? (100 words)
   A single third-person paragraph placing {{ name }} in their work context on a typical day.
   - What problem are they currently dealing with?
   - What decision is weighing on them?
   Make every detail match their segment. NO generic statements.

1. PERSONALITY TYPE & COGNITIVE STYLE (300 words)
   - MBTI type grounded in a specific professional behaviour observable in this segment
   - How they process new offerings (analytical vs. gut feel vs. consensus)
   - Key strengths and blindspots as a BUYER

2. EMOTIONAL TRIGGERS & MOTIVATION (300 words)
   - What excites them about the offering's category
   - What frustrates them about current alternatives
   - What they fear about making the wrong adoption decision

3. COMMUNICATION & COLLABORATION (200 words)
   - Preferred channel for learning about new offerings (peer, content, sales, analyst)
   - How they engage with vendors — demo-first vs. doc-first vs. pilot-first

4. DECISION PATTERNS (200 words)
   - How fast / slow they decide and why
   - Data vs. intuition vs. social proof in decision-making
   - Who else they bring into the decision

5. VALUES & WHAT WOULD SWAY THEM (200 words)
   - What they fundamentally care about (speed, cost, quality, risk, compliance)
   - The single argument or proof point that could change their mind

6. PREDICTED STANCE ON THIS OFFERING (200 words)
   - BULLISH, BEARISH, or CONDITIONAL — and why
   - Conditions they would require before adopting
   - Specific questions they would raise in an evaluation meeting

7. SIGNATURE QUOTE (60 words)
   A verbatim first-person quote {{ name }} would say when evaluating {{ feature.title }}.
   Must reveal their deepest concern or bias. Must be unmistakably from this segment.
   Format: "<quote>"

8. BUYER BACKSTORY (150 words)
   1-2 formative experiences that shaped how {{ name }} evaluates and buys solutions in this category.
   - What alternatives have they tried before?
   - What success or failure made them cautious or confident?

9. SEGMENT VOCABULARY (list only)
   Exactly 6 domain-specific terms {{ name }} uses — authentic to their segment, not generic buzzwords.

---

10. BUYING CONTEXT (300 words) [MANDATORY]
    - Company size and organisational structure relevant to this purchase
    - Budget authority: Can {{ name }} approve the purchase alone, or must they escalate? Who are the other approvers?
    - Typical procurement process: Is it informal (credit card purchase) or formal (RFP, legal review, security audit)?
    - How many people typically need to sign off, and what do each care about?
    - What happens if they get it wrong? (career risk, financial impact, operational risk)

11. BUYER JOURNEY (300 words) [MANDATORY]
    - AWARENESS: How would {{ name }} first hear about this type of offering? (peer, vendor, analyst, organic search, internal mandate)
    - TRIGGER: What specific event or pain forces them to start evaluating solutions RIGHT NOW?
    - EVALUATION: What do they do first — request a demo? read docs? find a case study? call a peer?
    - PROOF REQUIREMENTS: What specific evidence do they need before they feel confident enough to move forward?
    - DEAL-BREAKERS: Name 2-3 specific conditions that would stop the evaluation immediately.
    - TIMELINE: How long from first contact to decision? (days / weeks / months)

12. ROI & VALUE FRAMING (300 words) [MANDATORY]
    - CURRENT PAIN QUANTIFIED: Estimate time, cost, or risk they are carrying without this solution. Be specific (e.g., "4 hrs/week per team member = $X/year").
    - EXPECTED BENEFIT: What measurable improvement would success look like? (% reduction, $ saved, risk eliminated)
    - WILLINGNESS TO PAY: What price range do they consider "reasonable", "expensive", and "prohibitive"?
    - PAYBACK EXPECTATION: How quickly must they see ROI to feel the decision was correct? (months)
    - SUCCESS METRIC: How would they know, 6 months after adoption, that this was the right call?

13. COMPETITIVE ALTERNATIVES (200 words) [MANDATORY]
    - CURRENT STATE: What are they using today to address this problem? (manual process, competitor product, workaround, nothing)
    - SWITCH TRIGGER: What would make them replace their current approach with this offering?
    - COMPETITOR THREAT: Name 1-2 alternative solutions that could win this segment instead, and why.
    - SWITCHING COST: What would it cost them (time, money, change effort) to adopt this vs. staying with what they have?

14. ADOPTION BARRIERS (200 words) [MANDATORY]
    - REGULATORY / COMPLIANCE: Any legal or industry-specific restrictions that affect adoption?
    - INTEGRATION COMPLEXITY: How hard is it to fit this into their existing processes or systems?
    - CHANGE MANAGEMENT: Will their team/org resist adoption? What internal friction must they overcome?
    - VENDOR RISK: What concerns do they have about the vendor (stability, data security, lock-in)?
    - MITIGATION: What would reduce each barrier enough for them to move forward?""")

