# Under-the-Hood: How TSC Works When You Provide an Input

> **Scope:** This document covers every layer of the pipeline — data flow, LLM calls, retrieval logic, agent construction, simulation mechanics, scoring, and post-processing — traced directly from code.

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Layer 1 — Contextual Ingestor](#2-layer-1--contextual-ingestor)
3. [Layer 2 — Knowledge Graph (GraphRAG)](#3-layer-2--knowledge-graph-graphrag)
4. [Layer 3 — Persona Generation](#4-layer-3--persona-generation)
5. [OASIS User Persona Generator](#5-oasis-user-persona-generator)
6. [OASIS Simulation Engine](#6-oasis-simulation-engine)
7. [Post-Simulation: Scoring, Reports, and Executive Summary](#7-post-simulation-scoring-reports-and-executive-summary)
8. [Memory Architecture: Hindsight](#8-memory-architecture-hindsight)
9. [What Makes This Architecture Different](#9-what-makes-this-architecture-different)

---

## 1. System Overview

The TSC pipeline converts raw product and company documents into a live social simulation populated by psychologically grounded AI agents. The output is a quantitative market sentiment report with NPS, churn velocity, adoption momentum, and a VP-ready executive narrative.

**Full pipeline, in order:**

```
Raw Inputs
  ├── Feature Proposal (text/JSON)
  ├── Company Context (priorities, tech stack, competitors)
  ├── Tickets / Customer Interviews / Slack Logs (unstructured)
  └── Any attached documents

        │
        ▼
[ Layer 1 — Contextual Ingestor ]
  → Chunks, extracts, and classifies all inputs
  → Produces: ProblemContextBundle + HindsightSession (raw vector store)

        │
        ▼
[ Layer 2 — Knowledge Graph Builder ]
  → Extracts entities & typed relationships via LLM
  → Produces: KnowledgeGraph (nodes + edges, deterministic IDs)

        │
        ▼
[ Layer 3 — Persona Generator ]
  → Selects internal & external stakeholders from the graph
  → Generates 2500-word psychological profiles per stakeholder
  → Produces: List[FinalPersona] (with MBTI, OCEAN, stance, buyer journey)

        │
        ▼
[ OASIS User Persona Generator ]
  → Infers market segment distribution from real customer data
  → Instantiates diverse, demographically grounded product-user personas
  → Produces: List[OASISAgentProfile] (Big Five, 5-layer cognitive identity)

        │
        ▼
[ OASIS Simulation Engine ]
  → Builds social network topology (preferential attachment + homophily)
  → Injects LLM-generated seed posts as the stimulus
  → Runs N timesteps: each agent reads feed → reasons → acts
  → Game Master classifies every action into behavioral signals
  → Hindsight persists evolved beliefs across timesteps
  → Optional Focus Group Phase: structured interviews post-simulation

        │
        ▼
[ Report Orchestration (DAG) ]
  → Data Analyst Agent extracts exact metrics
  → Guardrail Fact-Checker Agent validates against raw JSON
  → Executive Writer Agent produces VP-ready narrative
  → Outputs: prediction_report.json + prediction_report.md
```

---

## 2. Layer 1 — Contextual Ingestor

**File:** `tsc/layers/layer1_ingestor.py`

### What Enters

The ingestor receives a `PipelineInput` object containing:
- `feature_proposal`: structured feature description
- `company_context`: tech stack, priorities, budget, competitors
- `raw_documents`: any mix of customer interviews, support tickets, Slack exports, or uploaded files

### Concurrent Chunking Pipeline

All document types are ingested in parallel using `asyncio.gather`. Each document goes through:

1. **Chunking:** Documents are split into semantic chunks. Each chunk gets a `priority_score` computed from keyword density (e.g., presence of "churn", "cancel", "compliance", "revenue") — higher-priority chunks bubble up for downstream use.

2. **Structured Extraction (LLM-first, regex-fallback):** Each chunk is passed to Gemini Flash with a strict JSON schema. The LLM extracts:
   - `entities` — named roles, products, companies, technical systems
   - `facts` — grounded claims, quotes, statistics
   - `constraints` — blockers, compliance requirements, timeline dependencies
   - `signals` — customer sentiment signals (positive, friction, exit intent)

   If the LLM call fails JSON validation, a regex-based extractor runs on the raw text.

3. **Hindsight Retention:** Every extracted chunk and its metadata is written into a vector store (Hindsight / Qdrant) under a named "bank" (e.g., `"world"`, `"personas"`). This is the raw retrieval substrate that all later layers query against via semantic recall.

### Output: `ProblemContextBundle`

```python
ProblemContextBundle(
    feature_proposal=FeatureProposal(...),
    company_context=CompanyContext(...),
    raw_chunks=List[PriorityChunk],   # sorted by priority_score
    extracted_entities=List[Entity],
    extracted_facts=List[Fact],
    extracted_constraints=List[Constraint],
    hindsight_session=HindsightSessionManager  # live reference to vector store
)
```

**What's unique here:** Chunks are not processed FIFO — they are scored and sorted by business criticality before being forwarded to downstream layers. A customer quote mentioning "GDPR" outranks a generic feature description chunk before it ever enters the graph or a persona context window.

---

## 3. Layer 2 — Knowledge Graph (GraphRAG)

**File:** `tsc/layers/layer2_graph.py`

### What Enters

Layer 2 receives the `ProblemContextBundle`. It focuses on the `raw_chunks` and `extracted_entities`.

### Relationship Extraction

Chunks are processed in batches of ~10. For each batch, a single LLM call to Gemini Flash receives the chunk texts and produces a JSON array of `Relationship` objects:

```json
{
  "source_entity": "CTO",
  "target_entity": "Data Privacy Policy",
  "relationship_type": "OWNS_DECISION",
  "weight": 0.85,
  "evidence": "CTO must sign off on any change touching user data per Q3 policy."
}
```

**Relationship types** are an enum that drives later simulation logic:
- `OWNS_DECISION` — decision authority edge
- `DEPENDS_ON` — technical dependency
- `OPPOSES` — known conflict
- `INFLUENCES` — indirect stakeholder pressure
- `REQUIRES_APPROVAL` — gate-keeping relationship
- `BENEFITS_FROM` / `HARMED_BY` — feature impact edges

### Deterministic Entity IDs

Entities get IDs derived from a canonical hash of their `name + type`. This prevents duplicate nodes across batches. When the same entity (e.g., "VP Engineering") appears in multiple chunks, its node accrues `mention_count` rather than duplicating.

### Output: `KnowledgeGraph`

```python
KnowledgeGraph(
    nodes=Dict[str, GraphEntity],   # entity_id → GraphEntity
    edges=List[GraphEdge],          # all typed relationships
)
```

**What's unique here:** The graph is not a bag-of-words index. Every edge carries a typed relationship, a weight, and a source evidence string. When the OASIS simulation engine later queries the graph during agent turns, it performs a zero-LLM neighborhood traversal — it finds entities mentioned in the current discussion context, then retrieves the top-K edges by weight and injects them as `[MANDATORY SYSTEM FACTS]` into the agent's prompt. This grounds agent reasoning in actual data from the documents rather than LLM hallucination.

---

## 4. Layer 3 — Persona Generation

**File:** `tsc/layers/layer3_personas.py`

Layer 3 is the most architecturally differentiated component. It does not produce generic personas from templates — it derives stakeholders from the knowledge graph and generates evidence-grounded psychological profiles.

### Step 1: Stakeholder Selection

The graph is queried for the top-20 entities by `mention_count`. For each entity, Layer 3 computes a `relevance_score`:

```
relevance_score = (
    graph_centrality_weight * 0.4
    + domain_coverage_score * 0.3
    + decision_authority_weight * 0.2
    + feature_impact_estimate * 0.1
)
```

Entities that score above a threshold become candidate **internal stakeholders** (employees, team leads, decision-makers). A separate LLM call then generates **external stakeholders** — market archetypes inferred from the product category (customers, regulators, investors, competitors).

External stakeholders also receive **MarketContext** and **BuyerJourney** metadata inferred from the LLM:
```python
MarketContext(
    company_size_band="mid-market",
    buyer_role="influencer",
    annual_solution_budget_usd=50000,
    pricing_sensitivity="medium",
    sales_cycle_weeks=8,
    deployment_preference="cloud",
    industry_vertical="technology",
    regulatory_burden="light"
)

BuyerJourney(
    awareness_channel="peer-recommendation",
    evaluation_trigger="...",
    key_proof_points=[...],
    deal_breakers=[...],
    success_metric="...",
    roi_threshold_months=12,
    willingness_to_pay_band="moderate"
)
```

### Step 2: Context Assembly (Per-Stakeholder RAG)

For every selected stakeholder, Layer 3 runs three parallel Hindsight queries:
1. **Personal facts** — anything in the corpus that mentions this person or role
2. **Org context** — the stakeholder's team, responsibilities, and known constraints
3. **Constraint context** — blockers, approval gates, risk factors they own

These are assembled into a `StakeholderContextBundle` — the evidence package passed to the profile generator.

### Step 3: Profile Generation (The 2500-Word Psychological Portrait)

Each stakeholder's context bundle is passed to Gemini with a structured 8-section prompt:

```
0. VIVID SCENE — A day-in-the-life narrative opening
1. PERSONALITY TYPE — MBTI + OCEAN scores with behavioral translation
2. CORE MOTIVATIONS — Intrinsic needs (Autonomy, Competence, Relatedness)
3. COMMUNICATION STYLE — Register, sentence patterns, channel preferences
4. DECISION PATTERN — Speed, data vs intuition preference, risk tolerance
5. EMOTIONAL TRIGGERS — What excites, frustrates, and scares them
6. PREDICTED STANCE — APPROVE / CONDITIONAL_APPROVE / REJECT + conditions
7. SIGNATURE QUOTE — A verbatim-style statement in their authentic voice
8. QUESTIONS THEY WILL ASK — The actual objections and clarifications they raise
```

The LLM is grounded by injecting the stakeholder's `personal_facts`, `org_context`, and `constraint_context` directly into the prompt. The profile cannot be fabricated from role stereotypes alone — it must explain predictions using sourced evidence.

### MBTI and OCEAN Extraction

After profile generation, a structured parser extracts psychological attributes:

- **MBTI:** Regex searches for `[EI][NS][TF][JP]` patterns. Each candidate match is scored by keyword proximity (`"personality"`, `"mbti"`, `"cognitive function"`) — the match with the highest contextual confidence wins.
- **OCEAN:** Regex searches for the literal format `OCEAN: O=0.xx, C=0.xx, E=0.xx, A=0.xx, N=0.xx` — the last occurrence in the text is used (to prefer the LLM's final stated values over any exploratory mentions).

### Confidence Scoring

Each persona receives a `profile_confidence` float built from four sub-scores:

| Signal | Max Weight |
|---|---|
| Evidence facts (personal + org + constraint) | +0.30 |
| Profile word count (>2000 words = full score) | +0.25 |
| Stakeholder relevance score | +0.15 |
| Profile structural quality (MBTI present, personality keywords) | +0.15 |
| Base score | +0.40 |

Capped at 1.0. This score surfaces in the UI so analysts can see which personas are evidence-backed vs. archetypal guesses.

### Fallback: Role-Archetypal Minimal Profile

If the LLM call fails, a deterministic fallback generator produces a ~200-word profile using the stakeholder's role string. It does **not** produce a generic skeleton — it selects from role-class archetypes:
- `engineer|architect` → INTJ, technically exacting, risk-averse to reliability impact
- `product|pm` → ENTJ, data-informed, demands success metrics
- `cto|vp|director` → ENTJ, thinks in quarters, evaluates by cost efficiency
- `security|compliance` → ISTJ, risk-first, veto until proven safe
- `design|ux` → ENFJ, user-advocate, blocks friction without payoff

The fallback profile is flagged with `[NOTE: This is a fallback profile — re-run to get the full narrative]` so it cannot be mistaken for a high-confidence output.

### Output: `List[FinalPersona]`

```python
FinalPersona(
    name="Alex Chen",
    role="VP Engineering",
    psychological_profile=PsychologicalProfile(
        mbti="INTJ",
        ocean_scores={"openness": 0.72, "conscientiousness": 0.88, ...},
        key_traits=[...],
        emotional_triggers=EmotionalTriggers(...),
        communication_style=CommunicationStyle(...),
        decision_pattern=DecisionPattern(...),
        predicted_stance=PredictedStance(
            feature="AI-Powered Code Review",
            prediction="CONDITIONAL_APPROVE",
            confidence=0.85,
            likely_conditions=["Requires SOC2 audit", "No latency regression"],
            potential_objections=["Training data provenance", "Model bias risk"]
        ),
        questions_they_will_ask=[...],
        full_profile_text="..."  # the raw 2500-word narrative
    ),
    evidence_sources=["customer_interviews.pdf", "slack_export.txt"],
    profile_word_count=2487,
    profile_confidence=0.91,
    persona_type="INTERNAL",
    grounding_quality=0.87,
    market_context=MarketContext(...),   # external only
    buyer_journey=BuyerJourney(...)      # external only
)
```

---

## 5. OASIS User Persona Generator

**File:** `tsc/oasis/oasis_persona_gen.py`

This is a **separate persona system** from Layer 3. While Layer 3 generates organizational stakeholders (internal employees, external institutional buyers), the OASIS User Persona Generator creates **product users** — the population that participates in the behavioral social simulation.

### Step 1: Customer Evidence Gathering

The generator first queries the Hindsight WorldDataBank (Qdrant) for all raw customer evidence:
```
"What user types, demographics, usage patterns, complaints, and feature requests
 are mentioned in the customer data?"
```
It also draws from up to 40 priority-sorted raw chunks from Layer 1 (customer interviews, support tickets, feedback).

### Step 2: Structured CoT Segment Inference (10-Category Coverage Mandate)

The system sends the customer evidence to the LLM using a four-step Chain-of-Thought reasoning prompt that mandates **10 distinct market categories**:

| Category | Example |
|---|---|
| Core Market: New/Activating | Users in onboarding, evaluating on time-to-value |
| Core Market: Current/Habitual | Power users, muscle-memory dependent |
| Core Market: Resurrected | Returned users, highly skeptical |
| Core Market: Dormant/Slipping | Actively evaluating competitors |
| Adversarial Market | Direct competitor agents |
| Narrative Market | Tech journalists, analysts |
| Capital Market | Investors, VCs, board members |
| Regulatory Market | Privacy officers, compliance auditors |
| Connected Market | API partners, suppliers |
| Internal GTM Market | Sales, legal, support |

The LLM must return a `coverage_check` dict verifying all 10 are present. If any are missing, the system logs a coverage gap warning.

**Proportioning:** The LLM assigns `proportion` and `revenue_proportion` per segment. Agent count distribution uses a 50/50 blend of headcount and revenue weight — ensuring high-ARR power-user segments receive proportionally more simulation agents even if they are a small headcount minority.

### Step 3: Per-Segment Persona Instantiation

For each segment, the generator calls the LLM (up to 5 retries, batches of 3) with a distinct system prompt depending on segment type:

**Core Market segments** use `CORE_PERSONA_GEN_SYSTEM`:
- Diversity mandate across 7 axes (seniority, tenure, emotional state, use-case, workaround, communication style, participation mode)
- OCEAN-to-behavior translation: each trait maps to concrete behavioral rules (e.g., `extraversion < 0.35` → lurker who only responds when directly challenged)
- Sycophancy guardrail: agents must have genuine self-interest and only change stance if arguments align with their Core Goals
- 5-layer cognitive identity packet (`[IDENTITY ANCHOR]`, `[LIVED EXPERIENCE & MOTIVATION]`, `[WORLDVIEW & COGNITIVE BIAS]`, `[COMMUNICATION FINGERPRINT]`, `[CORE GOALS & INCENTIVES]`)
- ENTROPY CONSTRAINT: unique rare Indian surnames, zero overlap in communication style or motivation

**Ecosystem segments** (competitors, VCs, regulators, journalists) use `ECOSYSTEM_PERSONA_GEN_SYSTEM`:
- Must use REAL Indian institutions (actual VC funds, regulatory bodies, publications)
- 3-layer institutional identity packet (no individual people — the entity itself speaks)
- Immune to peer pressure from ordinary users in the simulation
- Institutional epistemic humility: must flag unlisted details as speculative

### Step 4: Agent Description Build (Dual-Anchor Identity Injection)

Each persona is converted to an `OASISAgentProfile`. The `description` field is built by translating the 5 OCEAN floats into human behavioral sentences — this is what gets injected at the TOP of the agent's system prompt (primacy bias):

```
You are Ravi Menon, a Staff Engineer in the 'Core Market: Current/Habitual' segment.
RIGHT NOW: Currently at-risk; actively evaluating alternatives. Churn risk is 35%.
HOW YOU THINK: Resistant to change; needs proof before engaging with anything new.
              Reads full documentation before commenting; always cites sources.
HOW YOU ENGAGE: Lurker; reads silently and only responds when directly challenged.
               Blunt and direct; states disagreement immediately with no softening.
HOW YOU FEEL: Emotionally stable; measured language, does not amplify problems.
```

The `user_profile` field (the full 5-layer narrative) is injected at the BOTTOM of the system prompt (recency bias). This dual-anchor pattern — brief behavioral descriptor at top, full narrative at bottom — is based on research that LLMs give highest attention to primacy and recency positions in long context windows.

### Output: `List[OASISAgentProfile]`

```python
OASISAgentProfile(
    agent_id=12,
    source_persona_id="Ravi Menon",
    agent_type="Core Market: Current/Habitual",
    user_info_dict={
        "user_name": "ravi_menon",
        "name": "Ravi Menon",
        "description": "<OCEAN behavioral sentences>",
        "profile": {
            "user_profile": "<5-layer identity card>",
            "ocean_scores": {...},
            "other_info": {
                "segment": "Core Market: Current/Habitual",
                "pain_points": [...],
                "communication_style": "terse, technical, bullet points",
                ...
            }
        }
    },
    influence_strength=0.72,   # power users get 0.65–0.85
    receptiveness=0.38         # low: resistant to persuasion
)
```

---

## 6. OASIS Simulation Engine

**File:** `tsc/oasis/simulation_engine.py`

The simulation engine is built on **CAMEL-AI OASIS** — an open-source social platform simulation framework. The TSC engine wraps it with substantial custom logic: GraphRAG grounding, Hindsight memory, Game Master classification, social network topology construction, and a multi-agent DAG report pipeline.

### Infrastructure Setup

**Platform:** CAMEL's `Platform` object is initialized with a SQLite database for social actions (posts, comments, likes, follows, mutes, groups). A `Clock` object provides simulated time.

**Database isolation:** Each simulation run gets its own isolated SQLite file at `log/oasis_runs/{simulation_name}/{simulation_name}.sqlite`. The master metadata DB is also isolated per run, preventing cross-simulation contamination.

**Rate limiting:** Two mechanisms guard LLM calls:
- `asyncio.Semaphore(max_concurrency)` — throttles thundering herd (2 concurrent if RPM ≤ 5, else 4)
- `AsyncLimiter(GEMINI_FREE_RPM, 60.0)` — token bucket enforcing hard RPM cap (default 10/min, overridable via `GEMINI_FREE_RPM` env var)

**macOS deadlock immunity:** `tensorflow` and `codecarbon` are intercepted and set to `None` in `sys.modules` on darwin. CAMEL's `FunctionTool.get_openai_tool_schema` is monkey-patched to force `strict: False` on all tool schemas.

### Social Network Topology (3-Layer Construction)

Rather than a star graph where every agent only sees the proposer, the engine builds a realistic network:

**Layer 1 — Universal proposer follow:** Every agent follows agent_0 (the proposer) guaranteeing seed post visibility to all.

**Layer 2 — Preferential attachment with homophily:**
- Each agent samples 3–6 peers using weighted random selection
- Base weight = other agent's `influence_strength` (popular agents attract more followers)
- Homophily bonus: agents of the same `agent_type` get 2× weight
- Follower's `receptiveness` modulates how many connections they form

**Layer 3 — Stochastic reciprocity:** For each directional edge A→B, there is a 30% chance B→A is also created.

After construction, native igraph analytics are computed: graph density, clustering coefficient, and average betweenness centrality — emitted to the frontend for the 3D graph renderer.

### Seed Post Generation (AI-First, Template-Fallback)

Seed posts are the **only** information channel agents receive. Every fact in the proposal and context that is not embedded in a seed post does not exist for the simulation.

**AI-First path (v4 batched approach):**
1. An LLM call generates a compressed 300-word Executive Summary of the full context
2. Eight archetypal posts are generated in two parallel batches of 4:
   - Batch 1: `OFFICIAL_ANNOUNCEMENT`, `BUSINESS_ANALYST`, `TECHNICAL_DEVELOPER`, `COMPETITOR_OBSERVER`
   - Batch 2: `HISTORICAL_CONTEXT_CARRIER`, `SAFETY_REGULATORY_WATCHDOG`, `AFFECTED_STAKEHOLDER`, `EXIT_ULTIMATUM`
3. Each post is required to embed at least one specific data point from the brief

**Template-Fallback path:** If the LLM fails, controversy seeds are extracted from raw customer quotes in the market context — verbatim quotes from actual interviews become the seed posts. A set of 6 template seeds covers: angry power user, skeptical analyst, procurement evaluator, new user, churning user, and feature advocate.

**Human-in-the-Loop:** If an `interactive_cb` callback is registered, the simulation halts after seed generation for a human reviewer to inspect and edit the seeds before any agent sees them.

### Main Simulation Loop (Phase 1)

For each timestep `t` in `range(config.num_timesteps)`:

**Per-agent turn (sequential, rate-limited):**

1. **Hindsight Recall:** If memory is available, `recall_for_turn(agent_id)` is called — returns ~300 tokens of the agent's most relevant remembered experiences from prior turns (via semantic search over the agent's Hindsight bank).

2. **Platform Refresh:** `platform_obj.refresh(agent_id)` retrieves the agent's current social feed — up to 5 posts × 3 comments each. This is their view of the community discussion.

3. **GraphRAG Fact Injection:** If a `KnowledgeGraph` is provided, the engine performs a zero-LLM entity lookup:
   - Scans the context (platform_obs + hindsight) for entity names appearing in the graph
   - Traverses the neighborhood of active entities
   - Injects the top 15 edges (by weight) as `[MANDATORY SYSTEM FACTS — DO NOT HALLUCINATE]`

4. **Context Window Construction (P3 ordering):** The prompt is assembled in evidence-first order (lowest to highest attention in the LLM):
   ```
   <posts>     platform feed (data — middle position, lower bias OK)
   <social_relationships>  who they follow / who follows them
   <memory>    Hindsight context (narrative — middle)
   <journal>   agent's own emotional state (satisfaction, frustration, trust)
   <rules>     persona grounding + anti-sycophancy guardrails (recency — high attention)
   [MANDATORY SYSTEM FACTS]  GraphRAG edges (last before action cue)
   action_cue  "Choose ONE action..." (very last token — maximum focus)
   ```

5. **ReAct Loop (up to 3 steps):** The agent can take intermediate actions:
   - `search_feature_docs` — retrieves the raw feature spec (once per turn)
   - Any terminal action (`CREATE_COMMENT`, `CREATE_POST`, `LIKE`, `FOLLOW`, etc.) breaks the loop

6. **Phase-Aware Timestep Directive:** Early turns get `OPENING` directive ("State your initial position clearly"), mid-turns get `MID-DISCUSSION` ("React. Build on or push back"), final turns get `CLOSING` ("Has anything changed your position?").

7. **Anti-echo-chamber rule:** Agents must read the thread before posting. If their primary concern was already stated, they must agree briefly then pivot to a new, unmentioned angle.

8. **Action Logging:** The final `content` string is extracted from the CAMEL response. If the agent called a tool, the tool argument's `content` field is used; otherwise the raw message content. Thought blocks (`<thought>...</thought>`, `<thinking>...</thinking>`) are stripped. The final clean text is cross-verified against the SQLite platform DB — the actual stored comment/post content is used for accuracy.

### Game Master (GM) — Behavioral Signal Classification

After each agent action, the Game Master classifies the content into a behavioral signal vector.

**Routing logic:**
1. **Exact cache hit:** If the canonicalized text was seen before, return the cached resolution instantly (0 tokens)
2. **Semantic Jaccard cache hit:** If any cached key has ≥ 90% word overlap, return the cached resolution
3. **Regex fast-path:** 20 signal patterns are scanned (exit_intent, purchase_intent, trust_erosion, regulatory_risk, etc.). If no GM LLM client is configured, this produces a static delta result
4. **LLM structured classification:** If a GM LLM client is configured, the content plus the agent's current internal state is sent to the LLM which returns a `GameMasterResolution` Pydantic model with exact deltas for satisfaction, frustration, and trust

**Sycophancy collapse detection:** If an agent suddenly agrees with a statement despite prior high frustration, the GM flags this as a data validity warning. Calibrated by agreeableness: agreeable agents (>0.65) are allowed to find common ground naturally; this only flags collapse for stubborn or highly frustrated agents.

**State update:** The GM's resolution is weighted by the agent's `influence_strength` before being applied to the `DecisionJournal`:
```python
weighted_signal["satisfaction_delta"] = sig["satisfaction_delta"] * influence_strength
decision_journals[agent_id].update_from_signal(weighted_signal)
```

**GM → Platform Feedback Loop:** After every timestep, agents with `frustration > 0.75` are identified as `HIGH_RISK`. The GM instructs the platform to register a `dislike_post` on their current top feed item. This informs the RecSys (recommendation system) to serve different content to at-risk agents in the next timestep — simulating how platforms respond to engagement signals in the real world.

### Hindsight Memory (Act → Retain → Reflect Loop)

At the end of each agent turn, if Hindsight is available:
```
agent takes action
    → memory_manager.structured_retain(agent_id, agent_name, action_type, content, timestep)
```

After every timestep:
```
memory_manager.synthesize_post_timestep(timestep=t)
```

This synthesis step runs a reflection pass over all retained actions from the current timestep, extracting belief updates and storing them as evolved memories. On the next turn, `recall_for_turn(agent_id)` retrieves these evolved beliefs — giving agents continuity across the simulation's entire run.

---

## 7. Post-Simulation: Scoring, Reports, and Executive Summary

### Phase 2: Focus Group (Optional)

If `config.enable_interview_phase = True`, a stratified sample of agents is selected after the main loop:
- Top N/3 by satisfaction (champions)
- Top N/3 by frustration (detractors)
- Random N/3 from the remainder (lurkers)

Each selected agent undergoes `_interview_agent_with_hindsight`:
1. CAMEL social-feed memory is snapshotted and then cleared (token isolation)
2. Hindsight recall is performed — targeting the specific question being asked
3. A structured Q&A interview is conducted via `perform_interview()`
4. Each Q+A pair is retained back into Hindsight (tagged `"interview"`, `timestep=99`)
5. CAMEL memory is restored from the snapshot

Interview transcripts are then passed through `extract_business_metrics()` to extract:
- `willingness_to_pay_usd_monthly` — stated price tolerance
- `adoption_intent` — likelihood to adopt
- `churn_risk_delta` — expected change from baseline
- `primary_objection` — the single biggest stated barrier

### Population Extrapolation & Shadow Agents

The `PopulationSampler` supports declaring large populations (e.g., 1,000 agents) while only running LLM turns for a representative cohort (e.g., 30 agents). After the simulation:

1. Shadow agents inherit the behavioral state distribution of the active cohort, stratified by segment
2. `ClusterOnBehavioralState()` groups all agents (active + shadow) using igraph community detection on their `(satisfaction, frustration, trust)` state vectors
3. `build_extrapolated_report()` produces population-scale statistics with 95% confidence intervals and margin-of-error

### Business Metrics Computed

| Metric | Computation |
|---|---|
| **Net Promoter Score** | % promoters (satisfaction > 0.7) − % detractors (satisfaction < 0.4), scaled to [-100, +100] |
| **Churn Velocity** | Average rate of frustration increase per timestep across all agents |
| **Adoption Momentum** | Average rate of satisfaction increase per timestep across all agents |
| **Risk Distribution** | % of agents in HIGH / MODERATE / LOW risk buckets by frustration × trust |
| **Decision Events** | Any GM resolution that produced a signal with abs(intensity) > 0.6, logged with timestep, trigger, and confidence |

### Report Orchestration DAG (Anti-Hallucination Pipeline)

The executive summary is generated by a 3-agent DAG:

**Agent 1 — Data Analyst:** Receives the raw JSON metrics and extracts exact numerical facts into a `ReportFacts` Pydantic model. Prohibited from rounding, estimating, or fabricating.

**Agent 2 — Guardrail Fact-Checker:** Receives both the raw JSON AND the Analyst's extracted facts. Verifies every number and quote against the source. Returns `is_valid=True` or a list of specific errors.

**Retry loop:** If the fact-checker fails, the error is fed back to the Analyst as a correction prompt. Up to 3 retries. If validation fails all 3 times, the system returns a failure notice rather than a hallucinated summary.

**Agent 3 — Executive Writer:** Receives only the VALIDATED facts. Writes exactly 3 paragraphs with a fixed structure:
- Paragraph 1: Lead with the most surprising finding. Ship / Ship with changes / Do not ship verdict. Cites exact NPS, Churn Velocity, Adoption Momentum.
- Paragraph 2: Focus Group WTP, Adoption Intent, Churn Risk Delta. Includes one verbatim agent quote.
- Paragraph 3: Three actionable next steps, at least one addressing the top focus group objection.

---

## 8. Memory Architecture: Hindsight

Hindsight is the distributed memory system binding all layers together. It has two distinct usage modes in the TSC pipeline:

**WorldDataBank (Layer 1 output, persona gen input):**
- Stores all raw document chunks
- Queried by `OASISUserPersonaGenerator._gather_customer_data()` to ground segment inference in actual evidence
- Collection name: `"world"`

**HindsightOASISManager (simulation per-agent memory):**
- Each agent gets a named bank: `"{simulation_id}-{timestamp}"`
- `structured_retain()` writes each action (type, content, timestep) into the bank
- `recall_for_turn(agent_id, custom_query)` retrieves the most semantically relevant prior actions for the current context (default query: the current discussion topic)
- `synthesize_post_timestep(timestep)` runs a reflection pass extracting evolved beliefs
- On simulation restart: the previous run's banks are purged, new timestamped banks are created — preserving forensic history until overwritten by a new run

---

## 9. What Makes This Architecture Different

### GraphRAG as Live Agent Grounding, Not Just Retrieval

Most RAG systems retrieve documents at query time and inject them as context. TSC's graph is different: during the simulation, every agent turn performs an entity-recognition pass over the current discussion, traverses the knowledge graph neighborhood of active entities, and injects the top-weighted relationship edges as mandatory system facts. This means agents cannot contradict documented relationships (e.g., "CTO OWNS_DECISION on data privacy policy") even if the social pressure in the discussion pushes them to.

### Dual Persona System (Stakeholders vs. Users)

Layer 3 generates internal organizational stakeholders — the people who approve or reject a feature inside the company. The OASIS persona generator generates external product users — the people who will react to the feature in the market. These are separate systems with separate LLM prompts, separate evidence sources, and separate psychological frameworks (MBTI + clinical stances vs. Big Five + market segment behavior).

### 5-Layer Cognitive Identity Packet

Standard social simulation agents get a role description. TSC agents get a 5-section grounded biography:
1. **IDENTITY ANCHOR** — who they are factually
2. **LIVED EXPERIENCE & MOTIVATION** — what shaped their worldview
3. **WORLDVIEW & COGNITIVE BIAS** — their systematic blind spots
4. **COMMUNICATION FINGERPRINT** — sentence length, vocabulary register, punctuation habits
5. **CORE GOALS & INCENTIVES** — the 2–3 self-interested objectives they will always optimize for

The system does NOT script triggers that change their mind. It gives the LLM full autonomy to evaluate whether any given argument aligns with or threatens these core goals — producing emergent, non-scripted stance evolution.

### OCEAN-to-Behavior Translation (Not Abstract Floats)

OCEAN scores are not stored as `0.78` and left for the LLM to interpret. They are translated into explicit behavioral sentences at generation time:
- `openness > 0.65` → "Curious about new paradigms; tries features before they're polished."
- `neuroticism > 0.65` → "Emotionally reactive; uses strong language when frustrated ('broken', 'disaster')."
- `extraversion < 0.35` → "Lurker; reads silently and only responds when directly challenged."

These translated sentences are injected at the primacy position of the agent's system prompt. The full 5-layer narrative is injected at the recency position. Together they function as a two-anchor behavioral lock.

### Game Master as a Live Feedback Loop, Not Just Scoring

The Game Master does not merely score outputs — it writes negative engagement signals back to the CAMEL Platform's recommendation system for high-risk agents. This means the content an at-risk agent sees in their next feed is algorithmically adjusted by their current emotional state — the same feedback loop that real social platforms use. Echo chamber formation, filter bubble dynamics, and viral contagion are emergent properties of the simulation's structure, not scripted behaviors.

### Anti-Hallucination in Executive Output

The three-agent DAG (Analyst → Fact-Checker → Writer) is specifically designed to prevent the most common failure mode in LLM-generated reports: hallucinated numbers. The Writer never sees the raw data. It only receives a Pydantic-validated, fact-checker-approved structured object. If the Analyst fabricates or rounds a metric, the Fact-Checker catches it before the Writer ever runs — and the cycle retries up to 3 times before returning a failure notice rather than a corrupted report.

---

*Document generated from direct code analysis of `tsc/layers/layer1_ingestor.py`, `tsc/layers/layer2_graph.py`, `tsc/layers/layer3_personas.py`, `tsc/oasis/oasis_persona_gen.py`, and `tsc/oasis/simulation_engine.py`.*
