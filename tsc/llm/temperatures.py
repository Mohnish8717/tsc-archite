"""Centralized LLM temperature constants for the TSC pipeline.

Each constant is named after the task it controls, not the layer, so the
reason for the value is self-documenting.

Guidelines (Karpathy):
  - Temperature = 0.0  →  pure determinism, parser/classifier tasks.
  - Temperature = 0.1  →  near-deterministic, structured-output tasks.
  - Temperature = 0.2  →  light creativity, grounded by heavy context.
  - Temperature = 0.3  →  moderate reasoning, synthesis tasks.
  - Temperature = 0.4  →  balanced; used when both accuracy & variety matter.
  - Temperature = 0.5  →  exploratory; multi-option generation.
  - Temperature = 0.6  →  diverse personas / market coverage.
  - Temperature = 0.7  →  adversarial / red-team challenge mode.
  - Temperature = 0.8  →  maximum diversity; within-segment persona generation.

Never use values above 0.8 in production; they produce hallucinations faster
than they add creativity.
"""

# ─── Layer 1: Ingestion ──────────────────────────────────────────────────────
# Extracting structured entities / metadata from raw text. Must be precise.
L1_ENTITY_EXTRACTION: float = 0.1

# Generating clean problem description from noisy input.
L1_PROBLEM_SYNTHESIS: float = 0.2

# ─── Layer 2: Discovery / Graph ───────────────────────────────────────────────
# Market research query generation — needs deterministic, precise queries.
L2_RESEARCH_QUERY: float = 0.3

# Graph entity synthesis — balancing structure with mild creativity.
L2_GRAPH_SYNTHESIS: float = 0.3

# Graph relationship classification — deterministic.
L2_GRAPH_CLASSIFIER: float = 0.1

# Graph enrichment with additional market context.
L2_GRAPH_ENRICHMENT: float = 0.3

# ─── Layer 3: Personas ────────────────────────────────────────────────────────
# Selecting internal stakeholders — needs reasoning over org context.
L3_INTERNAL_STAKEHOLDER_SELECTION: float = 0.4

# Selecting external/customer personas — needs diversity across segments.
L3_EXTERNAL_STAKEHOLDER_SELECTION: float = 0.5

# Generating internal persona profiles — grounded by heavy evidence context.
L3_INTERNAL_PERSONA_PROFILE: float = 0.2

# Generating external/buyer persona profiles — slightly more creative.
L3_EXTERNAL_PERSONA_PROFILE: float = 0.35

# ─── Layer 4: Gates ───────────────────────────────────────────────────────────
# Standard gate evaluation — structured reasoning, not creative.
L4_GATE_EVALUATION: float = 0.2

# Red-team adversarial gate — needs creative, divergent failure-mode thinking.
L4_RED_TEAM: float = 0.6

# ─── Layer 5: Refinement ─────────────────────────────────────────────────────
# Suggesting feature refinements after gate failures.
L5_REFINEMENT: float = 0.4

# ─── OASIS: Simulation Engine ────────────────────────────────────────────────
# Persona segment inference — structured coverage map.
OASIS_SEGMENT_INFERENCE: float = 0.3

# Within-segment persona diversity generation — needs high variety.
OASIS_PERSONA_DIVERSITY: float = 0.8

# Game Master signal classifier — reads free-text agent posts, classifies into
# structured emotional states (satisfaction_delta, signal_type, reasoning).
# Must parse open-ended prose; 0.0 collapses borderline signals into single
# most-probable token, distorting the signal distribution. 0.1 gives the
# minimum headroom to classify ambiguous text accurately.
OASIS_SIMULATION_RESPONSE: float = 0.1

# ─── Layer 6: AG2 Debate Engine ───────────────────────────────────────────────
# Debate summary / transcript compression — should be accurate, low noise.
L6_DEBATE_SUMMARY: float = 0.1

# Research evidence synthesis during debate — structured but requires reasoning.
L6_DEBATE_RESEARCH: float = 0.3

# Compromise condition generation — creative but bounded by prior debate context.
L6_DEBATE_COMPROMISE: float = 0.3

# ─── Layer 7: Specification ───────────────────────────────────────────────────
# Generating the PRD/spec JSON — precision required; creativity hurts JSON fidelity.
L7_SPEC_GENERATION: float = 0.2

# ─── Layer 8: Handoff ────────────────────────────────────────────────────────
# Generating compliance summary — deterministic.
L8_COMPLIANCE_SUMMARY: float = 0.1

# Handoff narrative generation — light creativity to produce readable prose.
L8_HANDOFF_NARRATIVE: float = 0.2

# Final output consolidation — structured, must match expected schema.
L8_OUTPUT_CONSOLIDATION: float = 0.2

# ─── Memory / World RAG ───────────────────────────────────────────────────────
# Hindsight memory synthesis — requires some generalization.
MEMORY_HINDSIGHT: float = 0.3

# World RAG classification — deterministic lookup.
MEMORY_WORLD_RAG: float = 0.1

# ─── Selection Utilities ──────────────────────────────────────────────────────
# Synthetic expander for persona pool — moderate diversity.
SELECTION_SYNTHETIC_EXPANDER: float = 0.4

# Tension vector scoring — deterministic classifier.
SELECTION_TENSION_VECTOR: float = 0.0

# ─── Gates base (shared across all standard gates) ───────────────────────────
# Imported by tsc/gates/base.py; overridden per-gate as needed.
GATE_BASE_DEFAULT: float = 0.2

# ─── Memory: Embedded Query Fallback ─────────────────────────────────────────
# Used when Hindsight is unavailable; LLM synthesizes from regex-extracted context.
# Lower than the old 0.4 — this is structured Q&A, not creative generation.
MEMORY_QUERY_EMBEDDED: float = 0.3
