# TSC Pipeline v2.0 — Post-Run Fix Plan

Based on the comprehensive analysis report, this plan addresses 4 critical and 6 high/medium fixes.

## Proposed Changes

### 1. Verdict Extraction — Fix False REJECTED Parsing

#### [MODIFY] [layer6_debate.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer6_debate.py)

**Root cause:** `_extract_verdict()` checks reject patterns *first*, matching words like "not", "no", "cannot" which appear in nearly every conditional approval statement ("I cannot agree without..."). This causes conditional approvals to be parsed as REJECTED.

**Fix:** Restructure the pattern priority:
1. Check for **explicit approval** first (`"approve"`, `"approval"`, `"endorse"`)
2. Then check for **conditional** signals (`"conditional"`, `"subject to"`, `"provided that"`)
3. Only then check for **rejection** with strict patterns (`"I reject"`, `"should be rejected"`)
4. Prevent substring false positives (e.g., "I reject the notion" ≠ rejection of the feature)

---

### 2. Monte Carlo Gate 4.5 — Fix Empty Pain Points

#### [MODIFY] [market_fit_gate.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/gates/market_fit_gate.py)

**Root cause:** `_get_top_pain_points()` only looks for entities with `type == "PAIN_POINT"`, but no such entities exist in the graph (spaCy NER doesn't produce `PAIN_POINT` types). This returns an empty list, so agents have no resistance signals.

**Fix:**
1. Expand `_get_top_pain_points()` to also extract pain points from:
   - Entities mentioning negative keywords (crash, bug, drain, latency, freeze)
   - Graph node contexts containing complaint patterns
   - Fallback to hardcoded domain-relevant pain points from the feature description
2. In `_build_agent_decision_prompt()`, also inject the agent's own `world_state.pain_points` (which are populated from LHS distributions but never sent to the LLM)

---

### 3. Metric Extraction — Improve Coverage from 0% to ~50%

#### [MODIFY] [layer1_ingestor.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer1_ingestor.py)

**Root cause:** `_extract_metrics()` regex patterns are too narrow — they only match `N% of WORD`, `N users/tickets`, and `$N`. They miss temporal metrics (`20 minutes`, `2 seconds`), compound metrics (`10% reduction`), and percentage mentions without "of" (`95% uptime`).

**Fix:** Add new regex patterns:
- `(\d+)\s*(minutes?|seconds?|hours?|days?|weeks?|months?)` — temporal
- `(\d+(?:\.\d+)?)\s*%\s*(reduction|increase|improvement|uptime|churn|adoption)` — percentage with context
- `(\d+(?:\.\d+)?)\s*[xX]\s*(faster|slower|improvement)` — multiplier
- `(\d+(?:\.\d+)?)\s*(MB|GB|TB|ms|fps)` — technical units

---

### 4. Layer 5 Refinement — Don't Silently Skip on Validation Failure

#### [MODIFY] [layer5_refinement.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer5_refinement.py)

**Root cause:** `_validate_refinement()` rejects responses where the `analysis` field exceeds 500 chars. Since the LLM typically generates longer analysis text, this frequently triggers — and the pipeline returns the original unrefined `gates_summary`, silently discarding valid refinement suggestions.

**Fix:**
1. Increase the `analysis` max length from 500 → 2000 chars (analysis should be detailed)
2. Increase `revised_scope` max length from 500 → 1500 chars
3. When validation fails on *non-critical* issues (length only), truncate the field rather than rejecting entirely
4. Log a clear warning when truncation occurs

---

### 5. KG Circular Dependencies — Dampen Cycle Edges

#### [MODIFY] [layer2_graph.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer2_graph.py)

**Root cause:** `_detect_circular_dependencies()` only *detects* and *logs* cycles but takes no action. Downstream gates that depend on priority ordering can loop indefinitely.

**Fix:** After detection, reduce confidence of edges forming cycles by 50%, so they are weaker in downstream scoring. This creates a "soft break" without losing information.

---

### 6. Entity Skipping — Relax Name Length Validation

#### [MODIFY] [layer2_graph.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer2_graph.py)

**Root cause:** `_extract_entities()` rejects any normalized name shorter than 2 chars. This drops names like "D." (likely Diana) and single digits that may represent metrics.

**Fix:** When a name is < 2 chars, try to recover by checking the original `ent.text` and only skip if the original is also very short (< 2 chars). Preserve initials like "D." as-is without over-normalization.

---

### 7. Confidence Calculation — Fix gate_score Divisor

#### [MODIFY] [layer6_debate.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer6_debate.py)

**Root cause:** In `_calculate_confidence()`, `gate_score = gates_summary.overall_score / 10.0` divides by 10, but `overall_score` is already on a 0–1 scale (e.g., 0.61), producing an artificially low 0.061 gates component.

**Fix:** Remove the `/10.0` divisor if the score is already < 2.0 (indicating it's on a 0–1 scale).

---

### 8. KG Pain Point Entity Type — Add Missing Label

#### [MODIFY] [layer1_ingestor.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer1_ingestor.py)

**Root cause:** spaCy NER only outputs standard labels (PERSON, ORG, PRODUCT, etc.), never `PAIN_POINT`. The `_enrich_local()` method sets NER entities but never classifies complaint/negative sentences as `PAIN_POINT` entities.

**Fix:** Add a simple heuristic in `_enrich_local()`: if a sentence has negative sentiment + urgency ≥ 4, create a synthetic `PAIN_POINT` entity from the sentence, so the knowledge graph has `PAIN_POINT` nodes for Monte Carlo to use.

---

### 9. Layer 5 Validation — Increase Analysis Length Limit

This is covered by Fix #4 above (merged).

---

### 10. Zep Session Init — Graceful Fallback

#### No code change needed — already handled by the existing `(non-fatal)` warning and local fallback mechanism from AddFix-B.

---

---

---

### SOTA Implementation: Gemini 3 Flash Overhaul (SOTA-1 to SOTA-8)

**Objective:** Resolve semantic degradation and hallucination by transitioning to a fully grounded, evidence-backed pipeline using Gemini 3 Flash's 1M context and structured output.

#### 1. [SOTA-1] Recursive Semantic Chunking
- **[MODIFY] [layer1_ingestor.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer1_ingestor.py)**:
  - Replace `_semantic_chunk()` with `_semantic_chunk_llm()`.
  - Implement intent preservation, speaker attribution, and 2-sentence overlap.
  - Produce 15-25 chunks per interview (vs 1 previously).

#### 2. [SOTA-2 & 3] Grounded NER & Knowledge Graph
- **[MODIFY] [layer2_graph.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer2_graph.py)**:
  - **Constraint-Aware Extraction**: Extract entities linked to source quotes. Filter noise (e.g., "d", "001_power_user").
  - **Semantic Relationships**: Every edge must cite 2 source chunks and include semantic reasoning.

#### 3. [SOTA-4] Evidence-Based Persona Generation
- **[MODIFY] [layer3_personas.py](file:///Users/mohnish/Downloads/tsc%20architecture/tsc/layers/layer3_personas.py)**:
  - Cluster quotes by speaker.
  - Generate personas with 5-8 direct quotes each. Zero fabrication of traits.

#### 4. [SOTA-8] Adversarial Stakeholder Debate [COMPLETE]
- **Adversarial Critique**: Updated `layer6_debate.py` to include a dedicated critique round where stakeholders pressure-test each other using evidence.
- **Multi-stakeholder Rebuttal**: Refactored the final round into a rebuttal phase for all participants.
- **Evidence-Based**: Enforced mandatory citations [Evidence: <snippet>] in all debate turns via `prompts.py`.
- **Synthesis**: Final consensus is now a strategic resolution of the adversarial tensions raised during the debate.

#### [NEW] [SOTA-9] FastEmbed Local Integration [IN PROGRESS]
- **Replace Mock Embeddings**: Modify `layer1_ingestor.py` to use `fastembed` for local, high-performance embeddings.
- **macOS Stability**: Leverage ONNX runtime (via FastEmbed) to avoid the heavy Torch/Transformers mutex lock issues on macOS.
- **Semantic Chunking**: Restore 15-25 chunks per document by providing real vectors for semantic boundary detection.

---

---

---

## Verification Plan

### Automated Tests
```bash
python -m py_compile tsc/layers/layer1_ingestor.py
python -m py_compile tsc/layers/layer2_graph.py
python -m py_compile tsc/layers/layer3_personas.py
python -m py_compile tsc/layers/layer4_gates.py
python -m py_compile tsc/layers/layer6_debate.py
```

### Evidence Grounding Check
- Verify `pipeline_output.json` contains `evidence` arrays for every entity, relationship, and gate verdict.
- Confirm speaker attribution is preserved in all chunks.

### Zep Batching Verification
- Manual verification of Zep logs (if possible) or logging in `ZepGraphMemoryUpdater` to confirm batching occurs every 5 actions.
- Confirm `DO_NOTHING` actions are not present in Zep.

### Full Pipeline Re-run
```bash
python3 run_production_pipeline.py > output2.log 2>&1 &
```
