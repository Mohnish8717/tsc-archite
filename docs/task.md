# TSC Pipeline Repair — Task Checklist

## Original 10 Fixes
- [x] Fix 1: `layer1_ingestor.py` — cosine similarity zero-vector guard, mock-mode dedup bypass, semantic chunk fallback
- [x] Fix 2: `rate_limiter.py` [NEW] — token-bucket rate limiter for Groq
- [x] Fix 3: `layer3_personas.py` — remove `model=` kwarg from `generate()`, reduce max_tokens to 1500
- [x] Fix 4: `pipeline_config.py` [NEW] — centralized config dataclass
- [x] Fix 5: `red_team_gate.py` — halve domain weights, NaN guard, 0.85 per-step/domain cap, 0.88 overall cap
- [x] Fix 6: `market_fit_gate.py` — realistic affinity/urgency/tech bounds, min std=0.12
- [x] Fix 7: `layer4_gates.py` — clean up setattr for recommendation_reason + passed_gates
- [x] Fix 8: `layer4_gates.py` — fix score display from /10 to /1.0 (AddFix-D corrected targets)
- [x] Fix 9: `layer5_refinement.py` + `layer4_gates.py` — selective re-eval via `process_failed_only()`
- [x] Fix 10: `db/models.py` — `on_delete=` → `ondelete=` in ForeignKey

## Additional 7 Fixes
- [x] AddFix-A: `models/gates.py` — `ConfigDict(extra="allow")` + model_config on GatesSummary
- [x] AddFix-B: `memory/zep_client.py` — full rewrite from `.memory` to `.graph` API with local fallback
- [x] AddFix-C: `openai_provider.py` — remove `model=` kwarg from `generate()`, reduce default max_tokens
- [x] AddFix-D: (merged into Fix 8) — corrected score scale targets in layer4_gates.py
- [x] AddFix-E: `layer4_gates.py` — cache isolation in `process_failed_only()` with `selective_reeval` diagnostic flag
- [x] AddFix-F: `config/__init__.py` [NEW] — config package init
- [x] AddFix-G: `openai_provider.py` — rate limiter integration in `analyze()` and `generate()`

## Post-Analysis Fixes (v2.0 Report)
- [x] PA-1: `layer6_debate.py` — Fix verdict extraction (approve→conditional→reject order)
- [x] PA-2: `market_fit_gate.py` — Fix empty pain_points with 4 fallback strategies
- [x] PA-3: `layer1_ingestor.py` — Expand metric extraction regex + LLM hybrid extraction
- [x] PA-4: `layer5_refinement.py` — Increase validation limits (500→2000) + graceful truncation
- [x] PA-5: `layer2_graph.py` — Dampen KG cycle edges by 50% after detection
- [x] PA-6: `layer2_graph.py` — Recover short entity names (relax len<2 → len<1)
- [x] PA-7: `layer6_debate.py` — Fix confidence gate_score divisor (auto-detect 0-1 vs 0-10)
- [x] PA-8: `layer1_ingestor.py` — Synthesize PAIN_POINT entities from negative sentiment + urgency≥4
- [x] SOTA-9: FastEmbed Local Integration
- [x] Full Pipeline Verification (output.log)
- [x] Comprehensive Architecture Documentation (architecture_overview.md)
- [x] Detailed Gate Analysis (gate_analysis_deep_dive.md)
- [x] Detailed Persona Analysis (persona_generation_deep_dive.md)
- [x] PA-10: Create `memory_updater.py` (25-50 buffer, LLM batch summarization)
- [x] PA-11: Implement RRF reranking in `fact_retriever.py`
- [x] SOTA-1: [layer1_ingestor] Implement Semantic Chunking (Gemini 3 Flash)
- [x] SOTA-2: [layer2_graph] Implement Grounded NER with Evidence
- [x] SOTA-3: [layer2_graph] Implement Knowledge Graph with Semantic Evidence
- [x] SOTA-4: [layer3_personas] Implement Evidence-Based Persona Generation
- [x] SOTA-8: [layer6_debate] Implement Adversarial Stakeholder Debate
- [x] SOTA-9: [layer1_ingestor] Replace Mock Embeddings with FastEmbed
