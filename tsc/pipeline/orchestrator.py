"""Pipeline orchestrator: runs all layers of the Predictive Reality Engine pipeline."""

from __future__ import annotations

import json
import logging
import time
import asyncio
from pathlib import Path
from typing import Any, Optional

# Directory where OASIS simulation runs (and pipeline.jsonl) are written
_LOG_BASE = Path("log/oasis_runs")

from tsc.config import Settings, settings
from tsc.layers.layer1_ingestor import ContextualIngestor
from tsc.layers.layer2_graph import KnowledgeGraphBuilder
from tsc.layers.layer2_discovery import FeatureDiscoveryEngine
from tsc.layers.boardroom_personas import BoardroomPersonaFactory
from tsc.layers.layer6_ag2_debate import AG2DebateEngine
from tsc.layers.layer7_spec import SpecGenerator
from tsc.layers.layer8_handoff import HandoffGenerator
from tsc.llm.base import LLMClient
from tsc.llm.factory import create_llm_client
from tsc.memory.graph_store import GraphStore
from tsc.memory.hindsight_session import HindsightSessionManager
from tsc.memory.world_rag import WorldRAGEngine
from tsc.memory.world_bank import WorldDataBank, set_engine as _set_world_bank_engine
from tsc.oasis.oasis_persona_gen import OASISUserPersonaGenerator
from tsc.oasis.simulation_engine import RunOASISSimulation, OASISSimulationConfig
from tsc.models.inputs import DocumentType, InputDocument
from tsc.models.recommendation import FinalRecommendation

logger = logging.getLogger(__name__)


class TSCPipeline:
    """Orchestrates the autonomous product management pipeline."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        cfg: Optional[Settings] = None,
    ):
        self._cfg = cfg or settings
        self._llm = llm_client or create_llm_client(settings=self._cfg)

        # Memory: Universal Hindsight Session Backbone (agent experience only)
        self._session = HindsightSessionManager()

        # Memory: WorldRAGEngine — Qdrant + Neo4j + LazyGraphRAG
        # Handles ALL company knowledge / pipeline artifact storage.
        # Hindsight is preserved strictly for boardroom and OASIS agent memories.
        self._rag_engine = WorldRAGEngine()

        # Progress callback (for web UI)
        self._on_progress: Optional[Any] = None
        self._interactive_cb: Optional[Callable] = None

        # Path to the pipeline.jsonl event stream for the current run (set in evaluate)
        self._pipeline_jsonl: Optional[Path] = None

    def set_progress_callback(self, callback: Any) -> None:
        """Set a callback(layer_num, layer_name, status, details) for progress."""
        self._on_progress = callback

    def set_interactive_callback(self, callback: Callable) -> None:
        """Register a callback for interactive Human-in-the-Loop steps."""
        self._interactive_cb = callback

    async def evaluate(
        self,
        interviews: Optional[str] = None,
        support: Optional[str] = None,
        analytics: Optional[str] = None,
        context: Optional[str] = None,
        proposal: Optional[str] = None,
        num_simulations: Optional[int] = None,
        use_legacy_personas: bool = False,
    ) -> FinalRecommendation:
        """Run the full Predictive Reality Engine pipeline.

        Args:
            interviews: Path to customer interviews file.
            support: Path to support tickets file.
            analytics: Path to analytics data file.
            context: Path to company context JSON.
            proposal: Path to feature proposal JSON.
            num_simulations: Number of OASIS simulation agents.
            use_legacy_personas: If True, uses the full Layer 3 LLM-driven
                PersonaGenerator pipeline instead of BoardroomPersonaFactory.
                Default is False (static boardroom personas).

        Returns:
            FinalRecommendation with verdict, spec, and monitoring plan.
        """
        t0 = time.time()
        session_id = f"run-{int(t0)}"
        logger.info("=" * 60)
        logger.info("PREDICTIVE REALITY ENGINE PIPELINE — STARTING")
        logger.info("LLM: %s (%s)", self._llm.__class__.__name__, self._llm.model)
        if num_simulations:
            logger.info("Simulation Count: %d", num_simulations)
        logger.info("=" * 60)

        # ── Bootstrap WorldRAGEngine (Qdrant + Neo4j) ─────────────────────────
        try:
            await self._rag_engine.initialize(run_id=session_id)
            _set_world_bank_engine(self._rag_engine)   # wire WorldDataBank façade
            logger.info("WorldRAGEngine ready (run_id=%s)", session_id)
        except Exception as _rag_exc:
            logger.warning(
                "WorldRAGEngine init failed (%s) — pipeline continues in degraded mode",
                _rag_exc,
            )

        # ── Create run directory & pipeline event stream immediately ───────────
        run_dir = _LOG_BASE / session_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline_jsonl = run_dir / "pipeline.jsonl"
        self._pipeline_jsonl.write_text("")  # truncate / create fresh
        logger.info("Pipeline event stream: %s", self._pipeline_jsonl)

        # Emit initial "all layers waiting" event so the UI resets immediately
        self._write_jsonl_event({
            "type": "pipeline_reset",
            "session_id": session_id,
            "stages": {"layer1": "waiting", "layer3": "waiting", "layer5": "waiting"},
        })

        # Initialize Universal Memory Sessions
        await self._session.initialize(session_id)
        
        # Instantiate WorldDataBank facade for pipeline data
        world_bank = WorldDataBank()

        # If no interactive callback is set (e.g. CLI run), use a file-polling fallback
        if not self._interactive_cb:
            async def default_interactive_cb(action: str, payload: dict) -> dict:
                self._write_jsonl_event({
                    "type": "action_required",
                    "action": action,
                    "payload": payload
                })
                logger.info(f"⏸️ Pipeline paused for interactive action: {action}. Polling commands.json in {run_dir}...")
                commands_file = run_dir / "commands.json"
                while True:
                    if commands_file.exists():
                        try:
                            import json
                            data = json.loads(commands_file.read_text())
                            if data.get("type") == "action_response" and data.get("action") == action:
                                commands_file.unlink() # clear it after reading
                                return data.get("data", {})
                        except Exception as e:
                            logger.error(f"Error reading commands.json: {e}")
                    await asyncio.sleep(1)

            self._interactive_cb = default_interactive_cb

        # Build document list
        documents = self._build_document_list(
            interviews, support, analytics, context, proposal
        )
        logger.info("Input: %d documents", len(documents))

        # Layer 1: Ingest (Extract customer data & initial context)
        self._emit_progress(1, "Contextual Ingest", "running")
        ingestor = ContextualIngestor(self._llm, session=world_bank)
        bundle, feature, company = await ingestor.process(documents)

        self._emit_progress(1, "Contextual Ingest", "done", {
            "chunks": bundle.statistics.total_chunks,
        })

        # ── Emit Layer 1 ingestion nodes (real file names) ─────────────────────
        ingestion_nodes = self._build_ingestion_nodes(documents)
        self._write_jsonl_event({"type": "ingestion_sync", "nodes": ingestion_nodes})

        # ── Emit Live Knowledge Graph from Neo4j ──────────────────────────
        try:
            # Query Neo4j for the global company knowledge graph
            cypher = "MATCH (n)-[r]->(m) RETURN n.name AS source, type(r) AS rel, m.name AS target, labels(n) AS src_labels, labels(m) AS tgt_labels LIMIT 300"
            graph_results = await world_bank.query_graph(cypher, {})
            
            kg_nodes = {}
            kg_edges = []
            
            for res in graph_results:
                rec = res.metadata
                src_name = rec.get("source", "Unknown")
                tgt_name = rec.get("target", "Unknown")
                
                src_label = rec.get("src_labels", ["Entity"])[0] if rec.get("src_labels") else "Entity"
                tgt_label = rec.get("tgt_labels", ["Entity"])[0] if rec.get("tgt_labels") else "Entity"
                
                if src_name not in kg_nodes:
                    kg_nodes[src_name] = {"id": src_name, "label": src_name, "entityType": src_label, "mentions": 1}
                else:
                    kg_nodes[src_name]["mentions"] += 1
                    
                if tgt_name not in kg_nodes:
                    kg_nodes[tgt_name] = {"id": tgt_name, "label": tgt_name, "entityType": tgt_label, "mentions": 1}
                else:
                    kg_nodes[tgt_name]["mentions"] += 1
                    
                kg_edges.append({
                    "source": src_name,
                    "target": tgt_name,
                    "relationshipType": rec.get("rel", "RELATED_TO"),
                    "weight": 1
                })
                
            self._write_jsonl_event({
                "type": "knowledge_graph_sync",
                "nodes": list(kg_nodes.values()),
                "edges": kg_edges
            })
        except Exception as e:
            logger.warning(f"Failed to fetch Neo4j graph for UI: {e}")

        # Layer 2: OASIS Behavioral Analysis (Social Simulation)
        self._emit_progress(2, "Behavioral Analysis (OASIS)", "running")
        
        # Generate product-user personas grounded in customer interview data
        # FIX (Major): Persona generator uses world_bank (WorldDataBank/Qdrant).
        # Hindsight is agent-memory-only; persona profiles are pipeline run data
        # that must land in the persona_profiles Qdrant collection so downstream
        # layers can retrieve them via world_bank.recall("personas", ...).
        oasis_gen = OASISUserPersonaGenerator(self._llm, world_bank)
        profiles = await oasis_gen.generate(
            company=company,
            num_agents=num_simulations or 10,
            feature=feature if proposal else None,
            raw_chunks=bundle.chunks,
        )
        logger.info("Generated %d OASIS user personas from customer data", len(profiles))
        
        # Log all created personas to ensure diversity is visible
        logger.info("=" * 60)
        logger.info("PERSONAS CREATED FOR SIMULATION")
        logger.info("=" * 60)
        persona_payload = []
        for i, p in enumerate(profiles):
            info = p.user_info_dict
            other = info.get("profile", {}).get("other_info", {})
            name = info.get('name', f'Agent-{p.agent_id}')
            role = other.get('role', info.get('description', 'Simulation User'))
            logger.info(f"[{i+1}/{len(profiles)}] Agent ID: {p.agent_id}")
            logger.info(f"  Name: {name}")
            logger.info(f"  Segment: {role}")
            logger.info(f"  Description: {info.get('description', '')[:100]}...")
            logger.info(f"  Influence: {p.influence_strength:.2f} | Receptiveness: {p.receptiveness:.2f}")
            logger.info("-" * 40)
            # ── Safely extract buyer_journey: stored as dict in other_info ──────
            bj_raw = other.get("buyer_journey")
            if isinstance(bj_raw, dict):
                buyer_journey_stage = bj_raw.get("awareness_channel", "")
                buyer_journey_detail = bj_raw
            else:
                buyer_journey_stage = bj_raw or ""
                buyer_journey_detail = None

            persona_payload.append({
                "id": f"per_{p.agent_id}",
                "name": name,
                "role": role,
                "traits": other.get("traits", []) if isinstance(other.get("traits"), list) else [],
                "impact": round(p.influence_strength * 100),
                "bio": info.get("profile", {}).get("user_profile", "") or info.get("description", ""),
                # ── Psychological Profile ─────────────────────────────────────
                "mbti": other.get("mbti", ""),
                "mbti_description": other.get("mbti_description", ""),
                "key_traits": other.get("traits", []) if isinstance(other.get("traits"), list) else [],
                "emotional_triggers": other.get("emotional_triggers", {}),
                "communication_style": other.get("communication_style", {}),
                "decision_pattern": other.get("decision_pattern", {}),
                "predicted_stance": other.get("predicted_stance", {}),
                "questions_they_will_ask": other.get("questions_they_will_ask", []),
                # ── FinalPersona metadata ─────────────────────────────────────
                "domain_expertise": other.get("domain_expertise", []),
                "profile_confidence": other.get("profile_confidence", 0.0),
                "grounding_quality": other.get("grounding_quality", 1.0),
                "persona_type": other.get("persona_type", "INTERNAL"),
                "network_position_hint": other.get("network_position_hint", "peripheral"),
                "influence_strength": p.influence_strength,
                "receptiveness": p.receptiveness,
                "evidence_sources": other.get("evidence_sources", []),
                # ── Buyer Journey (external personas) ─────────────────────────
                "buyer_journey": buyer_journey_stage,        # string for stage indicator
                "buyer_journey_detail": buyer_journey_detail,  # full dict for detail panel
                # ── Market Context (external personas) ───────────────────────
                "market_context": other.get("market_context", None),
            })

        # ── Emit Layer 3 personas (real LLM-generated data) ────────────────────
        self._write_jsonl_event({"type": "persona_sync", "personas": persona_payload})
        
        sim_config = OASISSimulationConfig(
            simulation_name=session_id,
            num_agents=len(profiles),
            num_timesteps=10,
        )
        # FIX (Critical): Pass world_bank as the session for RunOASISSimulation.
        # The simulation retains its output (agent traces, comments, prediction
        # report summary) via session.retain("simulation", ...). This MUST write
        # to WorldDataBank's simulation_data Qdrant collection so that:
        #   - FeatureDiscoveryEngine (session=world_bank) can recall() it in Layer 3
        #   - AG2DebateEngine (world_bank=world_bank) can query_simulation() in Layer 5
        # The per-agent turn memory (HindsightOASISManager) remains internal to the
        # simulation engine — it reads HINDSIGHT_URL from env and is NOT affected here.
        behavioral_results = await RunOASISSimulation(
            config=sim_config,
            agent_profiles=profiles,
            feature=feature if proposal else None,
            context=company,
            mode="behavioral",
            session=world_bank,   # FIXED: was self._session (Hindsight) — caused Data Orphanage
            llm_client=self._llm,
            interactive_cb=self._interactive_cb,
        )
        self._emit_progress(2, "Behavioral Analysis", "done", {
            "agents": len(profiles),
            "interactions": len(behavioral_results.agent_interactions),
        })

        # Layer 3: Feature Discovery Engine
        self._emit_progress(3, "Feature Discovery", "running")
        discovery = FeatureDiscoveryEngine(self._llm, session=world_bank)
        discovered_features = await discovery.process(
            company=company,
            behavioral_results=behavioral_results,
            existing_proposal=feature if proposal else None,
            raw_chunks=bundle.chunks
        )
        feature = discovered_features[0]  # Take top ranked feature
        self._emit_progress(3, "Feature Discovery", "done", {
            "selected_feature": feature.title
        })

        # Layer 4: Boardroom Personas
        self._emit_progress(4, "Boardroom Assembly", "running")
        
        graph = None
        if use_legacy_personas:
            # Opt-in: Full LLM-driven persona generation pipeline (Layer 3)
            logger.info("Using LEGACY Layer 3 PersonaGenerator (user opt-in)")
            from tsc.layers.layer2_graph import KnowledgeGraphBuilder
            from tsc.layers.layer3_personas import PersonaGenerator
            from tsc.memory.graph_store import GraphStore
            
            # Init legacy memory just for this run
            await world_bank.initialize_session(session_id)
            gs = GraphStore(world_bank)
            
            graph_builder = KnowledgeGraphBuilder(self._llm, gs)
            graph = await graph_builder.process(bundle)
            persona_gen = PersonaGenerator(self._llm, gs)
            personas = await persona_gen.process(feature, company, graph, bundle)
        else:
            # Default: Static boardroom personas adapted to company context
            personas = BoardroomPersonaFactory.create_boardroom(
                company=company, feature=feature
            )
        
        # Emit Layer 4 boardroom personas
        boardroom_payload = []
        for p in personas:
            boardroom_payload.append({
                "name": p.name,
                "role": p.role,
                "role_short": getattr(p, "role_short", "") or "",
                "traits": getattr(p, "domain_expertise", []) or [],
                "bio": p.psychological_profile.full_profile_text if (p.psychological_profile and hasattr(p.psychological_profile, "full_profile_text")) else ""
            })
        self._write_jsonl_event({"type": "boardroom_persona_sync", "personas": boardroom_payload})
        
        self._emit_progress(4, "Boardroom Assembly", "done", {
            "personas": len(personas),
            "mode": "legacy_llm" if use_legacy_personas else "static_boardroom",
        })

        # Layer 5: AG2 Stakeholder Debate
        self._emit_progress(5, "Stakeholder Debate", "running")
        debate = AG2DebateEngine(self._llm)
        consensus = await debate.process(
            feature=feature, 
            company=company, 
            personas=personas, 
            graph=graph, 
            simulation_results=behavioral_results,
            session=self._session,
            world_bank=world_bank,
            pipeline_jsonl=self._pipeline_jsonl
        )
        self._emit_progress(5, "Stakeholder Debate", "done", {
            "verdict": consensus.overall_verdict,
        })

        # Layer 6: Specification Generation
        self._emit_progress(6, "Specification Generation", "running")
        spec_gen = SpecGenerator(self._llm)
        spec = await spec_gen.process(
            feature, company, consensus
        )
        self._emit_progress(6, "Specification Generation", "done", {
            "tasks": len(spec.development_tasks),
        })

        # Layer 7: Handoff
        self._emit_progress(7, "Handoff & Monitoring", "running")
        handoff = HandoffGenerator(self._llm)
        recommendation = await handoff.process(
            feature, company, personas, consensus, spec, behavioral_results, t0
        )
        self._emit_progress(7, "Handoff & Monitoring", "done")

        total = time.time() - t0
        logger.info("=" * 60)
        logger.info("TSC EVALUATION COMPLETE")
        logger.info("Verdict: %s | Confidence: %.2f", recommendation.final_verdict, recommendation.overall_confidence)
        logger.info("Total time: %.1f minutes", total / 60)
        logger.info("Tokens used: %d", self._llm.get_usage().total_tokens)
        logger.info("=" * 60)

        # G5: Persist FinalRecommendation to disk so the port-8080 WS bridge
        # can tail it and broadcast a final_recommendation event to the frontend.
        # This is the only route for the full verdict/spec/monitoring plan to reach
        # the UI — the port-8000 WebSocket is not consumed by the active frontend hook.
        try:
            rec_path = run_dir / "final_recommendation.json"
            rec_path.write_text(
                json.dumps({
                    "feature_name": recommendation.feature_name,
                    "final_verdict": recommendation.final_verdict,
                    "overall_confidence": recommendation.overall_confidence,
                    "summary_for_leadership": recommendation.summary_for_leadership,
                    "top_risks": [r.model_dump() for r in recommendation.top_risks],
                    "next_steps": [s.model_dump() for s in recommendation.next_steps],
                    "stakeholder_approvals": [a.model_dump() for a in recommendation.stakeholder_approvals],
                    "total_time_minutes": round(total / 60, 2),
                }, default=str),
                encoding="utf-8",
            )
            # G10: Emit full recommendation (was 4-field stub — now all fields)
            self._write_jsonl_event({
                "type": "final_recommendation",
                "feature_name": recommendation.feature_name,
                "final_verdict": recommendation.final_verdict,
                "overall_confidence": recommendation.overall_confidence,
                "summary_for_leadership": recommendation.summary_for_leadership,
                "top_risks": [r.model_dump() for r in recommendation.top_risks],
                "next_steps": [s.model_dump() for s in recommendation.next_steps],
                "stakeholder_approvals": [a.model_dump() for a in recommendation.stakeholder_approvals],
                "total_time_minutes": round(total / 60, 2),
            })
            # G5: Emit full consensus result (boardroom vote breakdown, debate rounds, phase spec)
            self._write_jsonl_event({
                "type": "consensus_result",
                "feature_name": consensus.feature_name,
                "overall_verdict": consensus.overall_verdict,
                "approval_confidence": consensus.approval_confidence,
                "stakeholder_verdicts": consensus.stakeholder_verdicts,
                "approvals": [a.model_dump() for a in consensus.approvals],
                "debate_rounds_count": len(consensus.debate_rounds),
                "phase_1": consensus.phase_1.model_dump(),
                "phase_2_gate": consensus.phase_2_gate.model_dump() if consensus.phase_2_gate else None,
                "mitigations": consensus.mitigations,
                "next_steps": consensus.next_steps,
                "simulation_key_quotes": consensus.simulation_key_quotes,
                "behavioral_insights": consensus.behavioral_insights,
                "tension_shifts": consensus.tension_shifts,
            })
            logger.info("📋 FinalRecommendation + ConsensusResult persisted: %s", rec_path)

        except Exception as _exc:
            logger.warning("Failed to persist FinalRecommendation: %s", _exc)

        return recommendation

    def _build_document_list(
        self,
        interviews: Optional[str],
        support: Optional[str],
        analytics: Optional[str],
        context: Optional[str],
        proposal: Optional[str],
    ) -> list[InputDocument]:
        docs = []
        if interviews:
            docs.append(InputDocument(type=DocumentType.INTERVIEWS, file_path=interviews))
        if support:
            docs.append(InputDocument(type=DocumentType.SUPPORT_TICKETS, file_path=support))
        if analytics:
            docs.append(InputDocument(type=DocumentType.ANALYTICS, file_path=analytics))
        if context:
            docs.append(InputDocument(type=DocumentType.COMPANY_CONTEXT, file_path=context))
        if proposal:
            docs.append(InputDocument(type=DocumentType.FEATURE_PROPOSAL, file_path=proposal))
        if not docs:
            raise ValueError("At least one input document is required.")
        return docs

    def _emit_progress(
        self,
        layer: int,
        name: str,
        status: str,
        details: Optional[dict] = None,
    ) -> None:
        logger.info("Layer %d/%d: %s — %s", layer, 7, name, status)
        # Write structured event to pipeline.jsonl so the WebSocket server can stream it
        self._write_jsonl_event({
            "type": "pipeline_progress",
            "layer": layer,
            "name": name,
            "status": status,
            "details": details or {},
        })
        if self._on_progress:
            try:
                self._on_progress(layer, name, status, details or {})
            except Exception:
                pass

    def _write_jsonl_event(self, event: dict) -> None:
        """Append a JSON event line to the active pipeline.jsonl stream."""
        if self._pipeline_jsonl is None:
            return
        try:
            with self._pipeline_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:
            logger.warning("Failed to write pipeline event: %s", exc)

    @staticmethod
    def _build_ingestion_nodes(documents: list) -> list:
        """Convert InputDocument list into IngestionNode dicts for the UI."""
        from tsc.models.inputs import DocumentType
        type_meta = {
            DocumentType.INTERVIEWS:       ("input",   "Customer Interviews"),
            DocumentType.SUPPORT_TICKETS:  ("input",   "Support Tickets"),
            DocumentType.ANALYTICS:        ("input",   "Analytics Data"),
            DocumentType.COMPANY_CONTEXT:  ("input",   "Company Context"),
            DocumentType.FEATURE_PROPOSAL: ("input",   "Feature Proposal"),
        }
        nodes = []
        for doc in documents:
            meta = type_meta.get(doc.type, ("input", str(doc.type)))
            file_name = Path(doc.file_path).name if doc.file_path else str(doc.type)
            nodes.append({
                "id": f"ing_{doc.type.value}",
                "label": f"{meta[1]}: {file_name}",
                "type": meta[0],
                "status": "active",
            })
        # Add a semantic extractor process node
        nodes.append({"id": "proc_semantic", "label": "Semantic Extractor", "type": "process", "status": "active"})
        # Add a tension cluster output node placeholder
        nodes.append({"id": "out_tension", "label": "Tension Cluster Analysis", "type": "output", "status": "active"})
        return nodes


async def run_evaluation(
    interviews: Optional[str] = None,
    support: Optional[str] = None,
    analytics: Optional[str] = None,
    context: Optional[str] = None,
    proposal: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    output: Optional[str] = None,
) -> FinalRecommendation:
    """Convenience function to run a full evaluation."""
    from tsc.config import LLMProvider

    cfg = settings

    # Override provider/model if specified
    if provider:
        cfg.llm_provider = LLMProvider(provider)
    if model:
        cfg.llm_model = model

    pipeline = TSCPipeline(cfg=cfg)
    result = await pipeline.evaluate(
        interviews=interviews,
        support=support,
        analytics=analytics,
        context=context,
        proposal=proposal,
    )

    # Save to file if output specified
    if output:
        out_path = Path(output)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Recommendation saved to %s", out_path)

    return result
