"""Pipeline orchestrator: runs all layers of the Predictive Reality Engine pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

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

        # Memory: Universal Hindsight Session Backbone
        self._session = HindsightSessionManager()

        # Progress callback (for web UI)
        self._on_progress: Optional[Any] = None

    def set_progress_callback(self, callback: Any) -> None:
        """Set a callback(layer_num, layer_name, status, details) for progress."""
        self._on_progress = callback

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

        # Initialize Universal Memory Sessions
        await self._session.initialize(session_id)

        # Build document list
        documents = self._build_document_list(
            interviews, support, analytics, context, proposal
        )
        logger.info("Input: %d documents", len(documents))

        # Layer 1: Ingest (Extract customer data & initial context)
        self._emit_progress(1, "Contextual Ingest", "running")
        ingestor = ContextualIngestor(self._llm, session=self._session)
        bundle, feature, company = await ingestor.process(documents)

        self._emit_progress(1, "Contextual Ingest", "done", {
            "chunks": bundle.statistics.total_chunks,
        })

        # Layer 2: OASIS Behavioral Analysis (Social Simulation)
        self._emit_progress(2, "Behavioral Analysis (OASIS)", "running")
        
        # Generate product-user personas grounded in customer interview data
        oasis_gen = OASISUserPersonaGenerator(self._llm, self._session)
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
        for i, p in enumerate(profiles):
            info = p.user_info_dict
            other = info.get("profile", {}).get("other_info", {})
            logger.info(f"[{i+1}/{len(profiles)}] Agent ID: {p.agent_id}")
            logger.info(f"  Name: {info.get('name', 'Unknown')}")
            logger.info(f"  Segment: {other.get('role', info.get('description', 'Unknown'))}")
            logger.info(f"  Description: {info.get('description', '')[:100]}...")
            logger.info(f"  Influence: {p.influence_strength:.2f} | Receptiveness: {p.receptiveness:.2f}")
            logger.info("-" * 40)
        
        sim_config = OASISSimulationConfig(
            simulation_name=session_id,
            num_agents=len(profiles),
        )
        behavioral_results = await RunOASISSimulation(
            config=sim_config,
            agent_profiles=profiles,
            feature=feature if proposal else None,
            context=company,
            mode="behavioral",
            session=self._session,
        )
        self._emit_progress(2, "Behavioral Analysis", "done", {
            "agents": len(profiles),
            "interactions": len(behavioral_results.agent_interactions),
        })

        # Layer 3: Feature Discovery Engine
        self._emit_progress(3, "Feature Discovery", "running")
        discovery = FeatureDiscoveryEngine(self._llm, self._session)
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
            from tsc.memory.world_bank import WorldDataBank
            from tsc.memory.graph_store import GraphStore
            
            # Init legacy memory just for this run
            wb = WorldDataBank()
            await wb.initialize_session(session_id)
            gs = GraphStore(wb)
            
            graph_builder = KnowledgeGraphBuilder(self._llm, gs)
            graph = await graph_builder.process(bundle)
            persona_gen = PersonaGenerator(self._llm, gs)
            personas = await persona_gen.process(feature, company, graph, bundle)
        else:
            # Default: Static boardroom personas adapted to company context
            personas = BoardroomPersonaFactory.create_boardroom(
                company=company, feature=feature
            )
        
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
            session=self._session
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
        if self._on_progress:
            try:
                self._on_progress(layer, name, status, details or {})
            except Exception:
                pass


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
