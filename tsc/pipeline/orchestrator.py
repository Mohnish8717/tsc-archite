"""Pipeline orchestrator: runs all 8 layers sequentially."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from tsc.config import Settings, settings
from tsc.layers.layer1_ingestor import ContextualIngestor
from tsc.layers.layer2_graph import KnowledgeGraphBuilder
from tsc.layers.layer3_personas import PersonaGenerator
from tsc.layers.layer4_gates import GateExecutor
from tsc.layers.layer5_refinement import RefinementEngine
from tsc.layers.layer6_debate import DebateEngine
from tsc.layers.layer6_ag2_debate import AG2DebateEngine
from tsc.layers.layer7_spec import SpecGenerator
from tsc.layers.layer8_handoff import HandoffGenerator
from tsc.llm.base import LLMClient
from tsc.llm.factory import create_llm_client
from tsc.memory.fact_retriever import FactRetriever
from tsc.memory.graph_store import GraphStore
# Zep removed entirely; orchestrator will use Hindsight globally
from tsc.models.inputs import DocumentType, InputDocument
from tsc.models.recommendation import FinalRecommendation

logger = logging.getLogger(__name__)


class TSCPipeline:
    """Orchestrates the full 8-layer TSC evaluation pipeline."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        cfg: Optional[Settings] = None,
    ):
        self._cfg = cfg or settings
        self._llm = llm_client or create_llm_client(settings=self._cfg)

        # Memory: Transitioning to Universal Hindsight World Data Bank
        from tsc.memory.world_bank import WorldDataBank
        self._world_bank = WorldDataBank()
        # GraphStore and FactRetriever will be refactored to use WorldDataBank
        self._graph_store = GraphStore(self._world_bank)
        self._fact_retriever = FactRetriever(self._world_bank)

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
    ) -> FinalRecommendation:
        """Run the full pipeline.

        Args:
            interviews: Path to customer interviews file.
            support: Path to support tickets file.
            analytics: Path to analytics data file.
            context: Path to company context JSON.
            proposal: Path to feature proposal JSON.
            num_simulations: Optional override for Monte Carlo simulation count.

        Returns:
            FinalRecommendation with verdict, spec, and monitoring plan.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("TSC v2.0 PIPELINE — STARTING EVALUATION")
        logger.info("LLM: %s (%s)", self._llm.__class__.__name__, self._llm.model)
        if num_simulations:
            logger.info("Simulation Count Override: %d", num_simulations)
        logger.info("=" * 60)

        # Initialize Universal Memory
        await self._world_bank.initialize_session("tsc-world")

        # Build document list
        documents = self._build_document_list(
            interviews, support, analytics, context, proposal
        )
        logger.info("Input: %d documents", len(documents))

        # Layer 1: Ingest
        self._emit_progress(1, "Contextual Ingestor", "running")
        ingestor = ContextualIngestor(self._llm)
        bundle, feature, company = await ingestor.process(documents)
        self._emit_progress(1, "Contextual Ingestor", "done", {
            "chunks": bundle.statistics.total_chunks,
            "entities": bundle.statistics.unique_entities,
        })

        # Layer 2: Knowledge Graph
        self._emit_progress(2, "Knowledge Graph Builder", "running")
        graph_builder = KnowledgeGraphBuilder(self._llm, self._graph_store)
        graph = await graph_builder.process(bundle)
        self._emit_progress(2, "Knowledge Graph Builder", "done", {
            "nodes": graph.metadata.total_nodes,
            "edges": graph.metadata.total_edges,
        })

        # Layer 3: Personas
        self._emit_progress(3, "Persona Generation", "running")
        persona_gen = PersonaGenerator(self._llm, self._graph_store)
        personas = await persona_gen.process(feature, company, graph, bundle)
        self._emit_progress(3, "Persona Generation", "done", {
            "personas": len(personas),
        })

        # Layer 4: Gates
        self._emit_progress(4, "Gate Evaluation", "running")
        gate_executor = GateExecutor(self._llm)
        gates_summary = await gate_executor.process(
            feature, company, graph, bundle, personas, num_simulations
        )
        self._emit_progress(4, "Gate Evaluation", "done", {
            "passed": len(gates_summary.results) - len(gates_summary.failed_gates),
            "total": len(gates_summary.results),
            "score": gates_summary.overall_score,
        })

        # Layer 5: Refinement
        self._emit_progress(5, "Iterative Refinement", "running")
        refinement = RefinementEngine(self._llm, self._cfg.gate_fail_threshold)
        gates_summary = await refinement.process(
            gates_summary, feature, company, graph, bundle, personas
        )
        self._emit_progress(5, "Iterative Refinement", "done")

        # Layer 6: Debate
        self._emit_progress(6, "Stakeholder Debate", "running")
        
        import os
        if os.getenv("DEBATE_ENGINE_TYPE", "ag2").lower() == "ag2":
            logger.info("Using deep-thinking AG2 Debate Engine")
            debate = AG2DebateEngine(self._llm)
        else:
            logger.info("Using legacy Debate Engine")
            debate = DebateEngine(self._llm)
            
        consensus = await debate.process(
            feature, company, graph, personas, gates_summary
        )
        self._emit_progress(6, "Stakeholder Debate", "done", {
            "verdict": consensus.overall_verdict,
            "confidence": consensus.approval_confidence,
        })

        # Layer 7: Specification
        self._emit_progress(7, "Specification Generation", "running")
        spec_gen = SpecGenerator(self._llm)
        spec = await spec_gen.process(
            feature, company, graph, personas, gates_summary, consensus
        )
        self._emit_progress(7, "Specification Generation", "done", {
            "word_count": len(spec.specification_markdown.split()),
        })

        # Layer 8: Handoff
        self._emit_progress(8, "Handoff & Monitoring", "running")
        handoff = HandoffGenerator(self._llm)
        recommendation = await handoff.process(
            feature, company, personas, gates_summary, consensus, spec, t0
        )
        self._emit_progress(8, "Handoff & Monitoring", "done")

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
        logger.info("Layer %d/%d: %s — %s", layer, 8, name, status)
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
