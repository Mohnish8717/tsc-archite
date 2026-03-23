from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional, list

from tsc.repositories.feature_repository import FeatureRepository
from tsc.repositories.persona_repository import PersonaRepository
from tsc.repositories.prediction_repository import PredictionRepository
from tsc.repositories.simulation_repository import SimulationRepository
from tsc.layers.layer1_ingestor import Ingestor
from tsc.layers.layer3_personas import PersonaGenerator
from tsc.layers.layer4_gates import GateExecutor
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona
from tsc.models.gates import GatesSummary

logger = logging.getLogger(__name__)

class SimulationEngine:
    """Orchestrates the full simulation pipeline with persistent storage."""

    def __init__(
        self,
        persona_gen: PersonaGenerator,
        gate_executor: GateExecutor,
        feature_repo: FeatureRepository,
        persona_repo: PersonaRepository,
        simulation_repo: SimulationRepository,
        prediction_repo: PredictionRepository,
        ingestor: Optional[Ingestor] = None,
    ):
        self._persona_gen = persona_gen
        self._gate_executor = gate_executor
        self._feature_repo = feature_repo
        self._persona_repo = persona_repo
        self._simulation_repo = simulation_repo
        self._prediction_repo = prediction_repo
        self._ingestor = ingestor

    async def run_simulation(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: Any,
        bundle: Any,
        num_simulations: Optional[int] = None,
        use_cached_personas: bool = True,
    ) -> uuid.UUID:
        """Execute a full simulation run and persist results."""
        t0 = time.time()
        logger.info("Starting simulation for feature: %s", feature.title)

        # 1. Ensure Feature is in DB
        if not hasattr(feature, 'id') or not feature.id:
            # Save or get existing
            feature_db = await self._feature_repo.get_feature_by_title(feature.title, company.company_id)
            if not feature_db:
                # We need to map FeatureProposal to dict or use feature_db model
                # For now, let's assume we can save it
                feature_id = await self._feature_repo.save_feature({
                    "company_id": company.company_id,
                    "title": feature.title,
                    "description": feature.description,
                    "target_market": getattr(company, 'market_segment', 'General'),
                    "metadata": feature.model_dump() if hasattr(feature, 'model_dump') else {}
                })
                feature.id = feature_id
            else:
                feature.id = feature_db.id

        # 2. Get Personas (Layer 3)
        personas: list[FinalPersona] = []
        if use_cached_personas and company.company_id:
            personas = await self._persona_gen.get_internal_personas_for_company(company.company_id)
            # Filter for external if gate needs them
            external_personas = [p for p in personas if getattr(p, 'persona_type', 'INTERNAL') == 'EXTERNAL']
            if external_personas:
                personas = external_personas
                logger.info("Retrieved %d cached external personas", len(personas))
        
        if not personas:
            personas = await self._persona_gen.process(feature, company, graph, bundle)
            logger.info("Generated %d new personas", len(personas))
            # Filter for external
            personas = [p for p in personas if getattr(p, 'persona_type', 'INTERNAL') == 'EXTERNAL']

        # 3. Execute Gates (Layer 4)
        summary: GatesSummary = await self._gate_executor.process(
            feature, company, graph, bundle, personas, num_simulations=num_simulations
        )
        logger.info("Gates execution complete. Overall score: %.2f", summary.overall_score)

        # 4. Persist Simulation Run
        run_data = {
            "feature_id": feature.id,
            "company_id": company.company_id,
            "approval_rate": summary.overall_score / 10.0,
            "sentiment_score": summary.overall_score / 10.0,
            "risk_assessment": summary.get_overall_risk() if hasattr(summary, 'get_overall_risk') else 0.5,
            "recommendations": summary.recommendation,
            "metadata_snapshot": {
                "num_simulations": num_simulations,
                "persona_count": len(personas),
                "duration_seconds": time.time() - t0,
                "gate_results": [r.model_dump() if hasattr(r, 'model_dump') else str(r) for r in summary.results]
            }
        }
        
        run_id = await self._simulation_repo.save_simulation_run(run_data)
        logger.info("Simulation run persisted with ID: %s", run_id)

        # 5. Persist Predictions (Segment Breakdown from Monte Carlo)
        # Find Monte Carlo gate result
        mc_result = next((r for r in summary.results if r.gate_id == "4.5"), None)
        if mc_result and mc_result.details:
            segment_breakdown = mc_result.details.get("segment_breakdown", {})
            
            # Map segment name (persona.role) back to persona_id
            persona_map = {p.role: p.id for p in personas if hasattr(p, 'id') and p.id}
            
            for role, adoption_rate in segment_breakdown.items():
                persona_id = persona_map.get(role)
                if persona_id:
                    prediction_data = {
                        "feature_id": feature.id,
                        "persona_id": persona_id,
                        "simulation_run_id": run_id,
                        "predicted_adoption_rate": float(adoption_rate),
                        "sentiment_score": float(adoption_rate), # Mocked
                        "qualitative_feedback": f"Detailed simulation for {role} role complete.",
                        "confidence_score": mc_result.score / 10.0,
                        "metadata": {"role": role}
                    }
                    await self._prediction_repo.save_prediction(prediction_data)
        
        return run_id

    async def get_run_history(self, company_id: uuid.UUID) -> list[Any]:
        """Retrieve history of simulation runs for a company."""
        return await self._simulation_repo.get_runs_by_company(company_id)
