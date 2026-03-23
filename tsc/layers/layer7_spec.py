"""Layer 7: Specification Generation.

Generates a detailed implementation spec with evidence citations.
"""

from __future__ import annotations

import logging
import time

from tsc.llm.base import LLMClient
from tsc.llm.prompts import SPEC_SYSTEM, SPEC_USER
from tsc.models.debate import ConsensusResult
from tsc.models.gates import GatesSummary
from tsc.models.graph import KnowledgeGraph
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona
from tsc.models.spec import DevelopmentTask, FeatureSpecification

logger = logging.getLogger(__name__)


class SpecGenerator:
    """Layer 7: Generate implementation specification."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def process(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        personas: list[FinalPersona],
        gates_summary: GatesSummary,
        consensus: ConsensusResult,
    ) -> FeatureSpecification:
        """Generate complete specification."""
        t0 = time.time()
        logger.info("Layer 7: Generating specification for %s", feature.title)

        prompt = SPEC_USER.render(
            feature=feature,
            company=company,
            consensus=consensus,
            gate_results=[
                {"gate_name": g.gate_name, "verdict": g.verdict.value, "score": g.score}
                for g in gates_summary.results
            ],
        )

        spec_markdown = await self._llm.generate(
            system_prompt=SPEC_SYSTEM,
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=6000,
        )

        # Extract dev tasks from the specification
        tasks = self._extract_tasks(spec_markdown, feature.title)

        # Build evidence citations
        citations = {
            g.gate_name: f"{g.verdict.value} (score: {g.score})"
            for g in gates_summary.results
        }

        spec = FeatureSpecification(
            feature_name=feature.title,
            specification_markdown=spec_markdown,
            development_tasks=tasks,
            evidence_citations=citations,
            total_effort_days=sum(t.effort_days for t in tasks),
            critical_path=[t.task_id for t in tasks if t.priority == "P0"],
        )

        logger.info(
            "Layer 7 complete: %d-word spec, %d tasks (%.1fs)",
            len(spec_markdown.split()),
            len(tasks),
            time.time() - t0,
        )
        return spec

    def _extract_tasks(
        self, spec_text: str, feature_name: str
    ) -> list[DevelopmentTask]:
        """Extract development tasks from the spec markdown."""
        # Default tasks if extraction fails
        prefix = feature_name.upper().replace(" ", "-")[:10]
        default_tasks = [
            DevelopmentTask(
                task_id=f"{prefix}-001",
                name="Core implementation",
                effort_days=3,
                priority="P0",
            ),
            DevelopmentTask(
                task_id=f"{prefix}-002",
                name="Integration and API",
                effort_days=2,
                priority="P0",
                dependency=f"{prefix}-001",
            ),
            DevelopmentTask(
                task_id=f"{prefix}-003",
                name="UI / UX implementation",
                effort_days=2,
                priority="P1",
                dependency=f"{prefix}-001",
            ),
            DevelopmentTask(
                task_id=f"{prefix}-004",
                name="Testing and QA",
                effort_days=2,
                priority="P1",
                dependency=f"{prefix}-002",
            ),
            DevelopmentTask(
                task_id=f"{prefix}-005",
                name="Documentation",
                effort_days=1,
                priority="P1",
                dependency=f"{prefix}-003",
            ),
            DevelopmentTask(
                task_id=f"{prefix}-006",
                name="Security audit",
                effort_days=0.5,
                priority="P0",
            ),
            DevelopmentTask(
                task_id=f"{prefix}-007",
                name="Launch coordination",
                effort_days=0.5,
                priority="P0",
                dependency=f"{prefix}-004",
            ),
        ]

        # Try to parse tasks from the spec (look for table rows)
        import re

        table_rows = re.findall(
            r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|",
            spec_text,
        )

        parsed_tasks = []
        for i, row in enumerate(table_rows):
            name = row[0].strip()
            if name.lower() in ("task", "name", "---", ""):
                continue
            effort_match = re.search(r"(\d+(?:\.\d+)?)", row[2] if len(row) > 2 else "1")
            effort = float(effort_match.group(1)) if effort_match else 1.0

            parsed_tasks.append(
                DevelopmentTask(
                    task_id=f"{prefix}-{i:03d}",
                    name=name[:100],
                    owner=row[1].strip() if len(row) > 1 else "",
                    effort_days=effort,
                    priority="P0" if i < 3 else "P1",
                )
            )

        return parsed_tasks if len(parsed_tasks) >= 3 else default_tasks
