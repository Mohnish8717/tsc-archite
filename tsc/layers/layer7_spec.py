"""Layer 7: Specification Generation.

Generates a detailed implementation spec with evidence citations.
"""

from __future__ import annotations

import logging
import time
import json
import re

from tsc.llm.base import LLMClient
from tsc.models.debate import ConsensusResult
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.spec import DevelopmentTask, FeatureSpecification

logger = logging.getLogger(__name__)

SPEC_SYSTEM_PROMPT = """You are a Staff Product Manager and Lead Engineer tasked with
writing a comprehensive, highly-structured Product Requirement Document (PRD).

You will receive:
1. The original Feature Proposal
2. The Company Context (tech stack, constraints)
3. The Boardroom Consensus (agreements, mitigations, simulation evidence)

Your output MUST be a highly detailed JSON object that follows this structure exactly:
{
  "title": "Clear, actionable title",
  "executive_summary": "1-paragraph summary of what we are building and why",
  "justification": "Why this is necessary, citing customer pain points and behavioral simulation evidence",
  "ui_changes": ["Detailed description of UI addition/modification 1", "Detailed description of UI addition/modification 2"],
  "data_model_changes": ["Detailed description of DB/Schema change 1"],
  "workflow_modifications": ["How the user workflow changes step-by-step"],
  "acceptance_criteria": ["Criteria 1", "Criteria 2"],
  "metrics": ["Success metric 1 (e.g. 15% increase in conversion)"],
  "tasks": [
    {
      "id": "TASK-1",
      "title": "Implement X",
      "description": "Technical details...",
      "domain": "frontend/backend/database/infrastructure",
      "dependencies": [],
      "effort_estimate": "small/medium/large"
    }
  ]
}

Ensure tasks cover frontend, backend, and testing. Ground your justification in the provided simulation evidence and boardroom consensus.
"""


class SpecGenerator:
    """Layer 7: Generate implementation specification."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def process(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        consensus: ConsensusResult,
    ) -> FeatureSpecification:
        """Generate complete specification."""
        t0 = time.time()
        logger.info("Layer 7: Generating specification for %s", feature.title)

        prompt = f"Feature: {feature.title}\nContext: {company.tech_stack}\nConsensus: {consensus.overall_summary}"

        spec_json_str = await self._llm.generate(
            system_prompt=SPEC_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=6000,
        )

        try:
            # Extract JSON from potential markdown wrapping
            match = re.search(r"\{.*\}", spec_json_str, re.DOTALL)
            result = json.loads(match.group(0) if match else spec_json_str)
        except Exception as e:
            logger.error("Failed to parse spec JSON: %s", e)
            raise

        tasks = [
            DevelopmentTask(
                task_id=t["id"],
                name=t["title"],
                effort_days=3 if t["effort_estimate"] == "large" else 1,
                priority="P0",
            )
            for t in result.get("tasks", [])
        ]

        exec_summary = result.get("executive_summary", "")
        
        ui_changes = result.get("ui_changes", [])
        data_changes = result.get("data_model_changes", [])
        workflow = result.get("workflow_modifications", [])
        
        detailed_stories = []
        if ui_changes:
            detailed_stories.append("UI Changes:\n" + "\n".join(f"- {ui}" for ui in ui_changes))
        if data_changes:
            detailed_stories.append("Data Model Changes:\n" + "\n".join(f"- {dc}" for dc in data_changes))
        if workflow:
            detailed_stories.append("Workflow Modifications:\n" + "\n".join(f"- {wf}" for wf in workflow))
            
        justification = result.get("justification", "")
        if justification:
            exec_summary += f"\n\nJustification (Evidence-Based):\n{justification}"

        spec = FeatureSpecification(
            feature_name=result.get("title", feature.title),
            specification_markdown=exec_summary,
            development_tasks=tasks,
            evidence_citations={"consensus": consensus.overall_summary},
            total_effort_days=sum(t.effort_days for t in tasks),
            critical_path=[t.task_id for t in tasks if t.priority == "P0"],
        )

        logger.info(
            "Layer 7 complete: %d tasks (%.1fs)",
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
