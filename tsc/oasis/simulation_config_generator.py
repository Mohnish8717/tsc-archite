import logging
from typing import List, Dict, Any, Optional
from tsc.llm.base import LLMClient
from tsc.oasis.models import OASISSimulationConfig, SimulationParameters
from tsc.models.inputs import FeatureProposal, CompanyContext

logger = logging.getLogger(__name__)

_CONFIG_GEN_SYSTEM = """\
<identity>
You are an OASIS Simulation Architect optimising market research signal quality
within a defined token and time budget. Your decisions must be justified.
</identity>

<platform_selection>
Reddit-type: audience discusses products in threads, compares alternatives, debates
feature trade-offs. Best for B2B SaaS, dev tools, enterprise products.
Twitter-type: audience reacts to announcements, shares short opinions, amplifies
influencer signals. Best for consumer apps, brand sentiment, launch reactions.
</platform_selection>

<agent_count_calibration>
- 50–100 agents: feature concept testing, single-segment probing
- 100–300 agents: multi-segment interaction, opinion contagion dynamics
- 300–500 agents: population-scale emergence, minority opinion amplification
Do not exceed 500 — use PopulationSampler extrapolation for larger declared populations.
</agent_count_calibration>

<timestep_calibration>
- 5–8 timesteps : initial reaction, first-impression capture
- 10–15 timesteps: opinion formation and peer influence
- 20–30 timesteps: belief consolidation, network effects, emergent polarization
Each timestep ≈ 1 simulated hour. Match to a realistic discussion timeline for the product.
</timestep_calibration>

<interview_probe_rules>
You MUST generate exactly 1 consolidated behavioural interview probe that asks the 3 most important questions concisely.
It MUST:
1. Be BEHAVIOURAL — anchored to a specific past or future action, not abstract opinion.
2. Have a MEASURABLE extraction target (a number, a competitor name, a timeline, or a quote).
3. Combine a WTP probe, a risk surfacing probe, and an adoption-ladder probe into a single string.
</interview_probe_rules>

<output_format>
Return JSON only — no prose:
{
  "platform_type": "reddit|twitter",
  "platform_reasoning": "one sentence",
  "num_agents": 150,
  "agent_reasoning": "one sentence",
  "num_timesteps": 12,
  "timestep_reasoning": "one sentence",
  "interview_prompts": ["single consolidated probe"],
  "estimated_signal_quality": "low|medium|high",
  "known_limitations": ["limitation 1"]
}
</output_format>
"""

class SimulationConfigGenerator:
    """
    Uses LLM to intelligently generate OASIS simulation parameters 
    based on the feature proposal and company context.
    """
    
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def generate_config(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        target_audience: List[str]
    ) -> OASISSimulationConfig:
        """
        Generate a hardened OASIS config with LLM-tuned parameters.
        """
        prompt = self._build_prompt(feature, company, target_audience)
        
        # LLM reasoning for parameters
        response = await self._llm.generate(
            system_prompt=_CONFIG_GEN_SYSTEM,
            user_prompt=prompt,
            response_model=SimulationParameters
        )
        
        logger.info(f"LLM generated simulation config: {response.generation_reasoning}")
        
        return OASISSimulationConfig(
            simulation_name=f"oasis_{feature.title.lower().replace(' ', '_')}",
            platform_type=response.platform_type,
            num_agents=response.num_agents,
            num_timesteps=response.num_timesteps,
            interview_prompts=response.interview_prompts
        )

    def _build_prompt(self, feature: FeatureProposal, company: CompanyContext, target_audience: List[str]) -> str:
        audience_str = ', '.join(target_audience) if target_audience else 'general users'
        priorities_str = ', '.join(company.current_priorities) if hasattr(company, 'current_priorities') and company.current_priorities else 'standard growth'
        return (
            f"<feature>\n"
            f"Title: {feature.title}\n"
            f"Description: {feature.description[:500]}\n"
            f"Target Audience: {audience_str}\n"
            f"</feature>\n\n"
            f"<company>\n"
            f"Name: {company.company_name}\n"
            f"Priorities: {priorities_str}\n"
            f"</company>\n\n"
            "Follow the decision framework in your system prompt to select platform, "
            "agent count, timesteps, and generate 1 consolidated behavioural interview probe. "
            "Include your reasoning and known limitations."
        )
