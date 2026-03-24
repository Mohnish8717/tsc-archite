import logging
from typing import List, Dict, Any, Optional
from tsc.llm.base import LLMClient
from tsc.oasis.models import OASISSimulationConfig, SimulationParameters
from tsc.models.inputs import FeatureProposal, CompanyContext

logger = logging.getLogger(__name__)

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
            system_prompt="You are an OASIS Simulation Architect. Your goal is to tune simulation parameters for maximum signal and minimal noise.",
            user_prompt=prompt,
            response_model=SimulationParameters # Use the and newly added model
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
        return f"""
        Generate simulation parameters for the following feature proposal:
        
        ### Feature: {feature.title}
        Description: {feature.description}
        Target Audience: {', '.join(target_audience)}
        
        ### Company Context:
        Name: {company.company_name}
        Priorities: {', '.join(company.current_priorities) if hasattr(company, 'current_priorities') else 'Standard growth'}
        
        Task:
        1. Decide on the best platform (twitter vs reddit) based on the audience.
        2. Decide on the number of agents (100-500) and timesteps (12-48) for significant interaction.
        3. Generate 3-5 high-signal interview questions to ask agents during the simulation.
        4. Provide reasoning for your choices.
        """
