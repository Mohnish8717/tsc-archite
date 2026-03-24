import os
import json
import logging
import time
import multiprocessing

# Fix gRPC deadlocks when forking on macOS/Linux
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force spawn method for multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
from typing import List, Dict, Any, Optional
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona
from tsc.oasis.models import OASISAgentProfile, OASISSimulationConfig, SimulationMetadata, SimulationStatus
from tsc.oasis.simulation_runner import SimulationRunner
from tsc.oasis.simulation_config_generator import SimulationConfigGenerator
from tsc.oasis.profile_builder import InitializeOASISAgents

logger = logging.getLogger(__name__)

class SimulationManager:
    """
    High-level orchestrator for OASIS simulations in TSC.
    Bridges Layer 3 personas with the hardened simulation runner.
    """
    
    def __init__(self, workspace_root: str = "/tmp/oasis_runs", llm_client: Any = None):
        self.workspace_root = workspace_root
        self.config_gen = SimulationConfigGenerator(llm_client) if llm_client else None

    async def prepare_simulation(
        self,
        simulation_id: str,
        feature: FeatureProposal,
        company: CompanyContext,
        personas: List[FinalPersona],
        market_context: Optional[Dict[str, Any]] = None
    ) -> SimulationMetadata:
        """
        Prepare an isolated workspace with personas and tuned configuration.
        """
        workspace_path = os.path.join(self.workspace_root, simulation_id)
        os.makedirs(workspace_path, exist_ok=True)
        
        logger.info(f"Preparing simulation workspace: {workspace_path}")
        
        # 1. Generate Intelligent Config
        if self.config_gen:
            logger.info("Generating LLM-tuned simulation parameters")
            config = await self.config_gen.generate_config(
                feature=feature,
                company=company,
                target_audience=[feature.target_users] # Pass as list
            )
        else:
            # Fallback to default
            config = OASISSimulationConfig(
                simulation_name=simulation_id,
                platform_type="reddit",
                num_agents=min(len(personas) * 10, 200),
                num_timesteps=24
            )

        # 2. Map Personas to OASIS Agents
        logger.info(f"Mapping {len(personas)} personas to OASIS agents")
        agent_profiles, _ = await InitializeOASISAgents(
            personas=personas,
            feature=feature,
            context=company,
            config=config,
            market_context=market_context
        )
        
        # 3. Save Preparation Artifacts
        metadata = SimulationMetadata(
            simulation_id=simulation_id,
            project_id=str(getattr(feature, 'id', 'unknown')),
            graph_id=str(getattr(company, 'id', 'unknown')),
            entities_count=len(personas),
            profiles_count=len(agent_profiles),
            entity_types=list(set([p.persona_type for p in personas])),
            config_reasoning="LLM Tuned" if self.config_gen else "Default"
        )
        
        with open(os.path.join(workspace_path, "config.json"), "w") as f:
            f.write(config.model_dump_json(indent=2))
            
        with open(os.path.join(workspace_path, "profiles.json"), "w") as f:
            profiles_list = [p.model_dump() for p in agent_profiles]
            json.dump(profiles_list, f, indent=2, default=str)
            
        with open(os.path.join(workspace_path, "metadata.json"), "w") as f:
            f.write(metadata.model_dump_json(indent=2))
            
        with open(os.path.join(workspace_path, "feature.json"), "w") as f:
            f.write(feature.model_dump_json(indent=2))
            
        with open(os.path.join(workspace_path, "company.json"), "w") as f:
            f.write(company.model_dump_json(indent=2))
            
        logger.info(f"Preparation complete for {simulation_id}. {len(agent_profiles)} agents ready.")
        return metadata

    def start_simulation(self, simulation_id: str) -> SimulationRunner:
        """
        Spawn the hardened SimulationRunner for a prepared workspace.
        """
        workspace_path = os.path.join(self.workspace_root, simulation_id)
        if not os.path.exists(os.path.join(workspace_path, "config.json")):
            raise FileNotFoundError(f"Simulation {simulation_id} is not prepared. Call prepare_simulation first.")
            
        runner = SimulationRunner(simulation_id)
        
        # Load all prepared artifacts
        with open(os.path.join(workspace_path, "config.json"), "r") as f:
            config_dict = json.load(f)
            
        with open(os.path.join(workspace_path, "profiles.json"), "r") as f:
            profiles_dict = json.load(f)
            agent_profiles = [OASISAgentProfile(**p) for p in profiles_dict]

        from tsc.models.inputs import FeatureProposal, CompanyContext
        with open(os.path.join(workspace_path, "feature.json"), "r") as f:
            feature = FeatureProposal.model_validate_json(f.read())
            
        with open(os.path.join(workspace_path, "company.json"), "r") as f:
            company = CompanyContext.model_validate_json(f.read())

        logger.info(f"Starting execution for {simulation_id}")
        
        # Reconstruct the config object for the runner (it expects OASISSimulationConfig)
        from .models import OASISSimulationConfig
        sim_config = OASISSimulationConfig(**config_dict)
        
        runner.start_simulation(
            config=sim_config,
            agent_profiles=agent_profiles,
            feature=feature,
            context=company
        )
        return runner
