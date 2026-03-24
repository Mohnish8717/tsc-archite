import logging
import asyncio

async def main():
    import os
    import sys
    import json
    
    # BLOCK TENSORFLOW: It triggers Abseil/gRPC deadlocks on macOS forks
    # even if not explicitly used by our logic (imported by deep dependencies)
    sys.modules["tensorflow"] = None
    
    # Deferred imports to avoid gRPC deadlocks
    from tsc.oasis.models import OASISSimulationConfig, OASISAgentProfile
    from tsc.models.inputs import FeatureProposal, CompanyContext
    from tsc.oasis.simulation_engine import RunOASISSimulation

    logger = logging.getLogger("tsc.oasis.worker")
    
    if len(sys.argv) < 2:
        logger.error("Usage: python worker.py <payload_json_path>")
        sys.exit(1)
        
    payload_path = sys.argv[1]
    if not os.path.exists(payload_path):
        logger.error(f"Payload file not found: {payload_path}")
        sys.exit(1)
        
    try:
        with open(payload_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            
        logger.info(f"Reconstructing simulation state from {payload_path}")
        
        # Reconstruct objects from JSON payload
        config = OASISSimulationConfig(**payload['config'])
        agent_profiles = [OASISAgentProfile(**p) for p in payload['agent_profiles']]
        feature = FeatureProposal(**payload['feature'])
        context = CompanyContext(**payload['context'])
        market_context = payload.get('market_context')
        base_dir = payload.get('base_dir', '/tmp/oasis_runs')
        
        # Run the actual simulation
        series = await RunOASISSimulation(
            config=config,
            agent_profiles=agent_profiles,
            feature=feature,
            context=context,
            market_context=market_context,
            base_dir=base_dir,
            zep_client=None
        )
        
        # Finalize and save result
        result_path = payload_path.replace(".json", "_result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(series.model_dump_json())
            
        logger.info(f"Simulation {config.simulation_name} COMPLETED. Results at {result_path}")
        
    except Exception as e:
        logger.exception(f"Fatal error in OASIS Worker: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
