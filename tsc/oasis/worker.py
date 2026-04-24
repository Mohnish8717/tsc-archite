# ── DEADLOCK PREVENTION: Must be set before ANY other imports ─────────────────
import os
import sys

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "false")
os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# ─────────────────────────────────────────────────────────────────────────────

import logging
import asyncio
import nest_asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("tsc.oasis.worker")


async def main():
    import json

    # BLOCK TENSORFLOW: It triggers Abseil/gRPC deadlocks on macOS forks
    sys.modules["tensorflow"] = None

    # Deferred imports to avoid gRPC deadlocks
    from tsc.oasis.models import OASISSimulationConfig, OASISAgentProfile
    from tsc.models.inputs import FeatureProposal, CompanyContext
    from tsc.oasis.simulation_engine import RunOASISSimulation

    if len(sys.argv) < 2:
        logger.error("Usage: python worker.py <payload_json_path>")
        sys.exit(1)

    payload_path = sys.argv[1]
    if not os.path.exists(payload_path):
        logger.error(f"Payload file not found: {payload_path}")
        sys.exit(1)

    try:
        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        logger.info(f"Reconstructing simulation state from {payload_path}")

        config = OASISSimulationConfig(**payload["config"])
        agent_profiles = [OASISAgentProfile(**p) for p in payload["agent_profiles"]]
        feature = FeatureProposal(**payload["feature"])
        context = CompanyContext(**payload["context"])
        market_context = payload.get("market_context")
        base_dir = payload.get("base_dir", "/tmp/oasis_runs")

        series = await RunOASISSimulation(
            config=config,
            agent_profiles=agent_profiles,
            feature=feature,
            context=context,
            market_context=market_context,
            base_dir=base_dir,
        )

        result_path = payload_path.replace(".json", "_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(series.model_dump_json())

        logger.info(f"Simulation {config.simulation_name} COMPLETED → {result_path}")

    except Exception as e:
        logger.exception(f"Fatal error in OASIS Worker: {e}")
        sys.exit(1)


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
