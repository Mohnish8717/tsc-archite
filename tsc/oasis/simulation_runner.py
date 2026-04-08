import os
import sys
import subprocess
import json
import time
import signal
import threading
import atexit
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from .models import (
    OASISSimulationConfig, 
    OASISAgentProfile, 
    MarketSentimentSeries,
    SimulationRunState,
    RunnerStatus,
    AgentAction
)
from tsc.models.inputs import FeatureProposal, CompanyContext

logger = logging.getLogger("tsc.oasis.runner")

# Platform Detection for specialized cleanup
IS_WINDOWS = sys.platform == "win32"

class SimulationRunner:
    """
    Main Process Manager for the OASIS Simulation.
    Responsible for Subprocess isolation, IPC control, robust cleanup, 
    and high-frequency observability via a dedicated monitor thread.
    """
    
    # Global cache to prevent multiple monitors/runners for the same ID in the same process
    _active_runners: Dict[str, 'SimulationRunner'] = {}

    def __init__(self, simulation_id: str, base_dir: str = "/tmp/oasis_runs"):
        self.simulation_id = simulation_id
        self.workspace = os.path.join(base_dir, simulation_id)
        os.makedirs(self.workspace, exist_ok=True)
        
        self.payload_file = os.path.join(self.workspace, "payload.json")
        self.command_file = os.path.join(self.workspace, "commands.json")
        self.result_file = os.path.join(self.workspace, "payload_result.json")
        self.state_file = os.path.join(self.workspace, "run_state.json")
        self.actions_file = os.path.join(self.workspace, "actions.jsonl")
        
        self.process: Optional[subprocess.Popen] = None
        self.run_state = self._load_run_state() or SimulationRunState(simulation_id=simulation_id)
        
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        
        SimulationRunner._active_runners[self.simulation_id] = self
        
    def _load_run_state(self) -> Optional[SimulationRunState]:
        """Load persistent status from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return SimulationRunState.model_validate_json(f.read())
            except Exception:
                pass
        return None

    def _save_run_state(self):
        """Persist current status to disk."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                f.write(self.run_state.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save run state: {e}")

    def start_simulation(
        self, 
        config: OASISSimulationConfig,
        agent_profiles: List[OASISAgentProfile],
        feature: FeatureProposal,
        context: CompanyContext,
        market_context: Optional[Dict[str, Any]] = None
    ) -> int:
        """Spawn the simulation subprocess and start monitoring."""
        if self.run_state.status in [RunnerStatus.RUNNING, RunnerStatus.STARTING]:
            logger.warning(f"Simulation {self.simulation_id} is already active.")
            return self.run_state.process_pid or -1

        # 0. Enforce identity consistency
        config.simulation_name = self.simulation_id

        # 1. Prepare Payload
        payload = {
            "config": config.model_dump(),
            "agent_profiles": [p.model_dump() for p in agent_profiles],
            "feature": feature.model_dump(),
            "context": context.model_dump(),
            "market_context": market_context,
            "base_dir": os.path.dirname(self.workspace)
        }
        
        with open(self.payload_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            
        # 2. Update State
        self.run_state.status = RunnerStatus.STARTING
        self.run_state.total_timesteps = config.num_timesteps
        self.run_state.platforms_active = [config.platform_type]
        self.run_state.started_at = datetime.utcnow()
        self._save_run_state()

        # 3. Spawn Subprocess
        worker_script = os.path.join(os.path.dirname(__file__), "worker.py")
        wrapper_script = os.path.join(os.path.dirname(__file__), "worker_wrapper.sh")
        log_file = os.path.join(self.workspace, "worker.log")
        
        # Open log file for output redirection
        log_handle = open(log_file, 'w', encoding='utf-8')
        
        # Build subprocess environment with macOS gRPC/torch deadlock prevention vars
        # These MUST be set before the subprocess starts — shell env is the only reliable way.
        subprocess_env = os.environ.copy()
        subprocess_env.update({
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            # CRITICAL: Prevents ObjC runtime crash when forking on macOS Sequoia/Sonoma
            "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
            # Prevents gRPC mutex deadlock inside asyncio + spawned subprocess
            "GRPC_ENABLE_FORK_SUPPORT": "false",
            "GRPC_POLL_STRATEGY": "poll",
            "GRPC_DNS_RESOLVER": "native",
            "AIOHTTP_CLIENT_TIMEOUT": "300",
            # Prevent torch from oversubscribing threads (worsens mutex contention)
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
        })

        # Use specialized subprocess flags for robust group management
        popen_args = {
            "args": [wrapper_script, worker_script, self.payload_file],
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": 'utf-8',
            "env": subprocess_env,
            "start_new_session": True if not IS_WINDOWS else False,
            "bufsize": 1  # Line buffered to prevent blocking
        }
        # Windows-specific process creation flags
        if IS_WINDOWS:
            # CREATE_NEW_PROCESS_GROUP = 0x00000200
            popen_args["creationflags"] = 0x00000200

        self.process = subprocess.Popen(**popen_args)

        
        self.run_state.process_pid = self.process.pid
        self.run_state.status = RunnerStatus.RUNNING
        self._save_run_state()
        
        # 4. Start Dashboard Monitoring Thread
        self._start_monitor()
        
        logger.info(f"OASIS Simulation {self.simulation_id} spawned with PID {self.process.pid}")
        return self.process.pid

    def _start_monitor(self):
        """Begin background tailing of action logs."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
            
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"OASISMonitor-{self.simulation_id}",
            daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self):
        """Tails the actions.jsonl file to update run_state in real-time."""
        pos = 0
        while not self._stop_monitor.is_set():
            if os.path.exists(self.actions_file):
                try:
                    with open(self.actions_file, 'r', encoding='utf-8') as f:
                        f.seek(pos)
                        for line in f:
                            if not line.strip(): continue
                            data = json.loads(line)
                            
                            # Handle Lifecycle Events
                            if "event_type" in data:
                                event = data["event_type"]
                                if event == "simulation_end":
                                    self.run_state.status = RunnerStatus.COMPLETED
                                    self.run_state.completed_at = datetime.utcnow()
                                    self._stop_monitor.set()
                                elif event == "round_end":
                                    self.run_state.current_timestep = data.get("timestep", 0)
                                    if self.run_state.total_timesteps > 0:
                                        self.run_state.percent_complete = round((self.run_state.current_timestep + 1) / self.run_state.total_timesteps * 100, 2)
                            
                            # Handle Agent Actions
                            elif "agent_id" in data:
                                action = AgentAction(**data)
                                self.run_state.add_action(action)
                        
                        pos = f.tell()
                except Exception as e:
                    logger.debug(f"Monitor read error (possible race condition): {e}")
            
            # Check if process died unexpectedly
            if self.process and self.process.poll() is not None:
                if self.run_state.status == RunnerStatus.RUNNING:
                    self.run_state.status = RunnerStatus.FAILED
                    self.run_state.error = self.process.stderr.read() if self.process.stderr else "Process exited unexpectedly"
                self._stop_monitor.set()

            self._save_run_state()
            time.sleep(1.0)
            
    def pause(self):
        self._send_command("pause")
        self.run_state.status = RunnerStatus.PAUSED
        self._save_run_state()
        
    def resume(self):
        self._send_command("resume")
        self.run_state.status = RunnerStatus.RUNNING
        self._save_run_state()
        
    def interview(self, questions: List[str]):
        try:
            with open(self.command_file, 'w', encoding='utf-8') as f:
                json.dump({"action": "interview", "questions": questions}, f)
            logger.info(f"Interview Signal Sent (Sim: {self.simulation_id})")
        except Exception as e:
            logger.error(f"Failed to send interview command: {e}")
            
    def stop(self):
        """Forcefully terminate simulation tree using cross-platform strategies."""
        self._send_command("stop")
        self.run_state.status = RunnerStatus.STOPPING
        self._save_run_state()
        self._stop_monitor.set()
        
        # Grace period for worker-level cleanup
        time.sleep(1.0) 
        
        if self.process and self.process.poll() is None:
            try:
                if IS_WINDOWS:
                    try:
                        subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T"], capture_output=True, timeout=5)
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], capture_output=True)
                else:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Process PGID {pgid} hanging. Escalating to SIGKILL.")
                        os.killpg(pgid, signal.SIGKILL)
                logger.warning(f"Cleanup performed on simulation group for PID: {self.process.pid}")
            except Exception as e:
                logger.error(f"Force cleanup failed: {e}")
        
        self.run_state.status = RunnerStatus.STOPPED
        self.run_state.completed_at = datetime.utcnow()
        self._save_run_state()

    def _send_command(self, action: str):
        try:
            with open(self.command_file, 'w', encoding='utf-8') as f:
                json.dump({"action": action}, f)
            logger.info(f"IPC Signal Sent: {action} (Sim: {self.simulation_id})")
        except Exception as e:
            logger.error(f"Failed to send IPC command: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return the latest persistent state snapshot."""
        # Ensure latest state is loaded if multiple runners exist
        refreshed_state = self._load_run_state()
        if refreshed_state:
            self.run_state = refreshed_state
            
        return self.run_state.model_dump()

    def get_agent_stats(self) -> List[Dict[str, Any]]:
        """Calculate per-agent activity stats from logs."""
        stats = {}
        if not os.path.exists(self.actions_file):
            return []
            
        try:
            with open(self.actions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if "agent_id" not in data: continue
                    aid = data["agent_id"]
                    if aid not in stats:
                        stats[aid] = {"agent_id": aid, "agent_name": data.get("agent_name", "Unknown"), "total_actions": 0, "types": {}}
                    stats[aid]["total_actions"] += 1
                    atype = data.get("action_type", "POST")
                    stats[aid]["types"][atype] = stats[aid]["types"].get(atype, 0) + 1
            return sorted(stats.values(), key=lambda x: x["total_actions"], reverse=True)
        except Exception:
            return []

    def get_interview_history(self) -> List[Dict[str, Any]]:
        """Fetch past interview responses from the workspace (cached JSON)."""
        interview_file = os.path.join(self.workspace, "mid_sim_interview_responses.json")
        if os.path.exists(interview_file):
            try:
                with open(interview_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def cleanup_simulation_logs(self):
        """Purge all logs and database to allow a fresh restart."""
        self.stop()
        import shutil
        # Delete logs and state but KEEP payload if we want a restart
        for f in [self.state_file, self.actions_file, os.path.join(self.workspace, "progress.json")]:
            if os.path.exists(f): os.remove(f)
        
        # Unique DB delete
        db_file = os.path.join(self.workspace, f"{self.simulation_id}.sqlite")
        if os.path.exists(db_file): os.remove(db_file)
        
        self.run_state = SimulationRunState(simulation_id=self.simulation_id)
        logger.info(f"Workspace {self.simulation_id} purged for fresh run.")

    def get_result(self) -> Optional[MarketSentimentSeries]:
        """Read the computed result from disk."""
        if os.path.exists(self.result_file):
            try:
                with open(self.result_file, 'r', encoding='utf-8') as f:
                    return MarketSentimentSeries.model_validate_json(f.read())
            except Exception as e:
                logger.error(f"Failed to read simulation result: {e}")
        return None

    _cleanup_registered = False

    @classmethod
    def cleanup_all_simulations(cls):
        """Clean all active simulation runners on application exit."""
        if not cls._active_runners:
            return
        logger.info("Atexit Hook Triggered: Cleaning up active OASIS simulations...")
        for sim_id, runner in list(cls._active_runners.items()):
            try:
                if runner.process and runner.process.poll() is None:
                    runner.stop()
            except Exception as e:
                logger.error(f"Failed to cleanup simulation {sim_id}: {e}")
        cls._active_runners.clear()

    @classmethod
    def register_cleanup(cls):
        """Register the global exit handlers strictly once."""
        if cls._cleanup_registered:
            return
        
        def cleanup_handler(signum, frame):
            cls.cleanup_all_simulations()
            sys.exit(0)
            
        atexit.register(cls.cleanup_all_simulations)
        try:
            signal.signal(signal.SIGTERM, cleanup_handler)
            signal.signal(signal.SIGINT, cleanup_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, cleanup_handler)
        except ValueError:
            pass # Subthreads cannot use signal hooks securely
        cls._cleanup_registered = True

# Register global cleanup upon import
SimulationRunner.register_cleanup()
