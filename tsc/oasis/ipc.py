import os
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger("tsc.oasis.ipc")

class CommandListener:
    """
    Listens for IPC commands (Pause/Resume/Stop) via a simple file-based mechanism.
    This avoids complex socket/pipe management for the MVP.
    """
    def __init__(self, simulation_id: str, base_dir: str):
        self.simulation_id = simulation_id
        self.base_dir = base_dir
        self.command_file = os.path.join(base_dir, "commands.json")
        self.is_paused = False
        self.should_stop = False
        
        # Ensure base_dir exists
        os.makedirs(self.base_dir, exist_ok=True)
        
    async def check_commands(self):
        """Poll for new commands."""
        if not os.path.exists(self.command_file):
            return
        
        try:
            with open(self.command_file, 'r', encoding='utf-8') as f:
                cmd = json.load(f)
            
            action = cmd.get("action")
            logger.info(f"IPC Command Received: {action}")
            
            if action == "pause":
                self.is_paused = True
            elif action == "resume":
                self.is_paused = False
            elif action == "stop":
                self.should_stop = True
            elif action == "interview":
                # Returns the interview details to be handled by the engine
                interview_payload = {
                    "questions": cmd.get("questions", []),
                    "target_agent_id": cmd.get("target_agent_id")
                }
                # Clear command file *after* extracting data
                os.remove(self.command_file)
                return interview_payload
                
            # Clear command file after reading
            if os.path.exists(self.command_file):
                os.remove(self.command_file)
        except Exception as e:
            logger.error(f"Failed to process IPC command: {e}")
        return None
            
    async def wait_if_paused(self, interview_callback=None):
        """Blocking loop for the worker if a 'pause' command is active, and handles interviews."""
        # Always check for new commands at least once
        payload = await self.check_commands()
        if payload and isinstance(payload, dict) and "questions" in payload and interview_callback:
            logger.info(f"Performing mid-simulation interview with {len(payload['questions'])} questions")
            await interview_callback(payload)

        if self.is_paused:
            logger.info("Simulation PAUSED. Waiting for resume...")
            
        while self.is_paused:
            await asyncio.sleep(1)
            payload = await self.check_commands()
            if payload and isinstance(payload, dict) and "questions" in payload and interview_callback:
                logger.info(f"Performing mid-simulation interview with {len(payload['questions'])} questions")
                await interview_callback(payload)
                
            if self.should_stop:
                break

class LocalActionLogger:
    """
    Writes agent actions to a local JSONL file for real-time dashboard tailing.
    This provides 'instant' feedback while Zep handles long-term memory.
    """
    def __init__(self, base_dir: str):
        self.log_file = os.path.join(base_dir, "actions.jsonl")
        os.makedirs(base_dir, exist_ok=True)
        
    def log_action(self, agent_id: str, agent_name: str, action_type: str, content: Any, timestep: int, platform: str = "reddit", metadata: Dict[str, Any] = None):
        """Append a single action to the JSONL log."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "agent_name": agent_name,
                "timestep": timestep,
                "action_type": action_type,
                "content": content,
                "platform": platform,
                # CRITICAL FIX: wrap as nested object, not top-level spread
                # Frontend reads data.metadata.target_id for network arc rendering
                "metadata": metadata or {}
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write local action log: {e}")

    def log_spawn(self, agent_id: str, agent_name: str, agent_type: str, role: str, traits: list, impact: float,
                  mbti: str = "", ocean_scores: Dict[str, float] = None, buyer_journey: str = "", bio: str = ""):
        """Emit an agent_spawn event so the frontend can build the initial agent registry."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "agent_spawn",
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_type": agent_type,
                "role": role,
                "traits": traits,
                "impact": round(impact * 100),
                "mbti": mbti,
                "ocean_scores": ocean_scores or {},
                "buyer_journey": buyer_journey,
                "bio": bio,
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write spawn log: {e}")


    def log_simulation_event(self, event_type: str, data: Dict[str, Any]):
        """Emit a structured simulation lifecycle event (simulation_start, progress, report, etc.)."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                **data
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write simulation event: {e}")

    def log_event(self, event_type: str, metadata: Dict[str, Any] = None):
        """Append a lifecycle event (e.g., simulation_end, round_start) to the log."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                **(metadata or {})
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write lifecycle event: {e}")

    def update_progress(self, timestep: int, total: int, status: str = "RUNNING"):
        """Update a progress heartbeat file."""
        progress_file = self.log_file.replace("actions.jsonl", "progress.json")
        try:
            data = {
                "last_update": datetime.now().isoformat(),
                "current_timestep": timestep,
                "total_timesteps": total,
                "percent_complete": round((timestep + 1) / total * 100, 2),
                "status": status
            }
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to update progress heartbeat: {e}")
