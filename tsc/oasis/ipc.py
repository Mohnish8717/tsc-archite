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
        self.should_abort = False
        
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
            elif action == "abort":
                self.should_abort = True
            elif action == "interview":
                # Returns the interview details to be handled by the engine
                interview_payload = {
                    "questions": cmd.get("questions", []),
                    "target_agent_id": cmd.get("target_agent_id")
                }
                # Clear command file *after* extracting data
                os.remove(self.command_file)
                # Auto-resume the simulation so it doesn't get stuck!
                self.is_paused = False
                return interview_payload
            elif action == "intervention":
                intervention_payload = {
                    "action": "intervention",
                    "event": cmd.get("event", ""),
                }
                os.remove(self.command_file)
                return intervention_payload
                
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
        if payload and isinstance(payload, dict):
            if "questions" in payload and interview_callback:
                logger.info(f"Performing mid-simulation interview with {len(payload['questions'])} questions")
                await interview_callback(payload)
            elif payload.get("action") == "intervention":
                return payload

        if self.is_paused:
            logger.info("Simulation PAUSED. Waiting for resume...")
            
        while self.is_paused:
            await asyncio.sleep(1)
            payload = await self.check_commands()
            if payload and isinstance(payload, dict):
                if "questions" in payload and interview_callback:
                    logger.info(f"Performing mid-simulation interview with {len(payload['questions'])} questions")
                    await interview_callback(payload)
                elif payload.get("action") == "intervention":
                    return payload
                
            if self.should_stop or self.should_abort:
                break

    def get_latest_command(self) -> Optional[Dict[str, Any]]:
        """Non-blocking poll for the latest command."""
        if not os.path.exists(self.command_file):
            return None
        try:
            with open(self.command_file, 'r', encoding='utf-8') as f:
                cmd = json.load(f)
            # Do NOT remove it here if it's a "stop" command, 
            # let check_commands process it natively later if needed, 
            # or we just process it manually.
            os.remove(self.command_file)
            return cmd
        except Exception as e:
            return None

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

    def log_spawn(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        role: str,
        traits: list,
        impact: float,
        # Core psychological identifiers
        mbti: str = "",
        mbti_description: str = "",
        ocean_scores: Dict[str, float] = None,
        # Buyer journey — stage string for UI indicator
        buyer_journey: str = "",
        # Full structured buyer journey dict (external personas)
        buyer_journey_detail: Optional[Dict[str, Any]] = None,
        bio: str = "",
        # Structured psychological sub-models
        emotional_triggers: Optional[Dict[str, Any]] = None,
        communication_style: Optional[Dict[str, Any]] = None,
        decision_pattern: Optional[Dict[str, Any]] = None,
        predicted_stance: Optional[Dict[str, Any]] = None,
        questions_they_will_ask: Optional[list] = None,
        # FinalPersona-level metadata
        domain_expertise: Optional[list] = None,
        profile_confidence: float = 0.0,
        grounding_quality: float = 1.0,
        persona_type: str = "INTERNAL",
        network_position_hint: str = "peripheral",
        influence_strength: float = 0.5,
        receptiveness: float = 0.5,
        # External persona context
        market_context: Optional[Dict[str, Any]] = None,
        evidence_sources: Optional[list] = None,
    ):
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
                # Core profile
                "mbti": mbti,
                "mbti_description": mbti_description,
                "ocean_scores": ocean_scores or {},
                "buyer_journey": buyer_journey,
                "buyer_journey_detail": buyer_journey_detail,
                "bio": bio,
                # Structured psychological fields
                "emotional_triggers": emotional_triggers or {},
                "communication_style": communication_style or {},
                "decision_pattern": decision_pattern or {},
                "predicted_stance": predicted_stance or {},
                "questions_they_will_ask": questions_they_will_ask or [],
                # Persona metadata
                "domain_expertise": domain_expertise or [],
                "profile_confidence": profile_confidence,
                "grounding_quality": grounding_quality,
                "persona_type": persona_type,
                "network_position_hint": network_position_hint,
                "influence_strength": round(influence_strength, 3),
                "receptiveness": round(receptiveness, 3),
                # External persona context
                "market_context": market_context,
                "evidence_sources": evidence_sources or [],
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
