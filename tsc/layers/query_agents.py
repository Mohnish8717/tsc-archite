#!/usr/bin/env python3
"""
V29: Post-Debate Agent Query Interface

Query individual agents from their evolved post-debate perspective.
Each agent answers based on their accumulated memory — positions taken,
commitments made, concessions accepted, and remaining concerns — not as
a generic LLM.

Usage:
    python query_agents.py --db live_debate_run.db --run-id <UUID>
    python query_agents.py --db live_debate_run.db --run-id <UUID> --agent Sarah_CISO
    python query_agents.py --memory-file evolved_memories.json --agent Sarah_CISO

Environment:
    GEMINI_API_KEY or GROQ_API_KEY must be set for LLM-backed responses.
"""

import argparse
import json
import os
import sys
import sqlite3
from pathlib import Path
from typing import Optional, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tsc.memory.hindsight_memory import HindsightBoardroom, LiveAgentMemory


def load_memories_from_db(db_path: str, run_id: str) -> Dict[str, dict]:
    """Load evolved agent memories from the simulation database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Try to load from simulation_metadata JSON
    cursor.execute(
        "SELECT simulation_metadata FROM simulation_runs WHERE id = ?",
        (run_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print(f"❌ No simulation run found with ID: {run_id}")
        sys.exit(1)
    
    metadata = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    memories = metadata.get("evolved_agent_memories", {})
    
    if not memories:
        print(f"⚠️  No evolved agent memories found in run {run_id}.")
        print("   This run may have been created before V29 (Stateful Memory).")
        print("   Available metadata keys:", list(metadata.keys()))
        sys.exit(1)
    
    return memories


def load_memories_from_file(file_path: str) -> Dict[str, dict]:
    """Load evolved agent memories from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def build_llm_config() -> Optional[dict]:
    """Build LLM config from environment variables."""
    model = os.getenv("TSC_LLM_MODEL", "gemma-4-31b-it")
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if gemini_key:
        return {
            "config_list": [{
                "model": model,
                "api_key": gemini_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/v1",
            }]
        }
    elif groq_key:
        return {
            "config_list": [{
                "model": model,
                "api_key": groq_key,
                "base_url": "https://api.groq.com/openai/v1",
            }]
        }
    return None


def interactive_query(boardroom: HindsightBoardroom, agent_name: str, llm_config: Optional[dict]):
    """Run an interactive query session with a single agent."""
    memory = boardroom.memories.get(agent_name)
    if not memory:
        print(f"❌ Agent '{agent_name}' not found.")
        print(f"   Available agents: {boardroom.get_agent_names()}")
        return
    
    # Print agent memory summary
    print(f"\n{'='*70}")
    print(f"🧠 AGENT: {memory.agent_name} ({memory.role})")
    print(f"   Feature debated: {memory.feature_title}")
    print(f"   Turns spoken: {memory.turn_count}")
    print(f"   Commitments: {len(memory.commitments)}")
    print(f"   Proposals: {len(memory.proposals)}")
    print(f"   Concessions: {len(memory.concessions)}")
    print(f"   Outstanding concerns: {len(memory.unresolved_concerns)}")
    print(f"   Veto status: {'RAISED' if memory.has_vetoed else 'NONE'}")
    print(f"{'='*70}")
    
    if not llm_config:
        print("\n⚠️  No LLM API key found. Showing memory dump instead of interactive chat.")
        print(boardroom.reflect_post_debate(agent_name))
        return
    
    print(f"\n💬 Ask {memory.agent_name} anything. Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input(f"You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Session ended.")
            break
        
        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            print(f"\n👋 {memory.agent_name} has left the boardroom.")
            break
        
        print(f"\n{memory.agent_name}: ", end="", flush=True)
        answer = boardroom.query_agent(agent_name, question, llm_config)
        print(answer)
        print()


def list_agents(boardroom: HindsightBoardroom):
    """Print a summary of all available agents and their memory state."""
    print(f"\n{'='*70}")
    print(f"📋 EVOLVED AGENTS ({len(boardroom.memories)} total)")
    print(f"{'='*70}")
    
    for name, memory in boardroom.memories.items():
        veto = "🔴 VETO" if memory.has_vetoed else "✅"
        print(f"  {veto} {name:30s} | {memory.turn_count:2d} turns | "
              f"{len(memory.commitments)} commits | {len(memory.proposals)} proposals | "
              f"{len(memory.unresolved_concerns)} concerns")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="V29: Query evolved agents from their post-debate perspective."
    )
    parser.add_argument("--db", type=str, help="Path to the simulation database (live_debate_run.db)")
    parser.add_argument("--run-id", type=str, help="Simulation run ID to load")
    parser.add_argument("--memory-file", type=str, help="Path to an evolved_memories.json file")
    parser.add_argument("--agent", type=str, help="Agent name to query (e.g., Sarah_CISO)")
    parser.add_argument("--list", action="store_true", help="List all available agents")
    parser.add_argument("--reflect", type=str, help="Print the evolved reflection for an agent")
    args = parser.parse_args()
    
    # Load memories
    if args.memory_file:
        raw_memories = load_memories_from_file(args.memory_file)
    elif args.db and args.run_id:
        raw_memories = load_memories_from_db(args.db, args.run_id)
    else:
        # Try default paths
        default_db = Path(__file__).resolve().parent.parent.parent / "live_debate_run.db"
        if default_db.exists():
            print(f"Using default DB: {default_db}")
            # Get the most recent run
            conn = sqlite3.connect(str(default_db))
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM simulation_runs ORDER BY simulation_timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            if row:
                raw_memories = load_memories_from_db(str(default_db), row[0])
            else:
                print("❌ No simulation runs found in the database.")
                sys.exit(1)
        else:
            parser.print_help()
            print("\n❌ Please specify --db + --run-id or --memory-file")
            sys.exit(1)
    
    # Reconstruct HindsightBoardroom from persisted memories
    boardroom = HindsightBoardroom()
    for agent_name, mem_dict in raw_memories.items():
        try:
            memory = LiveAgentMemory.from_dict(mem_dict)
            boardroom.memories[agent_name] = memory
        except Exception as e:
            print(f"⚠️  Failed to load memory for {agent_name}: {e}")
    
    if not boardroom.memories:
        print("❌ No agent memories loaded. Check the database or memory file.")
        sys.exit(1)
    
    # Build LLM config
    llm_config = build_llm_config()
    
    # Handle commands
    if args.list:
        list_agents(boardroom)
    elif args.reflect:
        reflection = boardroom.reflect_post_debate(args.reflect)
        if reflection:
            print(reflection)
        else:
            print(f"❌ Agent '{args.reflect}' not found.")
    elif args.agent:
        interactive_query(boardroom, args.agent, llm_config)
    else:
        # Default: list agents + interactive selection
        list_agents(boardroom)
        print("\nEnter an agent name to query (or 'quit' to exit):")
        while True:
            try:
                agent_name = input("Agent > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye.")
                break
            if agent_name.lower() in ('quit', 'exit', 'q'):
                break
            if agent_name in boardroom.memories:
                interactive_query(boardroom, agent_name, llm_config)
            else:
                print(f"❌ Unknown agent '{agent_name}'. Available: {boardroom.get_agent_names()}")


if __name__ == "__main__":
    main()
