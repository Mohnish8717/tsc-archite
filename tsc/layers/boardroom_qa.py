#!/usr/bin/env python3
"""
V29: Boardroom QA — "Ask the Board" Multi-Agent Query Mode

Ask a question and ALL evolved agents respond from their post-debate perspective.
Each answer is grounded in the agent's accumulated memory: positions, commitments,
concessions, proposals, and unresolved concerns from the debate.

Usage:
    python boardroom_qa.py --db live_debate_run.db --run-id <UUID>
    python boardroom_qa.py --memory-file evolved_memories.json

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
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tsc.memory.hindsight_memory import HindsightBoardroom, LiveAgentMemory


def load_memories(args) -> Dict[str, dict]:
    """Load evolved agent memories from DB or file."""
    if args.memory_file:
        with open(args.memory_file, 'r') as f:
            return json.load(f)
    
    db_path = args.db or str(Path(__file__).resolve().parent.parent.parent / "live_debate_run.db")
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if args.run_id:
        cursor.execute("SELECT simulation_metadata FROM simulation_runs WHERE id = ?", (args.run_id,))
    else:
        cursor.execute("SELECT simulation_metadata FROM simulation_runs ORDER BY simulation_timestamp DESC LIMIT 1")
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("❌ No simulation runs found.")
        sys.exit(1)
    
    metadata = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    memories = metadata.get("evolved_agent_memories", {})
    
    if not memories:
        print("⚠️  No evolved agent memories found. Run may predate V29.")
        sys.exit(1)
    
    return memories


def build_llm_config() -> Optional[dict]:
    """Build LLM config from environment variables."""
    model = os.getenv("TSC_LLM_MODEL", "gemma-4-31b-it")
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if gemini_key:
        return {"config_list": [{"model": model, "api_key": gemini_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/v1"}]}
    elif groq_key:
        return {"config_list": [{"model": model, "api_key": groq_key,
                "base_url": "https://api.groq.com/openai/v1"}]}
    return None


def ask_the_board(boardroom: HindsightBoardroom, question: str, llm_config: Optional[dict]) -> Dict[str, str]:
    """Ask all agents a question in parallel and return their responses."""
    results = {}
    
    if not llm_config:
        # Fallback: return memory summaries
        for name in boardroom.get_agent_names():
            results[name] = boardroom.reflect_post_debate(name)
        return results
    
    def _query_one(agent_name: str) -> tuple:
        try:
            answer = boardroom.query_agent(agent_name, question, llm_config)
            return agent_name, answer
        except Exception as e:
            return agent_name, f"[Query failed: {e}]"
    
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_query_one, name) for name in boardroom.get_agent_names()]
        for fut in as_completed(futures):
            name, answer = fut.result()
            results[name] = answer
    
    return results


def print_board_response(results: Dict[str, str], boardroom: HindsightBoardroom):
    """Pretty-print the board's responses."""
    ROLE_EMOJI = {
        "CEO": "👔", "CTO": "⚙️", "CFO": "💰", "CISO": "🔒",
        "CPO": "🎯", "Legal": "⚖️", "CMO": "📢", "CDO": "📊",
        "Sales": "📈", "HR": "👥", "Other": "🏢",
    }
    
    for name, answer in results.items():
        memory = boardroom.memories.get(name)
        role_short = memory.role_short if memory else "Other"
        emoji = ROLE_EMOJI.get(role_short, "🏢")
        
        print(f"\n{emoji} {name} ({memory.role if memory else 'Unknown'}):")
        print(f"   {answer}")
        print()


def main():
    parser = argparse.ArgumentParser(description="V29: Ask the Board — Multi-Agent Query Mode")
    parser.add_argument("--db", type=str, help="Path to simulation database")
    parser.add_argument("--run-id", type=str, help="Simulation run ID")
    parser.add_argument("--memory-file", type=str, help="Path to evolved_memories.json")
    parser.add_argument("--question", "-q", type=str, help="Single question to ask (non-interactive)")
    args = parser.parse_args()
    
    # Load memories
    raw_memories = load_memories(args)
    
    # Reconstruct boardroom
    boardroom = HindsightBoardroom()
    for name, mem_dict in raw_memories.items():
        try:
            boardroom.memories[name] = LiveAgentMemory.from_dict(mem_dict)
        except Exception as e:
            print(f"⚠️  Skipping {name}: {e}")
    
    print(f"\n{'='*70}")
    print(f"🏛️  BOARDROOM QA — {len(boardroom.memories)} Evolved Agents Ready")
    print(f"{'='*70}")
    
    for name, mem in boardroom.memories.items():
        veto = "🔴" if mem.has_vetoed else "✅"
        embedded_commitments = getattr(mem, '_embedded_commitments', [])
        print(f"  {veto} {name:30s} | {mem.turn_count} turns, {len(embedded_commitments)} fallback commitments")
    print(f"{'='*70}")
    
    llm_config = build_llm_config()
    if not llm_config:
        print("⚠️  No LLM API key found. Responses will be memory dumps only.")
    
    # Single question mode
    if args.question:
        print(f"\n📣 Question: {args.question}\n")
        results = ask_the_board(boardroom, args.question, llm_config)
        print_board_response(results, boardroom)
        return
    
    # Interactive mode
    print(f"\n💬 Ask the entire board a question. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Board session adjourned.")
            break
        
        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            print("\n🏛️  Board session adjourned.")
            break
        
        print(f"\n📣 Broadcasting to {len(boardroom.memories)} agents...\n")
        results = ask_the_board(boardroom, question, llm_config)
        print_board_response(results, boardroom)


if __name__ == "__main__":
    main()
