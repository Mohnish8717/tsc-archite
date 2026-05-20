import asyncio
from tsc.memory.world_rag import WorldRAGEngine

async def populate():
    engine = WorldRAGEngine()
    await engine.initialize("eval-run")
    
    # 1. discovery_data
    await engine.ingest_run_data(
        "discovery_data",
        "Customers strongly value trust and transparency. They indicated that autonomous healthcare agents need strict audit trails to be trusted.",
        {"source": "eval"},
        "eval-run"
    )
    
    # 2. simulation_data
    await engine.ingest_run_data(
        "simulation_data",
        "Enterprise segments, specifically large health systems and ACOs, strongly support the proposed feature.",
        {"source": "eval"},
        "eval-run"
    )
    
    # 3. regulatory_corpus
    await engine._upsert_chunks(
        "regulatory_corpus",
        ["The new compliance laws will lead to faster product cycles and HIPAA compliance strictness."],
        {"source": "eval", "run_id": "global"}
    )
    
    # 4. competitor_intel
    await engine._upsert_chunks(
        "competitor_intel",
        ["Startups and established players expanding into healthcare AI are our primary competitive threats."],
        {"source": "eval", "run_id": "global"}
    )
    print("Done populating dummy data for evaluation.")

if __name__ == "__main__":
    asyncio.run(populate())
