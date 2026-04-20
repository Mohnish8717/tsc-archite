import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tsc.memory.hindsight_memory import HindsightBoardroom

def test_sync_env():
    print("\n🧪 TESTING IN SYNC ENVIRONMENT...")
    load_dotenv()
    boardroom = HindsightBoardroom()
    
    class MockPersona:
        def __init__(self, name, role):
            self.name = name
            self.role = role
            self.role_short = "CEO"
            self.domain_expertise = ["Strategy", "Operations"]

    personas = [MockPersona("Test_User", "Chief Executive")]
    
    try:
        boardroom.initialize_agents(personas, "Test Feature", "This is a test feature description.")
        print("✅ Bank creation successful in sync env.")
        
        boardroom.extract_and_retain("Test_User", "I am proposing a major change.", ["Test_User"])
        print("✅ Retention successful in sync env.")
        
        recall = boardroom.recall_for_turn("Test_User")
        print(f"✅ Recall successful. Result length: {len(recall)}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed in sync env: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_env():
    print("\n🧪 TESTING IN ASYNC ENVIRONMENT (ALREADY RUNNING LOOP)...")
    # This simulates the boardroom engine if it's already in an async context
    return test_sync_env()

if __name__ == "__main__":
    success = test_sync_env()
    if success:
        # Now try to run it inside an event loop to see if it handles the conflict
        print("\n[INFO] Starting async test...")
        asyncio.run(test_async_env())
