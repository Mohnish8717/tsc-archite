#!/usr/bin/env python3
"""
V29: Hindsight Preflight Check

Verifies the Hindsight server is reachable and properly configured
before starting a full debate simulation.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Pathing setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def check_hindsight_server():
    url = os.getenv("HINDSIGHT_URL")
    api_key = os.getenv("HINDSIGHT_API_KEY")

    print("\n" + "═" * 60)
    print("🛰️  HINDSIGHT V29 PREFLIGHT CHECK")
    print("═" * 60)

    if not url:
        print("❌ HINDSIGHT_URL NOT SET in .env")
        print("   If you want to run with embedded memory, the simulation will still work,")
        print("   but it will be in DEGRADED mode.")
        return False

    print(f"🔗 Target: {url}")
    
    try:
        from hindsight_client import Hindsight
        h = Hindsight(base_url=url, api_key=api_key or None)
        
        # Test heartbeat/connection
        import time
        start_time = time.time()
        
        # Try to list banks to verify API key and connectivity
        import asyncio
        banks_coro = h.banks.list_banks()
        if asyncio.iscoroutine(banks_coro):
            response = asyncio.run(banks_coro)
        else:
            response = banks_coro
            
        # Extract count from response object if possible
        num_banks = len(getattr(response, 'banks', response)) if not isinstance(response, int) else response
            
        latency = (time.time() - start_time) * 1000
        
        print(f"✅ CONNECTION SUCCESSFUL ({latency:.1f}ms)")
        print(f"🏦 Active Banks detected.")
        
        print("\n🚀 READY: Hindsight server is responsive and ready for V29 debate.")
        return True
        
    except ImportError:
        print("❌ HINDSIGHT CLIENT NOT INSTALLED")
        print("   Run: pip install hindsight-client==0.5.3")
        return False
    except Exception as e:
        print(f"❌ SERVER REACHABLE BUT ERROR RETURNED: {e}")
        print("   Check if the server is running and the API key is correct.")
        return False

if __name__ == "__main__":
    success = check_hindsight_server()
    if not success:
        print("\n⚠️  PREFLIGHT FAILED")
        sys.exit(1)
    else:
        print("\n🔥 PREFLIGHT PASSED")
        sys.exit(0)
