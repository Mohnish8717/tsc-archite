import sqlite3
import json
import os

def retrieve_v29_logs(db_path):
    """
    Retrieves and prints the full structured transcript from the latest simulation run.
    Formatted to mimic V28 but specifically targeting V29 Hindsight integration.
    """
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get the latest simulation run (V29)
    cursor.execute("""
        SELECT id, simulation_metadata, simulation_timestamp 
        FROM simulation_runs 
        ORDER BY simulation_timestamp DESC 
        LIMIT 1
    """)
    run = cursor.fetchone()

    if not run:
        print("No simulation runs found in database.")
        conn.close()
        return

    print(f"═══ RETRIEVING V29 SIMULATION LOGS: {run['id']} ═══")
    print(f"═══ Timestamp: {run['simulation_timestamp']} ═══\n")

    metadata = json.loads(run['simulation_metadata'])
    transcript = metadata.get('ag2_transcript', [])

    if not transcript:
        print("No transcript found in simulation metadata.")
        conn.close()
        return

    for entry in transcript:
        if isinstance(entry, dict):
            # v28 style dict
            name = entry.get('stakeholder_name', entry.get('sender', entry.get('name', 'Unknown')))
            statement = entry.get('statement', entry.get('content', ''))
            verdict = entry.get('verdict', 'DEBATING')
            confidence = entry.get('confidence', 'N/A')
            print(f"[{verdict}] (Conf: {confidence}) 📢 {name}:")
            print(f"{statement}")
            print("-" * 80 + "\n")
        else:
            # plain string
            print(f"[DEBATING] (Conf: N/A) 📢 Agent:")
            print(f"{entry}")
            print("-" * 80 + "\n")

    print(f"═══ SUMMARY ═══")
    print(f"Verdict: {metadata.get('overall_verdict', 'N/A')}")
    print(f"Score: {str(metadata.get('overall_summary', 'N/A'))[:200]}...")
    
    conn.close()

if __name__ == "__main__":
    db_path = "live_debate_run.db"
    retrieve_v29_logs(db_path)
