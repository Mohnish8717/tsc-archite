import sqlite3
import json
import os

def retrieve_v27_logs(db_path):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get the latest simulation run
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

    print(f"═══ RETRIEVING LOGS FOR RUN: {run['id']} (Timestamp: {run['simulation_timestamp']}) ═══\n")

    metadata = json.loads(run['simulation_metadata'])
    transcript = metadata.get('ag2_transcript', [])

    if not transcript:
        print("No transcript found in simulation metadata.")
        conn.close()
        return

    for entry in transcript:
        # Pydantic model dump has keys standardizing names
        name = entry.get('stakeholder_name', 'Unknown')
        statement = entry.get('statement', '')
        verdict = entry.get('verdict', 'DEBATING')
        
        # Clean up statement (remove any stray AG2 markers if they survived the stripper)
        print(f"[{verdict}] 📢 {name}:")
        print(f"{statement}")
        print("-" * 80 + "\n")

    conn.close()

if __name__ == "__main__":
    db_path = "live_debate_run.db"
    retrieve_v27_logs(db_path)
