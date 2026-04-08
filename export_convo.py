import sqlite3
import os

sim_id = "e2e_sim_133511"
db_path = f"/tmp/oasis_runs/{sim_id}/{sim_id}.sqlite"
output_path = f"simulation_results/{sim_id}_conversation.md"

if not os.path.exists("simulation_results"):
    os.makedirs("simulation_results")

def export_convo():
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(output_path, "w") as f:
        f.write(f"# OASIS Simulation Conversation Log: {sim_id}\n\n")
        
        cursor.execute("SELECT post_id, user_id, content FROM post")
        posts = cursor.fetchall()
        
        for pid, uid, pcontent in posts:
            f.write(f"## [POST] User_{uid}\n")
            f.write(f"> {pcontent}\n\n")
            
            cursor.execute("SELECT user_id, content FROM comment WHERE post_id = ?", (pid,))
            comments = cursor.fetchall()
            for cuid, ccontent in comments:
                f.write(f"- **User_{cuid}**: {ccontent}\n")
            f.write("\n---\n\n")

    print(f"Successfully exported conversation to {output_path}")

if __name__ == "__main__":
    export_convo()
