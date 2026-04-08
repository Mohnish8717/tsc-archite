import sqlite3
import os
import sys

def retrieve_convo(db_path):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Static mapping from test_mock_sim.py
    PERSONA_NAMES = {
        0: "Sofia Chen",
        1: "Marcus 'Marc' Rodriguez",
        2: "Priya Kapoor",
        3: "James Wu",
        4: "Aisha Thompson",
        5: "Dev Patel",
        6: "Lena Bergsson",
        7: "Kai Okonkwo",
        8: "Rebecca 'Bex' Santos",
        9: "Amjad Hassan",
        10: "Dr. Zainab Malik",
        11: "Tommy Liu",
        12: "Eleanor Price",
        13: "Marcus 'Big Marc' Thompson",
        14: "Sophia 'Sophie' Nakamura",
        15: "Rashid Al-Dosari",
        16: "Nina Kowalski",
        17: "Hiroshi Tanaka"
    }

    # Get all posts
    cursor.execute("""
        SELECT post_id, user_id, content, created_at
        FROM post
        ORDER BY created_at ASC
    """)
    posts = cursor.fetchall()

    for post in posts:
        author = PERSONA_NAMES.get(post['user_id'], f"Agent_{post['user_id']}")
        print(f"[{post['created_at']}] 📢 {author} POSTED:")
        print(f"   \"{post['content']}\"\n")

        # Get comments for this post
        cursor.execute("""
            SELECT content, user_id, created_at
            FROM comment
            WHERE post_id = ?
            ORDER BY created_at ASC
        """, (post['post_id'],))
        comments = cursor.fetchall()

        for comment in comments:
            commenter = PERSONA_NAMES.get(comment['user_id'], f"Agent_{comment['user_id']}")
            print(f"   └─ [{comment['created_at']}] 💬 {commenter}: \"{comment['content']}\"")
        
        print("\n" + "-"*60 + "\n")

    conn.close()

if __name__ == "__main__":
    path = "simulation_results/sota_convergent_v18/sota_convergent_v18.sqlite"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    retrieve_convo(path)
