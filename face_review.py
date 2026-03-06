"""
face_review.py — Local web UI for reviewing face clusters.

Usage:
    python face_review.py
    python face_review.py --port 8050

Opens http://localhost:8050 in your browser.
"""

import argparse
import json
import sqlite3
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, jsonify, request, send_from_directory, send_file

DATA_DIR = Path("face_data")
DB_PATH = DATA_DIR / "faces.db"
THUMB_DIR = DATA_DIR / "thumbs"

app = Flask(__name__)


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── Static files ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("face_review_ui.html")


@app.route("/thumbs/<path:filename>")
def serve_thumb(filename):
    return send_from_directory(str(THUMB_DIR.resolve()), filename)


# ── API ───────────────────────────────────────────────────────────────────

@app.route("/api/clusters")
def api_clusters():
    """List all clusters with face count, ordered by size desc."""
    conn = get_db()
    rows = conn.execute("""
        SELECT c.id, c.name, c.birth_year, c.merged_into,
               COUNT(f.id) as face_count
        FROM clusters c
        LEFT JOIN faces f ON f.cluster_id = c.id
        WHERE c.merged_into IS NULL
        GROUP BY c.id
        ORDER BY face_count DESC
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/clusters/<int:cluster_id>/faces")
def api_cluster_faces(cluster_id):
    """Get all faces in a cluster, with photo info, sorted by date."""
    conn = get_db()
    rows = conn.execute("""
        SELECT f.id as face_id, f.thumb_path, f.photo_id,
               p.file_path, p.photo_date, p.date_source
        FROM faces f
        JOIN photos p ON p.id = f.photo_id
        WHERE f.cluster_id = ?
        ORDER BY
            CASE WHEN p.photo_date IS NOT NULL THEN 0 ELSE 1 END,
            p.photo_date,
            p.file_path
    """, (cluster_id,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/clusters/<int:cluster_id>", methods=["PUT"])
def api_update_cluster(cluster_id):
    """Update cluster name and/or birth_year."""
    data = request.json
    conn = get_db()
    if "name" in data:
        conn.execute("UPDATE clusters SET name = ? WHERE id = ?",
                      (data["name"], cluster_id))
    if "birth_year" in data:
        conn.execute("UPDATE clusters SET birth_year = ? WHERE id = ?",
                      (data["birth_year"], cluster_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/clusters/merge", methods=["POST"])
def api_merge_clusters():
    """Merge cluster source_id into target_id."""
    data = request.json
    source_id = data["source_id"]
    target_id = data["target_id"]
    conn = get_db()
    conn.execute("UPDATE faces SET cluster_id = ? WHERE cluster_id = ?",
                  (target_id, source_id))
    conn.execute("UPDATE clusters SET merged_into = ? WHERE id = ?",
                  (target_id, source_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    conn = get_db()
    photos = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    clusters = conn.execute(
        "SELECT COUNT(*) FROM clusters WHERE merged_into IS NULL"
    ).fetchone()[0]
    named = conn.execute(
        "SELECT COUNT(*) FROM clusters WHERE merged_into IS NULL AND name IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return jsonify({
        "photos": photos, "faces": faces,
        "clusters": clusters, "named": named,
    })


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Review face clusters")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Error: {DB_PATH} not found. Run face_scan.py first.")
        return

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    print(f"Starting review UI at http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
