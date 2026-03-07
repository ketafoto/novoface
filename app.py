"""
app.py — Unified web application for gedface.

Single entry point that provides both scan management and cluster review
in one browser tab. Replaces the separate face_scan.py + face_review.py workflow.

Usage:
    python app.py
    python app.py --port 8050
"""

import argparse
import json
import os
import sqlite3
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from threading import Timer

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from face_scan import (
    DATA_DIR,
    DB_PATH,
    THUMB_DIR,
    PHOTO_EXTENSIONS,
    init_db,
    process_photo,
    cluster_faces,
)

app = Flask(__name__)

# ── Scan state (read by SSE endpoint, written by scan worker) ─────────────

_scan = {
    "status": "idle",
    "current": 0,
    "total": 0,
    "faces_found": 0,
    "errors": 0,
    "rate": 0.0,
    "eta_seconds": 0,
    "current_file": "",
    "message": "",
    "started_at": None,
    "elapsed_seconds": 0,
    "photo_seconds": 0.0,
    "sec_per_img": 0.0,
}
_scan_thread = None
_scan_stop = threading.Event()


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db():
    DATA_DIR.mkdir(exist_ok=True)
    THUMB_DIR.mkdir(exist_ok=True)
    conn = init_db(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_folders (
            id INTEGER PRIMARY KEY,
            folder_path TEXT UNIQUE NOT NULL,
            added_at TEXT
        )
    """)
    conn.commit()
    conn.close()


# ── Scan worker ────────────────────────────────────────────────────────────

def _run_scan(folders, det_size, threshold):
    from insightface.app import FaceAnalysis

    try:
        _scan.update(
            status="loading_model",
            message="Loading face detection model...",
            current=0, total=0, faces_found=0, errors=0,
            rate=0.0, eta_seconds=0, current_file="",
            started_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=0, photo_seconds=0.0, sec_per_img=0.0,
        )

        fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(det_size, det_size))

        conn = init_db(DB_PATH)
        already = {r[0] for r in conn.execute("SELECT file_path FROM photos").fetchall()}

        files = []
        for folder in folders:
            fp = Path(folder)
            if not fp.is_dir():
                continue
            for root, _, names in os.walk(fp):
                for name in names:
                    if Path(name).suffix.lower() in PHOTO_EXTENSIONS:
                        full = Path(root) / name
                        if str(full) not in already:
                            files.append(full)

        total = len(files)
        _scan.update(status="scanning", total=total,
                     message=f"Scanning {total} new photos...")

        if total == 0:
            _scan.update(status="clustering", message="No new photos. Re-clustering...")
            cluster_faces(conn, threshold)
            _scan.update(status="done",
                         message="No new photos found. Clustering complete.")
            conn.close()
            return

        found = 0
        errs = 0
        t0 = time.time()

        for i, path in enumerate(files):
            if _scan_stop.is_set():
                _scan.update(status="clustering",
                             message=f"Stopped after {i}/{total}. Clustering...")
                cluster_faces(conn, threshold)
                _scan.update(status="done",
                             message=f"Stopped. {i} photos processed, {found} faces clustered.")
                conn.close()
                return

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 1 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            spi = elapsed / (i + 1) if i > 0 else 0
            _scan.update(current=i + 1, faces_found=found, errors=errs,
                         rate=round(rate, 2), eta_seconds=round(eta),
                         elapsed_seconds=round(elapsed),
                         sec_per_img=round(spi, 1),
                         current_file=path.name)

            photo_t0 = time.time()
            try:
                found += process_photo(path, conn, fa)
            except Exception:
                errs += 1
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO photos (file_path, processed_at) VALUES (?, ?)",
                        (str(path), datetime.now().isoformat()),
                    )
                    conn.commit()
                except Exception:
                    pass
            _scan["photo_seconds"] = round(time.time() - photo_t0, 1)

        _scan.update(current=total, faces_found=found, errors=errs,
                     status="clustering", message="Clustering faces...")
        cluster_faces(conn, threshold)
        _scan.update(status="done",
                     message=f"Done! {total} photos processed, {found} faces clustered.")
        conn.close()

    except Exception as e:
        _scan.update(status="error", message=str(e))


# ── Routes: static ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("face_review_ui.html")


@app.route("/thumbs/<path:filename>")
def serve_thumb(filename):
    return send_from_directory(str(THUMB_DIR.resolve()), filename)


# ── Routes: filesystem browser ─────────────────────────────────────────────

@app.route("/api/browse")
def api_browse():
    """List directories at the given path, or list available drives/root."""
    requested = request.args.get("path", "")

    if not requested:
        if os.name == "nt":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                try:
                    if Path(drive).exists():
                        drives.append({"name": f"{letter}:", "path": drive})
                except OSError:
                    pass
            return jsonify({"current": "", "parent": None, "dirs": drives})
        else:
            requested = "/"

    p = Path(requested).resolve()
    if not p.is_dir():
        return jsonify({"error": "Not a directory"}), 400

    parent = str(p.parent) if p.parent != p else None

    dirs = []
    try:
        for item in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            try:
                if item.is_dir():
                    dirs.append({"name": item.name, "path": str(item)})
            except (PermissionError, OSError):
                pass
    except (PermissionError, OSError):
        pass

    return jsonify({"current": str(p), "parent": parent, "dirs": dirs})


# ── Routes: scan management ────────────────────────────────────────────────

@app.route("/api/scan/folders", methods=["GET"])
def get_folders():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, folder_path, added_at FROM scan_folders ORDER BY id"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/scan/folders", methods=["POST"])
def add_folder():
    data = request.json
    fp = data.get("folder_path", "").strip()
    if not fp:
        return jsonify({"error": "Path required"}), 400
    p = Path(fp)
    if not p.is_dir():
        return jsonify({"error": f"Not a valid directory: {fp}"}), 400
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO scan_folders (folder_path, added_at) VALUES (?, ?)",
            (str(p.resolve()), datetime.now().isoformat()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Folder already added"}), 409
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/scan/folders/<int:fid>", methods=["DELETE"])
def remove_folder(fid):
    conn = get_db()
    conn.execute("DELETE FROM scan_folders WHERE id = ?", (fid,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/scan/start", methods=["POST"])
def start_scan():
    global _scan_thread
    if _scan_thread and _scan_thread.is_alive():
        return jsonify({"error": "Scan already running"}), 409

    data = request.json or {}
    det_size = int(data.get("det_size", 320))
    threshold = float(data.get("threshold", 0.35))

    conn = get_db()
    rows = conn.execute("SELECT folder_path FROM scan_folders").fetchall()
    conn.close()
    folders = [r["folder_path"] for r in rows]
    if not folders:
        return jsonify({"error": "No folders configured. Add at least one folder."}), 400

    _scan_stop.clear()
    _scan_thread = threading.Thread(
        target=_run_scan, args=(folders, det_size, threshold), daemon=True,
    )
    _scan_thread.start()
    return jsonify({"ok": True})


@app.route("/api/scan/stop", methods=["POST"])
def stop_scan():
    if not _scan_thread or not _scan_thread.is_alive():
        return jsonify({"error": "No scan running"}), 400
    _scan_stop.set()
    return jsonify({"ok": True})


@app.route("/api/scan/status")
def scan_status():
    return jsonify(_scan)


@app.route("/api/scan/progress")
def scan_progress():
    """SSE endpoint — polls _scan dict and pushes changes to the browser."""
    def generate():
        prev = None
        while True:
            cur = json.dumps(_scan)
            if cur != prev:
                yield f"data: {cur}\n\n"
                prev = cur
            time.sleep(0.5)
    return Response(
        generate(), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Routes: review ─────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    conn = get_db()
    result = {
        "photos": conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0],
        "faces": conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
        "clusters": conn.execute(
            "SELECT COUNT(*) FROM clusters WHERE merged_into IS NULL"
        ).fetchone()[0],
        "named": conn.execute(
            "SELECT COUNT(*) FROM clusters WHERE merged_into IS NULL AND name IS NOT NULL"
        ).fetchone()[0],
    }
    conn.close()
    return jsonify(result)


@app.route("/api/clusters")
def api_clusters():
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


@app.route("/api/clusters/<int:cid>/faces")
def api_cluster_faces(cid):
    conn = get_db()
    rows = conn.execute("""
        SELECT f.id as face_id, f.thumb_path, f.photo_id,
               p.file_path, p.photo_date, p.date_source
        FROM faces f
        JOIN photos p ON p.id = f.photo_id
        WHERE f.cluster_id = ?
        ORDER BY
            CASE WHEN p.photo_date IS NOT NULL THEN 0 ELSE 1 END,
            p.photo_date, p.file_path
    """, (cid,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/clusters/<int:cid>", methods=["PUT"])
def api_update_cluster(cid):
    data = request.json
    conn = get_db()
    if "name" in data:
        conn.execute("UPDATE clusters SET name = ? WHERE id = ?",
                     (data["name"], cid))
    if "birth_year" in data:
        conn.execute("UPDATE clusters SET birth_year = ? WHERE id = ?",
                     (data["birth_year"], cid))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/clusters/merge", methods=["POST"])
def api_merge_clusters():
    data = request.json
    conn = get_db()
    conn.execute("UPDATE faces SET cluster_id = ? WHERE cluster_id = ?",
                 (data["target_id"], data["source_id"]))
    conn.execute("UPDATE clusters SET merged_into = ? WHERE id = ?",
                 (data["target_id"], data["source_id"]))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/faces/<int:fid>/move", methods=["POST"])
def api_move_face(fid):
    data = request.json
    conn = get_db()
    conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?",
                 (data["cluster_id"], fid))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="gedface — face recognition for genealogy photos"
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    ensure_db()

    if not args.no_browser:
        Timer(2.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    print(f"gedface running at http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
