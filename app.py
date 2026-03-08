"""
app.py — Unified web application for gedface.

Single entry point that provides both scan management and cluster review
in one browser tab. Replaces the separate face_scan.py + face_review.py workflow.

Usage:
    python app.py
    python app.py --port 8050
"""

import argparse
import ctypes
import json
import os
import sys
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
    compute_file_hash,
)

# OpenVINO backend uses separate DB and thumbs
try:
    from openvino_pipeline import DB_PATH_OV, THUMB_DIR_OV
except ImportError:
    DB_PATH_OV = DATA_DIR / "faces_ov.db"
    THUMB_DIR_OV = DATA_DIR / "thumbs_ov"

app = Flask(__name__)

BACKEND_CONFIG = DATA_DIR / "backend.json"

def get_backend() -> str:
    """Current backend: 'cpu' (InsightFace) or 'openvino' (Iris Xe)."""
    if not BACKEND_CONFIG.exists():
        return "cpu"
    try:
        with open(BACKEND_CONFIG) as f:
            data = json.load(f)
            return "openvino" if data.get("backend") == "openvino" else "cpu"
    except Exception:
        return "cpu"

def set_backend(backend: str) -> None:
    if backend not in ("cpu", "openvino"):
        raise ValueError("backend must be 'cpu' or 'openvino'")
    DATA_DIR.mkdir(exist_ok=True)
    with open(BACKEND_CONFIG, "w") as f:
        json.dump({"backend": backend}, f)

# ── Scan state (per backend; read by SSE, written by scan worker) ─────────

def _scan_template():
    return {
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

_scan_cpu = _scan_template()
_scan_ov = _scan_template()
_scan_thread = None
_scan_thread_backend = None  # "cpu" or "openvino" while scan running
_scan_stop = threading.Event()


def _run_with_lower_priority(func, *args, **kwargs):
    """Run func(*args, **kwargs) with thread priority below normal (Windows only).
    Keeps UI and other processes responsive during long clustering.
    """
    if sys.platform != "win32":
        return func(*args, **kwargs)
    try:
        k32 = ctypes.windll.kernel32
        THREAD_PRIORITY_BELOW_NORMAL = -1
        THREAD_PRIORITY_NORMAL = 0
        h = k32.GetCurrentThread()
        k32.SetThreadPriority(h, THREAD_PRIORITY_BELOW_NORMAL)
        try:
            return func(*args, **kwargs)
        finally:
            k32.SetThreadPriority(h, THREAD_PRIORITY_NORMAL)
    except Exception:
        return func(*args, **kwargs)


def get_scan_folders_conn():
    """Connection to the DB that holds scan_folders and cluster_groups (always main DB)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def get_db(backend: str | None = None):
    """Connection to the data DB for current (or given) backend: photos, faces, clusters."""
    if backend is None:
        backend = get_backend()
    path = DB_PATH if backend == "cpu" else DB_PATH_OV
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn

def _scan_for_backend(backend: str):
    return _scan_cpu if backend == "cpu" else _scan_ov

def ensure_db():
    DATA_DIR.mkdir(exist_ok=True)
    THUMB_DIR.mkdir(exist_ok=True)
    THUMB_DIR_OV.mkdir(parents=True, exist_ok=True)
    conn = init_db(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_folders (
            id INTEGER PRIMARY KEY,
            folder_path TEXT UNIQUE NOT NULL,
            added_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cluster_groups (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(clusters)").fetchall()}
    if "group_id" not in cols:
        conn.execute("ALTER TABLE clusters ADD COLUMN group_id INTEGER REFERENCES cluster_groups(id)")
    conn.commit()
    conn.close()
    ov_conn = init_db(DB_PATH_OV)
    cols = {r[1] for r in ov_conn.execute("PRAGMA table_info(clusters)").fetchall()}
    if "group_id" not in cols:
        ov_conn.execute("ALTER TABLE clusters ADD COLUMN group_id INTEGER")
    ov_conn.execute("CREATE TABLE IF NOT EXISTS cluster_groups (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    ov_conn.commit()
    ov_conn.close()


# ── Scan worker ────────────────────────────────────────────────────────────

def _apply_cpu_limit(cpu_percent):
    """Restrict this process to a subset of CPU cores and lower priority."""
    import psutil
    p = psutil.Process()
    total_cores = psutil.cpu_count()
    use_cores = max(1, round(total_cores * cpu_percent / 100))
    all_cpus = list(range(total_cores))
    p.cpu_affinity(all_cpus[:use_cores])
    if os.name == "nt":
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        try:
            p.nice(10)
        except PermissionError:
            pass


def _run_scan(folders, det_size, threshold, cpu_percent, _scan_ref):
    from insightface.app import FaceAnalysis
    from face_scan import _face_providers

    try:
        _apply_cpu_limit(cpu_percent)

        _scan_ref.update(
            status="loading_model",
            message="Loading face detection model...",
            current=0, total=0, faces_found=0, errors=0,
            rate=0.0, eta_seconds=0, current_file="",
            started_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=0, photo_seconds=0.0, sec_per_img=0.0,
        )

        providers = _face_providers()
        fa = FaceAnalysis(name="buffalo_l", providers=providers)
        fa.prepare(ctx_id=-1, det_size=(det_size, det_size))

        conn = init_db(DB_PATH)
        already = {r[0] for r in conn.execute("SELECT file_path FROM photos").fetchall()}
        known_hashes = dict(
            conn.execute("SELECT file_hash, id FROM photos WHERE file_hash IS NOT NULL").fetchall()
        )
        prev_photos = len(already)
        prev_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

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

        new_count = len(files)
        grand_total = prev_photos + new_count
        _scan_ref.update(status="scanning", total=grand_total, current=prev_photos,
                     faces_found=prev_faces,
                     message=f"Resuming: {new_count} new photos ({prev_photos} already done)...")

        if new_count == 0:
            _scan_ref.update(status="clustering", message="No new photos. Re-clustering...")
            _run_with_lower_priority(
                cluster_faces, conn, threshold,
                progress_cb=lambda d, t: _scan_ref.update(message=f"Clustering... {d}/{t} faces"),
            )
            _scan_ref.update(status="done",
                         message="No new photos found. Clustering complete.")
            conn.close()
            return

        found = prev_faces
        errs = 0
        t0 = time.time()

        # If we have faces but no clusters (e.g. server was restarted before pause-handler ran clustering),
        # run clustering once so Review shows data without waiting for the next interim run.
        if prev_faces > 0:
            n_clusters = conn.execute(
                "SELECT COUNT(*) FROM clusters WHERE merged_into IS NULL"
            ).fetchone()[0]
            if n_clusters == 0:
                _scan_ref.update(status="clustering",
                             message="Re-clustering existing faces…")
                _run_with_lower_priority(
                    cluster_faces, conn, threshold,
                    progress_cb=lambda d, t: _scan_ref.update(message=f"Clustering... {d}/{t} faces"),
                )

        for i, path in enumerate(files):
            if _scan_stop.is_set():
                done_total = prev_photos + i
                _scan_ref.update(status="clustering",
                             message=f"Stopped after {done_total}/{grand_total}. Clustering...")
                _run_with_lower_priority(
                    cluster_faces, conn, threshold,
                    progress_cb=lambda d, t: _scan_ref.update(message=f"Clustering... {d}/{t} faces"),
                )
                _scan_ref.update(status="done",
                             message=f"Stopped. {done_total} photos processed, {found} faces clustered.")
                conn.close()
                return

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 1 else 0
            eta = (new_count - i - 1) / rate if rate > 0 else 0
            spi = elapsed / (i + 1) if i > 0 else 0
            _scan_ref.update(current=prev_photos + i + 1, faces_found=found, errors=errs,
                         rate=round(rate, 2), eta_seconds=round(eta),
                         elapsed_seconds=round(elapsed),
                         sec_per_img=round(spi, 1),
                         current_file=path.name)

            photo_t0 = time.time()
            try:
                fhash = compute_file_hash(path)
                if fhash in known_hashes:
                    conn.execute(
                        "UPDATE photos SET file_path = ? WHERE id = ?",
                        (str(path), known_hashes[fhash]),
                    )
                    conn.commit()
                    _scan_ref["photo_seconds"] = round(time.time() - photo_t0, 1)
                    continue
                found += process_photo(path, conn, fa, file_hash=fhash)
                known_hashes[fhash] = conn.execute(
                    "SELECT id FROM photos WHERE file_path = ?", (str(path),)
                ).fetchone()[0]
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
            _scan_ref["photo_seconds"] = round(time.time() - photo_t0, 1)

            if (i + 1) % 200 == 0 and found > 0:
                prev_msg = _scan_ref["message"]
                _scan_ref["message"] = f"Interim clustering after {prev_photos + i + 1} photos..."
                _run_with_lower_priority(
                    cluster_faces, conn, threshold,
                    progress_cb=lambda d, t: _scan_ref.update(message=f"Interim clustering... {d}/{t} faces"),
                )
                _scan_ref["message"] = prev_msg

        _scan_ref.update(current=grand_total, faces_found=found, errors=errs,
                     status="clustering", message="Final clustering...")
        _run_with_lower_priority(
            cluster_faces, conn, threshold,
            progress_cb=lambda d, t: _scan_ref.update(message=f"Clustering... {d}/{t} faces"),
        )
        _scan_ref.update(status="done",
                     message=f"Done! {grand_total} photos processed, {found} faces clustered.")
        conn.close()

    except Exception as e:
        _scan_ref.update(status="error", message=str(e))


def _run_scan_openvino(folders, threshold, cpu_percent, _scan_ref):
    from openvino_pipeline import OpenVINOHybridPipeline, process_photo as ov_process_photo

    try:
        _apply_cpu_limit(cpu_percent)

        _scan_ref.update(
            status="loading_model",
            message="Loading face detection (OpenVINO) and face recognition (InsightFace)…",
            current=0, total=0, faces_found=0, errors=0,
            rate=0.0, eta_seconds=0, current_file="",
            started_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=0, photo_seconds=0.0, sec_per_img=0.0,
        )

        pipeline = OpenVINOHybridPipeline(device="GPU")
        conn = init_db(DB_PATH_OV)
        already = {r[0] for r in conn.execute("SELECT file_path FROM photos").fetchall()}
        known_hashes = dict(
            conn.execute("SELECT file_hash, id FROM photos WHERE file_hash IS NOT NULL").fetchall()
        )
        prev_photos = len(already)
        prev_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

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

        new_count = len(files)
        grand_total = prev_photos + new_count
        _scan_ref.update(status="scanning", total=grand_total, current=prev_photos,
                     faces_found=prev_faces,
                     message=f"Resuming: {new_count} new photos ({prev_photos} already done)...")

        if new_count == 0:
            _scan_ref.update(status="clustering", message="No new photos. Re-clustering...")
            _run_with_lower_priority(
                cluster_faces, conn, threshold,
                progress_cb=lambda d, t: _scan_ref.update(message=f"Clustering... {d}/{t} faces"),
            )
            _scan_ref.update(status="done",
                         message="No new photos found. Clustering complete.")
            conn.close()
            return

        found = prev_faces
        errs = 0
        t0 = time.time()

        # If we have faces but no clusters (e.g. server was restarted before pause-handler ran clustering),
        # run clustering once so Review shows data without waiting for the next interim run.
        if prev_faces > 0:
            n_clusters = conn.execute(
                "SELECT COUNT(*) FROM clusters WHERE merged_into IS NULL"
            ).fetchone()[0]
            if n_clusters == 0:
                _scan_ref.update(status="clustering",
                             message="Re-clustering existing faces…")
                _run_with_lower_priority(
                    cluster_faces, conn, threshold,
                    progress_cb=lambda d, t: _scan_ref.update(message=f"Clustering... {d}/{t} faces"),
                )

        for i, path in enumerate(files):
            if _scan_stop.is_set():
                done_total = prev_photos + i
                _scan_ref.update(status="clustering",
                             message=f"Stopped after {done_total}/{grand_total}. Clustering...")
                _run_with_lower_priority(
                    cluster_faces, conn, threshold,
                    progress_cb=lambda d, t: _scan_ref.update(message=f"Clustering... {d}/{t} faces"),
                )
                _scan_ref.update(status="done",
                             message=f"Stopped. {done_total} photos processed, {found} faces clustered.")
                conn.close()
                return

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 1 else 0
            eta = (new_count - i - 1) / rate if rate > 0 else 0
            spi = elapsed / (i + 1) if i > 0 else 0
            _scan_ref.update(current=prev_photos + i + 1, faces_found=found, errors=errs,
                         rate=round(rate, 2), eta_seconds=round(eta),
                         elapsed_seconds=round(elapsed),
                         sec_per_img=round(spi, 1),
                         current_file=path.name)

            photo_t0 = time.time()
            try:
                fhash = compute_file_hash(path)
                if fhash in known_hashes:
                    conn.execute(
                        "UPDATE photos SET file_path = ? WHERE id = ?",
                        (str(path), known_hashes[fhash]),
                    )
                    conn.commit()
                    _scan_ref["photo_seconds"] = round(time.time() - photo_t0, 1)
                    continue
                found += ov_process_photo(path, conn, pipeline, THUMB_DIR_OV, file_hash=fhash)
                known_hashes[fhash] = conn.execute(
                    "SELECT id FROM photos WHERE file_path = ?", (str(path),)
                ).fetchone()[0]
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
            _scan_ref["photo_seconds"] = round(time.time() - photo_t0, 1)

            if (i + 1) % 200 == 0 and found > 0:
                prev_msg = _scan_ref["message"]
                _scan_ref["message"] = f"Interim clustering after {prev_photos + i + 1} photos..."
                _run_with_lower_priority(
                    cluster_faces, conn, threshold,
                    progress_cb=lambda d, t: _scan_ref.update(message=f"Interim clustering... {d}/{t} faces"),
                )
                _scan_ref["message"] = prev_msg

        _scan_ref.update(current=grand_total, faces_found=found, errors=errs,
                     status="clustering", message="Final clustering...")
        _run_with_lower_priority(
            cluster_faces, conn, threshold,
            progress_cb=lambda d, t: _scan_ref.update(message=f"Clustering... {d}/{t} faces"),
        )
        _scan_ref.update(status="done",
                     message=f"Done! {grand_total} photos processed, {found} faces clustered.")
        conn.close()

    except Exception as e:
        _scan_ref.update(status="error", message=str(e))


# ── Routes: static ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = send_file("face_review_ui.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/thumbs/<path:filename>")
def serve_thumb(filename):
    thumb_dir = THUMB_DIR if get_backend() == "cpu" else THUMB_DIR_OV
    return send_from_directory(str(thumb_dir.resolve()), filename)


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

def _openvino_ready():
    """Check if buffalo_l pack (rec + det for alignment) is present for OpenVINO hybrid. Returns (ok, message)."""
    try:
        from openvino_pipeline import _buffalo_dir
        d = _buffalo_dir()
        if d is None:
            return False, (
                "The GPU Iris Xe (OpenVINO) backend needs the InsightFace buffalo_l model pack. It was not found.\n\n"
                "To fix:\n"
                "1. Switch to CPU (InsightFace) and run a short scan (e.g. one folder with a few photos). That will download buffalo_l automatically.\n"
                "2. Then switch back to GPU Iris Xe (OpenVINO).\n\n"
                "If you use a custom model path, set the INSIGHTFACE_HOME environment variable before starting the app."
            )
        if not (d / "w600k_r50.onnx").exists() or not (d / "det_10g.onnx").exists():
            return False, (
                "The GPU Iris Xe (OpenVINO) backend needs the full buffalo_l pack (w600k_r50.onnx and det_10g.onnx for alignment).\n\n"
                "Run a CPU scan once so InsightFace downloads the full buffalo_l pack, then switch back to GPU Iris Xe (OpenVINO)."
            )
        return True, None
    except Exception as e:
        return False, str(e)


@app.route("/api/backend", methods=["GET"])
def api_get_backend():
    return jsonify({"backend": get_backend()})


@app.route("/api/openvino/ready", methods=["GET"])
def api_openvino_ready():
    """Check if OpenVINO hybrid can run (buffalo_l rec model present)."""
    ok, message = _openvino_ready()
    return jsonify({"ready": ok, "message": message})


@app.route("/api/backend", methods=["POST"])
def api_set_backend():
    global _scan_thread_backend
    if _scan_thread and _scan_thread.is_alive():
        return jsonify({"error": "Stop the scan before switching backend"}), 400
    data = request.json or {}
    backend = (data.get("backend") or "").strip().lower()
    if backend not in ("cpu", "openvino"):
        return jsonify({"error": "backend must be 'cpu' or 'openvino'"}), 400
    if backend == "openvino":
        ok, msg = _openvino_ready()
        if not ok:
            return jsonify({"error": "OpenVINO backend not ready", "detail": msg}), 400
    set_backend(backend)
    _scan_thread_backend = None
    return jsonify({"ok": True, "backend": backend})


@app.route("/api/scan/folders", methods=["GET"])
def get_folders():
    conn = get_scan_folders_conn()
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
    conn = get_scan_folders_conn()
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
    conn = get_scan_folders_conn()
    conn.execute("DELETE FROM scan_folders WHERE id = ?", (fid,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/scan/start", methods=["POST"])
def start_scan():
    global _scan_thread, _scan_thread_backend
    if _scan_thread and _scan_thread.is_alive():
        return jsonify({"error": "Scan already running"}), 409

    backend = get_backend()
    _scan_ref = _scan_cpu if backend == "cpu" else _scan_ov

    data = request.json or {}
    det_size = int(data.get("det_size", 320))
    threshold = float(data.get("threshold", 0.35))
    cpu_percent = int(data.get("cpu_percent", 60))

    conn = get_scan_folders_conn()
    rows = conn.execute("SELECT folder_path FROM scan_folders").fetchall()
    conn.close()
    folders = [r["folder_path"] for r in rows]
    if not folders:
        return jsonify({"error": "No folders configured. Add at least one folder."}), 400

    if backend == "openvino":
        ok, msg = _openvino_ready()
        if not ok:
            return jsonify({"error": "OpenVINO backend not ready", "detail": msg}), 400

    _scan_stop.clear()
    _scan_thread_backend = backend
    _scan_ref.update(
        status="loading_model",
        message="Starting scan…",
        current=0, total=0, faces_found=0, errors=0,
        rate=0.0, eta_seconds=0, current_file="",
        started_at=datetime.now().isoformat(timespec="seconds"),
        elapsed_seconds=0, photo_seconds=0.0, sec_per_img=0.0,
    )
    if backend == "cpu":
        _scan_thread = threading.Thread(
            target=_run_scan,
            args=(folders, det_size, threshold, cpu_percent, _scan_ref),
            daemon=True,
        )
    else:
        _scan_thread = threading.Thread(
            target=_run_scan_openvino,
            args=(folders, threshold, cpu_percent, _scan_ref),
            daemon=True,
        )
    _scan_thread.start()
    return jsonify({"ok": True})


@app.route("/api/scan/stop", methods=["POST"])
def stop_scan():
    global _scan_thread_backend
    if not _scan_thread or not _scan_thread.is_alive():
        return jsonify({"error": "No scan running"}), 400
    _scan_stop.set()
    return jsonify({"ok": True})


@app.route("/api/scan/cpu", methods=["POST"])
def set_cpu_limit():
    data = request.json or {}
    cpu_percent = int(data.get("cpu_percent", 60))
    _apply_cpu_limit(cpu_percent)
    return jsonify({"ok": True, "cpu_percent": cpu_percent})


@app.route("/api/scan/reset", methods=["POST"])
def reset_database():
    """Drop all scan data for the current backend so the next scan starts from scratch."""
    global _scan_thread_backend
    if _scan_thread and _scan_thread.is_alive():
        return jsonify({"error": "Cannot reset while a scan is running"}), 400
    backend = get_backend()
    conn = get_db(backend)
    if backend == "cpu":
        conn.executescript("""
            DELETE FROM faces;
            DELETE FROM photos;
            DELETE FROM clusters;
            DELETE FROM cluster_groups;
        """)
        import shutil
        if THUMB_DIR.is_dir():
            shutil.rmtree(THUMB_DIR)
            THUMB_DIR.mkdir(parents=True, exist_ok=True)
    else:
        conn.executescript("""
            DELETE FROM faces;
            DELETE FROM photos;
            DELETE FROM clusters;
            DELETE FROM cluster_groups;
        """)
        import shutil
        if THUMB_DIR_OV.is_dir():
            shutil.rmtree(THUMB_DIR_OV)
            THUMB_DIR_OV.mkdir(parents=True, exist_ok=True)
    conn.close()
    _scan_thread_backend = None
    return jsonify({"ok": True})


@app.route("/api/scan/status")
def scan_status():
    active = _scan_thread_backend if (_scan_thread and _scan_thread.is_alive()) else None
    scan_ref = _scan_for_backend(active or get_backend())
    return jsonify(scan_ref)


@app.route("/api/scan/progress")
def scan_progress():
    """SSE endpoint — polls active _scan dict and pushes changes to the browser."""
    def generate():
        prev = None
        while True:
            active = _scan_thread_backend if (_scan_thread and _scan_thread.is_alive()) else None
            scan_ref = _scan_for_backend(active or get_backend())
            cur = json.dumps(scan_ref)
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


@app.route("/api/scan/pending")
def scan_pending():
    """Count unprocessed photos and detect data from outside current folders."""
    folders_conn = get_scan_folders_conn()
    rows = folders_conn.execute("SELECT folder_path FROM scan_folders").fetchall()
    folders_conn.close()
    data_conn = get_db()
    all_photos = [r[0] for r in data_conn.execute("SELECT file_path FROM photos").fetchall()]
    data_conn.close()

    folder_prefixes = [row["folder_path"] for row in rows]
    already = set(all_photos)

    orphan = 0
    for p in all_photos:
        if not any(p.startswith(fp) for fp in folder_prefixes):
            orphan += 1

    pending = 0
    for row in rows:
        fp = Path(row["folder_path"])
        if not fp.is_dir():
            continue
        for root, _, names in os.walk(fp):
            for name in names:
                if Path(name).suffix.lower() in PHOTO_EXTENSIONS:
                    if str(Path(root) / name) not in already:
                        pending += 1
    return jsonify({"pending": pending, "orphan": orphan})


@app.route("/api/clusters")
def api_clusters():
    conn = get_db()
    rows = conn.execute("""
        SELECT c.id, c.name, c.birth_year, c.merged_into, c.group_id,
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
    limit = request.args.get("limit", type=int)
    conn = get_db()
    q = """
        SELECT f.id as face_id, f.thumb_path, f.photo_id,
               p.file_path, p.photo_date, p.date_source
        FROM faces f
        JOIN photos p ON p.id = f.photo_id
        WHERE f.cluster_id = ?
        ORDER BY
            CASE WHEN p.photo_date IS NOT NULL THEN 0 ELSE 1 END,
            p.photo_date, p.file_path
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = conn.execute(q, (cid,)).fetchall()
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


# ── Routes: cluster groups ─────────────────────────────────────────────────

@app.route("/api/groups")
def api_groups():
    conn = get_db()
    rows = conn.execute("SELECT id, name FROM cluster_groups ORDER BY name").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/groups", methods=["POST"])
def api_create_group():
    data = request.json
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    conn = get_db()
    cur = conn.execute("INSERT INTO cluster_groups (name) VALUES (?)", (name,))
    gid = cur.lastrowid
    conn.commit()
    conn.close()
    return jsonify({"id": gid, "name": name})


@app.route("/api/groups/<int:gid>", methods=["PUT"])
def api_update_group(gid):
    data = request.json
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    conn = get_db()
    conn.execute("UPDATE cluster_groups SET name = ? WHERE id = ?", (name, gid))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/groups/<int:gid>", methods=["DELETE"])
def api_delete_group(gid):
    conn = get_db()
    conn.execute("UPDATE clusters SET group_id = NULL WHERE group_id = ?", (gid,))
    conn.execute("DELETE FROM cluster_groups WHERE id = ?", (gid,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/clusters/<int:cid>/group", methods=["PUT"])
def api_set_cluster_group(cid):
    data = request.json
    group_id = data.get("group_id")
    conn = get_db()
    conn.execute("UPDATE clusters SET group_id = ? WHERE id = ?", (group_id, cid))
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
