"""
app.py — Unified web application for gedface.

Single entry point that provides both scan management and cluster review
in one browser tab. Replaces the separate face_scan.py + face_review.py workflow.

Usage:
    python app.py
    python app.py --port 8050
"""

import argparse
import atexit
import base64
import ctypes
import fnmatch
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
    connect_db,
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

CONFIG_FILE = Path(__file__).parent / "gedface_config.json"

def _load_app_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_app_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

BACKEND_CONFIG = DATA_DIR / "backend.json"
SETTINGS_PATH = DATA_DIR / "settings.json"


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_settings(s: dict):
    DATA_DIR.mkdir(exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2), encoding="utf-8")


def _parse_exclude_patterns(raw: str) -> list[str]:
    """Split comma-separated exclude pattern string into a cleaned list."""
    return [p.strip() for p in raw.split(",") if p.strip()]


def _path_matches_exclude(file_path: str, patterns: list[str]) -> bool:
    """Return True if any component of file_path matches any fnmatch pattern (case-insensitive)."""
    parts = Path(file_path).parts
    return any(
        fnmatch.fnmatch(part.lower(), pat.lower())
        for pat in patterns
        for part in parts
    )


def _cleanup_children():
    """Terminate any child processes spawned by this process (e.g. OpenVINO workers).
    Registered with atexit so it runs on normal exit and Ctrl-C.
    """
    try:
        import psutil
        children = psutil.Process().children(recursive=True)
        for ch in children:
            try:
                ch.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=3)
        for ch in alive:
            try:
                ch.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception:
        pass


atexit.register(_cleanup_children)

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
    conn = connect_db(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_db(backend: str | None = None):
    """Connection to the data DB for current (or given) backend: photos, faces, clusters."""
    if backend is None:
        backend = get_backend()
    path = DB_PATH if backend == "cpu" else DB_PATH_OV
    conn = connect_db(path)
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
    """Limit this process to cpu_percent% of total CPU and lower its I/O priority.

    Windows:
      - CPU: Job Object with JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP — accurate
        OS-level rate limiter.
      - I/O: THREAD_MODE_BACKGROUND_BEGIN on the calling thread — tells the OS
        to satisfy all I/O requests from this thread at background (lowest)
        priority, so photo file reads don't compete with interactive I/O.
    Linux: falls back to nice(10); proper cgroup limiting would require root.
    """
    cpu_percent = max(5, min(100, int(cpu_percent)))

    if os.name == "nt":
        try:
            import ctypes
            import ctypes.wintypes

            kernel32 = ctypes.windll.kernel32

            JOB_OBJECT_CPU_RATE_CONTROL_ENABLE   = 0x1
            JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4
            JobObjectCpuRateControlInformation   = 15

            class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("ControlFlags", ctypes.wintypes.DWORD),
                    # CpuRate: portion of cycles allowed, in units of 1/100 of a percent.
                    # 80 % -> 8000, 100 % -> 10000.
                    ("CpuRate",      ctypes.wintypes.DWORD),
                ]

            hJob = kernel32.CreateJobObjectW(None, None)
            if not hJob:
                raise ctypes.WinError()

            info = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION()
            info.ControlFlags = (JOB_OBJECT_CPU_RATE_CONTROL_ENABLE |
                                 JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP)
            info.CpuRate = cpu_percent * 100  # e.g. 80 % -> 8000

            ok = kernel32.SetInformationJobObject(
                hJob,
                JobObjectCpuRateControlInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                raise ctypes.WinError()

            hProcess = kernel32.GetCurrentProcess()
            if not kernel32.AssignProcessToJobObject(hJob, hProcess):
                raise ctypes.WinError()

        except OSError as e:
            # Nested job objects are supported on Win8+; older systems or
            # already-constrained environments may fail — log and continue.
            print(f"[cpu_limit] Job Object failed ({e}); no CPU cap applied.")

        # Lower I/O priority for this thread so photo reads don't compete
        # with interactive/system I/O.  THREAD_MODE_BACKGROUND_BEGIN sets
        # I/O priority to VeryLow for all I/O issued by this thread.
        try:
            THREAD_MODE_BACKGROUND_BEGIN = 0x00010000
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(),
                                       THREAD_MODE_BACKGROUND_BEGIN)
        except Exception:
            pass
    else:
        try:
            import psutil
            psutil.Process().nice(10)
        except (PermissionError, AttributeError):
            pass


def _run_scan(folders, det_size, threshold, cpu_percent, _scan_ref, exclude_patterns=None):
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

        if exclude_patterns:
            all_paths = [r[0] for r in conn.execute("SELECT file_path FROM photos").fetchall()]
            to_del = [p for p in all_paths if _path_matches_exclude(p, exclude_patterns)]
            if to_del:
                ph = ",".join("?" * len(to_del))
                conn.execute(f"DELETE FROM faces WHERE photo_id IN (SELECT id FROM photos WHERE file_path IN ({ph}))", to_del)
                conn.execute(f"DELETE FROM photos WHERE file_path IN ({ph})", to_del)
                conn.commit()

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
            for root, dirs, names in os.walk(fp):
                if exclude_patterns:
                    dirs[:] = [d for d in dirs if not any(
                        fnmatch.fnmatch(d.lower(), pat.lower()) for pat in exclude_patterns)]
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
                    # Same content already processed under a different path.
                    # Record this path too so it never shows as pending again.
                    conn.execute(
                        "INSERT OR IGNORE INTO photos (file_path, file_hash, processed_at) VALUES (?, ?, ?)",
                        (str(path), fhash, datetime.now().isoformat()),
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


def _run_scan_openvino(folders, threshold, cpu_percent, _scan_ref, exclude_patterns=None):
    from openvino_pipeline import OpenVINOHybridPipeline, process_photo as ov_process_photo

    try:
        _apply_cpu_limit(cpu_percent)

        _scan_ref.update(
            status="scanning",
            message="Checking for new photos…",
            current=0, total=0, faces_found=0, errors=0,
            rate=0.0, eta_seconds=0, current_file="",
            started_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=0, photo_seconds=0.0, sec_per_img=0.0,
        )

        conn = init_db(DB_PATH_OV)

        if exclude_patterns:
            all_paths = [r[0] for r in conn.execute("SELECT file_path FROM photos").fetchall()]
            to_del = [p for p in all_paths if _path_matches_exclude(p, exclude_patterns)]
            if to_del:
                ph = ",".join("?" * len(to_del))
                conn.execute(f"DELETE FROM faces WHERE photo_id IN (SELECT id FROM photos WHERE file_path IN ({ph}))", to_del)
                conn.execute(f"DELETE FROM photos WHERE file_path IN ({ph})", to_del)
                conn.commit()

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
            for root, dirs, names in os.walk(fp):
                if exclude_patterns:
                    dirs[:] = [d for d in dirs if not any(
                        fnmatch.fnmatch(d.lower(), pat.lower()) for pat in exclude_patterns)]
                for name in names:
                    if Path(name).suffix.lower() in PHOTO_EXTENSIONS:
                        full = Path(root) / name
                        if str(full) not in already:
                            files.append(full)

        new_count = len(files)
        grand_total = prev_photos + new_count

        if new_count == 0:
            _scan_ref.update(status="clustering", total=grand_total, current=prev_photos,
                         faces_found=prev_faces,
                         message="No new photos. Re-clustering...")
            _run_with_lower_priority(
                cluster_faces, conn, threshold,
                progress_cb=lambda d, t: _scan_ref.update(message=f"Clustering... {d}/{t} faces"),
            )
            _scan_ref.update(status="done",
                         message="No new photos found. Clustering complete.")
            conn.close()
            return

        # New photos exist — now load the model
        _scan_ref.update(
            status="loading_model",
            message="Loading face detection (OpenVINO) and face recognition (InsightFace)…",
            total=grand_total, current=prev_photos, faces_found=prev_faces,
        )
        pipeline = OpenVINOHybridPipeline(device="GPU")

        _scan_ref.update(status="scanning", total=grand_total, current=prev_photos,
                     faces_found=prev_faces,
                     message=f"Resuming: {new_count} new photos ({prev_photos} already done)...")

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
                    # Same content already processed under a different path.
                    # Record this path too so it never shows as pending again.
                    conn.execute(
                        "INSERT OR IGNORE INTO photos (file_path, file_hash, processed_at) VALUES (?, ?, ?)",
                        (str(path), fhash, datetime.now().isoformat()),
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
    resp = send_from_directory(str(thumb_dir.resolve()), filename)
    resp.headers["Cache-Control"] = "public, max-age=86400"  # 1 day
    return resp


@app.route("/api/thumbs", methods=["POST"])
def api_thumbs_batch():
    """Return multiple thumb images as base64. Body: {"paths": ["a.jpg", "b.jpg", ...]}. Max 200 paths."""
    data = request.get_json(silent=True) or {}
    paths = data.get("paths") or []
    if not paths or len(paths) > 200:
        return jsonify({})
    thumb_dir = Path(THUMB_DIR if get_backend() == "cpu" else THUMB_DIR_OV)
    thumb_dir = thumb_dir.resolve()
    out = {}
    for p in paths:
        if ".." in p or p.startswith("/"):
            continue
        path = (thumb_dir / p).resolve()
        if not str(path).startswith(str(thumb_dir)) or not path.is_file():
            continue
        try:
            out[p] = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            pass
    return jsonify(out)


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


@app.route("/api/scan/settings", methods=["GET"])
def get_scan_settings():
    return jsonify(load_settings())


@app.route("/api/scan/settings", methods=["POST"])
def post_scan_settings():
    data = request.json or {}
    s = load_settings()
    if "exclude_patterns" in data:
        s["exclude_patterns"] = str(data["exclude_patterns"])
    save_settings(s)
    return jsonify({"ok": True})


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

    settings = load_settings()
    exclude_patterns = _parse_exclude_patterns(settings.get("exclude_patterns", ""))

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
            args=(folders, det_size, threshold, cpu_percent, _scan_ref, exclude_patterns),
            daemon=True,
        )
    else:
        _scan_thread = threading.Thread(
            target=_run_scan_openvino,
            args=(folders, threshold, cpu_percent, _scan_ref, exclude_patterns),
            daemon=True,
        )
    _scan_thread.start()
    return jsonify({"ok": True})


@app.route("/api/scan/stop", methods=["POST"])
def stop_scan():
    global _scan_thread_backend
    if not _scan_thread or not _scan_thread.is_alive():
        # Thread is already dead — if status is stuck in an active state,
        # reset it so the UI can recover (e.g. after a C-level thread crash).
        scan_ref = _scan_for_backend(_scan_thread_backend or get_backend())
        if scan_ref.get("status") in ("scanning", "loading_model", "clustering"):
            scan_ref.update(status="idle", message="")
            _scan_thread_backend = None
        return jsonify({"ok": True})
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


@app.route("/api/db/repath", methods=["POST"])
def api_db_repath():
    """Replace a path prefix across all photo records (and matching scan folders).
    Pass dry_run=true to get the count without making changes."""
    if _scan_thread and _scan_thread.is_alive():
        return jsonify({"error": "Cannot repath while a scan is running"}), 400
    data = request.json or {}
    old_prefix = data.get("old_prefix", "").strip().rstrip("/\\")
    new_prefix = data.get("new_prefix", "").strip().rstrip("/\\")
    dry_run = bool(data.get("dry_run", False))
    if not old_prefix:
        return jsonify({"error": "old_prefix required"}), 400

    # Normalise separators to match what Python's Path.resolve() stores
    old_prefix = str(Path(old_prefix))
    if new_prefix:
        new_prefix = str(Path(new_prefix))

    like_pattern = old_prefix + "%"
    prefix_len = len(old_prefix) + 1  # SUBSTR is 1-indexed; +1 skips the prefix itself

    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE file_path LIKE ?", (like_pattern,)
    ).fetchone()[0]

    if dry_run:
        conn.close()
        return jsonify({"count": count})

    if count:
        conn.execute(
            "UPDATE photos SET file_path = ? || SUBSTR(file_path, ?) "
            "WHERE file_path LIKE ?",
            (new_prefix, prefix_len, like_pattern),
        )
        conn.commit()
    conn.close()

    # Also update matching scan_folders entries
    folders_conn = get_scan_folders_conn()
    folders_conn.execute(
        "UPDATE scan_folders SET folder_path = ? || SUBSTR(folder_path, ?) "
        "WHERE folder_path LIKE ?",
        (new_prefix, prefix_len, like_pattern),
    )
    folders_conn.commit()
    folders_conn.close()

    return jsonify({"ok": True, "count": count})


@app.route("/api/db/location", methods=["GET"])
def api_db_location_get():
    return jsonify({"path": str(DATA_DIR.resolve())})


@app.route("/api/db/location", methods=["POST"])
def api_db_location_move():
    """Move the entire DATA_DIR to a new location and persist it in gedface_config.json.
    Requires server restart to take effect."""
    if _scan_thread and _scan_thread.is_alive():
        return jsonify({"error": "Cannot move database while a scan is running"}), 400
    data = request.json or {}
    new_dir = data.get("new_dir", "").strip()
    if not new_dir:
        return jsonify({"error": "new_dir required"}), 400

    new_path = Path(new_dir)
    if not new_path.is_absolute():
        return jsonify({"error": "Please provide an absolute path"}), 400
    if new_path.resolve() == DATA_DIR.resolve():
        return jsonify({"error": "Same as current location"}), 400

    # Reject if new_path is inside current DATA_DIR
    try:
        new_path.resolve().relative_to(DATA_DIR.resolve())
        return jsonify({"error": "New location cannot be inside the current data directory"}), 400
    except ValueError:
        pass

    import shutil
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(DATA_DIR.resolve()), str(new_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    cfg = _load_app_config()
    cfg["data_dir"] = str(new_path)
    _save_app_config(cfg)
    return jsonify({"ok": True, "new_path": str(new_path)})


@app.route("/api/db/remove-path", methods=["POST"])
def api_db_remove_path():
    """Delete all photos (and their faces + thumbnails) whose path starts with the given prefix.
    Pass dry_run=true to get the count without making changes."""
    if _scan_thread and _scan_thread.is_alive():
        return jsonify({"error": "Cannot remove while a scan is running"}), 400
    data = request.json or {}
    prefix = data.get("prefix", "").strip().rstrip("/\\")
    dry_run = bool(data.get("dry_run", False))
    if not prefix:
        return jsonify({"error": "prefix required"}), 400

    prefix = str(Path(prefix))
    like_pattern = prefix + "%"

    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE file_path LIKE ?", (like_pattern,)
    ).fetchone()[0]

    if dry_run:
        conn.close()
        return jsonify({"count": count})

    if count:
        # Collect thumbnail paths before deletion so we can clean up files
        thumb_rows = conn.execute(
            "SELECT f.thumb_path FROM faces f "
            "JOIN photos p ON p.id = f.photo_id "
            "WHERE p.file_path LIKE ? AND f.thumb_path IS NOT NULL",
            (like_pattern,),
        ).fetchall()

        conn.execute(
            "DELETE FROM faces WHERE photo_id IN "
            "(SELECT id FROM photos WHERE file_path LIKE ?)", (like_pattern,)
        )
        conn.execute("DELETE FROM photos WHERE file_path LIKE ?", (like_pattern,))
        conn.commit()

        for (tp,) in thumb_rows:
            try:
                p = Path(tp)
                if not p.is_absolute():
                    p = Path.cwd() / p
                p.unlink(missing_ok=True)
            except Exception:
                pass

    conn.close()

    # Also remove matching scan_folders entries
    folders_conn = get_scan_folders_conn()
    folders_conn.execute(
        "DELETE FROM scan_folders WHERE folder_path LIKE ?", (like_pattern,)
    )
    folders_conn.commit()
    folders_conn.close()

    return jsonify({"ok": True, "count": count})


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
            # If the thread died but status is still 'active', emit an error so
            # the browser can recover from a stuck "Pause Scan" state.
            if active is None and scan_ref.get("status") in ("scanning", "loading_model", "clustering"):
                scan_ref = dict(scan_ref)
                scan_ref["status"] = "error"
                scan_ref["message"] = "Scan process ended unexpectedly"
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

@app.route("/api/photos/open", methods=["POST"])
def api_open_photo():
    """Open a photo in the system default application (like double-click in Explorer)."""
    path = (request.json or {}).get("path", "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    if not Path(path).is_file():
        return jsonify({"error": "File not found"}), 404
    import subprocess
    ctypes.windll.user32.AllowSetForegroundWindow(-1)  # ASFW_ANY — lets the launched app take focus
    subprocess.Popen(f'start "" "{path}"', shell=True)
    return jsonify({"ok": True})


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

    exclude_patterns = _parse_exclude_patterns(load_settings().get("exclude_patterns", ""))
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
        for root, dirs, names in os.walk(fp):
            if exclude_patterns:
                dirs[:] = [d for d in dirs if not any(
                    fnmatch.fnmatch(d.lower(), pat.lower()) for pat in exclude_patterns)]
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


@app.route("/api/clusters", methods=["POST"])
def api_create_cluster():
    """Create an empty cluster; returns { \"id\": new_id }."""
    conn = get_db()
    next_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM clusters").fetchone()[0] + 1
    conn.execute(
        "INSERT INTO clusters (id, name, birth_year, merged_into, group_id) VALUES (?, NULL, NULL, NULL, NULL)",
        (next_id,),
    )
    conn.commit()
    conn.close()
    return jsonify({"id": next_id})


@app.route("/api/clusters/first-faces", methods=["POST"])
def api_clusters_first_faces():
    """Return one thumb_path per cluster (first face by date/path). Request body: {"cluster_ids": [1,2,...]}."""
    data = request.get_json(silent=True) or {}
    ids = data.get("cluster_ids") or []
    if not ids:
        return jsonify({})
    conn = get_db()
    placeholders = ",".join("?" * len(ids))
    q = f"""
        WITH ordered AS (
            SELECT f.cluster_id, f.thumb_path,
                   ROW_NUMBER() OVER (
                       PARTITION BY f.cluster_id
                       ORDER BY p.photo_date IS NULL, p.photo_date, p.file_path
                   ) AS rn
            FROM faces f
            JOIN photos p ON p.id = f.photo_id
            WHERE f.cluster_id IN ({placeholders})
        )
        SELECT cluster_id, thumb_path FROM ordered WHERE rn = 1
    """
    rows = conn.execute(q, ids).fetchall()
    conn.close()
    return jsonify({str(r["cluster_id"]): r["thumb_path"] for r in rows})


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
    old = conn.execute("SELECT cluster_id FROM faces WHERE id = ?", (fid,)).fetchone()
    old_cluster_id = old[0] if old else None
    conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?",
                 (data["cluster_id"], fid))
    if old_cluster_id is not None:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM faces WHERE cluster_id = ?", (old_cluster_id,)
        ).fetchone()[0]
        if remaining == 0:
            conn.execute(
                "DELETE FROM clusters WHERE id = ? AND merged_into IS NULL",
                (old_cluster_id,)
            )
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
