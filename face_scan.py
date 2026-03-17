"""
face_scan.py — Scan a photo archive, detect faces, compute embeddings, extract dates.

Uses InsightFace (ArcFace) for face detection and recognition.

Usage:
    python face_scan.py "D:\\Photo Archive"
    python face_scan.py "D:\\Photo Archive" --det-size 320   (faster, default)
    python face_scan.py "D:\\Photo Archive" --det-size 640   (more accurate on group photos)
    python face_scan.py "D:\\Photo Archive" --model buffalo_s   (faster CPU, less accurate)

GPU: NVIDIA only — pip install onnxruntime-gpu (requires CUDA)

Output:
    face_data/faces.db  — SQLite database with face index
    face_data/thumbs/   — Small face thumbnail crops for quick review

The scan is resumable: re-running skips already-processed files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ExifTags

def _read_data_dir() -> Path:
    """Read data directory from gedface_config.json if present, else use CWD-relative default."""
    cfg = Path(__file__).parent / "gedface_config.json"
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text(encoding="utf-8")).get("data_dir")
            if d:
                return Path(d)
        except Exception:
            pass
    return Path("face_data")

DATA_DIR = _read_data_dir()
DB_PATH = DATA_DIR / "faces.db"
THUMB_DIR = DATA_DIR / "thumbs"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
THUMB_SIZE = 150
FACE_MARGIN = 0.5
SQLITE_BUSY_TIMEOUT_MS = 30000


def connect_db(db_path: Path) -> sqlite3.Connection:
    """Open SQLite with settings that tolerate concurrent UI + scan access."""
    conn = sqlite3.connect(str(db_path), timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = connect_db(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            file_hash TEXT,
            photo_date TEXT,
            date_source TEXT,
            file_size INTEGER,
            processed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            photo_id INTEGER NOT NULL REFERENCES photos(id),
            face_index INTEGER NOT NULL,
            top INTEGER, right_ INTEGER, bottom INTEGER, left_ INTEGER,
            encoding BLOB NOT NULL,
            thumb_path TEXT,
            cluster_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY,
            name TEXT,
            birth_year INTEGER,
            merged_into INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
    """)
    _migrate_db(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_hash ON photos(file_hash)")
    conn.commit()
    return conn


def _migrate_db(conn: sqlite3.Connection):
    """Add columns that may be missing in databases created by older versions."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(photos)").fetchall()}
    if "file_hash" not in cols:
        conn.execute("ALTER TABLE photos ADD COLUMN file_hash TEXT")


def compute_file_hash(file_path: Path) -> str:
    """SHA-256 of file contents. Reads in 64KB chunks to limit memory."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# Same regex for 4-digit years (1890–2040) in path/filename.
_YEAR_PATTERN = re.compile(r'(?<!\d)(189\d|19\d{2}|20[0-3]\d|2040)(?!\d)')


def _years_in_string(s: str) -> list[int]:
    """Return list of valid 4-digit years found in string (order of appearance)."""
    current_year = datetime.now().year
    out = []
    for m in _YEAR_PATTERN.findall(s):
        y = int(m)
        if 1890 <= y <= min(2040, current_year + 5):
            out.append(y)
    return out


def _is_grayscale_image(img, max_side: int = 200) -> bool:
    """True if image appears B&W (very low color saturation). Uses a small resize to keep cost low."""
    if img is None or img.size == 0:
        return False
    try:
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_sat = float(hsv[:, :, 1].mean())
        return mean_sat < 25
    except Exception:
        return False


def extract_date(file_path: Path, img=None, exif: dict | None = None) -> tuple:
    """Extract photo date from EXIF, then path/filename heuristics.

    When both a path year (e.g. folder 1960e) and a filename year (e.g. IMG_20170430)
    exist, prefers the path year if it is older than the filename year (digitization
    of an old photo: path = original era, filename = when the photo was scanned).
    If img (BGR) is provided and looks B&W, that reinforces preferring the path year.
    Pass exif from the loader to avoid opening the file again.
    """
    if exif is None:
        try:
            with Image.open(file_path) as pil_img:
                exif = pil_img._getexif()
        except Exception:
            exif = None
    if exif:
        try:
            for tag_id in (36867, 36868, 306):
                val = exif.get(tag_id)
                if val:
                    dt = datetime.strptime(val.strip(), "%Y:%m:%d %H:%M:%S")
                    return dt.strftime("%Y-%m-%d"), "exif"
        except Exception:
            pass

    path_str = str(file_path)
    parent_str = str(file_path.parent)
    name_str = file_path.name

    path_years = _years_in_string(parent_str)
    name_years = _years_in_string(name_str)
    path_year = int(path_years[-1]) if path_years else None
    name_year = int(name_years[-1]) if name_years else None

    if path_year is not None and name_year is not None:
        # Digitization pattern: path = original era, filename = when digitized.
        if path_year < name_year:
            return f"{path_year}", "path"
        if path_year > name_year:
            # If image looks B&W, prefer path year (old photo in a newer-named folder).
            if img is not None and _is_grayscale_image(img):
                return f"{path_year}", "path"
            return f"{name_year}", "filename"
        return f"{path_year}", "path"
    if path_year is not None:
        return f"{path_year}", "path"
    if name_year is not None:
        return f"{name_year}", "filename"
    return None, None


def load_image_cv2(file_path: Path, get_exif: bool = False):
    """Load image as BGR numpy array. If get_exif=True, return (img, exif). Uses cv2 for decode when possible (faster)."""
    try:
        if get_exif:
            img = cv2.imread(str(file_path))
            if img is not None:
                try:
                    with Image.open(file_path) as pil_img:
                        exif = pil_img._getexif()
                except Exception:
                    exif = None
                return img, exif
            pil_img = Image.open(file_path)
            exif = pil_img._getexif()
            pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img, exif
        img = cv2.imread(str(file_path))
        if img is None:
            pil_img = Image.open(file_path)
            pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return (None, None) if get_exif else None


def process_photo(
    file_path: Path,
    conn: sqlite3.Connection,
    face_app: FaceAnalysis,
    file_hash: str | None = None,
) -> int:
    """Process a single photo. Returns number of faces found."""
    out = load_image_cv2(file_path, get_exif=True)
    if out is None:
        return 0
    img, exif = out
    if img is None:
        return 0

    h, w = img.shape[:2]
    max_dim = 2048
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    photo_date, date_source = extract_date(file_path, img=img, exif=exif)

    try:
        detected_faces = face_app.get(img)
    except Exception:
        detected_faces = []

    cursor = conn.execute(
        "INSERT INTO photos (file_path, file_hash, photo_date, date_source, file_size, processed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(file_path), file_hash, photo_date, date_source,
         file_path.stat().st_size, datetime.now().isoformat()),
    )
    photo_id = cursor.lastrowid

    if not detected_faces:
        conn.commit()
        return 0

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for idx, face in enumerate(detected_faces):
        bbox = face.bbox.astype(int)
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        embedding = face.normed_embedding

        face_h = bottom - top
        face_w = right - left
        margin_h = int(face_h * FACE_MARGIN)
        margin_w = int(face_w * FACE_MARGIN)

        crop_top = max(0, top - margin_h)
        crop_bottom = min(h, bottom + margin_h)
        crop_left = max(0, left - margin_w)
        crop_right = min(w, right + margin_w)

        thumb = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        thumb_filename = f"{photo_id}_{idx}.jpg"
        thumb_path = THUMB_DIR / thumb_filename
        thumb.save(str(thumb_path), "JPEG", quality=80)

        conn.execute(
            "INSERT INTO faces (photo_id, face_index, top, right_, bottom, left_, encoding, thumb_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (photo_id, idx, top, right, bottom, left,
             embedding.astype(np.float32).tobytes(), str(thumb_filename)),
        )

    conn.commit()
    return len(detected_faces)


def cluster_faces(
    conn: sqlite3.Connection,
    threshold: float = 0.35,
    progress_cb: Callable[[int, int], None] | None = None,
):
    """Cluster faces that have no assignment yet (cluster_id IS NULL).
	Clusterisation is done by cosine similarity using greedy assignment.

    InsightFace normed_embedding is L2-normalized, so cosine similarity =
    dot product. A typical same-person threshold is ~0.3-0.4.

    Faces already assigned to any cluster — by a previous clustering run or by
    a manual merge in the UI — are never reassigned.  This prevents interim /
    pause-triggered re-clustering from overwriting manual work.

    progress_cb(done, total) is called periodically so UI can show progress.
    """
    print("\nClustering faces...")

    rows = conn.execute("SELECT id, encoding, cluster_id FROM faces ORDER BY id").fetchall()
    if not rows:
        print("No faces to cluster.")
        return

    face_ids  = [r[0] for r in rows]
    encodings = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
    cur_cluster = [r[2] for r in rows]   # current cluster_id; None for new faces
    total = len(encodings)
    report_every = max(1, total // 20)   # ~20 progress updates over the run

    # Build cluster representatives from every face that already has an assignment.
    # This seeds the greedy matcher with the full current state of the DB so that
    # newly-detected faces land in the right person's cluster.
    cluster_reps = []   # list of (cluster_db_id, [encoding indices])
    seen_clusters: dict[int, int] = {}   # cluster_id -> index in cluster_reps
    for i, cid in enumerate(cur_cluster):
        if cid is None:
            continue
        if cid not in seen_clusters:
            seen_clusters[cid] = len(cluster_reps)
            cluster_reps.append((cid, [i]))
        else:
            cluster_reps[seen_clusters[cid]][1].append(i)

    next_cluster_id = (conn.execute("SELECT COALESCE(MAX(id),0) FROM clusters").fetchone()[0]) + 1

    # assignments holds ONLY newly-assigned faces (those that were NULL).
    assignments: dict[int, int] = {}
    new_cluster_ids: list[int] = []

    for i, enc in enumerate(encodings):
        if (i + 1) % report_every == 0 or i == total - 1:
            if progress_cb:
                progress_cb(i + 1, total)
            else:
                print(f"  clustering {i + 1}/{total} faces...", end="\r")

        if cur_cluster[i] is not None:
            continue   # already assigned — never overwrite

        best_cluster = None
        best_sim = threshold

        for cid, indices in cluster_reps:
            centroid = np.mean([encodings[j] for j in indices], axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-10
            sim = np.dot(enc, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = (cid, indices)

        if best_cluster:
            cid, indices = best_cluster
            indices.append(i)
            assignments[face_ids[i]] = cid
        else:
            cluster_reps.append((next_cluster_id, [i]))
            assignments[face_ids[i]] = next_cluster_id
            new_cluster_ids.append(next_cluster_id)
            next_cluster_id += 1

    if not assignments:
        print("  No unassigned faces found.")
        return

    # Build the new assignments in a temp table first so the main DB write lock
    # is held only for the short final apply phase.
    conn.execute("DROP TABLE IF EXISTS temp.cluster_assignments")
    conn.execute("""
        CREATE TEMP TABLE cluster_assignments (
            face_id INTEGER PRIMARY KEY,
            cluster_id INTEGER NOT NULL
        )
    """)
    conn.executemany(
        "INSERT INTO cluster_assignments (face_id, cluster_id) VALUES (?, ?)",
        assignments.items(),
    )
    conn.commit()

    conn.execute("BEGIN IMMEDIATE")
    if new_cluster_ids:
        conn.executemany(
            "INSERT INTO clusters (id) VALUES (?)",
            ((cid,) for cid in new_cluster_ids),
        )
    # Guard: only write to faces that are still unassigned in the DB.
    # If the user merged a face while we were computing, their assignment wins.
    conn.execute("""
        UPDATE faces
        SET cluster_id = (
            SELECT ca.cluster_id
            FROM cluster_assignments ca
            WHERE ca.face_id = faces.id
        )
        WHERE id IN (SELECT face_id FROM cluster_assignments)
          AND cluster_id IS NULL
    """)
    conn.execute("""
        DELETE FROM clusters
        WHERE name IS NULL
          AND merged_into IS NULL
          AND id NOT IN (SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL)
          AND id NOT IN (SELECT DISTINCT cluster_id FROM cluster_assignments)
    """)
    conn.commit()
    conn.execute("DROP TABLE temp.cluster_assignments")
    conn.commit()

    print(f"  {len(assignments)} new faces assigned, {len(cluster_reps)} clusters total")


def _face_providers():
    """Use CUDA (NVIDIA) if available, else CPU. Intel integrated GPUs not supported."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def scan_archive(
    archive_path: Path,
    det_size: int,
    threshold: float = 0.35,
    model_name: str = "buffalo_l",
):
    from insightface.app import FaceAnalysis

    DATA_DIR.mkdir(exist_ok=True)
    THUMB_DIR.mkdir(exist_ok=True)

    conn = init_db(DB_PATH)

    processed_paths = set(
        r[0] for r in conn.execute("SELECT file_path FROM photos").fetchall()
    )
    known_hashes = dict(
        conn.execute("SELECT file_hash, id FROM photos WHERE file_hash IS NOT NULL").fetchall()
    )
    print(f"Already processed: {len(processed_paths)} files ({len(known_hashes)} with hash)")

    print(f"Scanning {archive_path} for photos...")
    photo_files = []
    for root, _, files in os.walk(archive_path):
        for fname in files:
            if Path(fname).suffix.lower() in PHOTO_EXTENSIONS:
                full = Path(root) / fname
                if str(full) not in processed_paths:
                    photo_files.append(full)

    total = len(photo_files)
    print(f"Found {total} new photos to process\n")

    if total == 0:
        print("Nothing new to process.")
        cluster_faces(conn, threshold)
        conn.close()
        return

    providers = _face_providers()
    print(f"Loading face detection model ({model_name}, {providers[0]})...")
    face_app = FaceAnalysis(name=model_name, providers=providers)
    face_app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    print("Model ready.\n")

    total_faces = 0
    start = time.time()
    errors = 0
    skipped = 0

    for i, fpath in enumerate(photo_files):
        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (total - i - 1) / rate if rate > 0 else 0

        print(
            f"\r[{i + 1}/{total}] "
            f"faces: {total_faces} | "
            f"errors: {errors} | "
            f"skipped: {skipped} | "
            f"{rate:.1f} img/s | "
            f"ETA: {remaining / 3600:.1f}h   ",
            end="", flush=True,
        )

        try:
            fhash = compute_file_hash(fpath)
            if fhash in known_hashes:
                conn.execute(
                    "UPDATE photos SET file_path = ? WHERE id = ?",
                    (str(fpath), known_hashes[fhash]),
                )
                conn.commit()
                skipped += 1
                continue
            n = process_photo(fpath, conn, face_app, file_hash=fhash)
            known_hashes[fhash] = conn.execute(
                "SELECT id FROM photos WHERE file_path = ?", (str(fpath),)
            ).fetchone()[0]
            total_faces += n
        except KeyboardInterrupt:
            print("\n\nInterrupted! Progress saved. Re-run to resume.")
            conn.close()
            sys.exit(0)
        except Exception:
            errors += 1
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO photos (file_path, processed_at) VALUES (?, ?)",
                    (str(fpath), datetime.now().isoformat()),
                )
                conn.commit()
            except Exception:
                pass

    elapsed = time.time() - start
    print(f"\n\nDone! {total} photos in {elapsed / 3600:.1f}h")
    print(f"  Faces found: {total_faces}")
    print(f"  Errors: {errors}")

    cluster_faces(conn, threshold)
    conn.close()
    print(f"\nDatabase saved: {DB_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan a photo archive for faces and cluster them."
    )
    parser.add_argument("archive", type=Path, help="Path to the photo archive folder")
    parser.add_argument(
        "--det-size", type=int, default=320,
        help="Detection size (320=fast, 640=accurate, default 320)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Clustering similarity threshold (higher=stricter, default 0.35)"
    )
    parser.add_argument(
        "--model", choices=("buffalo_l", "buffalo_s"), default="buffalo_l",
        help="Model: buffalo_l=accurate (default), buffalo_s=faster on CPU"
    )
    args = parser.parse_args()

    if not args.archive.is_dir():
        print(f"Error: {args.archive} is not a directory")
        sys.exit(1)

    scan_archive(args.archive, args.det_size, args.threshold, model_name=args.model)


if __name__ == "__main__":
    main()
