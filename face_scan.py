"""
face_scan.py — Scan a photo archive, detect faces, compute embeddings, extract dates.

Uses InsightFace (ArcFace) for face detection and recognition.

Usage:
    python face_scan.py "D:\\Photo Archive"
    python face_scan.py "D:\\Photo Archive" --det-size 320   (faster, default)
    python face_scan.py "D:\\Photo Archive" --det-size 640   (more accurate on group photos)

Output:
    face_data/faces.db  — SQLite database with face index
    face_data/thumbs/   — Small face thumbnail crops for quick review

The scan is resumable: re-running skips already-processed files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

DATA_DIR = Path("face_data")
DB_PATH = DATA_DIR / "faces.db"
THUMB_DIR = DATA_DIR / "thumbs"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
THUMB_SIZE = 150
FACE_MARGIN = 0.5


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
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


def extract_date(file_path: Path) -> tuple:
    """Extract photo date from EXIF, then fall back to folder/filename heuristics."""
    try:
        with Image.open(file_path) as img:
            exif = img._getexif()
            if exif:
                for tag_id in (36867, 36868, 306):
                    val = exif.get(tag_id)
                    if val:
                        dt = datetime.strptime(val.strip(), "%Y:%m:%d %H:%M:%S")
                        return dt.strftime("%Y-%m-%d"), "exif"
    except Exception:
        pass

    path_str = str(file_path)
    current_year = datetime.now().year
    year_matches = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', path_str)
    if year_matches:
        year = int(year_matches[-1])
        if 1850 <= year <= current_year:
            return f"{year}", "path"

    return None, None


def load_image_cv2(file_path: Path):
    """Load image as BGR numpy array, handling EXIF orientation."""
    try:
        img = cv2.imread(str(file_path))
        if img is None:
            # Try PIL for formats cv2 can't handle
            pil_img = Image.open(file_path)
            pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return None


def process_photo(
    file_path: Path,
    conn: sqlite3.Connection,
    face_app: FaceAnalysis,
    file_hash: str | None = None,
) -> int:
    """Process a single photo. Returns number of faces found."""
    img = load_image_cv2(file_path)
    if img is None:
        return 0

    h, w = img.shape[:2]
    max_dim = 2048
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    photo_date, date_source = extract_date(file_path)

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


def cluster_faces(conn: sqlite3.Connection, threshold: float = 0.35):
    """Cluster all faces by cosine similarity using greedy assignment.

    InsightFace normed_embedding is L2-normalized, so cosine similarity =
    dot product. A typical same-person threshold is ~0.3-0.4.
    """
    print("\nClustering faces...")

    rows = conn.execute("SELECT id, encoding FROM faces ORDER BY id").fetchall()
    if not rows:
        print("No faces to cluster.")
        return

    face_ids = [r[0] for r in rows]
    encodings = [np.frombuffer(r[1], dtype=np.float32) for r in rows]

    # Preserve manually named clusters and their assignments
    named = conn.execute(
        "SELECT id, name, birth_year FROM clusters WHERE name IS NOT NULL"
    ).fetchall()
    named_map = {r[0]: {"name": r[1], "birth_year": r[2]} for r in named}

    named_face_rows = conn.execute(
        "SELECT f.id, f.cluster_id, f.encoding FROM faces f "
        "JOIN clusters c ON c.id = f.cluster_id "
        "WHERE c.name IS NOT NULL"
    ).fetchall()
    pinned = {r[0]: r[1] for r in named_face_rows}

    # Clear old clustering
    conn.execute("UPDATE faces SET cluster_id = NULL")
    conn.execute("DELETE FROM clusters WHERE name IS NULL AND merged_into IS NULL")
    conn.commit()

    cluster_reps = []  # list of (cluster_db_id, [list of encoding indices])
    assignments = {}

    # Seed with named clusters
    for cid, info in named_map.items():
        indices = [i for i, fid in enumerate(face_ids) if fid in pinned and pinned[fid] == cid]
        if indices:
            cluster_reps.append((cid, indices))
            for idx in indices:
                assignments[face_ids[idx]] = cid

    next_cluster_id = (conn.execute("SELECT COALESCE(MAX(id),0) FROM clusters").fetchone()[0]) + 1

    for i, enc in enumerate(encodings):
        if face_ids[i] in assignments:
            continue

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
            conn.execute("INSERT INTO clusters (id) VALUES (?)", (next_cluster_id,))
            cluster_reps.append((next_cluster_id, [i]))
            assignments[face_ids[i]] = next_cluster_id
            next_cluster_id += 1

    for face_id, cluster_id in assignments.items():
        conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cluster_id, face_id))
    conn.commit()

    print(f"  {len(encodings)} faces -> {len(cluster_reps)} clusters")


def scan_archive(archive_path: Path, det_size: int, threshold: float = 0.35):
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

    print("Loading face detection model (first run downloads ~300MB)...")
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
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
    args = parser.parse_args()

    if not args.archive.is_dir():
        print(f"Error: {args.archive} is not a directory")
        sys.exit(1)

    scan_archive(args.archive, args.det_size, args.threshold)


if __name__ == "__main__":
    main()
