# gedface — Face recognition tool for genealogy photo archives

Scans a photo archive, detects and clusters faces, then provides a local web UI
to review clusters, name people, and quickly copy file paths for use with the
gedcom tree app.

No files are copied or moved — the tool generates small thumbnails for browsing
and references originals by path only. Scanning is fully resumable: you can stop
at any time and pick up exactly where you left off, even across reboots. If photos
are moved to a new location, the scanner recognises them by SHA-256 hash and
updates the stored path without reprocessing.

## Files

| File | Description |
|------|-------------|
| `app.py` | Unified Flask server — background scan thread, SSE progress, cluster review APIs |
| `face_scan.py` | CLI scanner — InsightFace detection, embedding extraction, greedy centroid clustering |
| `face_review.py` | Legacy standalone review server (use `app.py` instead) |
| `face_review_ui.html` | Dark-themed SPA — Scanner tab (folder management, live progress) and Review tab (cluster browsing, naming, merging, click-to-copy) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Unified web app (recommended)

```bash
python app.py                  # start and open browser
python app.py --no-browser     # start without opening a browser
python app.py --port 9000      # use a custom port
```

Opens a local web UI at http://localhost:8050 (by default) with two tabs:

**Scanner tab** — add archive folders, start/stop scans, and watch live progress:
- Add one or more photo folders (they're scanned recursively)
- Choose detection size (320 = fast, 640 = accurate for group photos)
- Set clustering similarity threshold (default 0.35, higher = stricter)
- Start a scan — progress updates live in the browser
- Stop and resume at any time; already-processed photos are skipped
- Moved files are detected by content hash and paths are updated automatically
- **Reset DB** button wipes all data so you can start a completely fresh scan

**Review tab** — browse and manage face clusters:
- Browse auto-detected face clusters sorted by size
- Name each cluster (person) and set birth year for auto-computed ages
- Merge clusters that are the same person at different ages
- Click a thumbnail to copy the file's absolute path to clipboard

### CLI scanner (alternative)

```bash
python face_scan.py "D:\Photo Archive"
python face_scan.py "D:\Photo Archive" --det-size 640 --threshold 0.30
```

Runs the scan from the command line with terminal progress output.
The scan is resumable — already-processed files are skipped on re-run.

### Standalone review server (legacy)

```bash
python face_review.py
```

Starts only the review UI without scan management. Use `app.py` instead.
