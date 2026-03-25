# novoface — Project Context for Claude Code

*Last updated: 2026-03-24*

> **For Claude:** Keep this file up to date. After every change that affects
> architecture, data flow, DB schema, API contracts, or key design decisions,
> update the relevant section(s) before finishing the task.

Face recognition tool for genealogy photo archives.
Scans a photo collection, detects faces, groups them into person-clusters,
and lets the user review and correct the groupings in a browser UI.

---

## Key Files

| File | Role |
|---|---|
| `main.py` | Desktop launcher — pywebview window, first-run setup dialog, starts Flask |
| `app.py` | Flask server — all API routes, scan management, CPU/IO throttling |
| `face_review_ui.html` | Single-page browser UI (HTML + vanilla JS, one large file) |
| `face_scan.py` | Photo scanning, face detection, DB schema, `cluster_faces()` |
| `openvino_pipeline.py` | Intel GPU (Iris Xe) backend; imports `cluster_faces` from face_scan |
| `novoface.spec` | PyInstaller build spec → `dist/novoface/novoface.exe` |
| `installer/novoface.iss` | Inno Setup 6 script → `installer/Output/novoface-setup.exe` |

---

## Data Storage

- **Database**: SQLite, WAL mode, 30 s busy timeout
- **Active backend** controlled by `backend.json` → `{"backend": "openvino"}`
- CPU backend DB: `faces.db`
- OpenVINO backend DB: `faces_ov.db`
- Thumbnails: `thumbs/` (CPU) or `thumbs_ov/` (OpenVINO)

### Data directory resolution (priority order)

1. `NOVOFACE_DATA_DIR` env var — set by `main.py` to the OS user-data dir:
   - Windows: `%LOCALAPPDATA%\novoface\`  (e.g. `C:\Users\Alice\AppData\Local\novoface\`)
   - Linux:   `~/.local/share/novoface/`
2. `novoface_config.json` `"data_dir"` key — power-user / CI override
3. `./face_data` relative to CWD — legacy dev workflow (`python app.py`)

When launched via `main.py` (packaged app), option 1 is always active.
When launched via `python app.py` directly (dev), option 3 is used — behaviour unchanged.

### Schema

```
photos   : id, file_path, file_hash, photo_date, date_source, file_size, processed_at
faces    : id, photo_id, face_index, top/right_/bottom/left_, encoding BLOB (512-dim f32), thumb_path, cluster_id
clusters : id, name (NULL=unnamed), birth_year, merged_into (NULL=active), group_id
cluster_groups : id, name
scan_folders   : id, folder_path, added_at  ← in faces.db only (always the CPU/main DB)
```

---

## Core Flows

### Scan
`POST /api/scan/start` → `_run_scan()` / `_run_scan_openvino()` in a daemon thread.
- Calls `process_photo()` for each new file → inserts into `photos` + `faces`
- Every 200 photos: interim `cluster_faces(conn, threshold)`
- On Pause (`_scan_stop` event): final `cluster_faces()` then exits
- CPU capped via `_apply_cpu_limit(cpu_percent)` — Windows Job Object HARD_CAP
- I/O priority lowered via `THREAD_MODE_BACKGROUND_BEGIN` on the scan thread

**Hash-duplicate handling (fixed 2026-03-11):**
- When a file's hash matches an already-processed file, the scan does `INSERT OR IGNORE`
  for the new path instead of `UPDATE`-ing the existing record's path.
- This prevents a permanent flip-flop where two files with the same content (e.g. the same
  photo in two archive folders) would alternately orphan each other from the DB on every scan.

### cluster_faces() — face_scan.py:290
Auto-assigns faces to clusters by greedy cosine similarity (ArcFace embeddings).

**Critical design rule (fixed 2026-03-10):**
- Only processes faces with `cluster_id IS NULL` — never reassigns existing assignments
- Seeds cluster representatives from ALL currently-assigned faces (named + unnamed)
- Final UPDATE has `AND cluster_id IS NULL` guard — concurrent user merges always win
- Runs on: scan start (if no clusters), every 200 photos, Pause Scan, scan completion

### Merge (UI → DB)
`POST /api/clusters/merge` → `api_merge_clusters()` (app.py near line 1010):
```sql
UPDATE faces    SET cluster_id = target_id WHERE cluster_id = source_id
UPDATE clusters SET merged_into = target_id WHERE id = source_id
```
Committed immediately. Source cluster is kept (archived via `merged_into`) for history.

### Face Move
`POST /api/faces/<id>/move` → `api_move_face()`: moves a single face to another cluster.

---

## Two Backends

| | CPU (InsightFace) | OpenVINO (Intel Iris Xe) |
|---|---|---|
| DB | `faces.db` | `faces_ov.db` |
| Thumbs | `thumbs/` | `thumbs_ov/` |
| Entry | `_run_scan()` | `_run_scan_openvino()` |

`get_db()` reads `backend.json` on every call — no restart needed to switch.
`get_scan_folders_conn()` always uses `faces.db` (scan_folders + cluster_groups live there).

---

## API Routes (app.py)

| Endpoint | Purpose |
|---|---|
| `GET  /api/clusters` | List active clusters with face counts |
| `POST /api/clusters` | Create empty cluster |
| `PUT  /api/clusters/<id>` | Rename / set birth_year |
| `POST /api/clusters/merge` | Merge source into target |
| `POST /api/clusters/first-faces` | Batch fetch one thumbnail per cluster |
| `GET  /api/clusters/<id>/faces` | All faces in a cluster |
| `POST /api/faces/<id>/move` | Move face to cluster |
| `GET  /api/scan/status` | SSE stream of scan progress |
| `POST /api/scan/start` | Start scan |
| `POST /api/scan/stop` | Pause scan (triggers cluster_faces on stop) |
| `POST /api/scan/cpu` | Adjust CPU cap live |
| `GET  /api/groups` | List cluster_groups |
| `POST /api/groups` | Create group |
| `PUT  /api/clusters/<id>/group` | Assign cluster to group |
| `GET  /api/log/settings` | Return `{enabled, max_mb, path}` |
| `POST /api/log/settings` | Update log settings + apply live |
| `POST /api/log/clear` | Truncate log file |
| `POST /api/log/open` | Open log in system default viewer |

---

## UI (face_review_ui.html) — Tabs

Three tabs: **Scanner** (folders, scan settings, progress), **Review** (cluster browsing/naming/merging), **Settings** (logging config, database tools).

### Key JS Functions

| Function | Purpose |
|---|---|
| `loadClusters()` | Fetch cluster list + group data, rebuild sidebar |
| `selectCluster(id)` | Load + render faces for a cluster |
| `showMergeModal()` | Show sorted list of merge targets |
| `doMerge(src, tgt)` | POST merge, refresh UI |
| `loadStats()` | Update photo/face/cluster counts in header |
| `loadLogSettings()` | Fetch log config from `/api/log/settings`, populate Settings tab |
| `saveLogSettings()` | POST updated log config; applied live via `reconfigure_logging()` in main.py |

---

## Desktop Packaging (Windows)

### Development run (unchanged)
```
python app.py          # uses ./face_data, opens browser automatically
```

### Packaged app run
```
python main.py         # uses %LOCALAPPDATA%\novoface\, opens pywebview window
```

### Build steps

Install prerequisites once (Inno Setup `--location` is required — without it winget
installs to an unresolvable path):
```powershell
pip install pyinstaller pywebview platformdirs   # add: pip install openvino  for GPU support
winget install --id JRSoftware.InnoSetup --location "C:\Program Files\Inno Setup 6" --accept-package-agreements --accept-source-agreements
```

Then build (or just run `.\installer\build.ps1`):
```powershell
python version.py           # → installer/version.iss + installer/version_info.txt
pyinstaller novoface.spec   # → dist/novoface/
& "C:\Program Files\Inno Setup 6\ISCC.exe" installer\novoface.iss
# → installer/Output/novoface-0.0.1-setup.exe
```

### First-run setup dialog
On first launch (no database found in data dir), `main.py` shows a tkinter dialog:
- **Data location** field (read-only) — shows the platformdirs path
- **Import existing face_data** field + Browse button — optional, user-provided path
- **Enable GPU acceleration** checkbox — shown only when `openvino` is bundled **and** an Intel Iris Xe / Arc GPU is detected via `wmic`. Pre-checked by default.
- **Start Fresh** / **Import & Start** buttons

If the user provides an import path, all files from that folder are copied into the data dir before Flask starts.

If the GPU checkbox is checked, a second progress dialog downloads the OpenVINO face models (~262 MB total) and writes `backend.json = {"backend": "openvino"}` so the GPU backend is pre-selected when the UI opens. On download failure, a warning is shown and the app falls back to CPU.

The setup screen never appears again once a database exists.

### PyInstaller notes
- `face_review_ui.html` is included as a data file; `app.py`'s `_base_dir()` resolves it correctly both frozen and non-frozen via `sys._MEIPASS`.
- `console=False` in the spec suppresses the black CMD window.
- First build always requires manual testing — insightface/onnxruntime native DLLs sometimes need explicit `--collect-all` entries (already in the spec).
- `openvino` is collected with `try/except` — if not installed in the build environment the bundle is CPU-only and the GPU option is hidden in the setup dialog. To include GPU support: `pip install openvino` before running `pyinstaller novoface.spec`.
- OpenVINO models (~262 MB) are **not** bundled — downloaded at first run via `urllib.request` (standard library, no extra deps).

---

## Known Issues / Design Notes

- Orphan child processes (e.g. OpenVINO workers) are cleaned up via `atexit` in app.py.
- `cluster_groups` and `scan_folders` live only in `faces.db` (the CPU DB), even when
  the active backend is OpenVINO. `get_scan_folders_conn()` always returns a CPU DB connection.
- When running as a packaged app, the CWD-relative `./face_data` fallback in `face_scan.py`
  is never reached because `main.py` always sets `NOVOFACE_DATA_DIR` before Flask starts.
