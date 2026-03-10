# gedface — Project Context for Claude Code

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
| `app.py` | Flask server — all API routes, scan management, CPU/IO throttling |
| `face_review_ui.html` | Single-page browser UI (HTML + vanilla JS, one large file) |
| `face_scan.py` | Photo scanning, face detection, DB schema, `cluster_faces()` |
| `openvino_pipeline.py` | Intel GPU (Iris Xe) backend; imports `cluster_faces` from face_scan |

---

## Data Storage

- **Database**: SQLite, WAL mode, 30 s busy timeout
- **Active backend** controlled by `face_data/backend.json` → `{"backend": "openvino"}`
- CPU backend DB: `face_data/faces.db`
- OpenVINO backend DB: `face_data/faces_ov.db`
- Thumbnails: `face_data/thumbs/` (CPU) or `face_data/thumbs_ov/` (OpenVINO)
- ⚠️ All paths are **relative to CWD** — server must be started from project root

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

---

## UI (face_review_ui.html) — Key JS Functions

| Function | Purpose |
|---|---|
| `loadClusters()` | Fetch cluster list + group data, rebuild sidebar |
| `selectCluster(id)` | Load + render faces for a cluster |
| `showMergeModal()` | Show sorted list of merge targets |
| `doMerge(src, tgt)` | POST merge, refresh UI |
| `loadStats()` | Update photo/face/cluster counts in header |

---

## Known Issues / Design Notes

- `DATA_DIR = Path("face_data")` in face_scan.py is relative to CWD — if the server
  is started from the wrong directory a fresh empty DB is created silently.
- Orphan child processes (e.g. OpenVINO workers) are cleaned up via `atexit` in app.py.
- `cluster_groups` and `scan_folders` live only in `faces.db` (the CPU DB), even when
  the active backend is OpenVINO. `get_scan_folders_conn()` always returns a CPU DB connection.
