# novoface — Project Context for Claude Code

> **Maintenance:** This is my own working memory, not user docs. Keep it as
> present-tense **rules with the one-line reason** that stops me from "helpfully"
> undoing them — not a changelog. After any change to architecture, data flow, DB
> schema, API contracts, or a load-bearing design decision, update the relevant
> section. No dated "(fixed …)" entries or "used to → now" stories; state how it
> works now and why it must stay that way.

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

**Hash-duplicate handling:** when a file's hash matches an already-processed file, the
scan does `INSERT OR IGNORE` for the new path — it must NOT `UPDATE` the existing record's
path. Reason: two files with identical content (same photo in two archive folders) would
otherwise alternately orphan each other from the DB on every scan (permanent flip-flop).

**Missing-file pruning — `_prune_missing_photos(conn, folders, scan_ref)` (app.py):**
deletes `photos` (+ their `faces` and thumbnails) whose `file_path` no longer exists on disk,
so Review never shows faces from files that were deleted.
- **Ordering rule (load-bearing):** runs *after* the scan loop finishes, NEVER during/on
  Pause. By then a **moved** file has already had its path updated in-place via the hash-match
  branch (no re-recognition), so only genuinely-deleted files are still missing. Pruning
  earlier would delete moved photos not yet re-encountered.
- **Scope:** only prunes photos under a currently-scanned folder (case-insensitive
  `os.path.normcase` prefix test); photos elsewhere (e.g. unmounted external/network drive)
  are left untouched. Also runs on the `new_count == 0` early-return path (all-known re-scan),
  where there are no new files to match moves against.

### cluster_faces() — face_scan.py:290
Auto-assigns faces to clusters by greedy cosine similarity (ArcFace embeddings).

**Load-bearing invariants (don't break these):**
- Only processes faces with `cluster_id IS NULL` — never reassigns existing assignments
- Seeds cluster representatives from ALL currently-assigned faces (named + unnamed)
- Final UPDATE has `AND cluster_id IS NULL` guard — so concurrent user merges always win
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

Three tabs: **Scanner** (folders, scan settings, progress), **Review** (cluster browsing/naming/merging), **Settings** (theme toggle, logging config, database tools).

**Theming:** Single source of truth per theme — every theme-varying color is a CSS custom property defined exactly once in `:root` (dark) and once in `[data-theme="light"]` (light), at the top of `<style>`. All CSS rules reference these via `var(--…)` only; no hardcoded theme colors live outside the two variable blocks. To change a color in either theme, edit one line in one block. The Settings → Appearance card has a Light/Dark segmented toggle; the choice is persisted in `localStorage["novoface_theme"]` and applied via an inline `<head>` script before first paint to avoid flash.

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

### Development run
```
python app.py          # uses ./face_data, opens browser automatically
```

### Packaged app run
```
python main.py         # uses %LOCALAPPDATA%\novoface\, opens pywebview window
```

### Build steps

**Build only from the project venv `venv-win`.** PyInstaller bundles from the Python it
runs under, so the build env must hold PyInstaller **+ every runtime dep at the
`requirements-lock.txt` versions** — `venv-win` is that curated environment. Building from
any other Python silently ships that env's packages instead. Two guardrails enforce this:
- `build.ps1` aborts unless `$env:VIRTUAL_ENV` is `venv-win`, and it invokes
  `python -m PyInstaller` (NOT bare `pyinstaller`) so it can never fall back to a global
  `pyinstaller.exe` that would bundle a different environment.
- PyInstaller + its deps (altgraph, pefile, pyinstaller-hooks-contrib, pywin32-ctypes) are in
  `requirements.txt`/`requirements-lock.txt`, so `pip install -r requirements-lock.txt`
  provisions the build toolchain into venv-win.

One-time prereqs (Inno Setup `--location` is required — without it winget installs to an
unresolvable path):
```powershell
winget install --id JRSoftware.InnoSetup --location "C:\Program Files\Inno Setup 6" --accept-package-agreements --accept-source-agreements
```
Build:
```powershell
.\venv-win\Scripts\Activate.ps1       # required — build.ps1 aborts otherwise
pip install -r requirements-lock.txt  # first run / after a pull; add `pip install openvino` for GPU
.\installer\build.ps1                 # → installer/Output/novoface-<ver>-setup.exe
```
`build.ps1` runs: `python version.py` → `python -m PyInstaller novoface.spec` →
manifest generation → `ISCC installer\novoface.iss`.

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
- Always smoke-test a fresh build (install → launch → run a scan on both backends). insightface/onnxruntime/numpy/scipy native pieces need explicit `collect_all` (already in the spec); a missing one only surfaces at runtime.
- `openvino` is collected with `try/except` — if absent from the build env the bundle is CPU-only and the GPU option is hidden. To include GPU support, `pip install openvino` into venv-win before building.
- OpenVINO models (~262 MB) are **not** bundled — downloaded at first run via `urllib.request` (stdlib, no extra deps).

**Bundle rules — file count matters (install-speed constraint):** endpoint anti-ransomware
(Check Point) throttles file-creation bursts — the installer stalls **~15 s after every ~100
files written**, so install time is dominated by *loose file count*. Every rule below either
keeps the app importable or trims dead files; don't casually reverse them.

- **`collect_all("numpy")` and `collect_all("scipy")` are required.** Both spread their runtime
  across pure-Python submodules (e.g. `numpy._core._exceptions`) and compiled `.pyd` that
  PyInstaller's transitive analysis under-collects. Symptoms of a missing piece: numpy →
  *"Importing the numpy C-extensions failed … No module named 'numpy._core._exceptions'"* at
  startup (Flask thread dies on `import cv2 → import numpy`, so the server never binds and
  `main.py`'s `_wait_for_flask` reports the misleading **"Flask server did not start within
  15.0s"**); scipy → *"The scipy install you are using seems to be broken (extension modules
  cannot be imported)."*
- **UPX is off** (`upx=False` in `EXE` and `COLLECT`) — **keep it off.** UPX compresses
  `.pyd`/`.dll` non-deterministically and corrupts numpy/scipy extensions, producing the scipy
  "broken install" error above; it "works" on one build and breaks the next. Only saves size.
- **matplotlib is excluded and stubbed** via runtime hook `installer/pyi_rth_mpl_stub.py`
  (registered in the spec's `runtime_hooks`). insightface's `__init__` eagerly imports
  `app.mask_renderer → thirdparty.face3d.mesh.vis`, which unconditionally does
  `import matplotlib.pyplot`; we never use 3D mesh rendering and won't bundle matplotlib
  (hundreds of files). The hook injects stub `matplotlib`/`mpl_toolkits` into `sys.modules`
  before app code runs so the import resolves (the stub is only imported, never called). This
  is the repo-owned fix — do NOT rely on hand-patching `site-packages/insightface`, which
  doesn't survive a venv rebuild.
- **Dead-weight trims:**
  - `_is_numeric_dead_weight` drops build artifacts (`.a .lib .h .pxd .pyx .tp .build`), type
    stubs (`.pyi`), and `*/tests/*` from numpy/scipy/sklearn/skimage *datas*; `*.tests` are also
    in `excludes`. **Only DATA is filtered — compiled extensions live in *binaries* and are never
    touched, so the packages stay importable.** Gotcha: PyInstaller data tuples are
    `(source_file, dest_DIRECTORY)`; the filter rebuilds the full path from the *source* filename
    to test the extension — testing `dest` (a folder) alone silently disables the trim.
  - `_ov_is_dead_weight` drops `openvino/tools/` (mo/ovc/benchmark) and `openvino/include/`
    (C++ headers) — we only use `import openvino as ov` + `Core()`/model loading. Keeps
    `openvino/{libs,runtime,frontend}`. Relax it if a feature ever needs the OV CLI tools.
  - `_is_tcl_dead_weight` drops the Tcl `tzdata/` and `msgs/` trees from `a.datas` (tkinter is
    used only for English first-run dialogs). Keeps core Tcl + Tk widgets + encodings.

### Installer incremental copy — skip by CONTENT HASH, not size
`build.ps1` writes `_filemanifest.txt` as `path|size|hash` (SHA1) per file. `novoface.iss`
`ShouldCopyFile()` skips rewriting a file only when the previous install's `size|hash` matches
exactly. **Never reduce this to size-only:** a file whose content changed at the same byte size
(e.g. a corrupt `.pyd` vs. a clean rebuild) must be rewritten, or a "reinstall to fix it"
silently keeps the bad file. (Size/hash comparison is in-memory, no per-file stat — see AV note
below.)

### ⚠ Bulk file deletion MUST go through cmd.exe (a trusted process)
Endpoint anti-ransomware (Check Point) throttles **deletes** from the unknown setup/uninstaller
process at ~150 ms each — 1000+ files = minutes of hang. cmd.exe is trusted and deletes at native
speed. Every bulk-delete path routes through it; if you add another, do the same (never loop
`DeleteFile()`/`unlink()`/`shutil.rmtree` over more than a handful on Windows):
1. **Uninstall** — `novoface.iss` `CurUninstallStepChanged()` → `cmd /c rmdir /s /q "…\_internal"`
   before Inno's own file removal.
2. **Installer stale-file cleanup** — `DeleteStaleFiles()` (at `ssInstall`) diffs old-vs-new
   manifest in memory, writes the stale paths to a temp `.bat` of `del /f /q` lines, runs it via
   `cmd /c`. (A version bump that drops many files can make 1000+ files stale.)
3. **App: Reset Database** — `app.py` `_fast_rmtree()` wipes the whole thumbs dir (one thumbnail
   per face → tens of thousands on a big archive) via `cmd /c rmdir /s /q`, falling back to
   `shutil.rmtree` on non-Windows / locked files.

Smaller app-side deletions (scan prune, delete-folder-by-prefix) use plain per-file `unlink` —
deliberately simple; they're low-volume.

---

## Known Issues / Design Notes

- Orphan child processes (e.g. OpenVINO workers) are cleaned up via `atexit` in app.py.
- `cluster_groups` and `scan_folders` live only in `faces.db` (the CPU DB), even when
  the active backend is OpenVINO. `get_scan_folders_conn()` always returns a CPU DB connection.
- When running as a packaged app, the CWD-relative `./face_data` fallback in `face_scan.py`
  is never reached because `main.py` always sets `NOVOFACE_DATA_DIR` before Flask starts.
