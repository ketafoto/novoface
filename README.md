# novoface — Face recognition tool for genealogy photo archives

*Last updated: 2026-03-24*

Scans a photo archive, detects and clusters faces, then provides a local web UI
to review clusters, name people, and quickly copy file paths for use with the
novotree app.

No files are copied or moved — the tool generates small thumbnails for browsing
and references originals by path only. Scanning is fully resumable: you can stop
at any time and pick up exactly where you left off, even across reboots. If photos
are moved to a new location, the scanner recognizes them by SHA-256 hash and
updates the stored path without reprocessing.

The database and thumbnails are stored in a single folder (`%LOCALAPPDATA%\novoface\`
on Windows, `~/.local/share/novoface/` on Linux). The folder is fully self-contained
and portable — to move novoface to another machine, or to back it up, simply copy
that folder. To start over, delete it (or use the **Reset DB** button in the Scanner
tab, which wipes only the scan data without touching the app).

---

## Windows — Installer

1. Download `novoface-setup.exe` from the releases page and run it.
2. Follow the installer wizard (no administrator rights required).
   - A **desktop shortcut** is created by default (uncheck if not wanted).
   - A **Start Menu entry** is always created.
3. Double-click the **novoface** shortcut — no terminal, no `python` command needed.
4. To **pin to the taskbar**: right-click the Start Menu entry → *Pin to taskbar*.
   Windows prevents installers from doing this automatically.
5. To **quit**: simply close the novoface window. The background server shuts down
   automatically — no terminal to close, nothing left running.

On first launch a setup screen appears:

- **Your data will be stored at** — shows where novoface will keep its database
  and thumbnails (`%LOCALAPPDATA%\novoface\` by default, read-only field).
- **Import existing face_data** — if you have a previous `face_data/` folder from
  an earlier development install, click **Browse…**, select it, then click
  **Import & Start**. Otherwise click **Start Fresh**.
- **Enable GPU acceleration** — if an Intel Iris Xe or Arc GPU is detected on your
  machine, this option appears pre-checked. Leaving it checked downloads the OpenVINO
  face models (~260 MB, one-time) and sets Intel GPU as the default scan engine.
  Uncheck it to skip and use CPU processing instead.

This screen appears only once. To finish, simply close the window — the server
shuts down automatically.

---

## Development setup

```bash
pip install -r requirements.txt
```

### Run (development)

```bash
python app.py                  # start and open browser at http://localhost:8050
python app.py --no-browser     # start without opening a browser
python app.py --port 9000      # use a custom port
```

Data is stored in `./face_data/` relative to CWD when running this way.

### Run (desktop launcher, uses platform data dir)

```bash
python main.py
```

Starts Flask, then opens the UI in a native pywebview window.
Data is stored in `%LOCALAPPDATA%\novoface\` (Windows) or `~/.local/share/novoface/` (Linux).
Closing the window stops the server.

---

## Files

| File | Description |
|------|-------------|
| `main.py` | Desktop launcher — pywebview window, first-run setup dialog, starts Flask |
| `app.py` | Unified Flask server — background scan thread, SSE progress, cluster review APIs |
| `face_scan.py` | Core scanner — InsightFace detection, embedding extraction, greedy centroid clustering |
| `face_review_ui.html` | Dark-themed SPA — Scanner tab (folder management, live progress) and Review tab (cluster browsing, naming, merging) |
| `requirements.txt` | Python dependencies |
| `novoface.spec` | PyInstaller build spec |
| `installer/novoface.iss` | Inno Setup 6 installer script |

---

## UI overview

Opens a local web UI with two tabs:

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

---

## Scan and clustering

Scan and clustering run in the **same worker thread**, one after the other. The
scan does not wait for "initial clustering": it loads the model, finds new photos,
then either (a) if there are no new photos it re-clusters existing faces and exits,
or (b) it processes each photo, runs interim clustering every 200 photos, then runs
final clustering at the end. Clustering is not run in parallel with the scan. On
Windows, the clustering phase runs at below-normal thread priority so the UI stays
responsive.

---

## Speed and GPU (large archives)

On CPU, scanning can be slow (~10 s/image with `buffalo_l`). For ~100K photos:

- **NVIDIA GPU** — Install [CUDA](https://developer.nvidia.com/cuda-downloads), then:
  ```bash
  pip uninstall onnxruntime
  pip install onnxruntime-gpu
  ```
  The app will use CUDA automatically.

- **Intel Iris Xe / Arc (experimental)** — If you installed via the Windows
  installer, GPU support is set up automatically during first launch (see above) —
  no manual steps needed.

  For the development workflow, OpenVINO's dependencies can downgrade numpy/networkx
  and may conflict with other packages, so use a **separate virtual environment**:
  ```bash
  python -m venv .venv-openvino
  .venv-openvino\Scripts\activate   # Windows
  pip install -r requirements.txt
  pip install openvino
  python scripts/download_openvino_models.py
  python app.py
  ```
  The script downloads the two face models over HTTP. In the Scanner tab, set
  **Backend** to **GPU Iris Xe (OpenVINO)**. Uses a separate DB (`faces_ov.db`)
  and thumbs (`thumbs_ov/`). Switch back to **CPU (InsightFace)** to restore and
  resume your normal scan.

- **Faster CPU model** — Use `--model buffalo_s` for quicker CPU scans (some accuracy
  tradeoff, CLI only).

---

## CLI scanner (alternative to the web app)

```bash
python face_scan.py "D:\Photo Archive"
python face_scan.py "D:\Photo Archive" --det-size 640 --threshold 0.30
python face_scan.py "D:\Photo Archive" --model buffalo_s
```

Runs the scan from the command line with terminal progress output.
The scan is resumable — already-processed files are skipped on re-run.
Data is always written to `./face_data/` when using the CLI directly.

---

## Building the Windows installer

### 1. Install prerequisites (once)

**Python packages:**
```powershell
pip install pyinstaller pywebview platformdirs
# Optional — adds Intel GPU acceleration support to the bundle:
pip install openvino
```

**Inno Setup 6** (installer compiler) — the `--location` flag is required,
otherwise winget installs to an unresolvable path:
```powershell
winget install --id JRSoftware.InnoSetup `
    --location "C:\Program Files\Inno Setup 6" `
    --accept-package-agreements --accept-source-agreements
```
Installed to: `C:\Program Files\Inno Setup 6\ISCC.exe`

### 2. Build

The easiest way — run the build script from the repo root:
```powershell
.\installer\build.ps1
```

Or step by step:
```powershell
python version.py           # → installer/version.iss + installer/version_info.txt
pyinstaller novoface.spec   # → dist/novoface/
& "C:\Program Files\Inno Setup 6\ISCC.exe" installer\novoface.iss
# → installer/Output/novoface-0.0.1-setup.exe
```

The first PyInstaller build always requires a test run on a clean machine — native
DLLs from insightface and onnxruntime are collected automatically by the spec but
should be verified.
