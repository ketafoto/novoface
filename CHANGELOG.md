# Changelog

All notable changes to novoface will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.1.0] — 2026-07-28

### Added
- **Full-photo hover preview in Review** — resting the cursor on a face thumbnail shows a floating preview of the whole source photo, so you can read the scene without opening the photo in an external viewer. If the original file was moved or deleted, the preview shows a clear "not found on disk" note suggesting a re-scan.
- **Thumbnail size slider in the Review detail pane** — resize face thumbnails from small (about 12 per row, for fast visual scanning) to large (about 5 per row), with a live "(N/row)" readout. Below a threshold the per-photo text is hidden for a compact contact-sheet layout. Your choice is remembered across sessions.
- **Face / Full-photo view toggle in Review** — switch the whole detail grid between tight face crops (best for naming and merging people) and the full source photos (best for finding shots with a particular composition, e.g. a person together with others). Full photos are shown whole (letterboxed), never cropped. Remembered across sessions.
- **Export selected photos to a folder** — pick any photos in a cluster (checkbox, Shift-click range, Ctrl-click) and copy the originals to a folder of your choice in one step, without opening each in a "Save as" dialog. A native folder picker opens at your last-used destination; files keep their names and existing files are skipped, never overwritten. A result banner reports how many were copied and offers to open the folder. The selection bar stays pinned to the top while you scroll a large cluster, so Export is always within reach.
- **Settings are now remembered between sessions** in the desktop app — your theme (light/dark), thumbnail size, Face/Full view, and last export folder now persist across app restarts and reinstalls.

### Changed
- **Faster, cleaner Review after a scan** — when a scan finishes, the Review pane now refreshes automatically so photos removed during the scan (files deleted or moved off disk) disappear right away, instead of lingering until you manually reopened the person.
- **Review detail header redesigned** — the person/scan controls and the new view controls are now organized into two clearly separated groups on a single, compact aligned row, replacing the previously cramped layout.
- Full-photo previews are cached on first view (per backend) for instant repeat display, and are cleared by Reset Database along with thumbnails.

### Fixed
- **Scan time estimate (ETA) is now accurate.** The remaining-time estimate previously grew larger the longer a scan ran (it counted slow start-up work and, on re-scans, already-known files against the average speed). It now reflects the current processing speed over a recent window, so it counts down sensibly toward zero.
- **Desktop app no longer forgot your settings on restart.** Theme, thumbnail size, and view choices were saved but discarded every time the app closed; they now stick.
- **Full-photo view no longer reverts to face crops** while scrolling a large cluster.

### Also includes the earlier 1.0.2 maintenance work
- Fixes for packaged-app scan crashes (numpy/scipy/matplotlib), anti-virus-throttled install/uninstall slowness, and stale-path pruning; the build is now pinned to the curated `venv-win` environment. *(Shipped in the 1.0.2 tag; folded in here for users upgrading directly from 1.0.1 or earlier.)*

---

## [1.0.2] — 2026-07-01

### Changed
- Scan failures now log a full traceback to `novoface.log` instead of only a short message, so intermittent backend/native-library errors can be diagnosed.
- Added exception logging across the codebase (face detection, image loading, OpenVINO pipeline, config/settings reads, and shutdown) to surface previously-silent failures.
- Installer now updates in place over an existing version, showing a brief note (with the detected version) confirming the update is safe and leaves your data untouched.
- Upgrades are dramatically faster on machines with endpoint security (e.g. Check Point), which scans every file the installer writes or deletes: the installer now ships a file manifest and (a) deletes only files the new build no longer contains, and (b) skips rewriting files that are already byte-identical on disk — so unchanged large native libraries are neither re-deleted nor re-copied, avoiding the per-file security scan that previously made upgrades take many minutes.
- Uninstall is likewise much faster on such machines: the bulk of the program files are removed with a single trusted `cmd.exe` operation rather than deleted one-by-one by the uninstaller process, bypassing the same per-file security scan.
- Trimmed the OpenVINO model-optimizer tools and C++ dev headers from the bundle (never used at runtime), cutting the install from ~4000 to ~2350 files.
- Installer always rebuilds from a clean PyInstaller cache (`--clean`) to prevent mismatched runtime modules from a previous build causing a startup crash.

---

## [1.0.1] — 2026-05-09

### Added
- Light / dark theme toggle in Settings → Appearance, persisted across sessions and applied before first paint to avoid flash
- Favicon for the browser UI
- Copy-to-clipboard button next to the support email in the About dialog; the email is also selectable for Ctrl+C
- Privacy disclosure on the Donate dialog noting that GitHub Sponsors and Ko-fi handle donations under their own privacy policies

### Changed
- About dialog: replaced the bottom Close button with an "X" in the top-right corner
- Updated the support contact email to a working address

### Removed
- Google Analytics tracking — unnecessary for a downloadable desktop application

---

## [1.0.0] — 2026-03-27

### Added

**Face Detection & Clustering**
- Automatic face detection and recognition using ArcFace embeddings (InsightFace)
- Intel Iris Xe / Arc GPU acceleration via OpenVINO backend — detected and configured at first launch
- Incremental scanning — only new photos are processed on subsequent scans
- Hash-based deduplication — renamed or moved files are never double-counted
- Configurable CPU limit to keep your machine responsive during long scans
- Background I/O priority so scanning does not slow down other work

**Review UI**
- Browser-based single-page UI — works in any modern browser
- Cluster sidebar with named clusters sorted alphabetically, unnamed by size
- Cluster groups with collapsible tree, drag-and-drop, and right-click menu
- Multi-select clusters (Shift+click range, Ctrl+click individual) for batch operations
- Merge clusters, move individual faces between clusters
- Search in cluster list and in merge target list
- Double-click any photo to open it in your system viewer
- Keyboard navigation (arrows, Enter) in all search overlays

**Scan Management**
- Pause and resume scan at any time
- Exclude folders by name pattern (e.g. `@eaDir`, `thumbs`)
- Live CPU cap adjustment during an active scan
- Cumulative progress stats across resumed sessions

**Data & Settings**
- SQLite database in WAL mode — safe for concurrent UI and scan access
- Relocate photo paths after moving your archive to a new drive
- Move the entire data directory without data loss
- Export / import database as a `.tar.gz` backup
- Rotating log file with configurable size limit
- Reset database to start from scratch

**Windows Desktop App**
- Packaged as a native `.exe` with a pywebview window — no browser setup needed
- Inno Setup installer with version info embedded in the executable
- First-run setup dialog: choose data location, import existing data, enable GPU acceleration
- OpenVINO face models downloaded automatically at first run (~262 MB)

---

[1.1.0]: https://github.com/ketafoto/novoface/releases/tag/v1.1.0
[1.0.2]: https://github.com/ketafoto/novoface/releases/tag/v1.0.2
[1.0.1]: https://github.com/ketafoto/novoface/releases/tag/v1.0.1
[1.0.0]: https://github.com/ketafoto/novoface/releases/tag/v1.0.0
