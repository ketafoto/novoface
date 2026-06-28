# Changelog

All notable changes to novoface will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.2] — 2026-06-28

### Changed
- Scan failures now log a full traceback to `novoface.log` instead of only a short message, so intermittent backend/native-library errors can be diagnosed.
- Added exception logging across the codebase (face detection, image loading, OpenVINO pipeline, config/settings reads, and shutdown) to surface previously-silent failures.
- Installer now updates in place over an existing version, showing a brief note (with the detected version) confirming the update is safe and leaves your data untouched.
- Upgrades are dramatically faster on machines with endpoint security (e.g. Check Point), which scans every file the installer writes or deletes: the installer now ships a file manifest and (a) deletes only files the new build no longer contains, and (b) skips rewriting files that are already byte-identical on disk — so unchanged large native libraries are neither re-deleted nor re-copied, avoiding the per-file security scan that previously made upgrades take many minutes.
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

[1.0.1]: https://github.com/ketafoto/novoface/releases/tag/v1.0.1
[1.0.0]: https://github.com/ketafoto/novoface/releases/tag/v1.0.0
