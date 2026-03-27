# Changelog

All notable changes to novoface will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

[1.0.0]: https://github.com/ketafoto/novoface/releases/tag/v1.0.0
