"""
main.py — Desktop launcher for novoface.

Starts the Flask server in a background daemon thread, shows a first-run
setup dialog (tkinter) if the data directory is empty, then opens the UI
in a native pywebview window backed by WebView2 (Edge).

Closing the window terminates the entire process — the daemon thread dies
with it, so no orphaned server is left running.

Usage (development):
    python main.py

Packaged usage (PyInstaller):
    novoface.exe  (built via:  pyinstaller novoface.spec)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import shutil
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Callable

import platformdirs

# ── Data directory ────────────────────────────────────────────────────────────
# Resolved once at startup.  Must be set as an env var BEFORE any face_scan /
# app imports so that DATA_DIR inside those modules picks up the right path.

APP_NAME = "novoface"
DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
os.environ["NOVOFACE_DATA_DIR"] = str(DATA_DIR)

# ── Logging ───────────────────────────────────────────────────────────────────
# Written to DATA_DIR/novoface.log.  Uses a RotatingFileHandler so the file
# never grows beyond log_max_mb (two backup files kept).
# Settings are read from settings.json on startup and can be changed live
# via the Settings tab in the UI (which calls reconfigure_logging()).

_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_ANSI_RE = __import__("re").compile(r"\x1b\[[0-9;]*m")


class _PlainFormatter(logging.Formatter):
    """Formatter that strips ANSI colour codes before writing to the log file."""
    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_RE.sub("", super().format(record))


def _read_log_settings() -> tuple[bool, int]:
    """Return (log_enabled, log_max_mb) from settings.json, with safe defaults."""
    try:
        s = json.loads((DATA_DIR / "settings.json").read_text(encoding="utf-8"))
        return bool(s.get("log_enabled", True)), int(s.get("log_max_mb", 5))
    except Exception:
        return True, 5


def _setup_logging(enabled: bool = True, max_mb: int = 5) -> Path:
    """Configure the root logger with a RotatingFileHandler."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log_path = DATA_DIR / "novoface.log"
    except Exception:
        import tempfile
        log_path = Path(tempfile.gettempdir()) / "novoface.log"

    handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=max_mb * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setFormatter(_PlainFormatter(_LOG_FMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if enabled else logging.WARNING)

    # Also redirect stderr so C-level / uncaught errors land in the log
    try:
        sys.stderr = open(log_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        pass

    return log_path


def reconfigure_logging(enabled: bool, max_mb: int) -> None:
    """Update logging live after the user changes settings in the UI."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if enabled else logging.WARNING)
    for h in root.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            h.maxBytes = max_mb * 1024 * 1024


_enabled, _max_mb = _read_log_settings()
_LOG_PATH = _setup_logging(_enabled, _max_mb)
logging.info("novoface starting — data dir: %s", DATA_DIR)

PORT = 8050

# ── OpenVINO model URLs ───────────────────────────────────────────────────────

_OV_BASE = (
    "https://storage.openvinotoolkit.org/repositories/"
    "open_model_zoo/2022.1/models_bin/2"
)
_OV_MODELS = [
    (
        f"{_OV_BASE}/face-detection-retail-0004/FP32/face-detection-retail-0004.xml",
        "intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml",
        "Detection model (small)",
    ),
    (
        f"{_OV_BASE}/face-detection-retail-0004/FP32/face-detection-retail-0004.bin",
        "intel/face-detection-retail-0004/FP32/face-detection-retail-0004.bin",
        "Detection model weights",
    ),
    (
        "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx",
        "public/face-recognition-resnet100-arcface-onnx/FP32/arcfaceresnet100-8.onnx",
        "Recognition model (~260 MB)",
    ),
]


# ── First-run detection ───────────────────────────────────────────────────────

def _is_first_run() -> bool:
    """True if neither the CPU nor the OpenVINO database exists yet."""
    return (
        not (DATA_DIR / "faces.db").exists()
        and not (DATA_DIR / "faces_ov.db").exists()
    )


# ── GPU / OpenVINO detection ──────────────────────────────────────────────────

def _detect_intel_gpu() -> str | None:
    """
    Return the GPU name string if an Intel Iris Xe / Arc GPU is found on
    this machine, else None.  Uses wmic — always available on Windows.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            name = line.strip()
            nl = name.lower()
            if "intel" in nl and any(k in nl for k in ("iris", "arc", "uhd")):
                return name
    except Exception:
        pass
    return None


def _openvino_available() -> bool:
    """True if the openvino runtime package is importable (bundled or installed)."""
    try:
        import openvino  # noqa: F401
        return True
    except ImportError:
        return False


def _openvino_models_exist(data_dir: Path) -> bool:
    """True if all required OpenVINO model files are already downloaded."""
    models_dir = data_dir / "openvino_models"
    return all((models_dir / rel).exists() for _, rel, _ in _OV_MODELS)


# ── Model download ────────────────────────────────────────────────────────────

def _download_openvino_models(
    data_dir: Path,
    on_file: "Callable[[int, int, str], None] | None" = None,
    on_progress: "Callable[[float], None] | None" = None,
) -> None:
    """
    Download OpenVINO face models into data_dir/openvino_models/.

    on_file(index, total, label)  — called when a new file starts
    on_progress(fraction 0-1)     — called repeatedly during each download
    Raises on network error.
    """
    models_dir = data_dir / "openvino_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    files_needed = [m for m in _OV_MODELS if not (models_dir / m[1]).exists()]

    for i, (url, rel, label) in enumerate(files_needed):
        if on_file:
            on_file(i, len(files_needed), label)

        dest = models_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        def _reporthook(block_count, block_size, total_size, _cb=on_progress):
            if _cb and total_size > 0:
                _cb(min(1.0, block_count * block_size / total_size))

        urllib.request.urlretrieve(url, dest, reporthook=_reporthook)

    if on_progress:
        on_progress(1.0)


# ── Download progress dialog ──────────────────────────────────────────────────

def _show_download_dialog(data_dir: Path) -> bool:
    """
    Show a modal tkinter window that downloads the OpenVINO models with a
    progress bar.  Returns True on success, False if the download failed.
    """
    import tkinter as tk
    from tkinter import ttk, messagebox

    success = [False]
    error_msg: list[str | None] = [None]

    root = tk.Tk()
    root.title("novoface — Downloading GPU models")
    root.resizable(False, False)

    w, h = 460, 160
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    # Prevent closing mid-download
    root.protocol("WM_DELETE_WINDOW", lambda: None)

    file_var = tk.StringVar(value="Preparing…")
    count_var = tk.StringVar(value="")

    tk.Label(root, textvariable=file_var, anchor="w").pack(
        fill="x", padx=16, pady=(18, 2)
    )
    tk.Label(root, textvariable=count_var, anchor="w", fg="#666666").pack(
        fill="x", padx=16, pady=(0, 8)
    )

    bar = ttk.Progressbar(root, length=420, mode="determinate", maximum=100)
    bar.pack(padx=16, pady=(0, 16))

    def _on_file(idx: int, total: int, label: str) -> None:
        root.after(0, lambda: file_var.set(label))
        root.after(0, lambda: count_var.set(f"File {idx + 1} of {total}"))
        root.after(0, lambda: bar.configure(value=0))

    def _on_progress(fraction: float) -> None:
        root.after(0, lambda: bar.configure(value=int(fraction * 100)))

    def _worker() -> None:
        try:
            _download_openvino_models(data_dir, on_file=_on_file, on_progress=_on_progress)
            success[0] = True
        except Exception as exc:
            error_msg[0] = str(exc)
        finally:
            root.after(0, root.destroy)

    threading.Thread(target=_worker, daemon=True).start()
    root.mainloop()

    if error_msg[0]:
        import tkinter.messagebox as mb
        mb.showwarning(
            "Download failed",
            f"Could not download GPU models:\n{error_msg[0]}\n\n"
            "The app will start with CPU processing instead.",
        )
        return False

    return success[0]


# ── First-run setup dialog ────────────────────────────────────────────────────

def _show_setup_dialog(gpu_name: str | None, models_present: bool = False) -> tuple[Path | None, bool]:
    """
    Show the first-run setup dialog.

    gpu_name   — detected Intel GPU name, or None (hides the GPU section)

    Returns (import_path, enable_gpu):
      import_path  — existing face_data folder the user wants to import, or None
      enable_gpu   — True if the user wants GPU acceleration enabled
    """
    import tkinter as tk
    from tkinter import filedialog

    chosen_import: list[Path | None] = [None]
    chosen_gpu = [False]

    root = tk.Tk()
    root.title("novoface — First Run Setup")
    root.resizable(False, False)

    # Height grows when GPU row is present
    w = 580
    h = 290 if gpu_name else 220
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    row = 0

    # ── Data location ─────────────────────────────────────────────────────
    tk.Label(root, text="Your data will be stored at:", anchor="w").grid(
        row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(18, 2)
    )
    row += 1
    dest_var = tk.StringVar(value=str(DATA_DIR))
    tk.Entry(root, textvariable=dest_var, state="readonly", width=64).grid(
        row=row, column=0, columnspan=3, padx=16, pady=(0, 4), sticky="ew"
    )
    row += 1

    # ── Import existing ───────────────────────────────────────────────────
    tk.Label(
        root, text="Import existing face_data (optional):", anchor="w"
    ).grid(row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(14, 2))
    row += 1

    import_var = tk.StringVar(value="")
    tk.Entry(root, textvariable=import_var, width=52).grid(
        row=row, column=0, columnspan=2, padx=(16, 4), pady=(0, 4), sticky="ew"
    )

    def _browse():
        folder = filedialog.askdirectory(
            title="Select your existing face_data folder", mustexist=True
        )
        if folder:
            import_var.set(folder)

    tk.Button(root, text="Browse…", command=_browse, width=10).grid(
        row=row, column=2, padx=(0, 16), pady=(0, 4)
    )
    row += 1

    # ── GPU acceleration (only when Intel GPU detected) ───────────────────
    if gpu_name:
        sep = tk.Frame(root, height=1, bg="#cccccc")
        sep.grid(row=row, column=0, columnspan=3, sticky="ew", padx=16, pady=(10, 0))
        row += 1

        gpu_var = tk.BooleanVar(value=True)

        gpu_frame = tk.Frame(root)
        gpu_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(6, 0))

        tk.Checkbutton(
            gpu_frame,
            text=f"Enable GPU acceleration  ({gpu_name})",
            variable=gpu_var,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")
        row += 1

        note = "Uses Intel GPU for faster face scanning  •  already installed" \
               if models_present else \
               "Uses Intel GPU for faster face scanning  •  requires ~260 MB download"
        tk.Label(
            root,
            text=note,
            anchor="w",
            fg="#666666",
            font=("Segoe UI", 8),
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=32, pady=(0, 4))
        row += 1
    else:
        gpu_var = None  # type: ignore[assignment]

    # ── Action buttons ────────────────────────────────────────────────────
    def _on_fresh():
        chosen_import[0] = None
        if gpu_var is not None:
            chosen_gpu[0] = gpu_var.get()
        root.destroy()

    def _on_import():
        p = import_var.get().strip()
        chosen_import[0] = Path(p) if p else None
        if gpu_var is not None:
            chosen_gpu[0] = gpu_var.get()
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.grid(row=row, column=0, columnspan=3, pady=18)
    tk.Button(btn_frame, text="Start Fresh", width=14, command=_on_fresh).pack(
        side="left", padx=10
    )
    tk.Button(
        btn_frame, text="Import & Start", width=14, command=_on_import, default="active"
    ).pack(side="left", padx=10)

    root.grid_columnconfigure(0, weight=1)
    root.mainloop()

    return chosen_import[0], chosen_gpu[0]


# ── Flask worker ──────────────────────────────────────────────────────────────

def _start_flask() -> None:
    """Import and start the Flask app (imports are deferred so the env var is set first)."""
    try:
        logging.info("importing app module")
        import app as flask_app  # noqa: PLC0415  (deferred intentionally)

        logging.info("calling ensure_db()")
        flask_app.ensure_db()
        logging.info("starting Flask on port %d", PORT)
        flask_app.app.run(
            host="127.0.0.1",
            port=PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    except Exception:
        logging.exception("Flask thread crashed")
        raise


def _wait_for_flask(
    timeout: float = 15.0,
    tick: Callable[[], None] | None = None,
) -> None:
    """Block until Flask responds or raise RuntimeError.
    tick() is called every 0.2 s so callers can keep a UI window alive.
    """
    url = f"http://127.0.0.1:{PORT}/"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            if tick:
                try:
                    tick()
                except Exception:
                    pass
            time.sleep(0.2)
    raise RuntimeError(
        f"Flask server did not start within {timeout}s — check for port conflicts."
    )


# ── Startup splash ────────────────────────────────────────────────────────────
# A slim non-modal window shown while Flask is initialising.
# On first run it also stays visible during the data-import copy.
# Closed just before the pywebview window opens.

def _show_splash(message: str):
    """Create a small 'starting…' splash window. Returns the Tk root."""
    import tkinter as tk
    root = tk.Tk()
    root.title("novoface")
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", lambda: None)   # prevent accidental close
    w, h = 340, 72
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
    var = tk.StringVar(value=message)
    tk.Label(root, textvariable=var, font=("Segoe UI", 10), fg="#888888").pack(expand=True)
    root._msg_var = var  # type: ignore[attr-defined]
    root.update()
    return root


def _update_splash(root, message: str) -> None:
    try:
        root._msg_var.set(message)
        root.update()
    except Exception:
        pass


def _close_splash(root) -> None:
    try:
        root.destroy()
    except Exception:
        pass


# ── Backend selection ─────────────────────────────────────────────────────────

def _set_backend(data_dir: Path, backend: str) -> None:
    """Write backend.json so the app opens with the chosen backend pre-selected."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "backend.json").write_text(
        json.dumps({"backend": backend}), encoding="utf-8"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("main() started")

    # ── First-run setup ───────────────────────────────────────────────────
    if _is_first_run():
        logging.info("first run detected — showing setup dialog")
        # Only offer GPU option if OpenVINO is bundled AND an Intel GPU is present
        gpu_name = (
            _detect_intel_gpu()
            if _openvino_available()
            else None
        )
        logging.info("gpu_name=%s openvino_available=%s", gpu_name, _openvino_available())

        import_path, enable_gpu = _show_setup_dialog(
            gpu_name, models_present=_openvino_models_exist(DATA_DIR)
        )
        logging.info("setup dialog: import_path=%s enable_gpu=%s", import_path, enable_gpu)

        # Copy existing face_data if the user chose to import
        if import_path and import_path.exists():
            logging.info("importing face_data from %s", import_path)
            splash = _show_splash("Importing your data, please wait…")
            for item in import_path.iterdir():
                dest = DATA_DIR / item.name
                if item.is_dir():
                    shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(item), str(dest))
                _update_splash(splash, f"Importing: {item.name}")
            logging.info("import complete")
            _update_splash(splash, "Starting novoface…")
        else:
            splash = _show_splash("Starting novoface…")

        # Download OpenVINO models and pre-select the GPU backend
        if enable_gpu and not _openvino_models_exist(DATA_DIR):
            _close_splash(splash)
            downloaded = _show_download_dialog(DATA_DIR)
            splash = _show_splash("Starting novoface…")
            if downloaded:
                _set_backend(DATA_DIR, "openvino")
        elif enable_gpu and _openvino_models_exist(DATA_DIR):
            # Models already present (e.g. imported from existing face_data)
            _set_backend(DATA_DIR, "openvino")
    else:
        splash = _show_splash("Starting novoface…")

    # ── Start Flask server ────────────────────────────────────────────────
    logging.info("starting Flask thread")
    flask_thread = threading.Thread(target=_start_flask, daemon=True, name="flask")
    flask_thread.start()
    logging.info("waiting for Flask to be ready")
    _wait_for_flask(tick=lambda: _update_splash(splash, "Starting novoface…"))
    logging.info("Flask is ready")
    _close_splash(splash)

    # ── Open pywebview window ─────────────────────────────────────────────
    # Importing webview here (not at the top) keeps startup fast and avoids
    # WebView2 initialisation before tkinter is done.
    logging.info("opening pywebview window")
    import webview  # noqa: PLC0415

    webview.create_window(
        "novoface",
        f"http://127.0.0.1:{PORT}",
        width=1400,
        height=900,
        min_size=(900, 600),
    )
    webview.start()

    # ── Graceful shutdown after window close ──────────────────────────────
    # webview.start() has returned — the window is gone.  Signal any running
    # scan to stop so it can finish the current photo and flush its DB write,
    # rather than being killed mid-write by the daemon thread exit.
    # The join timeout is short: we don't want to block the user's desktop.
    try:
        import app as flask_app  # already imported and cached by _start_flask
        if flask_app._scan_thread and flask_app._scan_thread.is_alive():
            flask_app._scan_stop.set()
            flask_app._scan_thread.join(timeout=5)
    except Exception:
        pass
    # Process now exits normally:
    #   • atexit fires _cleanup_children() → kills any OpenVINO subprocesses
    #   • remaining daemon threads (Flask, scan if still alive) are killed


if __name__ == "__main__":
    try:
        main()
    except Exception:
        msg = traceback.format_exc()
        logging.critical("fatal error:\n%s", msg)
        # Show a visible error box — console=False means the user sees nothing otherwise
        try:
            import tkinter.messagebox as _mb
            _mb.showerror(
                "novoface — startup error",
                f"novoface failed to start.\n\nLog: {_LOG_PATH}\n\n{msg}",
            )
        except Exception:
            pass
        sys.exit(1)
