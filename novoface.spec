# novoface.spec — PyInstaller build spec
#
# Build (CPU-only, no OpenVINO):
#   pip install pyinstaller pywebview platformdirs
#   pyinstaller novoface.spec
#
# Build (with Intel GPU support):
#   pip install pyinstaller pywebview platformdirs openvino
#   pyinstaller novoface.spec
#
# Output: dist/novoface/novoface.exe  (--onedir, no console window)
#
# Notes:
# - insightface and onnxruntime ship native DLLs that PyInstaller needs help
#   collecting; collect_all() handles that automatically.
# - openvino is collected only when installed in the build environment.
#   If absent, the GPU option is silently hidden in the first-run dialog.
# - pywebview on Windows uses WebView2 (Edge), which is pre-installed on all
#   Windows 10/11 machines since ~2021.  No extra runtime bundling required.
# - The face_review_ui.html is included as a data file in the bundle root so
#   that app.py's _base_dir() / "face_review_ui.html" resolves correctly.
# - OpenVINO models (~262 MB) are NOT bundled — they are downloaded at first
#   run when the user enables GPU acceleration in the setup dialog.

import sys as _sys
_sys.path.insert(0, '.')
from version import __version__, _version_tuple  # noqa: E402

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# ── Collect data + binaries from packages that need it ───────────────────────
ins_datas, ins_binaries, ins_hidden = collect_all("insightface")
ort_datas, ort_binaries, ort_hidden = collect_all("onnxruntime")
wv_datas,  wv_binaries,  wv_hidden  = collect_all("webview")

# scipy ships compiled extension modules (.pyd) that must all be present, or scipy
# raises "The scipy install you are using seems to be broken (extension modules
# cannot be imported), please try reinstalling." Collect it explicitly so no .pyd
# is silently dropped by transitive analysis. (numpy is pulled in with it.)
#
# BUT collect_all("scipy") also drags in ~1500 loose DATA files (C/Fortran build
# artifacts, Cython sources, dev headers, and test fixtures) that the app never
# imports at runtime. Every extra loose file inflates install time badly, because
# endpoint anti-ransomware (Check Point) throttles file-creation bursts — the
# installer stalls ~15 s after every ~100 files written. We therefore filter the
# scipy/sklearn/skimage DATA lists down to just what's imported at runtime.
#
# CRITICAL: only DATA files are filtered here. Compiled extensions (.pyd/.dll/.so)
# live in the *binaries* list, which is left untouched — so the scipy "broken
# install" guarantee (all extension modules present) is preserved.
sp_datas, sp_binaries, sp_hidden = collect_all("scipy")

# numpy 2.x splits its runtime across many pure-Python submodules under numpy/_core/
# (e.g. numpy._core._exceptions) that PyInstaller's transitive analysis does NOT
# reliably discover — they are imported indirectly by the compiled _multiarray_umath
# extension. When one is missing the app dies at startup with:
#   "Importing the numpy C-extensions failed … No module named 'numpy._core._exceptions'"
# (Flask thread crashes on `import cv2 -> import numpy`, so the server never starts and
# main.py's _wait_for_flask times out at 15 s.) collect_all forces EVERY numpy submodule
# into the bundle. Same DATA-file trimming as scipy applies below (tests/build artifacts);
# numpy.tests modules are dropped via the Analysis `excludes`.
np_datas, np_binaries, np_hidden = collect_all("numpy")

# File extensions that are build/dev artifacts — needed to COMPILE these packages
# from source, never to RUN the already-compiled wheel we bundle.
_DEV_ARTIFACT_EXTS = (
    ".a", ".lib", ".o", ".obj",          # static libs / object files
    ".h", ".hpp", ".hxx",                # C/C++ headers
    ".c", ".cpp", ".cc", ".cxx",         # C/C++ sources
    ".f", ".f90", ".f77", ".pyf",        # Fortran sources / f2py signatures
    ".pyx", ".pxd", ".pxi", ".tp",       # Cython sources / templates (.pyx.tp -> .tp)
    ".pyi",                              # type stubs (runtime doesn't read them)
    ".build",                            # meson build descriptors
)

def _is_numeric_dead_weight(src, dest):
    """True for numpy/scipy/sklearn/skimage DATA files the runtime never opens:
    build/dev artifacts and bundled test suites + their fixtures. Only filters DATA
    files — compiled extensions live in the *binaries* list and are never touched, so
    numpy/scipy stay fully importable.

    NOTE: PyInstaller data tuples are (source_file, dest_DIRECTORY) — `dest` is the
    in-bundle *folder*, not the file. We reconstruct the full relative path from the
    source filename + dest folder so the extension test actually works (a directory
    never ends in ``.pyi``/``.h``/…). Getting this wrong silently disables the trim."""
    fname = src.replace("\\", "/").rsplit("/", 1)[-1]
    rel = (dest.replace("\\", "/").rstrip("/") + "/" + fname).lower()
    top = rel.split("/", 1)[0]
    if top not in ("numpy", "scipy", "sklearn", "skimage"):
        return False
    if rel.endswith(_DEV_ARTIFACT_EXTS):
        return True
    # Bundled test packages and their large data fixtures (.mat/.npz/.arff/.wav/…).
    if "/tests/" in rel or "/test/" in rel:
        return True
    return False

def _trim_numeric_datas(datas, label):
    before = len(datas)
    kept = [(src, dest) for (src, dest) in datas if not _is_numeric_dead_weight(src, dest)]
    print(f"[novoface.spec] {label} data files: {before} -> {len(kept)} "
          f"(dropped {before - len(kept)} build/test files)")
    return kept

sp_datas = _trim_numeric_datas(sp_datas, "scipy")
np_datas = _trim_numeric_datas(np_datas, "numpy")
ins_datas = _trim_numeric_datas(ins_datas, "insightface")   # pulls sklearn/skimage

# OpenVINO: collected only when installed in the build environment.
# GPU acceleration is silently unavailable in the app if openvino is absent.
try:
    ov_datas, ov_binaries, ov_hidden = collect_all("openvino")
except Exception:
    ov_datas, ov_binaries, ov_hidden = [], [], []

# Trim OpenVINO dead weight: the runtime (libs/runtime/frontend) is all we use
# (`import openvino as ov` + Core/model loading). The model-optimizer & converter
# CLI tools (openvino/tools/mo, ovc, benchmark) and the C++ dev headers
# (openvino/include) are never imported at runtime, but together they add ~1650
# loose files — a large share of the bundle's file count, which is what makes
# install/uninstall slow under on-access AV scanning. Drop them from the bundle.
def _ov_is_dead_weight(dest):
    # dest is the in-bundle directory path, e.g. "openvino\tools\mo\ops" or
    # "openvino\include\openvino\op". Match the tools/ and include/ subtrees.
    dest = dest.replace("\\", "/")
    return (
        dest == "openvino/tools" or dest.startswith("openvino/tools/")
        or dest == "openvino/include" or dest.startswith("openvino/include/")
    )

_ov_before = len(ov_datas)
ov_datas = [(src, dest) for (src, dest) in ov_datas if not _ov_is_dead_weight(dest)]
print(f"[novoface.spec] OpenVINO data files: {_ov_before} -> {len(ov_datas)} "
      f"(dropped {_ov_before - len(ov_datas)} tools/include files)")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=ins_binaries + ort_binaries + wv_binaries + sp_binaries + np_binaries + ov_binaries,
    datas=[
        # App resources
        ("face_review_ui.html",      "."),
        ("face_scan.py",             "."),
        ("openvino_pipeline.py",     ".",),
        ("app.py",                   "."),
        ("version.py",               "."),
        ("installer/novoface.ico",   "."),
    ] + ins_datas + ort_datas + wv_datas + sp_datas + np_datas + ov_datas,
    hiddenimports=[
        # pywebview Windows backend
        "webview.platforms.winforms",
        # Flask internals sometimes missed
        "flask",
        "flask.templating",
        "jinja2",
        "werkzeug",
        "werkzeug.serving",
        # Common missed imports
        "pkg_resources",
        "pkg_resources.py2_compat",
        "platformdirs",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors.typedefs",
        "sklearn.neighbors.quad_tree",
        "sklearn.tree._utils",
        # OpenVINO internals (no-op if not installed)
        "openvino.runtime",
        "openvino._pyopenvino",
    ] + ins_hidden + ort_hidden + wv_hidden + sp_hidden + np_hidden + ov_hidden
      + collect_submodules("sklearn"),
    hookspath=[],
    hooksconfig={},
    # Runtime hook: stub `matplotlib`/`mpl_toolkits` before app code runs, so
    # insightface's eager `app.mask_renderer -> thirdparty.face3d.mesh.vis` import
    # chain (which does an unconditional `import matplotlib.pyplot`) does not crash
    # the scan with "No module named 'matplotlib'". matplotlib is intentionally not
    # bundled (see excludes); we never use 3D mesh rendering. This fix lives in the
    # repo so it survives venv recreation / building on another machine — unlike the
    # fragile hand-patch to site-packages/insightface/app/__init__.py it replaces.
    runtime_hooks=["installer/pyi_rth_mpl_stub.py"],
    excludes=[
        # Keep bundle lean — these are never used at runtime
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "sphinx",
        # Bundled test suites of the numeric stack: hundreds of .py modules plus
        # large data fixtures, none of which the app imports. Excluding the test
        # PACKAGES here stops collect_submodules("sklearn") and transitive analysis
        # from dragging them (and their fixtures) into the bundle — a major slice of
        # the file count that throttles install under endpoint anti-ransomware.
        "sklearn.tests",
        "scipy.tests",
        "numpy.tests",
        "skimage.tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── Trim the Tcl/Tk runtime (tkinter) ────────────────────────────────────────
# tkinter is used ONLY for the small first-run setup dialogs (progress bar, folder
# picker, message boxes) in main.py — all English, no timezone use. PyInstaller's
# tkinter hook bundles the full Tcl data tree though, including the timezone
# database (tzdata/, ~700 files) and localized message catalogs (msgs/*.msg) that
# these dialogs never touch. Dropping them removes ~800 loose files — the single
# biggest slice of the install-time file count (endpoint anti-ransomware throttles
# ~15 s after every ~100 files created). The Tk widget/encoding files are KEPT so
# the dialogs still render correctly.
def _is_tcl_dead_weight(dest):
    d = dest.replace("\\", "/").lower()
    return (
        "/tzdata/" in d
        or d.endswith("/msgs") or "/msgs/" in d
    ) and ("_tcl_data/" in d or "tcl" in d.split("/")[0])

_tcl_before = len(a.datas)
a.datas = [(dest, src, typ) for (dest, src, typ) in a.datas
           if not _is_tcl_dead_weight(dest)]
print(f"[novoface.spec] Tcl/Tk data files: {_tcl_before} -> {len(a.datas)} "
      f"(dropped {_tcl_before - len(a.datas)} tzdata/msgs files)")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="novoface",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX is DISABLED: compressing scipy/numpy compiled extension (.pyd) files
    # corrupts them non-deterministically, producing "scipy install seems broken
    # (extension modules cannot be imported)" at runtime — a fix that appears to
    # work on one build and regresses on the next. UPX only shaves file size; it
    # is not worth breaking the numerical stack. (2026-07-04)
    upx=False,
    console=False,      # No black console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version="installer/version_info.txt",   # Windows VERSIONINFO (generated by: python version.py)
    icon="installer/novoface.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,   # See EXE(upx=False) above — UPX corrupts scipy/numpy .pyd files.
    upx_exclude=[],
    name="novoface",
)
