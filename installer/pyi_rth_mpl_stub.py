# PyInstaller runtime hook — inject a stub `matplotlib` if the real one is absent.
#
# WHY THIS EXISTS
# --------------
# insightface's package __init__ eagerly imports its `app` subpackage, which pulls
# in `app.mask_renderer` -> `thirdparty.face3d.mesh.vis`, and that module does an
# unconditional `import matplotlib.pyplot as plt` (plus `from mpl_toolkits.mplot3d
# import Axes3D`). We never use 3D mesh rendering, and matplotlib is deliberately
# NOT bundled (it would add hundreds of files). So in the frozen app, importing any
# insightface submodule (e.g. `from insightface.model_zoo import scrfd` in
# openvino_pipeline.py) crashes the scan with "No module named 'matplotlib'".
#
# Some environments have a hand-patched insightface `app/__init__.py` that wraps the
# mask_renderer import in try/except — but that patch lives in site-packages, not in
# this repo, so it does NOT survive a venv recreation or a build on another machine.
# Relying on it is exactly the kind of fix that silently regresses. This hook makes
# the frozen app self-sufficient: it runs BEFORE any app code, so by the time
# insightface imports, `matplotlib.pyplot` resolves to a harmless stub.
#
# The stubbed `plt`/`Axes3D` are only ever *imported* by vis.py, never *called* on
# our code path (we do detection + recognition + alignment only). If real matplotlib
# is ever present in the environment, we use it instead of the stub.

import sys
import types


def _install_matplotlib_stub():
    try:
        import matplotlib  # noqa: F401 — real matplotlib available; nothing to do.
        return
    except ImportError:
        pass

    def _make(name, parent=None):
        mod = types.ModuleType(name)
        mod.__stub__ = True
        sys.modules[name] = mod
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[-1], mod)
        return mod

    class _Any:
        """Callable/subscriptable no-op — tolerates attribute access if ever touched."""
        def __call__(self, *a, **k):
            return self
        def __getattr__(self, _):
            return self
        def __getitem__(self, _):
            return self

    mpl = _make("matplotlib")
    mpl.__version__ = "0.0.0-stub"
    pyplot = _make("matplotlib.pyplot", mpl)
    # Any attribute access on pyplot (plt.subplot, plt.title, …) returns the no-op.
    pyplot.__getattr__ = lambda _name: _Any()

    mpl_toolkits = _make("mpl_toolkits")
    mplot3d = _make("mpl_toolkits.mplot3d", mpl_toolkits)
    mplot3d.Axes3D = _Any


_install_matplotlib_stub()
