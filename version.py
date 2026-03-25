"""
version.py — Single source of truth for the novoface version number.

Import __version__ from here anywhere in the Python codebase.

Run directly to generate build artifacts consumed by the installer toolchain:
    python version.py
    → installer/version.iss       (Inno Setup #define)
    → installer/version_info.txt  (PyInstaller Windows VERSIONINFO)
"""

__version__ = "0.0.1"
__date__    = "2026-03-24"   # date of this release


def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse "X.Y.Z" → (X, Y, Z, 0) — always 4 components for Windows VERSIONINFO."""
    parts = [int(x) for x in v.split(".")]
    return tuple(parts + [0] * (4 - len(parts)))


if __name__ == "__main__":
    from pathlib import Path

    Path("installer").mkdir(exist_ok=True)

    # ── Inno Setup: #include "version.iss" ───────────────────────────────
    iss_path = Path("installer/version.iss")
    iss_path.write_text(f'#define MyAppVersion "{__version__}"\n', encoding="utf-8")
    print(f"  wrote {iss_path}")

    # ── PyInstaller: Windows VERSIONINFO resource ─────────────────────────
    vt = _version_tuple(__version__)
    ver_str = ".".join(str(x) for x in vt)
    desc = "novoface \u2014 face recognition for genealogy photos"

    vinfo_path = Path("installer/version_info.txt")
    vinfo_path.write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={vt},
    prodvers={vt},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [StringStruct('CompanyName', 'novoface'),
         StringStruct('FileDescription', '{desc}'),
         StringStruct('FileVersion', '{ver_str}'),
         StringStruct('InternalName', 'novoface'),
         StringStruct('LegalCopyright', ''),
         StringStruct('OriginalFilename', 'novoface.exe'),
         StringStruct('ProductName', 'novoface'),
         StringStruct('ProductVersion', '{ver_str}')])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""",
        encoding="utf-8",
    )
    print(f"  wrote {vinfo_path}")

    print(f"\nVersion {__version__} artifacts ready.")
    print("Next steps:")
    print("  pyinstaller novoface.spec")
    print("  iscc installer\\novoface.iss")
