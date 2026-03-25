<#
.SYNOPSIS
    Build the novoface Windows installer (novoface-X.Y.Z-setup.exe).

.DESCRIPTION
    Full packaging pipeline -- run this once to produce a distributable installer:

      Step 1  python version.py
              Reads __version__ from version.py and writes two build artifacts:
                installer/version.iss        -- Inno Setup version #define
                installer/version_info.txt   -- Windows VERSIONINFO resource
                                               (embedded in the .exe by PyInstaller)

      Step 2  pyinstaller novoface.spec
              Bundles the Python app + all dependencies (Flask, InsightFace,
              onnxruntime, pywebview, etc.) into a self-contained folder:
                dist/novoface/novoface.exe

              If openvino is installed in the active Python environment, it is
              also bundled and the GPU acceleration option appears in the app's
              first-run setup dialog.  Otherwise it is silently omitted and the
              app runs CPU-only.

      Step 3  iscc installer\novoface.iss
              Inno Setup Compiler packages dist/novoface/ into a single
              installer executable:
                installer/Output/novoface-X.Y.Z-setup.exe

.PREREQUISITES
    pip install pyinstaller pywebview platformdirs
    pip install openvino          (optional -- enables GPU acceleration in the bundle)

    Inno Setup 6.7.1 -- the --location flag is required; without it winget installs
    to a path that cannot be resolved by scripts:

        winget install --id JRSoftware.InnoSetup `
            --location "C:\Program Files\Inno Setup 6" `
            --accept-package-agreements --accept-source-agreements

    Installs ISCC.exe to: C:\Program Files\Inno Setup 6\ISCC.exe
    This script auto-discovers that path -- no PATH changes needed.

.USAGE
    From the repo root in PowerShell:
        .\installer\build.ps1

    To skip the PyInstaller step when only the .iss script changed:
        .\installer\build.ps1 -SkipPyInstaller
#>

param(
    [switch]$SkipPyInstaller
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -- Locate repo root (one level up from this script) -------------------------
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

# -- Locate iscc.exe (Inno Setup Compiler) ------------------------------------
$IsccCandidates = @(
    "$env:ProgramFiles\Inno Setup 6\iscc.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\iscc.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) {
    Write-Error "Inno Setup 6 not found.  Install it with:  winget install JRSoftware.InnoSetup"
}

# -- Helper: run a command and stop on failure --------------------------------
function Invoke-Step {
    param([string]$Label, [scriptblock]$Command)
    Write-Host ""
    Write-Host "--- $Label ---" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Label failed (exit code $LASTEXITCODE)."
    }
}

# -- Step 1: generate version artifacts ---------------------------------------
Invoke-Step "Step 1/3 -- Generate version artifacts" {
    python version.py
}

# Read the version back for the final summary message
$Version = (python -c "from version import __version__; print(__version__)")

# -- Step 2: PyInstaller -------------------------------------------------------
if (-not $SkipPyInstaller) {
    Invoke-Step "Step 2/3 -- PyInstaller (bundle app)" {
        pyinstaller novoface.spec --noconfirm
    }
} else {
    Write-Host ""
    Write-Host "--- Step 2/3 -- PyInstaller  [skipped via -SkipPyInstaller] ---" -ForegroundColor DarkGray
}

# -- Step 3: Inno Setup -------------------------------------------------------
Invoke-Step "Step 3/3 -- Inno Setup (create installer)" {
    & $Iscc "installer\novoface.iss"
}

# -- Cleanup ------------------------------------------------------------------
Remove-Item -Recurse -Force "dist\novoface" -ErrorAction SilentlyContinue

# -- Done ---------------------------------------------------------------------
$Output = "installer\Output\novoface-$Version-setup.exe"
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Done!  ->  $Output" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
