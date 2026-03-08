"""
Download OpenVINO face models for the Iris Xe backend (no omz_downloader needed).

Run from repo root:
  python scripts/download_openvino_models.py

Uses face_data/openvino_models/ unless you set OPENVINO_MODELS. Requires no openvino-dev.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "face_data"
DEFAULT_OUT = DATA_DIR / "openvino_models"

# Detection: OpenVINO storage (IR .xml+.bin)
BASE = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2"
DOWNLOADS = [
    (f"{BASE}/face-detection-retail-0004/FP32/face-detection-retail-0004.xml", "intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"),
    (f"{BASE}/face-detection-retail-0004/FP32/face-detection-retail-0004.bin", "intel/face-detection-retail-0004/FP32/face-detection-retail-0004.bin"),
]
# Recognition: Hugging Face ONNX (OMZ IR URLs often return 404 HTML)
RECOGNITION_ONNX_URL = "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx"
RECOGNITION_ONNX_REL = "public/face-recognition-resnet100-arcface-onnx/FP32/arcfaceresnet100-8.onnx"


def main():
    out_dir = Path(os.environ.get("OPENVINO_MODELS", str(DEFAULT_OUT)))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    for url, rel_path in DOWNLOADS:
        dest = out_dir / rel_path
        if dest.exists():
            print(f"[exists] {rel_path}")
            continue
        print(f"Downloading {rel_path} ...")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)
            print(f"  -> {dest}")
        except Exception as e:
            print(f"  Failed: {e}")

    # Recognition model: ONNX from Hugging Face (~260 MB)
    rec_dest = out_dir / RECOGNITION_ONNX_REL
    if not rec_dest.exists():
        print("Downloading face recognition model (ONNX, ~260 MB) ...")
        try:
            rec_dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(RECOGNITION_ONNX_URL, rec_dest)
            print(f"  -> {rec_dest}")
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print(f"[exists] {RECOGNITION_ONNX_REL}")

    print("\nDone. Start the app and set Backend to GPU Iris Xe (OpenVINO).")


if __name__ == "__main__":
    main()
