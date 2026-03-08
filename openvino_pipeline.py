"""
OpenVINO face detection + recognition pipeline for Intel Iris Xe (or CPU).

Uses Open Model Zoo models; same DB schema as face_scan (512-dim embeddings).
Install: pip install openvino
Models: run once (see README):
  omz_downloader --name face-detection-retail-0004
  omz_downloader --name face-recognition-resnet100-arcface-onnx
Then set OPENVINO_MODELS to the download output dir, or place models in
face_data/openvino_models/ (see _models_dir()).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Reuse from face_scan where possible
from face_scan import (
    DATA_DIR,
    PHOTO_EXTENSIONS,
    THUMB_SIZE,
    FACE_MARGIN,
    init_db,
    cluster_faces,
    compute_file_hash,
    extract_date,
)

DB_PATH_OV = DATA_DIR / "faces_ov.db"
THUMB_DIR_OV = DATA_DIR / "thumbs_ov"
DET_INPUT_H, DET_INPUT_W = 300, 300
REC_INPUT_SIZE = 112
CONF_THRESHOLD = 0.6


def _models_dir() -> Path:
    d = os.environ.get("OPENVINO_MODELS")
    if d:
        return Path(d)
    return DATA_DIR / "openvino_models"


def _find_model(name: str, ext: str = "xml") -> Path | None:
    base = _models_dir()
    # OMZ layout: intel/face-detection-retail-0004/FP32/... or public/.../...
    for pattern in (
        f"**/intel/{name}/**/{name}.{ext}",
        f"**/public/{name}/**/{name}.{ext}",
        f"**/{name}.{ext}",
    ):
        found = list(base.glob(pattern))
        if found:
            return found[0]
    return None


def load_image_cv2(file_path: Path):
    try:
        img = cv2.imread(str(file_path))
        if img is None:
            pil_img = Image.open(file_path)
            pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return None


class OpenVINOPipeline:
    """Face detection + recognition using OpenVINO (GPU or CPU)."""

    def __init__(self, device: str = "GPU"):
        import openvino as ov

        self.device = "GPU" if device.upper() == "GPU" else "CPU"
        core = ov.Core()

        det_path = _find_model("face-detection-retail-0004") or _find_model(
            "face-detection-retail-0004", "onnx"
        )
        # Prefer ONNX for recognition (OMZ IR .xml/.bin often 404; Hugging Face has .onnx)
        rec_path = _find_model("face-recognition-resnet100-arcface-onnx", "onnx") or _find_model(
            "face-recognition-resnet100-arcface-onnx"
        ) or _find_model("arcfaceresnet100-8", "onnx")
        if not det_path or not rec_path:
            raise FileNotFoundError(
                "OpenVINO models not found. Run:\n"
                "  pip install openvino-dev\n"
                "  omz_downloader --name face-detection-retail-0004\n"
                "  omz_downloader --name face-recognition-resnet100-arcface-onnx\n"
                f"Then set OPENVINO_MODELS to the output dir, or put models under {_models_dir()}"
            )

        self.det = core.compile_model(det_path, self.device)
        self.rec = core.compile_model(rec_path, self.device)
        self.det_out_key = list(self.det.outputs)[0]
        # ArcFace ONNX: use the 512-dim embedding output (first output is usually correct)
        rec_outputs = list(self.rec.outputs)
        self.rec_out_key = rec_outputs[0]
        for out in rec_outputs:
            try:
                sh = getattr(out, "get_shape", lambda: None)()
                if sh is not None:
                    dims = list(sh)
                    if dims and dims[-1] == 512:
                        self.rec_out_key = out
                        break
                name = getattr(out, "get_any_name", lambda: "")()
                if name and ("fc1" in (name or "").lower() or "pre_fc1" in (name or "").lower()):
                    self.rec_out_key = out
                    break
            except Exception:
                pass

    def get_faces(self, img_bgr: np.ndarray):
        """Return list of (bbox_xyxy, embedding_512), L2-normalized."""
        h, w = img_bgr.shape[:2]
        # Detection expects [1,3,300,300] BGR
        blob = cv2.dnn.blobFromImage(
            img_bgr, 1.0, (DET_INPUT_W, DET_INPUT_H), (104, 177, 123), swapRB=False
        )
        det_out = self.det([blob])[self.det_out_key]
        # [1,1,N,7] -> (img_id, label, conf, xmin, ymin, xmax, ymax)
        dets = det_out.squeeze()
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        faces = []
        for d in dets:
            if d[2] < CONF_THRESHOLD:
                continue
            xmin, ymin, xmax, ymax = d[3], d[4], d[5], d[6]
            # OMZ detection often in 0-1 range
            if xmax <= 1 and ymax <= 1:
                xmin, ymin, xmax, ymax = xmin * w, ymin * h, xmax * w, ymax * h
            xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
            xmax, ymax = min(w, int(xmax)), min(h, int(ymax))
            if xmax <= xmin or ymax <= ymin:
                continue
            # Crop and resize to 112x112 for recognition
            # Model (ArcFace ResNet100) expects RGB, normalized (pixel - 127.5) / 128.0
            crop = img_bgr[ymin:ymax, xmin:xmax]
            crop = cv2.resize(crop, (REC_INPUT_SIZE, REC_INPUT_SIZE))
            rec_blob = cv2.dnn.blobFromImage(
                crop, 1.0 / 128.0, (REC_INPUT_SIZE, REC_INPUT_SIZE), (127.5, 127.5, 127.5), swapRB=True
            )
            emb = self.rec([rec_blob])[self.rec_out_key].squeeze()
            norm = np.linalg.norm(emb) + 1e-10
            emb = (emb / norm).astype(np.float32)
            bbox_xyxy = (xmin, ymin, xmax, ymax)
            faces.append((bbox_xyxy, emb))
        return faces


def _buffalo_dir() -> Path | None:
    """Base dir for buffalo_l pack (must contain det_10g.onnx and w600k_r50.onnx)."""
    base = Path(os.environ.get("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface")))
    d = base / "models" / "buffalo_l"
    return d if d.is_dir() else None


def _buffalo_rec_path() -> Path | None:
    """Path to buffalo_l recognition ONNX (w600k_r50.onnx)."""
    d = _buffalo_dir()
    if d is None:
        return None
    rec = d / "w600k_r50.onnx"
    return rec if rec.exists() else None


class OpenVINOHybridPipeline:
    """OpenVINO for detection (fast on Iris); SCRFD 5-kps + align + InsightFace rec (reliable clustering)."""

    REC_SIZE = 112  # buffalo_l recognition input (aligned)

    def __init__(self, device: str = "GPU"):
        import openvino as ov
        import onnxruntime as ort
        from face_scan import _face_providers

        self.device = "GPU" if device.upper() == "GPU" else "CPU"
        core = ov.Core()
        det_path = _find_model("face-detection-retail-0004") or _find_model(
            "face-detection-retail-0004", "onnx"
        )
        if not det_path:
            raise FileNotFoundError(
                "OpenVINO detection model not found. Run:\n"
                "  python scripts/download_openvino_models.py\n"
                f"Or set OPENVINO_MODELS. See {_models_dir()}"
            )
        self.det = core.compile_model(det_path, self.device)
        self.det_out_key = list(self.det.outputs)[0]

        buffalo = _buffalo_dir()
        if not buffalo:
            raise FileNotFoundError(
                "InsightFace buffalo_l model pack not found.\n"
                "Run a CPU scan once so InsightFace downloads buffalo_l, or set INSIGHTFACE_HOME."
            )
        rec_path = buffalo / "w600k_r50.onnx"
        if not rec_path.exists():
            raise FileNotFoundError(
                "InsightFace buffalo_l recognition model (w600k_r50.onnx) not found."
            )
        det_scrfd_path = buffalo / "det_10g.onnx"
        if not det_scrfd_path.exists():
            raise FileNotFoundError(
                "InsightFace buffalo_l detector (det_10g.onnx) not found for alignment."
            )
        providers = _face_providers()
        self.rec = ort.InferenceSession(
            str(rec_path), sess_options=ort.SessionOptions(), providers=providers
        )
        self._rec_in_name = self.rec.get_inputs()[0].name
        # SCRFD on crops to get 5 keypoints for ArcFace alignment. Use 640x640 so output
        # shapes match the model's static dims (avoids ONNX shape mismatch warnings).
        from insightface.model_zoo import scrfd
        scrfd_sess = ort.InferenceSession(
            str(det_scrfd_path), sess_options=ort.SessionOptions(), providers=providers
        )
        self.scrfd = scrfd.SCRFD(model_file=str(det_scrfd_path), session=scrfd_sess)
        self.scrfd.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)

    def get_faces(self, img_bgr: np.ndarray):
        """Detect with OpenVINO; align each face with SCRFD 5-kps + norm_crop; then run rec."""
        from insightface.utils import face_align

        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            img_bgr, 1.0, (DET_INPUT_W, DET_INPUT_H), (104, 177, 123), swapRB=False
        )
        det_out = self.det([blob])[self.det_out_key]
        dets = det_out.squeeze()
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        bboxes = []
        aligned_list = []  # 112x112 RGB per face
        for d in dets:
            if d[2] < CONF_THRESHOLD:
                continue
            xmin, ymin, xmax, ymax = d[3], d[4], d[5], d[6]
            if xmax <= 1 and ymax <= 1:
                xmin, ymin, xmax, ymax = xmin * w, ymin * h, xmax * w, ymax * h
            xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
            xmax, ymax = min(w, int(xmax)), min(h, int(ymax))
            if xmax <= xmin or ymax <= ymin:
                continue
            face_h, face_w = ymax - ymin, xmax - xmin
            margin_h = int(face_h * FACE_MARGIN)
            margin_w = int(face_w * FACE_MARGIN)
            t = max(0, ymin - margin_h)
            b = min(h, ymax + margin_h)
            left = max(0, xmin - margin_w)
            right = min(w, xmax + margin_w)
            crop = img_bgr[t:b, left:right]
            # SCRFD on crop to get 5 keypoints for alignment (same as full InsightFace pipeline)
            det_boxes, kpss = self.scrfd.detect(crop, input_size=(640, 640), max_num=1)
            if kpss is not None and len(kpss) > 0:
                kps_crop = kpss[0].astype(np.float32)  # 5x2 in crop coords
                kps_full = kps_crop + np.array([left, t], dtype=np.float32)
                aimg = face_align.norm_crop(img_bgr, kps_full, image_size=self.REC_SIZE)
                aligned_list.append(cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB))
                bboxes.append((xmin, ymin, xmax, ymax))
            else:
                # Fallback: unaligned resize (worse but better than skipping)
                crop_resized = cv2.resize(crop, (self.REC_SIZE, self.REC_SIZE))
                aligned_list.append(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
                bboxes.append((xmin, ymin, xmax, ymax))
        if not aligned_list:
            return []
        # Recognition: buffalo_l rec expects (pixel - 127.5) / 127.5 (ArcFaceONNX input_std)
        batch = np.stack(aligned_list, axis=0).astype(np.float32)
        batch = (batch - 127.5) / 127.5
        batch = np.transpose(batch, (0, 3, 1, 2))
        embs_list = []
        for i in range(len(aligned_list)):
            out = self.rec.run(None, {self._rec_in_name: batch[i : i + 1]})
            embs_list.append(out[0].squeeze(0))
        embs = np.stack(embs_list, axis=0)
        norm = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        embs = (embs / norm).astype(np.float32)
        return list(zip(bboxes, embs))


def process_photo(
    file_path: Path,
    conn: sqlite3.Connection,
    pipeline: OpenVINOPipeline,
    thumb_dir: Path,
    file_hash: str | None = None,
) -> int:
    """Process one photo; same contract as face_scan.process_photo. Returns face count."""
    img = load_image_cv2(file_path)
    if img is None:
        return 0

    h, w = img.shape[:2]
    max_dim = 2048
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    photo_date, date_source = extract_date(file_path, img=img)

    try:
        faces = pipeline.get_faces(img)
    except Exception:
        faces = []

    cursor = conn.execute(
        "INSERT INTO photos (file_path, file_hash, photo_date, date_source, file_size, processed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(file_path), file_hash, photo_date, date_source,
         file_path.stat().st_size, datetime.now().isoformat()),
    )
    photo_id = cursor.lastrowid

    if not faces:
        conn.commit()
        return 0

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for idx, (bbox_xyxy, embedding) in enumerate(faces):
        xmin, ymin, xmax, ymax = bbox_xyxy
        left, top, right, bottom = xmin, ymin, xmax, ymax
        face_h = bottom - top
        face_w = right - left
        margin_h = int(face_h * FACE_MARGIN)
        margin_w = int(face_w * FACE_MARGIN)
        crop_top = max(0, top - margin_h)
        crop_bottom = min(h, bottom + margin_h)
        crop_left = max(0, left - margin_w)
        crop_right = min(w, right + margin_w)
        thumb = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        thumb_filename = f"{photo_id}_{idx}.jpg"
        thumb_path = thumb_dir / thumb_filename
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(str(thumb_path), "JPEG", quality=80)
        conn.execute(
            "INSERT INTO faces (photo_id, face_index, top, right_, bottom, left_, encoding, thumb_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (photo_id, idx, top, right, bottom, left,
             embedding.tobytes(), str(thumb_filename)),
        )

    conn.commit()
    return len(faces)
