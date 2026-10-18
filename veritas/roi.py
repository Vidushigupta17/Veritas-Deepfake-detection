"""Step 1: face + ROI extraction (OpenCV + MediaPipe Face Landmarker)."""
from __future__ import annotations
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .config import REGIONS

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")

# FaceMesh landmark index sets (cheeks are mirror pairs). Convex hull is taken, so order is irrelevant.
ROI_LANDMARKS = {
    "forehead": [67, 109, 10, 338, 297, 103, 332, 104, 333, 69, 299, 66, 296, 107, 336, 9],
    "left_cheek": [50, 101, 118, 117, 123, 187, 205, 36, 119, 100, 142],     # image-left
    "right_cheek": [280, 330, 347, 346, 352, 411, 425, 266, 348, 329, 371],  # image-right
}
ROI_COLORS = {"forehead": (255, 160, 0), "left_cheek": (0, 200, 0), "right_cheek": (0, 120, 255)}


def ensure_model(path: str = MODEL_PATH) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            print(f"Downloading Face Landmarker model -> {path}")
            urllib.request.urlretrieve(MODEL_URL, path)
        except Exception as e:
            raise RuntimeError(
                f"Could not download the model automatically ({e}).\n"
                f"Download it manually from:\n  {MODEL_URL}\nand save it as:\n  {path}")
    return path


class FaceROI:
    def __init__(self, model_path: str | None = None):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(model_path or MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1)
        self.lm = vision.FaceLandmarker.create_from_options(opts)

    def landmarks(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)

    def close(self):
        self.lm.close()


def roi_masks(points, shape):
    """points: (N,2) pixel landmarks -> dict region -> uint8 mask."""
    masks = {}
    for name in REGIONS:
        hull = cv2.convexHull(points[ROI_LANDMARKS[name]].astype(np.int32))
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(m, hull, 1)
        masks[name] = m
    return masks


def mean_rgb(frame_bgr, masks):
    """Step 2a: mean R,G,B inside each ROI."""
    out = {}
    for name, m in masks.items():
        px = frame_bgr[m.astype(bool)]
        out[name] = px[:, ::-1].mean(axis=0) if len(px) else np.full(3, np.nan)
    return out


def draw_rois(frame_bgr, masks, alpha=0.35, colors=None):
    colors = colors or ROI_COLORS
    out = frame_bgr.copy()
    for name, m in masks.items():
        layer = out.copy()
        layer[m.astype(bool)] = colors[name]
        out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
    return out


def extract_rgb_traces(video_path, max_frames=None, progress=None, keep_frame_at=None):
    """Run the whole video. Returns (traces{region:(T,3)}, fps, valid_mask(T,), sample_frame, sample_masks).
    Frames with no detected face are NaN and later interpolated."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fr = FaceROI()
    traces = {r: [] for r in REGIONS}
    valid, sample, sample_masks = [], None, None
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and i >= max_frames):
                break
            pts = fr.landmarks(frame)
            if pts is None:
                for r in REGIONS: traces[r].append(np.full(3, np.nan))
                valid.append(False)
            else:
                masks = roi_masks(pts, frame.shape)
                rgb = mean_rgb(frame, masks)
                for r in REGIONS: traces[r].append(rgb[r])
                valid.append(True)
                if sample is None and (keep_frame_at is None or i >= keep_frame_at):
                    sample, sample_masks = frame.copy(), masks
            i += 1
            if progress and total: progress(min(i / (max_frames or total), 1.0))
    finally:
        cap.release(); fr.close()
    return {r: np.array(v) for r, v in traces.items()}, float(fps), np.array(valid), sample, sample_masks
