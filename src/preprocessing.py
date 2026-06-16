"""
src/preprocessing.py

Replicates the exact preprocessing from 01_preprocess.py and the
training DataLoader so that inference inputs match training inputs.

Pipeline per video:
  1. Read 150 consecutive frames (one clip)
  2. Detect faces (MediaPipe + Haar fallback), forward/backward fill
  3. Crop all 150 frames to 224×224 BGR
  4. Sample 5 evenly-spaced crops → face_crops  (5, 224, 224, 3) uint8 BGR
  5. Compute FFT maps with Hanning window + centre-crop → (5, 96, 96) float32
  6. Extract POS rPPG from all 150 crops → (6, 150) float32
  7. Compute Pearson coherence → (6, 6) float32
  8. Normalize for model: faces→RGB→ImageNet norm, rPPG→z-score per ROI
"""

import cv2  # noqa: F401
import numpy as np
import logging
from pathlib import Path
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

# ── ROI definitions (identical to 01_preprocess.py) ──────────────────────────
ROIS = [
    (slice(10, 50),   slice(67, 157)),    # Forehead centre
    (slice(110, 170), slice(15, 85)),     # Left cheek
    (slice(110, 170), slice(139, 209)),   # Right cheek
    (slice(95, 145),  slice(90, 134)),    # Nose bridge
    (slice(65, 100),  slice(25, 95)),     # Left periorbital
    (slice(65, 100),  slice(129, 199)),   # Right periorbital
]

# ── rPPG filter constants (identical to 01_preprocess.py) ────────────────────
LOW_HZ = 0.7
HIGH_HZ = 4.0
FILTER_ORDER = 5
TARGET_FPS = 24.0

# ── Face detection constants ─────────────────────────────────────────────────
DET_WIDTH = 640
DETECT_STRIDE = 5
FACE_EXPAND = 0.20

# ── MediaPipe setup (optional) ───────────────────────────────────────────────
_MP_DETECTOR = None
_USE_MEDIAPIPE = False

try:
    import mediapipe as mp
    _MP_DETECTOR = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    _USE_MEDIAPIPE = True
    logger.info("MediaPipe loaded for face detection")
except ImportError:
    logger.info("MediaPipe not available, using Haar cascade only")

# ── Haar cascade (always available) ──────────────────────────────────────────
_HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def bandpass_filter(sig, low, high, fs, order):
    """Zero-phase Butterworth bandpass (identical to 01_preprocess.py)."""
    min_len = 3 * (2 * order + 1)
    if sig.shape[-1] < min_len:
        return sig
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig, axis=-1)


def _detect_face(frame_bgr):
    """Detect one face, return (x, y, w, h) in original coords or None."""
    oh, ow = frame_bgr.shape[:2]
    nw = DET_WIDTH
    nh = int(oh * DET_WIDTH / ow)
    small = cv2.resize(frame_bgr, (nw, nh))
    sx, sy = ow / nw, oh / nh

    def _scale(bbox):
        x, y, w, h = bbox
        return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))

    # MediaPipe first
    if _USE_MEDIAPIPE and _MP_DETECTOR is not None:
        try:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = _MP_DETECTOR.process(rgb)
            if res.detections:
                bb = res.detections[0].location_data.relative_bounding_box
                x = max(0, int(bb.xmin * nw))
                y = max(0, int(bb.ymin * nh))
                w = int(bb.width * nw)
                h = int(bb.height * nh)
                return _scale((x, y, w, h))
        except Exception:
            pass

    # Haar fallback
    if not _HAAR.empty():
        try:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = _HAAR.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
            )
            if len(faces) > 0:
                return _scale(tuple(faces[0]))
        except Exception:
            pass

    return None


def _crop_face(frame_bgr, bbox, face_sz=224):
    """Crop face with expansion, resize to face_sz × face_sz."""
    fh, fw = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    px = int(bw * FACE_EXPAND)
    py = int(bh * FACE_EXPAND)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(fw, x + bw + px)
    y2 = min(fh, y + bh + py)
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    return cv2.resize(patch, (face_sz, face_sz), interpolation=cv2.INTER_AREA)


def _compute_fft_map(crop_bgr, fft_sz=96):
    """
    FFT map with Hanning window + centre-crop.
    Identical to 01_preprocess.py compute_fft_map().
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray * win))))
    rng = mag.max() - mag.min()
    mag = (mag - mag.min()) / (rng + 1e-8)
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    half = fft_sz // 2
    return mag[cy - half: cy + half, cx - half: cx + half].astype(np.float32)


def _extract_rppg(all_crops, n_rois=6, clip_f=150):
    """
    POS rPPG extraction (Wang et al. 2017).
    Identical to 01_preprocess.py extract_rppg().

    Args:
        all_crops: list of clip_f BGR uint8 arrays, each (224, 224, 3)

    Returns:
        signals   (n_rois, clip_f) float32
        coherence (n_rois, n_rois) float32
    """
    T = len(all_crops)
    raw = np.zeros((n_rois, T, 3), dtype=np.float32)

    for t, bgr in enumerate(all_crops):
        if bgr is None:
            continue
        rgb = bgr[:, :, ::-1].astype(np.float32)
        for r, (rs, cs) in enumerate(ROIS):
            patch = rgb[rs, cs].reshape(-1, 3)
            if patch.shape[0] > 0:
                raw[r, t] = patch.mean(axis=0)

    signals = np.zeros((n_rois, T), dtype=np.float32)
    for r in range(n_rois):
        C = raw[r]
        mu = C.mean(axis=0)
        if np.any(mu < 1.0):
            continue
        Cn = C / (mu + 1e-8) - 1.0
        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = Cn[:, 1] + Cn[:, 2] - 2.0 * Cn[:, 0]
        alpha = float(np.std(S1)) / (float(np.std(S2)) + 1e-8)
        signals[r] = S1 - alpha * S2

    signals = bandpass_filter(signals, LOW_HZ, HIGH_HZ, TARGET_FPS, FILTER_ORDER)

    # Pearson coherence
    valid_mask = signals.std(axis=1) > 1e-8
    normed = np.zeros_like(signals)
    for r in range(n_rois):
        if valid_mask[r]:
            s = signals[r]
            normed[r] = (s - s.mean()) / (s.std() + 1e-8)
    coherence = np.clip((normed @ normed.T) / T, -1.0, 1.0).astype(np.float32)

    return signals.astype(np.float32), coherence


def preprocess_video(video_path: str, cfg: dict):
    """
    Full preprocessing pipeline for one video.

    Reads up to clip_f (150) frames, detects/crops faces,
    extracts all features matching training format.

    Args:
        video_path: Path to .mp4 file
        cfg: Model config dict with n_faces, face_sz, fft_sz, n_rois, clip_f

    Returns:
        dict with keys:
            'faces'     : torch.Tensor (1, 5, 3, 224, 224) float32, ImageNet-normalized RGB
            'fft'       : torch.Tensor (1, 5, 96, 96) float32
            'rppg'      : torch.Tensor (1, 6, 150) float32, z-scored per ROI
            'coherence' : torch.Tensor (1, 6, 6) float32

    Raises:
        FileNotFoundError, ValueError on failures
    """
    import torch

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    n_faces = cfg["n_faces"]    # 5
    face_sz = cfg["face_sz"]    # 224
    fft_sz = cfg["fft_sz"]      # 96
    n_rois = cfg["n_rois"]      # 6
    clip_f = cfg["clip_f"]      # 150

    logger.info(f"Preprocessing: {video_path.name}")

    # ── Step 1: Read up to clip_f frames ─────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"  Video has {total_frames} frames, reading up to {clip_f}")

    frames = []
    while len(frames) < clip_f:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 30:
        raise ValueError(f"Video too short: {len(frames)} frames (need ≥30)")

    # Pad to clip_f if needed
    while len(frames) < clip_f:
        frames.append(frames[-1].copy())

    frames = frames[:clip_f]
    logger.info(f"  Using {len(frames)} frames")

    # ── Step 2: Face detection (sparse + fill) ───────────────────────────
    bboxes = [None] * clip_f
    for i in range(0, clip_f, DETECT_STRIDE):
        bboxes[i] = _detect_face(frames[i])

    det_count = sum(1 for b in bboxes if b is not None)
    n_tried = len(range(0, clip_f, DETECT_STRIDE))
    logger.info(f"  Face detections: {det_count}/{n_tried} attempts")

    if det_count == 0:
        raise ValueError("No face detected in any frame")

    # Forward fill
    last = None
    for i in range(clip_f):
        if bboxes[i] is not None:
            last = bboxes[i]
        elif last is not None:
            bboxes[i] = last

    # Backward fill
    last = None
    for i in range(clip_f - 1, -1, -1):
        if bboxes[i] is not None:
            last = bboxes[i]
        elif last is not None:
            bboxes[i] = last

    # ── Step 3: Crop all frames ──────────────────────────────────────────
    all_crops = []
    for i in range(clip_f):
        if bboxes[i] is not None:
            crop = _crop_face(frames[i], bboxes[i], face_sz)
            all_crops.append(crop)
        else:
            all_crops.append(None)

    # Fill None crops
    last_crop = None
    for i in range(clip_f):
        if all_crops[i] is not None:
            last_crop = all_crops[i]
        elif last_crop is not None:
            all_crops[i] = last_crop

    last_crop = None
    for i in range(clip_f - 1, -1, -1):
        if all_crops[i] is not None:
            last_crop = all_crops[i]
        elif last_crop is not None:
            all_crops[i] = last_crop

    if any(c is None for c in all_crops):
        raise ValueError("Could not crop face from any frame")

    # ── Step 4: Sample n_faces evenly-spaced crops ───────────────────────
    indices = np.linspace(0, clip_f - 1, n_faces, dtype=int)
    sampled_crops = [all_crops[i] for i in indices]
    face_arr = np.stack(sampled_crops, axis=0).astype(np.uint8)  # (5, 224, 224, 3) BGR

    # ── Step 5: FFT maps (Hanning + centre-crop, matching training) ──────
    fft_arr = np.stack(
        [_compute_fft_map(c, fft_sz) for c in sampled_crops], axis=0
    )  # (5, 96, 96) float32

    # ── Step 6: rPPG from all clip_f crops ───────────────────────────────
    rppg_arr, coh_arr = _extract_rppg(all_crops, n_rois, clip_f)
    # rppg_arr: (6, 150) float32
    # coh_arr:  (6, 6)   float32

    # ── Step 7: Convert to tensors matching training DataLoader output ───

    # Faces: BGR uint8 → RGB float32 → ImageNet normalize
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    faces_rgb = face_arr[:, :, :, ::-1].copy().astype(np.float32) / 255.0
    faces_t = torch.from_numpy(faces_rgb).permute(0, 3, 1, 2)  # (5, 3, 224, 224)
    faces_t = (faces_t - IMAGENET_MEAN) / IMAGENET_STD
    faces_t = faces_t.unsqueeze(0)  # (1, 5, 3, 224, 224)

    # FFT: already float32, just add batch dim
    fft_t = torch.from_numpy(fft_arr).unsqueeze(0)  # (1, 5, 96, 96)

    # rPPG: z-score per ROI (matching training DataLoader)
    mu = rppg_arr.mean(axis=1, keepdims=True)
    std = rppg_arr.std(axis=1, keepdims=True) + 1e-8
    rppg_normed = (rppg_arr - mu) / std
    rppg_t = torch.from_numpy(rppg_normed).unsqueeze(0)  # (1, 6, 150)

    # Coherence: as-is
    coh_t = torch.from_numpy(coh_arr).unsqueeze(0)  # (1, 6, 6)

    logger.info(f"  Tensors: faces={tuple(faces_t.shape)}, fft={tuple(fft_t.shape)}, "
                f"rppg={tuple(rppg_t.shape)}, coh={tuple(coh_t.shape)}")

    return {
        "faces": faces_t,
        "fft": fft_t,
        "rppg": rppg_t,
        "coherence": coh_t,
    }