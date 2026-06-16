"""
scripts/01_preprocess.py

Converts raw DFD videos → single HDF5 archive (~6 GB).
Extracts per 150-frame clip:
  face_crops   (5, 224, 224, 3)  uint8   — spatial branch
  fft_maps     (5,  96,  96)     float16 — frequency branch
  rppg_signals (6, 150)          float32 — physiological branch
  coherence    (6,   6)          float32 — cross-ROI coherence

Key properties:
  ✓ Resumable      — restart after crash, picks up where it left off
  ✓ Memory-safe    — reads one clip at a time, never full video in RAM
  ✓ Fast detection — face detected every 5 frames, filled for rest
  ✓ True det-rate  — gate evaluated BEFORE fill (fixed from draft)
  ✓ Dual detector  — MediaPipe primary, OpenCV Haar fallback
  ✓ Full log       — written to data/preprocess.log

Runtime estimate: 4–8 hours on CPU for the full dataset.
Start it before sleeping. Do NOT delete raw videos until complete.

Usage:
    python scripts/01_preprocess.py
"""

import re
import sys
import json
import time
import random
import logging
import datetime
import traceback
from pathlib import Path

import numpy as np
import cv2
import h5py
import yaml
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe — optional. Haar cascade is always available as fallback.
# ─────────────────────────────────────────────────────────────────────────────
USE_MEDIAPIPE = False
MP_DETECTOR   = None

try:
    import mediapipe as mp
    MP_DETECTOR   = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    USE_MEDIAPIPE = True
    print("[INFO] MediaPipe loaded successfully.")
except Exception as _mp_err:
    print(f"[WARN] MediaPipe not available ({_mp_err}). Using Haar cascade only.")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(".")
CONFIG_P   = ROOT / "configs" / "config.yaml"
HDF5_P     = ROOT / "data"   / "deepfake_physio.h5"
SPLITS_P   = ROOT / "data"   / "splits.json"
PROGRESS_P = ROOT / "data"   / "progress.json"
LOG_P      = ROOT / "data"   / "preprocess.log"

(ROOT / "data").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging — file + console
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_P, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
with open(CONFIG_P, encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

REAL_DIR  = Path(cfg["data"]["real_dir"])
FAKE_DIR  = Path(cfg["data"]["fake_dir"])
REAL_PAT  = re.compile(cfg["parsing"]["real_pattern"])
FAKE_PAT  = re.compile(cfg["parsing"]["fake_pattern"])

FACE_SZ    = int(cfg["preprocessing"]["face_size"])              # 224
FFT_SZ     = int(cfg["preprocessing"]["fft_size"])               # 96
N_ROIS     = int(cfg["preprocessing"]["n_rois"])                 # 6
CLIP_F     = int(cfg["preprocessing"]["clip_frames"])            # 150
N_FACES    = int(cfg["preprocessing"]["face_sample_per_clip"])   # 5
MIN_FRAMES = int(cfg["preprocessing"]["min_valid_frames"])       # 100
MIN_DET    = float(cfg["preprocessing"]["min_detection_rate"])   # 0.85
FPS        = float(cfg["preprocessing"]["target_fps"])           # 24

VAL_N  = int(cfg["splits"]["val_actors"])   # 4
TEST_N = int(cfg["splits"]["test_actors"])  # 6
SEED   = int(cfg["splits"]["seed"])         # 42

LOW_HZ   = float(cfg["rppg"]["low_hz"])     # 0.7
HIGH_HZ  = float(cfg["rppg"]["high_hz"])    # 4.0
FILT_ORD = int(cfg["rppg"]["filter_order"]) # 5

# Detection optimisation
DETECT_STRIDE = 5    # Run detector every 5 frames → 30 detections per clip
DET_WIDTH     = 640  # Downscale to this width before detection for speed

# ─────────────────────────────────────────────────────────────────────────────
# 6 ROI definitions on 224×224 face crop (row_slice, col_slice)
# ─────────────────────────────────────────────────────────────────────────────
ROIS = [
    (slice(10,  50),  slice(67,  157)),   # 0 — Forehead centre
    (slice(110, 170), slice(15,   85)),   # 1 — Left cheek
    (slice(110, 170), slice(139, 209)),   # 2 — Right cheek
    (slice(95,  145), slice(90,  134)),   # 3 — Nose bridge
    (slice(65,  100), slice(25,   95)),   # 4 — Left periorbital
    (slice(65,  100), slice(129, 199)),   # 5 — Right periorbital
]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Signal processing
# ═════════════════════════════════════════════════════════════════════════════

def bandpass_filter(sig: np.ndarray, low: float, high: float,
                    fs: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth bandpass. Skips safely if signal too short."""
    min_len = 3 * (2 * order + 1)
    if sig.shape[-1] < min_len:
        return sig
    nyq  = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig, axis=-1)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Face detection and cropping
# ═════════════════════════════════════════════════════════════════════════════

def detect_face(frame_bgr: np.ndarray, haar) -> tuple | None:
    """
    Returns (x, y, w, h) in original pixel coordinates, or None.
    Internally resizes to DET_WIDTH for speed.
    Tries MediaPipe first, then Haar cascade.
    """
    oh, ow = frame_bgr.shape[:2]
    nw     = DET_WIDTH
    nh     = int(oh * DET_WIDTH / ow)
    small  = cv2.resize(frame_bgr, (nw, nh))
    sx     = ow / nw
    sy     = oh / nh

    def _scale(bbox):
        x, y, w, h = bbox
        return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))

    # MediaPipe
    if USE_MEDIAPIPE and MP_DETECTOR is not None:
        try:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = MP_DETECTOR.process(rgb)
            if res.detections:
                bb = res.detections[0].location_data.relative_bounding_box
                x  = max(0, int(bb.xmin  * nw))
                y  = max(0, int(bb.ymin  * nh))
                w  = int(bb.width  * nw)
                h  = int(bb.height * nh)
                return _scale((x, y, w, h))
        except Exception:
            pass

    # Haar cascade fallback
    if haar is not None and not haar.empty():
        try:
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
            )
            if len(faces) > 0:
                return _scale(tuple(faces[0]))
        except Exception:
            pass

    return None


def crop_face(frame_bgr: np.ndarray, bbox: tuple,
              expand: float = 0.20) -> np.ndarray | None:
    """
    Expand bbox by `expand` ratio on each side, clamp to frame boundaries,
    then resize to FACE_SZ × FACE_SZ uint8 BGR.
    """
    fh, fw      = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    px = int(bw * expand)
    py = int(bh * expand)
    x1 = max(0,  x  - px)
    y1 = max(0,  y  - py)
    x2 = min(fw, x + bw + px)
    y2 = min(fh, y + bh + py)
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    return cv2.resize(patch, (FACE_SZ, FACE_SZ), interpolation=cv2.INTER_AREA)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FFT map
# ═════════════════════════════════════════════════════════════════════════════

def compute_fft_map(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Log-magnitude 2-D FFT of a face crop, centre-cropped to FFT_SZ × FFT_SZ.
    Hanning window applied before FFT to suppress spectral leakage.
    Returns float16 array of shape (FFT_SZ, FFT_SZ).
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win  = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    mag  = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray * win))))
    rng  = mag.max() - mag.min()
    mag  = (mag - mag.min()) / (rng + 1e-8)        # Normalise → [0, 1]
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    half   = FFT_SZ // 2
    return mag[cy - half : cy + half,
               cx - half : cx + half].astype(np.float16)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — rPPG extraction (POS algorithm, Wang et al. 2017)
# ═════════════════════════════════════════════════════════════════════════════

def extract_rppg(
    face_crops: list,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    POS rPPG extraction across 6 facial ROIs.

    Parameters
    ----------
    face_crops : list of CLIP_F arrays, each (224, 224, 3) BGR uint8

    Returns
    -------
    signals   : (N_ROIS, CLIP_F) float32  bandpass-filtered rPPG per ROI
    coherence : (N_ROIS, N_ROIS) float32  Pearson cross-ROI correlation
    valid     : bool  True when ≥ 4 ROIs produced non-trivial signals
    """
    T   = len(face_crops)
    raw = np.zeros((N_ROIS, T, 3), dtype=np.float32)

    for t, bgr in enumerate(face_crops):
        if bgr is None:
            continue
        rgb = bgr[:, :, ::-1].astype(np.float32)   # BGR → RGB
        for r, (rs, cs) in enumerate(ROIS):
            patch = rgb[rs, cs].reshape(-1, 3)
            if patch.shape[0] > 0:
                raw[r, t] = patch.mean(axis=0)

    signals = np.zeros((N_ROIS, T), dtype=np.float32)
    for r in range(N_ROIS):
        C  = raw[r]                                      # (T, 3)
        mu = C.mean(axis=0)
        if np.any(mu < 1.0):                             # Near-black → skip
            continue
        Cn    = C / (mu + 1e-8) - 1.0                   # Temporal normalise
        S1    = Cn[:, 1] - Cn[:, 2]                     # G − B
        S2    = Cn[:, 1] + Cn[:, 2] - 2.0 * Cn[:, 0]  # G + B − 2R
        alpha = float(np.std(S1)) / (float(np.std(S2)) + 1e-8)
        signals[r] = S1 - alpha * S2

    signals = bandpass_filter(signals, LOW_HZ, HIGH_HZ, FPS, FILT_ORD)

    # Pearson coherence matrix
    valid_mask = signals.std(axis=1) > 1e-8
    normed     = np.zeros_like(signals)
    for r in range(N_ROIS):
        if valid_mask[r]:
            s = signals[r]
            normed[r] = (s - s.mean()) / (s.std() + 1e-8)
    coherence = np.clip((normed @ normed.T) / T, -1.0, 1.0).astype(np.float32)
    valid     = bool(valid_mask.sum() >= 4)

    return signals.astype(np.float32), coherence, valid


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Split generation
# ═════════════════════════════════════════════════════════════════════════════

def build_or_load_splits() -> dict:
    """
    Load splits.json if it exists; otherwise build and save it.
    Pair-safe rule: a fake video enters split X only if BOTH its target
    actor AND source actor belong to split X.
    """
    if SPLITS_P.exists():
        log.info(f"Loading existing splits from {SPLITS_P}")
        with open(SPLITS_P, encoding="utf-8") as f:
            return json.load(f)

    log.info("Building pair-safe actor splits …")

    real_vids, fake_vids = [], []
    for p in sorted(REAL_DIR.glob("*.mp4")):
        m = REAL_PAT.match(p.name)
        if m:
            real_vids.append({
                "path": str(p), "actor": m.group(1), "label": 0
            })

    for p in sorted(FAKE_DIR.glob("*.mp4")):
        m = FAKE_PAT.match(p.name)
        if m:
            fake_vids.append({
                "path": str(p), "target": m.group(1),
                "source": m.group(2), "method": m.group(4), "label": 1,
            })

    all_actors = sorted({v["actor"] for v in real_vids})
    rng        = random.Random(SEED)
    shuffled   = all_actors.copy()
    rng.shuffle(shuffled)

    test_actors  = set(shuffled[:TEST_N])
    val_actors   = set(shuffled[TEST_N : TEST_N + VAL_N])
    train_actors = set(shuffled[TEST_N + VAL_N :])

    def get_split(aid: str) -> str | None:
        if aid in train_actors: return "train"
        if aid in val_actors:   return "val"
        if aid in test_actors:  return "test"
        return None

    splits: dict = {"train": [], "val": [], "test": []}

    for v in real_vids:
        sp = get_split(v["actor"])
        if sp:
            splits[sp].append(v)

    dropped = 0
    for v in fake_vids:
        t_sp = get_split(v["target"])
        s_sp = get_split(v["source"])
        if t_sp and t_sp == s_sp:
            splits[t_sp].append(v)
        else:
            dropped += 1

    splits["_meta"] = {
        "train_actors":  sorted(train_actors),
        "val_actors":    sorted(val_actors),
        "test_actors":   sorted(test_actors),
        "seed":          SEED,
        "dropped_fakes": dropped,
        "created":       datetime.datetime.now().isoformat(),
    }

    for sp in ("train", "val", "test"):
        r  = sum(1 for v in splits[sp] if v["label"] == 0)
        fk = sum(1 for v in splits[sp] if v["label"] == 1)
        log.info(f"  {sp:<6}: {r} real   {fk} fake")
    log.info(f"  Fake videos dropped (pair-unsafe): {dropped}")

    with open(SPLITS_P, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    log.info(f"Splits saved → {SPLITS_P}")
    return splits


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Progress / resumability
# ═════════════════════════════════════════════════════════════════════════════

def load_progress() -> set:
    if PROGRESS_P.exists():
        with open(PROGRESS_P, encoding="utf-8") as f:
            data = json.load(f)
        done = set(data.get("completed", []))
        log.info(f"Resuming — {len(done)} videos already done, skipping them.")
        return done
    return set()


def save_progress(done: set):
    with open(PROGRESS_P, "w", encoding="utf-8") as f:
        json.dump({
            "completed": sorted(done),
            "count":     len(done),
            "updated":   datetime.datetime.now().isoformat(),
        }, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — HDF5 initialisation and writing
# ═════════════════════════════════════════════════════════════════════════════

def init_hdf5(path: Path) -> h5py.File:
    """
    Open (or create) the HDF5 file and ensure all resizable datasets exist.
    Safe to call on a partially-written file — will not overwrite existing data.
    """
    f      = h5py.File(path, "a")
    str_dt = h5py.string_dtype(encoding="utf-8")

    def _ensure(grp_name: str):
        if grp_name not in f:
            f.create_group(grp_name)
        g = f[grp_name]

        # Numeric datasets — compressed with LZF (fast)
        specs = {
            "face_crops":   ((0, N_FACES, FACE_SZ, FACE_SZ, 3),
                             np.uint8,   (1, N_FACES, FACE_SZ, FACE_SZ, 3)),
            "fft_maps":     ((0, N_FACES, FFT_SZ, FFT_SZ),
                             np.float16, (1, N_FACES, FFT_SZ, FFT_SZ)),
            "rppg_signals": ((0, N_ROIS, CLIP_F),
                             np.float32, (1, N_ROIS, CLIP_F)),
            "coherence":    ((0, N_ROIS, N_ROIS),
                             np.float32, (1, N_ROIS, N_ROIS)),
            "labels":       ((0,), np.int8,   (512,)),
            "rppg_valid":   ((0,), np.bool_,  (512,)),
        }
        for name, (shape, dtype, chunks) in specs.items():
            if name not in g:
                maxshape = (None,) + shape[1:]
                g.create_dataset(name, shape=shape, maxshape=maxshape,
                                 dtype=dtype, chunks=chunks, compression="lzf")

        # String datasets — variable length, no compression
        for name in ("actor_ids", "video_paths", "splits"):
            if name not in g:
                g.create_dataset(name, shape=(0,), maxshape=(None,),
                                 dtype=str_dt, chunks=(512,))

    for sp in ("train", "val", "test"):
        _ensure(sp)

    return f


def append_clip(hf: h5py.File, split: str,
                face_arr: np.ndarray, fft_arr: np.ndarray,
                rppg_arr: np.ndarray, coh_arr: np.ndarray,
                label: int, rppg_ok: bool,
                actor_id: str, vid_path: str):
    """Append one processed clip to the correct HDF5 split group."""
    g = hf[split]
    n = g["labels"].shape[0]   # Current clip count — resize point

    for ds in g.values():
        ds.resize(n + 1, axis=0)

    g["face_crops"][n]   = face_arr
    g["fft_maps"][n]     = fft_arr
    g["rppg_signals"][n] = rppg_arr
    g["coherence"][n]    = coh_arr
    g["labels"][n]       = label
    g["rppg_valid"][n]   = rppg_ok
    g["actor_ids"][n]    = actor_id
    g["video_paths"][n]  = vid_path
    g["splits"][n]       = split


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Single-clip processing
# ═════════════════════════════════════════════════════════════════════════════

def process_clip(frames: list, haar) -> tuple | None:
    """
    Process one CLIP_F-frame clip end-to-end.

    Steps
    -----
    1. Detect faces every DETECT_STRIDE frames.
    2. COUNT detections BEFORE fill (true detection-rate gate).
    3. Forward + backward fill bounding boxes.
    4. Crop every frame to 224×224.
    5. Sample N_FACES evenly-spaced crops → face_arr.
    6. Compute FFT maps for those N_FACES crops → fft_arr.
    7. Extract rPPG + coherence from all CLIP_F crops.

    Returns (face_arr, fft_arr, rppg_arr, coh_arr, rppg_ok) or None.
    """
    T = len(frames)

    # ── Step 1: Sparse detection ───────────────────────────────────────────
    bboxes: list = [None] * T
    for i in range(0, T, DETECT_STRIDE):
        bboxes[i] = detect_face(frames[i], haar)

    # ── Step 2: TRUE detection rate — measured before fill ─────────────────
    #   FIX: counting here (not after fill) gives the real per-frame rate.
    #   A clip with only 1 detection would pass post-fill but is skipped here.
    raw_det_count = sum(1 for b in bboxes if b is not None)
    # We only ran the detector on every DETECT_STRIDE-th frame, so the
    # denominator is the number of frames we actually tried.
    n_tried = len(range(0, T, DETECT_STRIDE))
    if raw_det_count / n_tried < MIN_DET:
        return None

    # ── Step 3: Forward + backward fill ───────────────────────────────────
    last = None
    for i in range(T):
        if bboxes[i] is not None:
            last = bboxes[i]
        elif last is not None:
            bboxes[i] = last

    last = None
    for i in range(T - 1, -1, -1):
        if bboxes[i] is not None:
            last = bboxes[i]
        elif last is not None:
            bboxes[i] = last

    # ── Step 4: Crop every frame ───────────────────────────────────────────
    crops: list = []
    for i in range(T):
        if bboxes[i] is not None:
            crops.append(crop_face(frames[i], bboxes[i]))
        else:
            crops.append(None)

    # Fill any None crops (crop_face can return None on degenerate patches)
    last_crop = None
    for i in range(T):
        if crops[i] is not None:
            last_crop = crops[i]
        elif last_crop is not None:
            crops[i] = last_crop

    last_crop = None
    for i in range(T - 1, -1, -1):
        if crops[i] is not None:
            last_crop = crops[i]
        elif last_crop is not None:
            crops[i] = last_crop

    if any(c is None for c in crops):   # Entire clip had no usable face
        return None

    # ── Step 5: Sample N_FACES evenly-spaced crops ────────────────────────
    indices  = np.linspace(0, T - 1, N_FACES, dtype=int)
    sampled  = [crops[i] for i in indices]
    face_arr = np.stack(sampled, axis=0).astype(np.uint8)   # (N_FACES,224,224,3)

    # ── Step 6: FFT maps ──────────────────────────────────────────────────
    fft_arr = np.stack([compute_fft_map(c) for c in sampled], axis=0)  # (N_FACES,96,96)

    # ── Step 7: rPPG + coherence ──────────────────────────────────────────
    rppg_arr, coh_arr, rppg_ok = extract_rppg(crops)

    return face_arr, fft_arr, rppg_arr, coh_arr, rppg_ok


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Full-video processing
# ═════════════════════════════════════════════════════════════════════════════

def process_video(vid_info: dict, split: str,
                  hf: h5py.File, haar) -> int:
    """
    Stream video frame-by-frame, slice into non-overlapping CLIP_F-frame
    clips, process each, write to HDF5.
    Returns the number of clips successfully written.
    """
    path  = vid_info["path"]
    label = int(vid_info["label"])
    actor = vid_info.get("actor") or vid_info.get("target") or "??"

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.warning(f"Cannot open: {path}")
        return 0

    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < MIN_FRAMES:
        cap.release()
        return 0

    clips_written = 0
    buffer: list  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)

        if len(buffer) == CLIP_F:
            result = process_clip(buffer, haar)
            if result is not None:
                face_arr, fft_arr, rppg_arr, coh_arr, rppg_ok = result
                append_clip(hf, split,
                            face_arr, fft_arr, rppg_arr, coh_arr,
                            label, rppg_ok, actor, path)
                clips_written += 1
            buffer = []   # Non-overlapping — discard fully

    cap.release()
    return clips_written


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Integrity check
# ═════════════════════════════════════════════════════════════════════════════

def integrity_check(hf: h5py.File):
    """Verify dataset shapes and print per-split summary."""
    log.info("=" * 60)
    log.info("  INTEGRITY CHECK")
    log.info("=" * 60)

    all_ok      = True
    total_clips = 0

    for sp in ("train", "val", "test"):
        if sp not in hf:
            log.error(f"  [{sp}] Group missing!")
            all_ok = False
            continue

        g = hf[sp]
        n = g["labels"].shape[0]
        total_clips += n

        real_n  = int((g["labels"][:] == 0).sum())
        fake_n  = int((g["labels"][:] == 1).sum())
        valid_n = int(g["rppg_valid"][:].sum())
        log.info(f"  [{sp}]  clips={n:>6}  real={real_n:>5}  "
                 f"fake={fake_n:>5}  rppg_valid={valid_n:>5}")

        expected = {
            "face_crops":   (n, N_FACES, FACE_SZ, FACE_SZ, 3),
            "fft_maps":     (n, N_FACES, FFT_SZ,  FFT_SZ),
            "rppg_signals": (n, N_ROIS,  CLIP_F),
            "coherence":    (n, N_ROIS,  N_ROIS),
            "labels":       (n,),
        }
        for ds_name, exp_shape in expected.items():
            actual = g[ds_name].shape
            if actual != exp_shape:
                log.error(f"    SHAPE MISMATCH {ds_name}: "
                          f"expected {exp_shape}, got {actual}")
                all_ok = False

    # Spot-check 3 random clips from train
    if "train" in hf and hf["train"]["labels"].shape[0] >= 3:
        g    = hf["train"]
        idxs = np.random.choice(g["labels"].shape[0], 3, replace=False)
        log.info("  Spot-check (3 random train clips):")
        for i in idxs:
            lbl  = int(g["labels"][i])
            rv   = bool(g["rppg_valid"][i])
            act  = g["actor_ids"][i]
            log.info(f"    idx={i:<5}  label={'FAKE' if lbl else 'REAL'}"
                     f"  rppg_valid={rv}  actor={act}")

    size_gb = HDF5_P.stat().st_size / 1e9
    log.info(f"  Total clips   : {total_clips:,}")
    log.info(f"  HDF5 file size: {size_gb:.2f} GB")
    log.info("  [PASS] All shape checks passed." if all_ok
             else "  [FAIL] Shape check failed — see log above.")
    log.info("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Main
# ═════════════════════════════════════════════════════════════════════════════

def fmt_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("  TriGuard-DF  |  01_preprocess.py")
    log.info(f"  Started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    splits = build_or_load_splits()

    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    haar      = cv2.CascadeClassifier(haar_path)
    if haar.empty():
        log.warning("Haar cascade not found — relying on MediaPipe only.")
        haar = None

    done_set = load_progress()
    hf       = init_hdf5(HDF5_P)
    log.info(f"HDF5 opened : {HDF5_P}")

    total_clips   = 0
    total_skipped = 0
    save_every    = 50

    try:
        for sp in ("train", "val", "test"):
            vid_list = splits[sp]
            pending  = [v for v in vid_list if v["path"] not in done_set]

            real_c = sum(1 for v in vid_list if v["label"] == 0)
            fake_c = sum(1 for v in vid_list if v["label"] == 1)
            log.info(f"\n{'─' * 50}")
            log.info(f"  Split [{sp.upper()}]  total={len(vid_list)}"
                     f"  real={real_c}  fake={fake_c}")
            log.info(f"  Done={len(vid_list) - len(pending)}  "
                     f"Pending={len(pending)}")
            log.info(f"{'─' * 50}")

            if not pending:
                log.info("  All videos already processed. Skipping.")
                continue

            pbar = tqdm(pending, desc=sp, unit="vid", dynamic_ncols=True)
            for idx, vid_info in enumerate(pbar):
                try:
                    n = process_video(vid_info, sp, hf, haar)
                    total_clips   += n
                    total_skipped += (n == 0)
                    done_set.add(vid_info["path"])
                    pbar.set_postfix(clips=total_clips, skipped=total_skipped)

                    if (idx + 1) % save_every == 0:
                        hf.flush()
                        save_progress(done_set)
                        log.info(f"  Checkpoint — {len(done_set)} videos, "
                                 f"{total_clips} clips, "
                                 f"{fmt_elapsed(time.time() - t0)}")

                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error(f"  ERROR {vid_info['path']}: {exc}")
                    log.debug(traceback.format_exc())
                    total_skipped += 1
                    done_set.add(vid_info["path"])

            hf.flush()
            save_progress(done_set)

    except KeyboardInterrupt:
        log.warning("Interrupted. Progress saved — safe to resume.")
        hf.flush()
        save_progress(done_set)

    finally:
        log.info("\nRunning integrity check …")
        integrity_check(hf)
        hf.close()

        elapsed = time.time() - t0
        log.info(f"\nTotal clips written : {total_clips:,}")
        log.info(f"Videos skipped      : {total_skipped}")
        log.info(f"Total time          : {fmt_elapsed(elapsed)}")
        log.info(f"HDF5 saved at       : {HDF5_P.resolve()}")
        log.info("\nPreprocessing complete.")
        log.info("Next step → python scripts/02_train.py")


if __name__ == "__main__":
    main()