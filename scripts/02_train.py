"""
scripts/02_train.py

TriGuard-DF — Three-Branch Cross-Modal Deepfake Detector
=========================================================

Architecture
─────────────────────────────────────────────────────────────────
Branch 1 │ Spatial   │ EfficientNet-B4 + Temporal Attention Pool
Branch 2 │ Spectral  │ Residual CNN on 2-D FFT maps
Branch 3 │ Physio    │ ROI-Transformer + Coherence MLP
Fusion   │           │ CLS-Token Cross-Modal Transformer (4 layers)
Head     │           │ MLP → scalar logit

Training highlights
─────────────────────────────────────────────────────────────────
  WeightedRandomSampler  — fixes 1 : 5.2 class imbalance
  Differential LR        — backbone 10× lower than heads
  Linear warmup + cosine annealing (per step)
  Mixed precision AMP    — auto-disabled on CPU
  Label-smoothing BCE    — no pos_weight (sampler handles balance)
  Early stopping on val AUC (patience = 10)
  Best checkpoint → models/triguard_best.pt
  Full metrics  → data/training_log.csv + data/train.log

Install once:
    pip install timm scikit-learn

Usage:
    python scripts/02_train.py
"""

import csv
import json
import math
import random
import sys
import time
import logging
import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

try:
    import timm
except ImportError:
    print("timm not found. Run: pip install timm")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT    = Path(".")
HDF5_P  = ROOT / "data"   / "deepfake_physio.h5"
LOG_P   = ROOT / "data"   / "train.log"
CSV_P   = ROOT / "data"   / "training_log.csv"
MODEL_D = ROOT / "models"
CKPT_P  = MODEL_D / "triguard_best.pt"
CFG_P   = MODEL_D / "triguard_config.json"

MODEL_D.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
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
# Hyperparameters — edit these freely
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Paths
    "hdf5_path":  str(HDF5_P),
    "seed":       42,
    # Training
    "epochs":     50,
    "batch_size": 8,
    "num_workers": 0,          # Keep 0 on Windows
    # Optimiser
    "lr_backbone":  2e-5,      # EfficientNet-B4 pretrained layers
    "lr_head":      2e-4,      # All new layers
    "weight_decay": 1e-4,
    "grad_clip":    1.0,
    # Scheduler
    "warmup_epochs": 3,
    # Regularisation
    "label_smooth": 0.05,
    "dropout":      0.4,
    # Early stopping
    "patience": 10,
    # Architecture
    "embed_dim":       256,
    "n_heads":         8,
    "n_fusion_layers": 4,
    # Data shapes — must match 01_preprocess.py
    "n_faces": 5,
    "face_sz": 224,
    "n_rois":  6,
    "clip_f":  150,
    "fft_sz":  96,
}

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Dataset
# ═════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DeepfakeDataset(Dataset):
    """
    Lazy-opens HDF5 per worker (Windows / multiprocessing safe).

    Returns
    -------
    faces     (N_FACES, 3, 224, 224) float32  ImageNet-normalised RGB
    fft       (N_FACES, 96, 96)      float32
    rppg      (N_ROIS,  150)         float32  z-scored per ROI
    coherence (N_ROIS,  N_ROIS)      float32
    label     int64 scalar
    """

    def __init__(self, hdf5_path: str, split: str, augment: bool = False):
        self.path    = hdf5_path
        self.split   = split
        self.augment = augment
        self._hf     = None   # opened lazily in __getitem__

        # Cache labels in RAM — small (just ints)
        with h5py.File(hdf5_path, "r") as f:
            self.labels = f[split]["labels"][:].astype(np.int64)
        self.n = len(self.labels)

    def _open(self):
        if self._hf is None:
            self._hf = h5py.File(self.path, "r")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        self._open()
        g = self._hf[self.split]

        # ── Face crops  (5, 224, 224, 3) uint8 BGR ─────────────────────────
        raw   = g["face_crops"][idx].astype(np.float32) / 255.0
        raw   = raw[:, :, :, ::-1].copy()                      # BGR → RGB
        faces = torch.from_numpy(raw).permute(0, 3, 1, 2)     # (5,3,H,W)
        faces = (faces - IMAGENET_MEAN) / IMAGENET_STD

        # ── FFT maps  (5, 96, 96) float16 → float32 ───────────────────────
        fft = torch.from_numpy(g["fft_maps"][idx].astype(np.float32))

        # ── rPPG  (6, 150) float32  — z-score per ROI ─────────────────────
        rppg = g["rppg_signals"][idx].astype(np.float32)
        mu   = rppg.mean(axis=1, keepdims=True)
        std  = rppg.std(axis=1,  keepdims=True) + 1e-8
        rppg = torch.from_numpy((rppg - mu) / std)

        # ── Coherence  (6, 6) float32 ─────────────────────────────────────
        coh = torch.from_numpy(g["coherence"][idx].astype(np.float32))

        label = int(self.labels[idx])

        if self.augment:
            faces, fft, rppg = self._augment(faces, fft, rppg)

        return (faces, fft, rppg, coh,
                torch.tensor(label, dtype=torch.long))

    # ── Augmentation ──────────────────────────────────────────────────────
    def _augment(self, faces, fft, rppg):
        # Horizontal flip — consistent across all frames
        if random.random() < 0.5:
            faces = torch.flip(faces, dims=[-1])
            fft   = torch.flip(fft,   dims=[-1])

        # Brightness jitter
        if random.random() < 0.5:
            faces = faces * (1.0 + random.uniform(-0.2, 0.2))

        # Random erase one frame
        if random.random() < 0.3:
            t  = random.randint(0, faces.shape[0] - 1)
            r0 = random.randint(0, faces.shape[2] - 56)
            c0 = random.randint(0, faces.shape[3] - 56)
            faces[t, :, r0:r0 + 56, c0:c0 + 56] = 0.0

        # Light Gaussian noise on FFT and rPPG
        fft  = fft  + torch.randn_like(fft)  * 0.02
        rppg = rppg + torch.randn_like(rppg) * 0.05

        return faces, fft, rppg

    def sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        n_real = float((self.labels == 0).sum())
        n_fake = float((self.labels == 1).sum())
        w_real = 1.0 / max(n_real, 1.0)
        w_fake = 1.0 / max(n_fake, 1.0)
        w = np.where(self.labels == 0, w_real, w_fake)
        return torch.from_numpy(w.astype(np.float32))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model
# ═════════════════════════════════════════════════════════════════════════════

# ── 2a. Temporal attention pooling ────────────────────────────────────────────

class TemporalAttentionPool(nn.Module):
    """
    Learnable soft-attention over T frame tokens → one vector.
    Input : (B, T, D)
    Output: (B, D)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):                    # (B, T, D)
        w = torch.softmax(self.scorer(x), dim=1)   # (B, T, 1)
        return (x * w).sum(dim=1)           # (B, D)


# ── 2b. Spatial branch ────────────────────────────────────────────────────────

class SpatialBranch(nn.Module):
    """
    EfficientNet-B4 backbone encodes each of T frames independently.
    Temporal attention fuses T frame embeddings into one vector.
    Output: (B, embed_dim)
    """
    BACKBONE_DIM = 1792   # EfficientNet-B4 feature dim

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True,
            num_classes=0, global_pool="avg",
        )
        self._freeze_early()
        self.pool = TemporalAttentionPool(self.BACKBONE_DIM)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.BACKBONE_DIM),
            nn.Dropout(dropout),
            nn.Linear(self.BACKBONE_DIM, embed_dim),
            nn.GELU(),
        )

    def _freeze_early(self):
        """Freeze stem + blocks 0-4. Train blocks 5-6 and head."""
        freeze = {"conv_stem", "bn1",
                  "blocks.0", "blocks.1", "blocks.2",
                  "blocks.3", "blocks.4"}
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(p) for p in freeze):
                param.requires_grad_(False)

    def forward(self, faces):               # (B, T, 3, 224, 224)
        B, T, C, H, W = faces.shape
        x = self.backbone(faces.view(B * T, C, H, W))   # (B*T, 1792)
        x = x.view(B, T, -1)                             # (B, T, 1792)
        return self.proj(self.pool(x))                   # (B, embed_dim)


# ── 2c. Spectral branch ───────────────────────────────────────────────────────

class ResBlock2d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class SpectralBranch(nn.Module):
    """
    Residual CNN processes each FFT map independently.
    Temporal mean-pool fuses T frequency maps.
    Input : (B, T, 96, 96)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.GELU(), ResBlock2d(32),
            nn.Conv2d(32, 64,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.GELU(), ResBlock2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(), ResBlock2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.GELU(),
        )

    def forward(self, fft):                  # (B, T, 96, 96)
        B, T, H, W = fft.shape
        x = self.cnn(fft.view(B * T, 1, H, W))  # (B*T, 256, 1, 1)
        x = self.proj(x)                         # (B*T, embed_dim)
        return x.view(B, T, -1).mean(dim=1)      # (B, embed_dim)


# ── 2d. Physio branch ─────────────────────────────────────────────────────────

class PhysioBranch(nn.Module):
    """
    Dual-path physiological encoder.

    Path A  rPPG Transformer:
        Treats each ROI signal (len=150) as a token.
        Transformer captures cross-ROI physiological dynamics.

    Path B  Coherence MLP:
        Encodes the (6×6) synchrony matrix into a compact descriptor.

    Output: (B, embed_dim)
    """

    def __init__(self, n_rois: int, clip_f: int,
                 embed_dim: int, dropout: float):
        super().__init__()
        roi_d = 128

        # Path A
        self.roi_proj = nn.Linear(clip_f, roi_d)
        self.roi_pos  = nn.Parameter(torch.randn(1, n_rois, roi_d) * 0.02)
        enc_a = nn.TransformerEncoderLayer(
            d_model=roi_d, nhead=4, dim_feedforward=256,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.roi_tr   = nn.TransformerEncoder(enc_a, num_layers=2)
        self.roi_out  = nn.Sequential(nn.Linear(roi_d, 64), nn.GELU())

        # Path B
        self.coh_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_rois * n_rois, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
        )

        # Merge
        self.merge = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, rppg, coh):        # (B,6,150), (B,6,6)
        x = self.roi_proj(rppg) + self.roi_pos   # (B, 6, 128)
        x = self.roi_out(self.roi_tr(x)).mean(1)  # (B, 64)
        y = self.coh_mlp(coh)                     # (B, 64)
        return self.merge(torch.cat([x, y], dim=-1))  # (B, embed_dim)


# ── 2e. Cross-modal fusion ────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    BERT-style CLS-token transformer over 3 modal tokens.
    Sequence: [CLS | spatial | spectral | physio]
    Output  : CLS embedding → (B, embed_dim)
    """

    def __init__(self, embed_dim: int, n_heads: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, s, f, p):              # each (B, D)
        B = s.shape[0]
        tokens = torch.cat([
            self.cls.expand(B, -1, -1),
            s.unsqueeze(1), f.unsqueeze(1), p.unsqueeze(1),
        ], dim=1)                            # (B, 4, D)
        out = self.transformer(tokens + self.pos)
        return self.norm(out[:, 0])          # (B, D)  — CLS


# ── 2f. Full model ────────────────────────────────────────────────────────────

class TriGuardNet(nn.Module):
    """
    TriGuard-DF complete model.
    Returns raw logit (B,) — apply sigmoid for probability.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D    = cfg["embed_dim"]
        drop = cfg["dropout"]

        self.spatial  = SpatialBranch(D, drop)
        self.spectral = SpectralBranch(D, drop)
        self.physio   = PhysioBranch(
            cfg["n_rois"], cfg["clip_f"], D, drop
        )
        self.fusion   = CrossModalFusion(
            D, cfg["n_heads"], cfg["n_fusion_layers"], drop
        )
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(D, D // 2), nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    def forward(self, faces, fft, rppg, coh):
        s = self.spatial(faces)
        f = self.spectral(fft)
        p = self.physio(rppg, coh)
        z = self.fusion(s, f, p)
        return self.head(z).squeeze(-1)        # (B,)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Loss, optimiser, scheduler
# ═════════════════════════════════════════════════════════════════════════════

class LabelSmoothBCE(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.s = smoothing

    def forward(self, logits, labels):
        targets = labels.float() * (1.0 - self.s) + 0.5 * self.s
        return F.binary_cross_entropy_with_logits(logits, targets)


def build_optimiser(model: TriGuardNet, cfg: dict):
    bb_ids = {id(p) for p in model.spatial.backbone.parameters()}
    heads  = [p for p in model.parameters()
              if id(p) not in bb_ids and p.requires_grad]
    return torch.optim.AdamW([
        {"params": [p for p in model.spatial.backbone.parameters()
                    if p.requires_grad],
         "lr": cfg["lr_backbone"]},
        {"params": heads, "lr": cfg["lr_head"]},
    ], weight_decay=cfg["weight_decay"])


def build_scheduler(opt, cfg: dict, steps_per_epoch: int):
    """Linear warmup then cosine decay, stepped per batch."""
    warmup = cfg["warmup_epochs"] * steps_per_epoch
    total  = cfg["epochs"]        * steps_per_epoch

    def _fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, _fn)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Train / eval loops
# ═════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, opt,
                scheduler, scaler, device, cfg):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []
    use_amp = device.type == "cuda"

    for faces, fft, rppg, coh, labels in loader:
        faces  = faces.to(device,  non_blocking=True)
        fft    = fft.to(device,    non_blocking=True)
        rppg   = rppg.to(device,   non_blocking=True)
        coh    = coh.to(device,    non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        opt.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(faces, fft, rppg, coh)
            loss   = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

        scheduler.step()
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    auc = (roc_auc_score(all_labels, all_probs)
           if len(set(all_labels)) > 1 else 0.0)
    return avg_loss, auc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    for faces, fft, rppg, coh, labels in loader:
        faces  = faces.to(device,  non_blocking=True)
        fft    = fft.to(device,    non_blocking=True)
        rppg   = rppg.to(device,   non_blocking=True)
        coh    = coh.to(device,    non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(faces, fft, rppg, coh)
        total_loss += criterion(logits, labels).item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs,  dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)
    avg_loss   = total_loss / len(loader)

    has_both = len(set(all_labels.tolist())) > 1
    auc  = roc_auc_score(all_labels, all_probs)          if has_both else 0.0
    ap   = average_precision_score(all_labels, all_probs) if has_both else 0.0
    pred = (all_probs >= 0.5).astype(np.int32)
    f1   = f1_score(all_labels, pred, zero_division=0)
    acc  = float((pred == all_labels).mean())

    return avg_loss, auc, ap, f1, acc


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main
# ═════════════════════════════════════════════════════════════════════════════

def count_params(m: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def fmt_time(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h:02d}h {m:02d}m {sec:02d}s"


def main():
    t0 = time.time()
    set_seed(CFG["seed"])

    log.info("=" * 65)
    log.info("  TriGuard-DF  |  02_train.py")
    log.info(f"  Started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 65)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    if device.type == "cuda":
        log.info(f"GPU    : {torch.cuda.get_device_name(0)}")
    else:
        log.info("No GPU found — training on CPU (will be slow).")

    # ── Datasets & loaders ────────────────────────────────────────────────
    log.info("Loading datasets …")
    ds_train = DeepfakeDataset(CFG["hdf5_path"], "train", augment=True)
    ds_val   = DeepfakeDataset(CFG["hdf5_path"], "val",   augment=False)
    ds_test  = DeepfakeDataset(CFG["hdf5_path"], "test",  augment=False)
    log.info(f"  Train : {len(ds_train):,} clips")
    log.info(f"  Val   : {len(ds_val):,} clips")
    log.info(f"  Test  : {len(ds_test):,} clips")

    sampler = WeightedRandomSampler(
        ds_train.sample_weights(),
        num_samples=len(ds_train),
        replacement=True,
    )
    pin = device.type == "cuda"
    nw  = CFG["num_workers"]
    bs  = CFG["batch_size"]

    train_loader = DataLoader(ds_train, batch_size=bs,
                              sampler=sampler, num_workers=nw,
                              pin_memory=pin)
    val_loader   = DataLoader(ds_val,   batch_size=bs * 2,
                              shuffle=False, num_workers=nw,
                              pin_memory=pin)
    test_loader  = DataLoader(ds_test,  batch_size=bs * 2,
                              shuffle=False, num_workers=nw,
                              pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────
    log.info("Building TriGuardNet …")
    model = TriGuardNet(CFG).to(device)
    total, trainable = count_params(model)
    log.info(f"  Total params     : {total:,}")
    log.info(f"  Trainable params : {trainable:,}")

    with open(CFG_P, "w") as f:
        json.dump(CFG, f, indent=2)
    log.info(f"  Config saved → {CFG_P}")

    # ── Optimiser / scheduler / loss ──────────────────────────────────────
    criterion = LabelSmoothBCE(CFG["label_smooth"])
    opt       = build_optimiser(model, CFG)
    scheduler = build_scheduler(opt, CFG, len(train_loader))
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── CSV log ───────────────────────────────────────────────────────────
    csv_f  = open(CSV_P, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow([
        "epoch", "train_loss", "train_auc",
        "val_loss", "val_auc", "val_ap", "val_f1", "val_acc",
        "lr_backbone", "lr_head", "epoch_sec",
    ])

    # ── Training loop ─────────────────────────────────────────────────────
    best_auc, patience_ctr, best_epoch = 0.0, 0, 0

    log.info("")
    log.info(f"{'Ep':>4} │ {'TrLoss':>7} {'TrAUC':>7} │ "
             f"{'VaLoss':>7} {'VaAUC':>7} {'VaAP':>6} "
             f"{'VaF1':>6} {'VaAcc':>6} │ {'Time':>10}")
    log.info("─" * 80)

    for epoch in range(1, CFG["epochs"] + 1):
        ep0 = time.time()

        tr_loss, tr_auc = train_epoch(
            model, train_loader, criterion, opt, scheduler, scaler, device, CFG,
        )
        va_loss, va_auc, va_ap, va_f1, va_acc = eval_epoch(
            model, val_loader, criterion, device,
        )

        ep_t   = time.time() - ep0
        lr_bb  = opt.param_groups[0]["lr"]
        lr_hd  = opt.param_groups[1]["lr"]

        log.info(
            f"{epoch:>4} │ {tr_loss:>7.4f} {tr_auc:>7.4f} │ "
            f"{va_loss:>7.4f} {va_auc:>7.4f} {va_ap:>6.4f} "
            f"{va_f1:>6.4f} {va_acc:>6.4f} │ {fmt_time(ep_t):>10}"
        )
        writer.writerow([
            epoch, f"{tr_loss:.6f}", f"{tr_auc:.6f}",
            f"{va_loss:.6f}", f"{va_auc:.6f}", f"{va_ap:.6f}",
            f"{va_f1:.6f}", f"{va_acc:.6f}",
            f"{lr_bb:.2e}", f"{lr_hd:.2e}", f"{ep_t:.1f}",
        ])
        csv_f.flush()

        if va_auc > best_auc:
            best_auc     = va_auc
            best_epoch   = epoch
            patience_ctr = 0
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "opt":   opt.state_dict(), "val_auc": va_auc,
                "cfg":   CFG,
            }, CKPT_P)
            log.info(f"       ✓ Best val AUC {va_auc:.4f} — saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= CFG["patience"]:
                log.info(f"\nEarly stop — no improvement for "
                         f"{CFG['patience']} epochs.")
                break

    csv_f.close()
    log.info(f"\nBest val AUC : {best_auc:.4f}  (epoch {best_epoch})")

    # ── Test evaluation ───────────────────────────────────────────────────
    log.info("\nLoading best checkpoint for final test evaluation …")
    ckpt = torch.load(CKPT_P, map_location=device)
    model.load_state_dict(ckpt["model"])

    te_loss, te_auc, te_ap, te_f1, te_acc = eval_epoch(
        model, test_loader, criterion, device,
    )

    log.info("=" * 65)
    log.info("  FINAL TEST RESULTS")
    log.info("=" * 65)
    log.info(f"  AUC-ROC            : {te_auc:.4f}")
    log.info(f"  Avg Precision (AP) : {te_ap:.4f}")
    log.info(f"  F1 Score           : {te_f1:.4f}")
    log.info(f"  Accuracy           : {te_acc:.4f}")
    log.info(f"  Loss               : {te_loss:.4f}")
    log.info("=" * 65)
    log.info(f"\nTotal time          : {fmt_time(time.time() - t0)}")
    log.info(f"Best checkpoint     : {CKPT_P.resolve()}")
    log.info(f"Training CSV        : {CSV_P.resolve()}")
    log.info("\nDone. Next step → python scripts/03_evaluate.py")


if __name__ == "__main__":
    main()