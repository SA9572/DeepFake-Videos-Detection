"""
src/model_loader.py

Loads triguard_best.pt checkpoint into TriGuardNet architecture.
"""

import torch
import logging
from pathlib import Path

from .architecture import TriGuardNet

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "cpu") -> tuple:
    """
    Load TriGuardNet from checkpoint.

    Args:
        model_path: Path to triguard_best.pt
        device: 'cpu' or 'cuda'

    Returns:
        (model, cfg) — loaded model in eval mode and the config dict
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading checkpoint from {model_path} "
                f"({model_path.stat().st_size / 1024**2:.1f} MB)")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    cfg = checkpoint["cfg"]
    logger.info(f"  Checkpoint epoch : {checkpoint.get('epoch', '?')}")
    logger.info(f"  Checkpoint AUC   : {checkpoint.get('val_auc', '?')}")
    logger.info(f"  embed_dim={cfg['embed_dim']}, n_faces={cfg['n_faces']}, "
                f"n_rois={cfg['n_rois']}, clip_f={cfg['clip_f']}, fft_sz={cfg['fft_sz']}")

    model = TriGuardNet(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded: {n_params:,} parameters on {device}")

    return model, cfg