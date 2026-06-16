"""
src/inference.py

End-to-end inference: video path → prediction.
"""

import torch
import logging
import json
from pathlib import Path
from datetime import datetime

from .model_loader import load_model
from .preprocessing import preprocess_video

logger = logging.getLogger(__name__)


class TriGuardInference:
    """
    Main inference engine.

    Usage:
        engine = TriGuardInference("models/triguard_best.pt")
        result = engine.predict("path/to/video.mp4")
        print(result)
    """

    def __init__(self, model_path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model, self.cfg = load_model(model_path, device)
        logger.info(f"TriGuardInference ready on {device}")

    def predict(self, video_path: str, threshold: float = 0.5) -> dict:
        """
        Process one video and return prediction.

        Args:
            video_path: Path to video file
            threshold: Probability above this → FAKE

        Returns:
            dict with prediction, confidence, probabilities, metadata
        """
        video_path = str(video_path)
        start_time = datetime.now()

        try:
            # Preprocess
            tensors = preprocess_video(video_path, self.cfg)

            # Move to device
            faces = tensors["faces"].to(self.device)
            fft = tensors["fft"].to(self.device)
            rppg = tensors["rppg"].to(self.device)
            coh = tensors["coherence"].to(self.device)

            # Inference
            with torch.no_grad():
                logit = self.model(faces, fft, rppg, coh)
                prob_fake = torch.sigmoid(logit).item()

            prediction = "FAKE" if prob_fake > threshold else "REAL"
            elapsed = (datetime.now() - start_time).total_seconds()

            result = {
                "video": video_path,
                "prediction": prediction,
                "confidence": round(prob_fake if prediction == "FAKE"
                                    else 1.0 - prob_fake, 4),
                "probability_fake": round(prob_fake, 4),
                "probability_real": round(1.0 - prob_fake, 4),
                "threshold": threshold,
                "inference_time_sec": round(elapsed, 2),
                "device": self.device,
                "timestamp": start_time.isoformat(),
            }

            logger.info(f"  → {prediction} (fake_prob={prob_fake:.4f}, "
                        f"{elapsed:.1f}s)")
            return result

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"  Error: {e}")
            return {
                "video": video_path,
                "prediction": "ERROR",
                "error": str(e),
                "inference_time_sec": round(elapsed, 2),
                "timestamp": start_time.isoformat(),
            }

    def predict_batch(self, video_paths: list, threshold: float = 0.5,
                      output_file: str = None) -> list:
        """Process multiple videos."""
        results = []
        for i, vp in enumerate(video_paths):
            logger.info(f"[{i + 1}/{len(video_paths)}] {Path(vp).name}")
            result = self.predict(vp, threshold)
            results.append(result)

        # Summary
        ok = [r for r in results if r["prediction"] != "ERROR"]
        fakes = sum(1 for r in ok if r["prediction"] == "FAKE")
        reals = sum(1 for r in ok if r["prediction"] == "REAL")
        errors = len(results) - len(ok)
        logger.info(f"Batch done: {len(ok)} processed, "
                    f"{fakes} FAKE, {reals} REAL, {errors} errors")

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")

        return results