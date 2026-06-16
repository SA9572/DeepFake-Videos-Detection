"""
main.py — TriGuard-DF v1 Inference Entry Point

Usage:
    python main.py                              # Interactive mode
    python main.py video.mp4                    # Single video
    python main.py video1.mp4 video2.mp4        # Multiple videos
    python main.py --dir videos/                # All .mp4 in directory
"""

import sys
import logging
from pathlib import Path

# Create output dirs before logging tries to write
Path("output/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("output/logs/inference.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 65)
    print("  TriGuard-DF v1 — Deepfake Detection Inference")
    print("=" * 65)

    from src.inference import TriGuardInference

    MODEL_PATH = "models/triguard_best.pt"

    if not Path(MODEL_PATH).exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Make sure triguard_best.pt is in the models/ directory.")
        sys.exit(1)

    # Initialize
    engine = TriGuardInference(MODEL_PATH, device="auto")
    print()

    # Parse arguments
    args = sys.argv[1:]

    if not args:
        # Interactive mode
        print("No video path provided. Enter paths interactively.")
        print("Type 'quit' to exit.\n")
        while True:
            path = input("Video path: ").strip().strip('"').strip("'")
            if path.lower() in ("quit", "exit", "q"):
                break
            if not path:
                continue
            result = engine.predict(path)
            _print_result(result)
            print()
        return

    # Check for --dir flag
    if args[0] == "--dir":
        if len(args) < 2:
            print("Usage: python main.py --dir <directory>")
            sys.exit(1)
        video_dir = Path(args[1])
        video_paths = sorted(
            list(video_dir.glob("*.mp4")) +
            list(video_dir.glob("*.avi")) +
            list(video_dir.glob("*.mov"))
        )
        if not video_paths:
            print(f"No video files found in {video_dir}")
            sys.exit(1)
        print(f"Found {len(video_paths)} videos in {video_dir}\n")
        results = engine.predict_batch(
            [str(p) for p in video_paths],
            output_file="output/predictions.json"
        )
        print("\n" + "=" * 65)
        for r in results:
            _print_result(r)
        return

    # Process listed files
    if len(args) == 1:
        result = engine.predict(args[0])
        _print_result(result)
    else:
        results = engine.predict_batch(
            args,
            output_file="output/predictions.json"
        )
        print("\n" + "=" * 65)
        for r in results:
            _print_result(r)


def _print_result(result):
    """Pretty-print a single result."""
    if result["prediction"] == "ERROR":
        print(f"  ❌ {Path(result['video']).name}: ERROR — {result['error']}")
        return

    icon = "🔴" if result["prediction"] == "FAKE" else "🟢"
    print(f"  {icon} {Path(result['video']).name}")
    print(f"     Prediction : {result['prediction']}")
    print(f"     Confidence : {result['confidence']}")
    print(f"     P(fake)    : {result['probability_fake']}")
    print(f"     P(real)    : {result['probability_real']}")
    print(f"     Time       : {result['inference_time_sec']}s")


if __name__ == "__main__":
    main()