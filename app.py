"""
app.py — TriGuard-DF Flask API
==============================

Endpoints:
    GET  /                        → serve index.html
    GET  /health                  → model status
    POST /predict                 → single video inference
    POST /batch/start             → start batch job
    GET  /batch/status/<batch_id> → poll batch progress
    GET  /batch/download/<batch_id> → download results CSV/JSON
"""

import os
import uuid
import json
import csv
import io
import time
import threading
import logging
import traceback
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, request, jsonify,
    render_template, send_file, send_from_directory
)
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

Path("output/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("output/logs/flask.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Temp upload folder
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_FOLDER = Path("output/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}

# ─────────────────────────────────────────────────────────────────────────────
# Load model ONCE at startup
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/triguard_best.pt"

engine      = None
model_error = None

def _load_engine():
    global engine, model_error
    try:
        log.info("Loading TriGuard model …")
        from src.inference import TriGuardInference
        engine = TriGuardInference(MODEL_PATH, device="auto")
        log.info("Model loaded successfully.")
    except Exception as e:
        model_error = str(e)
        log.error(f"Failed to load model: {e}")
        log.error(traceback.format_exc())

# Load in background thread so Flask starts instantly
threading.Thread(target=_load_engine, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# In-memory batch job store
# { batch_id: { status, total, done, results, error, files } }
# ─────────────────────────────────────────────────────────────────────────────

batch_jobs: dict = {}
batch_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def save_upload(file) -> Path:
    """Save an uploaded file to UPLOAD_FOLDER with a unique name."""
    original  = secure_filename(file.filename)
    ext       = Path(original).suffix.lower()
    uid       = uuid.uuid4().hex[:8]
    safe_name = f"{uid}_{original}"
    dest      = UPLOAD_FOLDER / safe_name
    file.save(str(dest))
    return dest


def cleanup_file(path: Path):
    """Delete a temp file silently."""
    try:
        if path and path.exists():
            path.unlink()
    except Exception:
        pass


def format_result_for_api(result: dict, original_name: str) -> dict:
    """
    Normalise inference result for frontend consumption.
    Adds display_name so frontend shows original filename,
    not the temp path.
    """
    return {
        "video":            original_name,
        "prediction":       result.get("prediction", "ERROR"),
        "confidence":       result.get("confidence", 0),
        "probability_fake": result.get("probability_fake", 0),
        "probability_real": result.get("probability_real", 0),
        "inference_time_sec": result.get("inference_time_sec", 0),
        "device":           result.get("device", "unknown"),
        "timestamp":        result.get("timestamp", datetime.now().isoformat()),
        "error":            result.get("error", None),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

# ── Index ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """
    Returns model readiness status.
    Frontend polls this on load to show ready/loading state.
    """
    if engine is not None:
        return jsonify({
            "status":  "ready",
            "device":  engine.device,
            "model":   MODEL_PATH,
        })
    elif model_error:
        return jsonify({
            "status": "error",
            "error":  model_error,
        }), 500
    else:
        return jsonify({
            "status": "loading",
        }), 202


# ── Single video prediction ───────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Form data: file = video file

    Returns JSON result dict.
    """
    # Model ready?
    if engine is None:
        return jsonify({
            "prediction": "ERROR",
            "error": "Model is still loading. Please wait a moment.",
        }), 503

    # File present?
    if "file" not in request.files:
        return jsonify({
            "prediction": "ERROR",
            "error": "No file uploaded.",
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "prediction": "ERROR",
            "error": "Empty filename.",
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "prediction": "ERROR",
            "error": f"Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}",
        }), 400

    original_name = secure_filename(file.filename)
    temp_path     = None

    try:
        # Save upload
        temp_path = save_upload(file)
        log.info(f"Single predict: {original_name} → {temp_path}")

        # Run inference
        raw_result = engine.predict(str(temp_path))
        result     = format_result_for_api(raw_result, original_name)

        log.info(f"Result: {result['prediction']} "
                 f"(fake={result['probability_fake']:.4f})")
        return jsonify(result)

    except Exception as e:
        log.error(f"Predict error: {e}")
        log.error(traceback.format_exc())
        return jsonify({
            "video":      original_name,
            "prediction": "ERROR",
            "error":      str(e),
        }), 500

    finally:
        cleanup_file(temp_path)


# ── Batch: start job ──────────────────────────────────────────────────────────

@app.route("/batch/start", methods=["POST"])
def batch_start():
    """
    POST /batch/start
    Form data: files[] = multiple video files

    Saves all files, starts background worker,
    returns { batch_id, total } immediately.
    Frontend then polls /batch/status/<batch_id>.
    """
    if engine is None:
        return jsonify({
            "error": "Model is still loading. Please wait.",
        }), 503

    files = request.files.getlist("files[]")

    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    # Filter valid files
    valid_files = [f for f in files
                   if f.filename and allowed_file(f.filename)]

    if not valid_files:
        return jsonify({
            "error": f"No valid video files. Supported: "
                     f"{', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Save all uploads first
    saved = []   # list of (temp_path, original_name)
    for f in valid_files:
        try:
            temp_path = save_upload(f)
            saved.append((temp_path, secure_filename(f.filename)))
        except Exception as e:
            log.error(f"Failed to save {f.filename}: {e}")

    if not saved:
        return jsonify({"error": "Failed to save any uploaded files."}), 500

    # Create batch job
    batch_id = uuid.uuid4().hex

    with batch_lock:
        batch_jobs[batch_id] = {
            "status":   "running",    # running | done | error
            "total":    len(saved),
            "done":     0,
            "results":  [],
            "files":    saved,        # cleaned up after done
            "created":  datetime.now().isoformat(),
        }

    log.info(f"Batch job {batch_id}: {len(saved)} videos queued")

    # Start background worker
    t = threading.Thread(
        target=_batch_worker,
        args=(batch_id, saved),
        daemon=True,
    )
    t.start()

    return jsonify({
        "batch_id": batch_id,
        "total":    len(saved),
    })


def _batch_worker(batch_id: str, saved: list):
    """
    Background thread: processes videos one by one.
    Updates batch_jobs[batch_id] after each video.
    Frontend polls /batch/status to see live progress.
    """
    log.info(f"Batch worker started: {batch_id}")

    for temp_path, original_name in saved:
        try:
            log.info(f"  [{batch_id}] Processing: {original_name}")
            raw_result = engine.predict(str(temp_path))
            result     = format_result_for_api(raw_result, original_name)
        except Exception as e:
            log.error(f"  [{batch_id}] Error on {original_name}: {e}")
            result = {
                "video":            original_name,
                "prediction":       "ERROR",
                "confidence":       0,
                "probability_fake": 0,
                "probability_real": 0,
                "inference_time_sec": 0,
                "device":           "unknown",
                "timestamp":        datetime.now().isoformat(),
                "error":            str(e),
            }
        finally:
            cleanup_file(temp_path)

        with batch_lock:
            batch_jobs[batch_id]["results"].append(result)
            batch_jobs[batch_id]["done"] += 1
            log.info(f"  [{batch_id}] Done "
                     f"{batch_jobs[batch_id]['done']}/"
                     f"{batch_jobs[batch_id]['total']}")

    # Mark complete
    with batch_lock:
        batch_jobs[batch_id]["status"] = "done"

    log.info(f"Batch job {batch_id} complete.")


# ── Batch: poll status ────────────────────────────────────────────────────────

@app.route("/batch/status/<batch_id>")
def batch_status(batch_id: str):
    """
    GET /batch/status/<batch_id>

    Returns:
    {
        status : "running" | "done" | "error",
        total  : int,
        done   : int,
        results: [ result, ... ]   ← grows as videos complete
    }

    Frontend polls this every 2 seconds.
    """
    with batch_lock:
        job = batch_jobs.get(batch_id)

    if job is None:
        return jsonify({"error": "Batch job not found."}), 404

    return jsonify({
        "status":  job["status"],
        "total":   job["total"],
        "done":    job["done"],
        "results": job["results"],
    })


# ── Batch: download results ───────────────────────────────────────────────────

@app.route("/batch/download/<batch_id>")
def batch_download(batch_id: str):
    """
    GET /batch/download/<batch_id>?format=csv   → CSV file
    GET /batch/download/<batch_id>?format=json  → JSON file

    Generates file on the fly from in-memory results.
    Nothing is stored on disk.
    """
    with batch_lock:
        job = batch_jobs.get(batch_id)

    if job is None:
        return jsonify({"error": "Batch job not found."}), 404

    results = job["results"]
    fmt     = request.args.get("format", "json").lower()
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── JSON ──────────────────────────────────────────────────────────────
    if fmt == "json":
        output = {
            "batch_id":   batch_id,
            "generated":  datetime.now().isoformat(),
            "total":      job["total"],
            "results":    results,
            "summary": {
                "fake":  sum(1 for r in results if r["prediction"] == "FAKE"),
                "real":  sum(1 for r in results if r["prediction"] == "REAL"),
                "error": sum(1 for r in results if r["prediction"] == "ERROR"),
            }
        }
        buf = io.BytesIO(
            json.dumps(output, indent=2).encode("utf-8")
        )
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/json",
            as_attachment=True,
            download_name=f"triguard_results_{ts}.json",
        )

    # ── CSV ───────────────────────────────────────────────────────────────
    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Header
        writer.writerow([
            "video", "prediction", "confidence",
            "probability_fake", "probability_real",
            "inference_time_sec", "device", "timestamp", "error",
        ])

        # Rows
        for r in results:
            writer.writerow([
                r.get("video", ""),
                r.get("prediction", ""),
                r.get("confidence", ""),
                r.get("probability_fake", ""),
                r.get("probability_real", ""),
                r.get("inference_time_sec", ""),
                r.get("device", ""),
                r.get("timestamp", ""),
                r.get("error", ""),
            ])

        buf.seek(0)
        return send_file(
            io.BytesIO(buf.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"triguard_results_{ts}.csv",
        )

    else:
        return jsonify({
            "error": "Invalid format. Use ?format=csv or ?format=json"
        }), 400


# ── Cleanup old batch jobs (simple memory management) ────────────────────────

def _cleanup_old_jobs():
    """
    Remove batch jobs older than 1 hour from memory.
    Runs every 30 minutes in background.
    """
    while True:
        time.sleep(1800)   # 30 minutes
        now = datetime.now()
        to_delete = []

        with batch_lock:
            for bid, job in batch_jobs.items():
                try:
                    created = datetime.fromisoformat(job["created"])
                    age_hrs = (now - created).total_seconds() / 3600
                    if age_hrs > 1.0:
                        to_delete.append(bid)
                except Exception:
                    pass

            for bid in to_delete:
                del batch_jobs[bid]
                log.info(f"Cleaned up old batch job: {bid}")

threading.Thread(target=_cleanup_old_jobs, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  TriGuard-DF Flask Server")
    log.info("  http://127.0.0.1:5000")
    log.info("=" * 60)
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,    # Keep False — model loads once, debug reloads it twice
        threaded=True,  # Required for batch polling to work
    )