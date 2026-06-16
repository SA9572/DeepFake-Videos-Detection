"""
scripts/00_verify_dataset.py

Run this BEFORE preprocessing. Takes ~2 minutes.
Confirms:
  1. Both dataset folders are found and readable
  2. Filename parsing extracts correct actor/method IDs
  3. Pair-safe split logic works on your actual files
  4. Class distribution and method code breakdown
  5. Video length samples (checks 10 random videos)

Usage:
    python scripts/00_verify_dataset.py
"""

import os
import re
import sys
import random
import cv2
import yaml
from pathlib import Path
from collections import defaultdict

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/config.yaml")
if not CONFIG_PATH.exists():
    print("[ERROR] configs/config.yaml not found. Run from DeepFake Detection/ root.")
    sys.exit(1)

with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

REAL_DIR  = Path(cfg["data"]["real_dir"])
FAKE_DIR  = Path(cfg["data"]["fake_dir"])
REAL_PAT  = re.compile(cfg["parsing"]["real_pattern"])
FAKE_PAT  = re.compile(cfg["parsing"]["fake_pattern"])
VAL_N     = cfg["splits"]["val_actors"]
TEST_N    = cfg["splits"]["test_actors"]
SEED      = cfg["splits"]["seed"]

# ── Separator helper ──────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHECK FOLDERS EXIST
# ─────────────────────────────────────────────────────────────────────────────
section("1. Folder Check")

for folder, name in [(REAL_DIR, "Real"), (FAKE_DIR, "Fake")]:
    if folder.exists():
        count = len(list(folder.glob("*.mp4")))
        print(f"  [OK]  {name} folder found : {folder}")
        print(f"        {count} .mp4 files detected")
    else:
        print(f"  [FAIL] {name} folder NOT found : {folder}")
        print(f"         Check your config.yaml paths.")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PARSE ALL FILENAMES
# ─────────────────────────────────────────────────────────────────────────────
section("2. Filename Parsing")

real_videos = []   # list of dicts
fake_videos = []
parse_errors = []

# -- Real videos --
for f in sorted(REAL_DIR.glob("*.mp4")):
    m = REAL_PAT.match(f.name)
    if m:
        real_videos.append({
            "path":     str(f),
            "actor":    m.group(1),
            "scene":    m.group(2),
            "label":    0,
        })
    else:
        parse_errors.append(("REAL", f.name))

# -- Fake videos --
for f in sorted(FAKE_DIR.glob("*.mp4")):
    m = FAKE_PAT.match(f.name)
    if m:
        fake_videos.append({
            "path":     str(f),
            "target":   m.group(1),
            "source":   m.group(2),
            "scene":    m.group(3),
            "method":   m.group(4),
            "label":    1,
        })
    else:
        parse_errors.append(("FAKE", f.name))

print(f"  Real videos parsed   : {len(real_videos)}")
print(f"  Fake videos parsed   : {len(fake_videos)}")
print(f"  Total                : {len(real_videos) + len(fake_videos)}")
print(f"  Parse errors         : {len(parse_errors)}")

if parse_errors:
    print("\n  [WARNING] Files that didn't match expected pattern:")
    for kind, name in parse_errors[:10]:
        print(f"    [{kind}] {name}")
    if len(parse_errors) > 10:
        print(f"    ... and {len(parse_errors) - 10} more")

# Show 3 examples of each
print("\n  Sample real filenames parsed:")
for v in real_videos[:3]:
    print(f"    actor={v['actor']}  scene={v['scene']}")

print("\n  Sample fake filenames parsed:")
for v in fake_videos[:3]:
    print(f"    target={v['target']}  source={v['source']}  "
          f"method={v['method']}  scene={v['scene']}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ACTOR & METHOD BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
section("3. Actor & Method Code Breakdown")

# Unique real actors
real_actors = sorted(set(v["actor"] for v in real_videos))
print(f"  Unique actors (real videos) : {len(real_actors)}")
print(f"  Actor IDs : {real_actors}")

# Actor appearance in fake videos
fake_targets = sorted(set(v["target"] for v in fake_videos))
fake_sources = sorted(set(v["source"] for v in fake_videos))
all_fake_actors = sorted(set(fake_targets) | set(fake_sources))
print(f"\n  Actors appearing as TARGET in fakes : {fake_targets}")
print(f"  Actors appearing as SOURCE in fakes : {fake_sources}")
print(f"  All unique actor IDs across fakes   : {all_fake_actors}")

# Method codes
method_counts = defaultdict(int)
for v in fake_videos:
    method_counts[v["method"]] += 1

print(f"\n  Deepfake method codes found:")
for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
    print(f"    {method:<20} {count:>5} videos")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PAIR-SAFE SPLIT SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
section("4. Pair-Safe Split Simulation")

all_actors = sorted(set(real_actors) | set(all_fake_actors))
print(f"  Total unique actor IDs : {len(all_actors)}")

random.seed(SEED)
shuffled = all_actors.copy()
random.shuffle(shuffled)

test_actors  = set(shuffled[:TEST_N])
val_actors   = set(shuffled[TEST_N: TEST_N + VAL_N])
train_actors = set(shuffled[TEST_N + VAL_N:])

print(f"\n  Train actors ({len(train_actors)}) : {sorted(train_actors)}")
print(f"  Val   actors ({len(val_actors)})  : {sorted(val_actors)}")
print(f"  Test  actors ({len(test_actors)}) : {sorted(test_actors)}")

# Apply pair-safe rule to fake videos
def get_split(actor_id):
    if actor_id in train_actors: return "train"
    if actor_id in val_actors:   return "val"
    if actor_id in test_actors:  return "test"
    return "unknown"

split_counts = defaultdict(lambda: {"real": 0, "fake": 0, "fake_dropped": 0})

for v in real_videos:
    split = get_split(v["actor"])
    split_counts[split]["real"] += 1

for v in fake_videos:
    t_split = get_split(v["target"])
    s_split = get_split(v["source"])
    if t_split == s_split:
        split_counts[t_split]["fake"] += 1
    else:
        # Pair-safe rule: drop this video
        split_counts["dropped"]["fake_dropped"] += 1

print(f"\n  After pair-safe filtering:")
print(f"  {'Split':<10} {'Real':>8} {'Fake':>8} {'Total':>8}")
print(f"  {'-'*38}")
for split in ["train", "val", "test"]:
    r = split_counts[split]["real"]
    fk = split_counts[split]["fake"]
    print(f"  {split:<10} {r:>8} {fk:>8} {r+fk:>8}")
dropped = split_counts["dropped"]["fake_dropped"]
print(f"\n  Fake videos dropped (pair-unsafe) : {dropped}")
print(f"  Fake videos retained              : "
      f"{sum(split_counts[s]['fake'] for s in ['train','val','test'])}")

# Warn if test set is too small
for split in ["val", "test"]:
    total = split_counts[split]["real"] + split_counts[split]["fake"]
    if total < 20:
        print(f"\n  [WARNING] {split} set has only {total} videos.")
        print(f"           Consider reducing val_actors/test_actors in config.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VIDEO LENGTH SPOT CHECK (10 random videos)
# ─────────────────────────────────────────────────────────────────────────────
section("5. Video Length Spot Check (10 random samples)")

all_videos = real_videos + fake_videos
sample = random.sample(all_videos, min(10, len(all_videos)))

short_count = 0
print(f"  {'File':<45} {'FPS':>6} {'Frames':>8} {'Secs':>6} {'OK?':>5}")
print(f"  {'-'*72}")

for v in sample:
    cap = cv2.VideoCapture(v["path"])
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    secs = frames / fps
    ok = "YES" if frames >= 100 else "SHORT"
    if frames < 100:
        short_count += 1
    name = Path(v["path"]).name[:43]
    print(f"  {name:<45} {fps:>6.1f} {frames:>8} {secs:>6.1f} {ok:>5}")

if short_count > 0:
    print(f"\n  [WARNING] {short_count}/10 sampled videos are shorter than 100 frames.")
    print(f"           rPPG branch will be marked invalid for these clips.")
else:
    print(f"\n  [OK] All sampled videos are long enough for rPPG extraction.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. STORAGE ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────
section("6. HDF5 Storage Estimate")

total_videos = len(real_videos) + len(fake_videos)
clips_per_video  = 3          # Conservative estimate
samples = total_videos * clips_per_video
faces_per_clip   = 8

# Face crops: 8 × (256×256×3) uint8 per clip
face_bytes  = samples * faces_per_clip * 256 * 256 * 3
# FFT maps:   8 × (96×96×1) float16 per clip
fft_bytes   = samples * faces_per_clip * 96 * 96 * 2
# rPPG:       (6×150) float32 per clip
rppg_bytes  = samples * 6 * 150 * 4
# Coherence:  (6×6) float32 per clip
coh_bytes   = samples * 6 * 6 * 4

total_gb = (face_bytes + fft_bytes + rppg_bytes + coh_bytes) / 1e9

print(f"  Estimated clips        : ~{samples:,}")
print(f"  Face crops             : ~{face_bytes/1e9:.2f} GB")
print(f"  FFT maps               : ~{fft_bytes/1e9:.2f} GB")
print(f"  rPPG signals           : ~{rppg_bytes/1e9:.3f} GB")
print(f"  Coherence matrices     : ~{coh_bytes/1e9:.3f} GB")
print(f"  ─────────────────────────────────")
print(f"  Estimated HDF5 total   : ~{total_gb:.2f} GB")
print(f"  (Actual will vary with real clip counts per video)")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
section("VERIFICATION RESULT")

errors = len(parse_errors)
if errors == 0:
    print("  [PASS] Dataset structure verified. Safe to run 01_preprocess.py")
    print(f"\n  Next step:")
    print(f"    python scripts/01_preprocess.py")
else:
    print(f"  [WARN] {errors} files could not be parsed.")
    print(f"         Review the filenames above and update parsing patterns")
    print(f"         in configs/config.yaml before preprocessing.")