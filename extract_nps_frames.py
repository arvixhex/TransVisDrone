"""
Extract frames from NPS Videos into AllFrames/train, val, test using OpenCV.
Frames are extracted with 1-based indexing (Clip_1_00001.png, Clip_1_00002.png, ...)
so that nps_to_visdrone.py's `frame_id += 1` offset correctly maps annotation
frame 0 → disk file 00001, annotation frame 1 → disk file 00002, etc.

All frames are always extracted (no subsampling) as required by TransVisDrone's
Video Swin Transformer for temporal continuity.

Usage:
    python extract_nps_frames.py                    # extract all splits
    python extract_nps_frames.py --split train      # extract one split only
    python extract_nps_frames.py --skip-existing    # skip clips already extracted
"""

import sys
import argparse
import cv2
from pathlib import Path

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
VIDEOS_DIR = Path("datasets/NPS/Videos")
OUTPUT_DIR = Path("datasets/NPS/AllFrames")

SPLITS = {
    "train": range(1,  37),
    "val":   range(37, 41),
    "test":  range(41, 51),
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def check_opencv():
    """Verify OpenCV is available and has video support."""
    try:
        import cv2
        assert hasattr(cv2, "VideoCapture")
    except (ImportError, AssertionError):
        print("[ERROR] OpenCV (cv2) not found. Install with: pip install opencv-python")
        sys.exit(1)


def get_total_frames(video_path: Path) -> int:
    """Use OpenCV to get the declared frame count from video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def already_extracted(outdir: Path, clip_index: int, expected_frames: int) -> bool:
    """Return True if the expected number of frames already exist on disk."""
    existing = len(list(outdir.glob(f"Clip_{clip_index}_*.png")))
    return existing > 0 and (expected_frames < 0 or existing >= expected_frames)


# ──────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────
def extract_clip(clip_index: int, split: str, skip_existing: bool) -> bool:
    """
    Extract all frames for a single clip using OpenCV.
    Frames are named Clip_{clip_index}_{frame_idx:05d}.png starting from 00001.

    1-based indexing is required because NPS annotations are 0-based, and
    nps_to_visdrone.py does `frame_id += 1` to find the corresponding disk file.
    So annotation frame 0 → Clip_X_00001.png, frame 1 → Clip_X_00002.png, etc.

    Returns True on success.
    """
    video_file = VIDEOS_DIR / f"Clip_{clip_index}.mov"
    outdir     = OUTPUT_DIR / split
    outdir.mkdir(parents=True, exist_ok=True)

    if not video_file.exists():
        print(f"  [SKIP] Clip_{clip_index}: video file not found at {video_file}")
        return False

    if skip_existing:
        expected = get_total_frames(video_file)
        if already_extracted(outdir, clip_index, expected):
            existing = len(list(outdir.glob(f"Clip_{clip_index}_*.png")))
            print(f"  [SKIP] Clip_{clip_index}: {existing} frames already on disk")
            return True

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"  [ERROR] Clip_{clip_index}: OpenCV could not open {video_file}")
        return False

    native_fps     = cap.get(cv2.CAP_PROP_FPS)
    total_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx   = 1   # 1-based: first frame saved as Clip_X_00001.png
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_path = outdir / f"Clip_{clip_index}_{frame_idx:05d}.png"
        success  = cv2.imwrite(str(out_path), frame)
        if not success:
            print(f"  [WARN] Clip_{clip_index}: failed to write frame {frame_idx}")
        else:
            saved_count += 1

        frame_idx += 1

    cap.release()

    if saved_count == 0:
        print(f"  [ERROR] Clip_{clip_index}: no frames written (video may be corrupt)")
        return False

    print(f"  [OK]   Clip_{clip_index} → {split}/  ({saved_count} frames, native {native_fps:.1f}fps, declared={total_declared})")
    return True


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract all NPS UAV dataset frames via OpenCV (1-based index, matches nps_to_visdrone.py)."
    )
    p.add_argument("--split", choices=["train", "val", "test"],
                   help="Process only one split (default: all)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip clips whose frames are already on disk")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_opencv()

    splits_to_run = {args.split: SPLITS[args.split]} if args.split else SPLITS

    total_ok   = 0
    total_fail = 0

    for split, clip_range in splits_to_run.items():
        print(f"\n── {split.upper()} ──────────────────────────")
        for i in clip_range:
            ok = extract_clip(i, split, skip_existing=args.skip_existing)
            if ok:
                total_ok += 1
            else:
                total_fail += 1

    # Summary
    print("\n── SUMMARY ──────────────────────────────────")
    for split, clip_range in splits_to_run.items():
        split_dir   = OUTPUT_DIR / split
        frame_count = len(list(split_dir.glob("*.png"))) if split_dir.exists() else 0
        print(f"  {split:<6}  {frame_count} frames")

    print(f"\n  Clips OK: {total_ok}  |  Failed: {total_fail}")
    if total_fail:
        print("[WARN] Some clips failed. Check errors above.")
    else:
        print("[OK] All done — frames are 1-indexed, ready for nps_to_visdrone.py")