"""
Download UAV NPS dataset, extract, and organize into train/val/test symlinks.

Usage:
    python download_and_setup_nps.py               # full run
    python download_and_setup_nps.py --skip-download  # re-use existing zip
    python download_and_setup_nps.py --skip-extract   # re-use already-extracted clips
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
URL        = "https://engineering.purdue.edu/~bouman/UAV_Dataset/Videos.zip"
NPS_DIR    = Path("datasets/NPS")
VIDEOS_DIR = NPS_DIR / "Videos"
ZIP_PATH   = NPS_DIR / "Videos.zip"

SPLITS = {
    "train": range(1,  37),   # Clips  1-36
    "val":   range(37, 41),   # Clips 37-40
    "test":  range(41, 51),   # Clips 41-50
}

EXPECTED_TOTAL = sum(len(r) for r in SPLITS.values())  # 50

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def run(cmd: str, label: str) -> int:
    """Run a shell command, print it, return exit code."""
    print(f"  $ {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"[ERROR] {label} failed with exit code {ret}.")
    return ret


def check_tool(name: str):
    if shutil.which(name) is None:
        print(f"[ERROR] '{name}' not found in PATH. Please install it and retry.")
        sys.exit(1)


def already_extracted() -> bool:
    """True if at least Clip_1.mov exists directly under VIDEOS_DIR."""
    return (VIDEOS_DIR / "Clip_1.mov").exists()


# ──────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────
def step_download(skip: bool):
    print("\n[Step 1] Download")
    if skip:
        if not ZIP_PATH.exists():
            print(f"[ERROR] --skip-download set but {ZIP_PATH} not found.")
            sys.exit(1)
        print(f"  Skipping download — using existing {ZIP_PATH}")
        return

    NPS_DIR.mkdir(parents=True, exist_ok=True)
    check_tool("wget")
    ret = run(f'wget -c --show-progress "{URL}" -O "{ZIP_PATH}"', "wget download")
    if ret != 0 or not ZIP_PATH.exists():
        print("[ERROR] Download failed.")
        sys.exit(1)
    print(f"  Downloaded → {ZIP_PATH}  ({ZIP_PATH.stat().st_size / 1e9:.2f} GB)")


def step_extract(skip: bool):
    print("\n[Step 2] Extract")
    if skip:
        if not already_extracted():
            print(f"[ERROR] --skip-extract set but clips not found in {VIDEOS_DIR}.")
            sys.exit(1)
        print(f"  Skipping extraction — clips already present in {VIDEOS_DIR}")
        return

    if already_extracted():
        print(f"  Clips already extracted in {VIDEOS_DIR}, skipping.")
        return

    check_tool("7z")
    # Extract to NPS_DIR; the zip contains a Videos/ folder, so clips land at
    # datasets/NPS/Videos/Clip_X.mov  (NOT datasets/NPS/Videos/Videos/Clip_X.mov)
    ret = run(f'7z x "{ZIP_PATH}" -o"{NPS_DIR}" -y', "7z extract")
    if ret != 0:
        sys.exit(1)

    if not already_extracted():
        print(f"[ERROR] Extraction finished but Clip_1.mov not found in {VIDEOS_DIR}.")
        print("        Check if the zip uses a different folder name or clip naming.")
        sys.exit(1)

    print("  Extraction complete.")


def step_symlinks():
    print("\n[Step 3] Create train/val/test symlinks")
    missing, linked, skipped = [], 0, 0

    for split, clip_range in SPLITS.items():
        split_dir = VIDEOS_DIR / split
        split_dir.mkdir(exist_ok=True)

        for i in clip_range:
            src = VIDEOS_DIR / f"Clip_{i}.mov"
            dst = split_dir  / f"Clip_{i}.mov"

            if not src.exists():
                missing.append(src.name)
                continue

            if dst.is_symlink():
                # Fix broken symlinks silently
                if not dst.exists():
                    dst.unlink()
                else:
                    skipped += 1
                    continue

            dst.symlink_to(src.resolve())
            linked += 1

    print(f"  Linked: {linked}  |  Already existed: {skipped}  |  Source missing: {len(missing)}")
    if missing:
        print(f"  [WARN] Missing clips: {', '.join(missing)}")


def step_verify():
    print("\n[Step 4] Verify")
    total = 0
    all_ok = True

    for split, clip_range in SPLITS.items():
        split_dir  = VIDEOS_DIR / split
        expected   = len(clip_range)
        # Count only .mov symlinks, ignore stray files
        found      = len([f for f in split_dir.iterdir() if f.suffix == ".mov"])
        status     = "OK" if found == expected else "!!"
        total     += found
        if found != expected:
            all_ok = False
        print(f"  [{status}]  {split:<6}  {found}/{expected} clips")

    print(f"\n  Total: {total}/{EXPECTED_TOTAL}")
    if not all_ok:
        print("[WARN] Some splits are incomplete. Check warnings above.")
    else:
        print("[OK] Dataset ready.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Download and set up the NPS UAV dataset.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download and use an existing Videos.zip")
    p.add_argument("--skip-extract",  action="store_true",
                   help="Skip extraction and use already-extracted clips")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    step_download(skip=args.skip_download)
    step_extract(skip=args.skip_extract)
    step_symlinks()
    step_verify()