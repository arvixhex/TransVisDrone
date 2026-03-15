#!/usr/bin/env python3
"""
download_aot_images.py — AOT Image Downloader
===============================================
Downloads flight images from the AOT (Airborne Object Tracking) S3 bucket.

IMPORTANT — WHERE TO RUN THIS SCRIPT:
    Run from your project root (the folder containing 'aotcore/').
    Data is saved relative to that root under datasets/AOT/

Directory layout produced:
    datasets/AOT/part{N}/Images/
        train/   ← trainflightids{N}.json
        val/     ← valflightids{N}.json  (or validationflightids{N}.json)
        test/    ← testflightidsfull{N}.json

Usage:
    python download_aot_images.py                       # fully interactive
    python download_aot_images.py --part 1              # skip part prompt
    python download_aot_images.py --part 1 --split test # skip both prompts
    python download_aot_images.py --base-dir ~/tvd
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S3_BUCKET = "s3://airborne-obj-detection-challenge-training"

VALID_PARTS  = [1, 2, 3]
VALID_SPLITS = ["train", "val", "test"]

# Flight ID JSON filenames per part per split.
# Keys: (part, split) → filename inside aot_flight_ids/
# Only part1 files are currently known to exist — add others as obtained.
FLIGHT_ID_FILES: dict[tuple[int, str], str] = {
    (1, "train"): "trainflightidsfull1.json",
    (1, "val"):   "valflightidsfull1.json",
    (1, "test"):  "testflightidsfull1.json",
    (2, "train"): "trainflightidsfull2.json",       # not yet available
    (2, "val"):   "valflightidsfull2.json",          # not yet available
    (2, "test"):  "testflightidsfull2.json",     # not yet available
    (3, "train"): "trainflightidsfull3.json",        # not yet available
    (3, "val"):   "valflightidsfull3.json",          # not yet available
    (3, "test"):  "testflightidsfull3.json",     # not yet available
}

AVG_FRAMES_PER_FLIGHT = 1_200
AVG_SIZE_GB_PER_FLIGHT = 3.6


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


# ---------------------------------------------------------------------------
# aotcore / base-dir resolution
# ---------------------------------------------------------------------------

def find_aotcore_root() -> Path | None:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "aotcore").is_dir():
            return candidate
    return None


def resolve_base_dir(cli_arg: str | None) -> Path:
    if cli_arg:
        p = Path(cli_arg).expanduser().resolve()
        if not (p / "aotcore").is_dir():
            log("WARN", f"'aotcore' folder NOT found under: {p}")
            log("WARN", "Continuing anyway since --base-dir was explicit.")
        else:
            log("OK", f"aotcore confirmed at: {p / 'aotcore'}")
        log("INFO", f"Using --base-dir: {p}")
        return p

    root = find_aotcore_root()
    if root is not None:
        log("OK", f"aotcore confirmed at: {root / 'aotcore'}")
        log("INFO", f"Auto-detected project root: {root}")
        return root

    print()
    print("=" * 60)
    log("ERROR", "Cannot locate the 'aotcore' folder.")
    log("ERROR", "Run this script from your project root, e.g.:")
    log("ERROR", "    cd ~/tvd   &&   python scripts/download_aot_images.py")
    log("ERROR", "Or pass the root explicitly:")
    log("ERROR", "    python download_aot_images.py --base-dir ~/tvd")
    print("=" * 60)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Flight-ID JSON helpers
# ---------------------------------------------------------------------------

def get_flight_ids_dir(base: Path) -> Path:
    """Find the aot_flight_ids directory, checking several candidate locations."""
    candidates = [
        base / "aot_flight_ids",
        base / "aotcore" / "aot_flight_ids",
        base / "TransVisDrone" / "aot_flight_ids",
        Path("/home/project/Drone/tvd/TransVisDrone/aot_flight_ids"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return base / "aot_flight_ids"   # best-guess fallback (may not exist yet)


def load_flight_ids(flight_ids_dir: Path, part: int, split: str) -> list[str] | None:
    """
    Load flight IDs for the given part + split.
    Returns the list on success, or None with a detailed error if the file
    is missing — caller decides whether to abort or skip.
    """
    key = (part, split)
    filename = FLIGHT_ID_FILES.get(key)
    if filename is None:
        log("ERROR", f"No flight-ID filename configured for part{part}/{split}.")
        return None

    filepath = flight_ids_dir / filename
    if not filepath.exists():
        print()
        print("=" * 60)
        log("ERROR", f"Flight ID file NOT found for part{part} / {split}:")
        log("ERROR", f"  Expected : {filepath}")
        log("ERROR", "")
        log("ERROR", "This file is required before images can be downloaded.")
        log("ERROR", "Possible fixes:")
        log("ERROR", f"  • Obtain '{filename}' and place it in:")
        log("ERROR", f"    {flight_ids_dir}/")
        log("ERROR", f"  • Or run annotation setup first:")
        log("ERROR", f"    python setup_aot.py --part {part}")
        log("WARN",  f"Skipping part{part}/{split} — no flight ID list available.")
        print("=" * 60)
        return None

    ids: list[str] = json.loads(filepath.read_text())
    log("OK", f"Loaded {len(ids):,} flight IDs  [{split}]  from: {filepath.name}")
    return ids


# ---------------------------------------------------------------------------
# AWS CLI check
# ---------------------------------------------------------------------------

def check_aws_cli() -> bool:
    try:
        r = subprocess.run(["aws", "--version"], capture_output=True, text=True, check=True)
        log("OK", f"AWS CLI: {(r.stdout or r.stderr).strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("ERROR", "AWS CLI not found.")
        log("ERROR", "Install: pip install awscli  OR  conda install -c conda-forge awscli")
        return False


# ---------------------------------------------------------------------------
# Part + split selection
# ---------------------------------------------------------------------------

def select_part_interactive() -> int:
    print()
    print("  Which part do you want to download images for?")
    print("  Options : 1 | 2 | 3")
    print("  Default : 1  (press Enter to accept)")
    raw = input("  Your choice: ").strip()
    if not raw:
        log("INFO", "No input — using default: part1")
        return 1
    try:
        n = int(raw)
        if n in VALID_PARTS:
            return n
    except ValueError:
        pass
    log("WARN", f"Invalid input '{raw}' — using default: part1")
    return 1


def select_split_interactive() -> str:
    print()
    print("  Which split do you want to download?")
    print("  [train]  training flights")
    print("  [val]    validation flights")
    print("  [test]   test flights  (default)")
    raw = input("  Your choice (train/val/test): ").strip().lower()
    if not raw:
        log("INFO", "No input — using default: test")
        return "test"
    if raw in VALID_SPLITS:
        return raw
    log("WARN", f"Invalid input '{raw}' — using default: test")
    return "test"


# ---------------------------------------------------------------------------
# Download mode
# ---------------------------------------------------------------------------

def choose_flights(flight_ids: list[str]) -> list[str]:
    """
    Interactively pick which flights to download (all complete flights).
    Returns a flat list of flight_id strings.
    """
    total = len(flight_ids)

    print()
    print("─" * 60)
    print("  DOWNLOAD MODES  (all modes download complete flights)")
    print("  Note: partial / random-frame download is not supported —")
    print("        it breaks temporal continuity required for detection.")
    print()
    print("  [1] Complete flights by count")
    print(f"      Choose how many full flights (max {total}).")
    print()
    print("  [2] Specific flight IDs")
    print("      Paste your own flight hex IDs one by one.")
    print()
    print("  [3] Continuous — first N flights")
    print("      Simplest; good for initial pipeline testing.")
    print()
    print(f"  [4] All flights  ({total} total)")
    print("      Download every flight in this split's JSON.")
    print()
    mode = input("  Choose mode [1/2/3/4]: ").strip()

    selected: list[str] = []

    # ── Mode 1: count ──────────────────────────────────────────────────────
    if mode == "1":
        n = int(input(f"\n  How many complete flights? (max {total}): ").strip())
        n = min(n, total)
        which = (
            input("  Take from [S]tart, [E]nd, or [R]andom? (S/E/R) [S]: ")
            .strip().upper() or "S"
        )
        if which == "E":
            selected = flight_ids[-n:]
        elif which == "R":
            selected = random.sample(flight_ids, n)
            log("INFO", f"Randomly selected {n} flights.")
        else:
            selected = flight_ids[:n]

        _print_estimate(len(selected))

    # ── Mode 2: specific IDs ───────────────────────────────────────────────
    elif mode == "2":
        show = input("\n  Show all flight IDs with index numbers? (y/n): ").strip().lower()
        if show == "y":
            print()
            for i, fid in enumerate(flight_ids):
                print(f"    [{i:>4}] {fid}")
            print()

        print("  Paste flight IDs one per line. Type DONE or leave blank to finish.")
        while True:
            line = input("  flight_id: ").strip()
            if line.upper() == "DONE" or line == "":
                break
            if line in flight_ids:
                selected.append(line)
                print(f"    [OK]   valid flight ID")
            else:
                print(f"    [WARN] not in flight list — adding anyway")
                selected.append(line)

        log("INFO", f"Will download {len(selected)} specific flights")

    # ── Mode 3: first N continuous ─────────────────────────────────────────
    elif mode == "3":
        n = int(input(f"\n  How many flights from start? (max {total}): ").strip())
        n = min(n, total)
        selected = flight_ids[:n]
        _print_estimate(len(selected))

    # ── Mode 4: all flights ────────────────────────────────────────────────
    elif mode == "4":
        selected = list(flight_ids)
        print()
        log("INFO", f"All {total} flights selected.")
        _print_estimate(len(selected))

    else:
        log("ERROR", "Invalid mode. Exiting.")
        sys.exit(1)

    return selected


def _print_estimate(n: int) -> None:
    print()
    log("INFO", f"Flights selected : {n}")
    log("INFO", f"Est. frames      : ~{n * AVG_FRAMES_PER_FLIGHT:,}")
    log("INFO", f"Est. size        : ~{n * AVG_SIZE_GB_PER_FLIGHT:.1f} GB")


# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------

def confirm_download(n_flights: int, dest_dir: Path) -> bool:
    print()
    print("─" * 60)
    print("  READY TO DOWNLOAD")
    print(f"  Flights          : {n_flights}")
    print(f"  Estimated size   : ~{n_flights * AVG_SIZE_GB_PER_FLIGHT:.1f} GB")
    print(f"  Destination      : {dest_dir}")
    print("─" * 60)
    ans = input("\n  Start download? (y/n): ").strip().lower()
    return ans == "y"


# ---------------------------------------------------------------------------
# Download loop
# ---------------------------------------------------------------------------

def download_flights(
    flight_ids: list[str],
    images_dir: Path,
    part: int,
    split: str,
) -> list[str]:
    """
    Downloads each flight completely via `aws s3 sync`.
    Returns list of failed flight IDs.
    """
    failed: list[str] = []
    total = len(flight_ids)

    for i, flight_id in enumerate(flight_ids):
        dest = images_dir / flight_id
        dest.mkdir(parents=True, exist_ok=True)

        # Skip if looks already downloaded
        existing = len(list(dest.iterdir()))
        if existing > 100:
            log("SKIP", f"[{i+1:>4}/{total}] Already exists ({existing} frames): {flight_id[:24]}...")
            continue

        s3_path = f"{S3_BUCKET}/part{part}/Images/{flight_id}/"
        print(f"[INFO] [{i+1:>4}/{total}] Downloading: {flight_id[:24]}...")

        result = subprocess.run([
            "aws", "s3", "sync", s3_path, str(dest),
            "--no-sign-request", "--quiet"
        ])

        if result.returncode == 0:
            n = len(list(dest.iterdir()))
            print(f"[OK]   [{i+1:>4}/{total}] Done — {n:,} frames  (~{n * 3 / 1000:.1f} MB)")
        else:
            log("ERROR", f"[{i+1:>4}/{total}] FAILED: {flight_id}")
            failed.append(flight_id)

    return failed


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(images_dir: Path, flight_ids: list[str], failed: list[str]) -> None:
    total_frames = sum(
        len(list((images_dir / fid).iterdir()))
        for fid in flight_ids
        if (images_dir / fid).exists()
    )
    size_result = subprocess.run(
        ["du", "-sh", str(images_dir)],
        capture_output=True, text=True
    )
    size_str = size_result.stdout.split()[0] if size_result.returncode == 0 else "unknown"

    print()
    print("=" * 60)
    print("  DOWNLOAD COMPLETE")
    print(f"  Total frames on disk : {total_frames:,}")
    print(f"  Total size on disk   : {size_str}")
    print(f"  Destination          : {images_dir}")
    print(f"  Failed flights       : {len(failed)}")
    if failed:
        print("  Failed IDs:")
        for fid in failed:
            print(f"    {fid}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download AOT flight images. "
            "Run from project root (the folder containing 'aotcore/')."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--part", type=int, choices=VALID_PARTS,
        help="Part number (1/2/3). Prompted interactively if omitted."
    )
    parser.add_argument(
        "--split", choices=VALID_SPLITS,
        help="Split to download: train | val | test. Prompted interactively if omitted."
    )
    parser.add_argument(
        "--base-dir", metavar="PATH",
        help="Project root (must contain 'aotcore/'). Auto-detected if omitted."
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  AOT Image Downloader")
    print("=" * 60)
    print()
    log("WARN", "Run this script from your project root")
    log("WARN", "(the folder that directly contains 'aotcore/').")
    log("WARN", "Images will be saved to:")
    log("WARN", "  <project_root>/datasets/AOT/part{N}/Images/{split}/")
    print()

    # Step 1: base dir
    base = resolve_base_dir(args.base_dir)
    flight_ids_dir = get_flight_ids_dir(base)
    log("INFO", f"Flight ID files directory: {flight_ids_dir}")

    # Step 2: AWS CLI
    print()
    if not check_aws_cli():
        sys.exit(1)

    # Step 3: part selection
    part = args.part if args.part else select_part_interactive()
    log("INFO", f"Part selected: part{part}")

    # Step 4: split selection
    split = args.split if args.split else select_split_interactive()
    log("INFO", f"Split selected: {split}")

    # Step 5: load flight IDs — hard stop if file missing
    flight_ids = load_flight_ids(flight_ids_dir, part, split)
    if flight_ids is None:
        sys.exit(1)

    print()
    print(f"  Total flights in '{split}' split  : {len(flight_ids):,}")
    print(f"  Avg frames per flight             : ~{AVG_FRAMES_PER_FLIGHT:,}")
    print(f"  Avg size per flight               : ~{AVG_SIZE_GB_PER_FLIGHT} GB")
    print(f"  Total if all downloaded           : ~{len(flight_ids) * AVG_SIZE_GB_PER_FLIGHT:.0f} GB")

    # Step 6: choose flights interactively
    selected = choose_flights(flight_ids)
    if not selected:
        log("WARN", "No flights selected. Exiting.")
        sys.exit(0)

    # Step 7: confirm
    #   Save path:  datasets/AOT/part{N}/Images/{split}/
    images_dir = base / "datasets" / "AOT" / f"part{part}" / "Images" 
    images_dir.mkdir(parents=True, exist_ok=True)

    if not confirm_download(len(selected), images_dir):
        log("INFO", "Cancelled by user.")
        sys.exit(0)

    # Step 8: download
    print()
    failed = download_flights(selected, images_dir, part, split)

    # Step 9: summary
    print_summary(images_dir, selected, failed)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()