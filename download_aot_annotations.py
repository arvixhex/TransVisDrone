#!/usr/bin/env python3
"""
setup_aot.py — AOT Dataset Setup Script
========================================
Downloads the AOT (Airborne Object Tracking) groundtruth and encounter list
from the public S3 bucket.

IMPORTANT — WHERE TO RUN THIS SCRIPT:
    Run this script from your project root directory, i.e. the folder that
    directly contains the 'aotcore' sub-directory.

    Example:
        cd ~/tvd          # <-- this folder must contain aotcore/
        python scripts/setup_aot.py

    You can also pass --base-dir explicitly, but the default auto-detection
    is recommended so all other scripts stay consistent.

Directory layout produced (relative to project root):
    datasets/
    └── AOT/
        └── part1/          (or part2 / part3 depending on selection)
            ├── ImageSets/
            │   ├── groundtruth.json
            │   └── valid_encounters_maxRange700_maxGap3_minEncLen30.json
            └── Images/     (empty, ready for image downloads)

Usage:
    python setup_aot.py                     # interactive part selection (default: part1)
    python setup_aot.py --part 1            # non-interactive, download part1
    python setup_aot.py --part 2 3          # download part2 and part3
    python setup_aot.py --part all          # download all three parts
    python setup_aot.py --base-dir ~/tvd    # explicit base dir override
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S3_BUCKET = "s3://airborne-obj-detection-challenge-training"

# Files to download per part.
# Each entry: (s3_key_relative_to_part_root, local_path_relative_to_part_root)
PART_FILES = [
    (
        "ImageSets/groundtruth.json",
        "ImageSets/groundtruth.json",
    ),
    (
        "ImageSets/valid_encounters_maxRange700_maxGap3_minEncLen30.json",
        "ImageSets/valid_encounters_maxRange700_maxGap3_minEncLen30.json",
    ),
]

VALID_PARTS = [1, 2, 3]
REQUIRED_SUBDIRS = ["ImageSets", "Images"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


# ---------------------------------------------------------------------------
# aotcore detection
# ---------------------------------------------------------------------------

def find_aotcore_root() -> Path | None:
    """
    Walk upward from this script's location until we find a directory
    that directly contains an 'aotcore' sub-directory.
    Returns that directory, or None if not found.
    """
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "aotcore").is_dir():
            return candidate
    return None


def resolve_base_dir(cli_arg: str | None) -> Path:
    """
    Determine the base (project root) directory.
    Priority:  CLI --base-dir  >  auto-detect via aotcore presence
    Exits with a clear message if neither works.
    """
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

    # Hard stop — aotcore not found anywhere
    print()
    print("=" * 60)
    log("ERROR", "Cannot locate the 'aotcore' folder.")
    log("ERROR", "This script must be run from your project root —")
    log("ERROR", "the directory that directly contains 'aotcore/'.")
    print()
    log("ERROR", "Fix options:")
    log("ERROR", "  1. cd into your project root, then re-run:")
    log("ERROR", "  2. Pass the root explicitly:")
    log("ERROR", "       python setup_aot.py --base-dir ~/example/path")
    print("=" * 60)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Part selection
# ---------------------------------------------------------------------------

def select_parts_interactive() -> list[int]:
    """
    Prompt the user to pick which parts to download.
    Default (just pressing Enter) = part 1 only.
    """
    print()
    print("  Which AOT parts do you want to download annotation files for?")
    print("  Options : 1 | 2 | 3 | 1 2 | 1 3 | 2 3 | 1 2 3 | all")
    print("  Default : 1  (press Enter to accept)")
    print()
    raw = input("  Your choice: ").strip()

    if not raw:
        log("INFO", "No input — using default: part1")
        return [1]

    if raw.lower() == "all":
        return [1, 2, 3]

    chosen = []
    for token in raw.replace(",", " ").split():
        try:
            n = int(token)
            if n not in VALID_PARTS:
                raise ValueError
            if n not in chosen:
                chosen.append(n)
        except ValueError:
            log("WARN", f"Ignoring invalid part value: '{token}'")

    if not chosen:
        log("WARN", "No valid parts selected — falling back to default: part1")
        return [1]

    return sorted(chosen)


def resolve_parts(cli_parts: list[str] | None) -> list[int]:
    """Parse --part CLI values, or fall back to interactive prompt."""
    if cli_parts is None:
        return select_parts_interactive()

    if len(cli_parts) == 1 and cli_parts[0].lower() == "all":
        return [1, 2, 3]

    chosen = []
    for token in cli_parts:
        try:
            n = int(token)
            if n not in VALID_PARTS:
                raise ValueError
            if n not in chosen:
                chosen.append(n)
        except ValueError:
            log("WARN", f"Ignoring invalid --part value: '{token}'")

    if not chosen:
        log("WARN", "No valid --part values — using default: part1")
        return [1]

    return sorted(chosen)


# ---------------------------------------------------------------------------
# AWS CLI check
# ---------------------------------------------------------------------------

def check_aws_cli() -> bool:
    try:
        result = subprocess.run(
            ["aws", "--version"],
            capture_output=True, text=True, check=True
        )
        version_str = (result.stdout or result.stderr).strip()
        log("OK", f"AWS CLI found: {version_str}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("ERROR", "AWS CLI not found.")
        log("ERROR", "Install it with:  pip install awscli  OR  conda install -c conda-forge awscli")
        return False


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def create_part_dirs(base: Path, part: int) -> None:
    for sub in REQUIRED_SUBDIRS:
        target = base / "datasets" / "AOT" / f"part{part}" / sub
        target.mkdir(parents=True, exist_ok=True)
        log("OK", f"Directory ready: {target}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(s3_key: str, dest: Path) -> bool:
    """
    Download one file from the S3 bucket using --no-sign-request.
    Skips if file already exists and is non-empty.
    Returns True on success.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        log("SKIP", f"Already exists ({dest.stat().st_size:,} B): {dest.name}")
        return True

    s3_uri = f"{S3_BUCKET}/{s3_key}"
    log("INFO", f"Downloading: {s3_uri}")
    log("INFO", f"         -> {dest}")

    result = subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(dest), "--no-sign-request"],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        size = dest.stat().st_size if dest.exists() else 0
        log("OK", f"Saved ({size:,} B): {dest.name}")
        return True
    else:
        log("ERROR", f"Failed: {s3_key}")
        if result.stderr.strip():
            log("ERROR", result.stderr.strip())
        return False


def download_part(base: Path, part: int) -> list[tuple[str, bool]]:
    """Download all annotation files for one part. Returns list of (label, ok)."""
    print()
    log("INFO", f"--- Downloading part{part} ---")
    create_part_dirs(base, part)

    results = []
    for s3_rel, local_rel in PART_FILES:
        s3_key = f"part{part}/{s3_rel}"
        dest   = base / "datasets" / "AOT" / f"part{part}" / local_rel
        ok = download_file(s3_key, dest)
        results.append((f"part{part}/{local_rel}", ok))
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def show_summary(base: Path, parts: list[int]) -> None:
    print()
    log("INFO", "Final layout:")
    for part in parts:
        image_sets = base / "datasets" / "AOT" / f"part{part}" / "ImageSets"
        print(f"    datasets/AOT/part{part}/ImageSets/")
        if image_sets.exists():
            files = sorted(image_sets.iterdir())
            if files:
                for f in files:
                    print(f"        {f.stat().st_size:>12,} B   {f.name}")
            else:
                print("        (empty — all downloads failed)")
        else:
            print("        (directory missing)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Set up AOT dataset folder structure and download annotation files. "
            "Run from your project root (the folder that contains 'aotcore/')."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--part",
        nargs="+",
        metavar="N",
        help="Part(s) to download: 1, 2, 3, or 'all'. Omit for interactive prompt.",
    )
    parser.add_argument(
        "--base-dir",
        metavar="PATH",
        help=(
            "Project root directory (must contain 'aotcore/'). "
            "Auto-detected from this script's location if omitted."
        ),
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  AOT Dataset Setup")
    print("=" * 60)
    print()
    log("WARN", "Make sure you are running this from your project root")
    log("WARN", "(the folder that directly contains 'aotcore/').")
    log("WARN", "Data will be saved to:  <project_root>/datasets/AOT/")
    print()

    # Step 1: resolve base dir (enforces aotcore check)
    base = resolve_base_dir(args.base_dir)

    # Step 2: check AWS CLI
    print()
    if not check_aws_cli():
        sys.exit(1)

    # Step 3: part selection
    parts = resolve_parts(args.part)
    print()
    log("INFO", f"Parts selected: {[f'part{p}' for p in parts]}")

    # Step 4: download each part
    all_results: list[tuple[str, bool]] = []
    for part in parts:
        all_results.extend(download_part(base, part))

    # Step 5: summary
    show_summary(base, parts)

    print()
    failed = [(name, ok) for name, ok in all_results if not ok]
    if not failed:
        log("OK", "All annotation files downloaded successfully.")
    else:
        log("WARN", f"{len(failed)} file(s) failed:")
        for name, _ in failed:
            log("WARN", f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()