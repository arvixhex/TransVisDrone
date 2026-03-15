"""
Download NPS-Drones-Dataset annotations from the Dogfight GitHub repo:
  https://github.com/mwaseema/Drone-Detection/tree/main/annotations/NPS-Drones-Dataset

Annotations are saved to:
  datasets/NPS/annotations/

Annotation format (from Dogfight repo):
  frame_number, num_boxes, x1, y1, x2, y2 [, x2_1, y2_1, x2_2, y2_2, ...]
  Frame index is 0-based (frame 0 = first frame).

Usage:
    python download_nps_annotations.py                  # download all
    python download_nps_annotations.py --dry-run        # list files without downloading
    python download_nps_annotations.py --skip-existing  # skip already downloaded files
"""

import sys
import time
import argparse
import urllib.request
import urllib.error
import json
from pathlib import Path

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
GITHUB_API_URL = (
    "https://api.github.com/repos/mwaseema/Drone-Detection"
    "/contents/annotations/NPS-Drones-Dataset"
)
OUTPUT_DIR = Path("datasets/NPS/annotations")

# GitHub API returns download URLs pointing here:
# https://raw.githubusercontent.com/mwaseema/Drone-Detection/main/annotations/NPS-Drones-Dataset/<file>
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/mwaseema/Drone-Detection"
    "/main/annotations/NPS-Drones-Dataset"
)

HEADERS = {
    "User-Agent": "TransVisDrone-NPS-Downloader",
    "Accept": "application/vnd.github.v3+json",
}

RETRY_ATTEMPTS = 3
RETRY_DELAY    = 2.0   # seconds between retries


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def fetch_json(url: str) -> dict | list:
    """Fetch a URL and return parsed JSON."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def download_file(url: str, dest: Path, retries: int = RETRY_ATTEMPTS) -> bool:
    """Download a single file with retry logic. Returns True on success."""
    req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return True
        except urllib.error.URLError as e:
            if attempt < retries:
                print(f"    [RETRY {attempt}/{retries}] {e} — retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [ERROR] Failed after {retries} attempts: {e}")
    return False


def list_annotation_files() -> list[dict]:
    """
    Use GitHub API to list all files in the NPS-Drones-Dataset annotations folder.
    Returns list of dicts with 'name' and 'download_url'.
    """
    print(f"[INFO] Fetching file list from GitHub API...")
    try:
        contents = fetch_json(GITHUB_API_URL)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("[ERROR] GitHub API rate limit hit. Wait ~60s and retry, or set GITHUB_TOKEN.")
        elif e.code == 404:
            print("[ERROR] Repo path not found. Check the URL.")
        else:
            print(f"[ERROR] GitHub API error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not reach GitHub API: {e}")
        sys.exit(1)

    if not isinstance(contents, list):
        print(f"[ERROR] Unexpected API response: {contents}")
        sys.exit(1)

    files = [
        {"name": item["name"], "download_url": item["download_url"]}
        for item in contents
        if item["type"] == "file"
    ]
    print(f"[INFO] Found {len(files)} annotation files.")
    return files


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Download NPS-Drones-Dataset annotations from Dogfight GitHub."
    )
    p.add_argument("--dry-run", action="store_true",
                   help="List files without downloading")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip files already present on disk")
    p.add_argument("--token", type=str, default=None,
                   help="GitHub personal access token (avoids API rate limits)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Optionally add auth header to avoid GitHub rate limits
    if args.token:
        HEADERS["Authorization"] = f"token {args.token}"

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving annotations to: {OUTPUT_DIR.resolve()}\n")

    # Get file list from GitHub API
    files = list_annotation_files()

    if args.dry_run:
        print("\n── DRY RUN — files that would be downloaded ──")
        for f in files:
            print(f"  {f['name']}")
        print(f"\nTotal: {len(files)} files")
        sys.exit(0)

    # Download each file
    total_ok   = 0
    total_fail = 0
    total_skip = 0

    for i, f in enumerate(files, 1):
        dest = OUTPUT_DIR / f["name"]

        if args.skip_existing and dest.exists():
            print(f"  [{i:02d}/{len(files)}] [SKIP] {f['name']} (already exists)")
            total_skip += 1
            continue

        print(f"  [{i:02d}/{len(files)}] Downloading {f['name']} ...", end=" ", flush=True)
        ok = download_file(f["download_url"], dest)

        if ok:
            size_kb = dest.stat().st_size / 1024
            print(f"✅  ({size_kb:.1f} KB)")
            total_ok += 1
        else:
            total_fail += 1

        # Small polite delay to avoid hammering GitHub
        time.sleep(0.1)

    # Summary
    print("\n── SUMMARY ──────────────────────────────────")
    print(f"  Downloaded : {total_ok}")
    print(f"  Skipped    : {total_skip}")
    print(f"  Failed     : {total_fail}")
    print(f"  Location   : {OUTPUT_DIR.resolve()}")

    if total_fail:
        print("\n[WARN] Some files failed. Re-run with --skip-existing to retry only failed ones.")
    else:
        print("\n[OK] All annotation files downloaded.")
        print("[NOTE] Annotations are 0-indexed (frame 0 = first frame).")
        print("       Run nps_to_visdrone.py next to offset to 1-based VisDrone style.")