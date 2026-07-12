"""Download selected additional AJILE12 subjects from DANDI 000055 for cross-subject work.

Fetches a small, explicit set of *distinct-subject* sessions (not more sub-01) into
``ajile12-nwb-data/``, chosen as the smallest distinct-subject files to respect limited
local disk. Uses the DANDI REST API per-asset download URL with resume + a disk-space
guard so a nearly-full disk fails safely instead of wedging the system.

Run (in the dbs-ml env, from the repo root):
    python download_ajile_subjects.py
"""

import os
import shutil
import sys
import time
import urllib.request

DANDISET = "000055"
API = "https://api.dandiarchive.org/api/dandisets/{}/versions/draft/assets/".format(DANDISET)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ajile12-nwb-data")

# Distinct-subject sessions to fetch (smallest per subject). sub-01 already local.
WANT = [
    "sub-04/sub-04_ses-3_behavior+ecephys.nwb",
    "sub-06/sub-06_ses-3_behavior+ecephys.nwb",
    "sub-07/sub-07_ses-3_behavior+ecephys.nwb",
]

# Refuse to start / continue a file if free space would drop below this (bytes).
MIN_FREE_BYTES = 10 * 1024 ** 3  # 10 GB safety margin


def list_assets():
    import json
    with urllib.request.urlopen(API + "?page_size=100", timeout=60) as r:
        return json.load(r)["results"]


def download_asset(asset_id, size, dest):
    url = API + "{}/download/".format(asset_id)
    tmp = dest + ".part"
    resume_from = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    free = shutil.disk_usage(os.path.dirname(dest)).free
    need = size - resume_from
    if free - need < MIN_FREE_BYTES:
        raise SystemExit(
            "ABORT: {:.1f} GB free, need {:.1f} GB for {} (+{:.0f} GB margin). "
            "Free space first.".format(free / 1e9, need / 1e9, os.path.basename(dest),
                                       MIN_FREE_BYTES / 1e9))

    req = urllib.request.Request(url)
    if resume_from:
        req.add_header("Range", "bytes={}-".format(resume_from))
        print("  resuming from {:.2f} GB".format(resume_from / 1e9), flush=True)

    mode = "ab" if resume_from else "wb"
    t0 = time.time()
    done = resume_from
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, mode) as fh:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if int(done / 1e9) != int((done - len(chunk)) / 1e9):  # ~ every 1 GB
                rate = (done - resume_from) / max(1e-9, time.time() - t0) / 1e6
                print("  {:.0f}/{:.0f} GB  ({:.1f} MB/s)".format(
                    done / 1e9, size / 1e9, rate), flush=True)
    os.replace(tmp, dest)
    print("  DONE {} ({:.2f} GB)".format(os.path.basename(dest), done / 1e9), flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assets = list_assets()
    by_path = {a["path"]: a for a in assets}

    print("Downloading {} assets into {}".format(len(WANT), OUT_DIR), flush=True)
    for i, path in enumerate(WANT, 1):
        a = by_path.get(path)
        if a is None:
            print("[{}/{}] MISSING on server: {}".format(i, len(WANT), path), flush=True)
            continue
        dest = os.path.join(OUT_DIR, os.path.basename(path))
        if os.path.exists(dest) and os.path.getsize(dest) == a["size"]:
            print("[{}/{}] already complete: {}".format(i, len(WANT), path), flush=True)
            continue
        print("[{}/{}] {} ({:.2f} GB)".format(i, len(WANT), path, a["size"] / 1e9), flush=True)
        download_asset(a["asset_id"], a["size"], dest)

    print("All requested downloads finished.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
