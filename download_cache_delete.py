"""Expand the patient cohort past N=3 without needing 850 GB of disk: for each remaining
AJILE12 subject, download its smallest session, extract the band-power cube cache (the
only thing later analysis actually needs, ~0.2-1.3 MB), then delete the 13-20 GB raw NWB
file before moving to the next subject. Peak extra disk usage is ~1 raw file (~20 GB) at
a time, not the full remaining cohort.

Run (dbs-ml env, from the repo root):
    python download_cache_delete.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_ajile_subjects import list_assets, download_asset, MIN_FREE_BYTES, OUT_DIR
from phase4_decompose_transfer import build_cache, cache_path

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase4_cache")

# Remaining subjects' smallest session each (sub-01/04/06/07 already cached).
WANT = [
    ("03", "sub-03/sub-03_ses-6_behavior+ecephys.nwb"),
    ("02", "sub-02/sub-02_ses-5_behavior+ecephys.nwb"),
    ("08", "sub-08/sub-08_ses-6_behavior+ecephys.nwb"),
    ("11", "sub-11/sub-11_ses-4_behavior+ecephys.nwb"),
    ("05", "sub-05/sub-05_ses-7_behavior+ecephys.nwb"),
    ("10", "sub-10/sub-10_ses-5_behavior+ecephys.nwb"),
    ("12", "sub-12/sub-12_ses-7_behavior+ecephys.nwb"),
    ("09", "sub-09/sub-09_ses-7_behavior+ecephys.nwb"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    assets = list_assets()
    by_path = {a["path"]: a for a in assets}

    for i, (sid, path) in enumerate(WANT, 1):
        t0 = time.time()
        print("\n========== [{}/{}] sub-{} ==========".format(i, len(WANT), sid), flush=True)

        if os.path.exists(cache_path(CACHE_DIR, sid)):
            print("  already cached; skipping", flush=True)
            continue

        a = by_path.get(path)
        if a is None:
            print("  MISSING on server: {}".format(path), flush=True)
            continue
        dest = os.path.join(OUT_DIR, os.path.basename(path))

        print("  downloading {} ({:.2f} GB)...".format(path, a["size"] / 1e9), flush=True)
        try:
            download_asset(a["asset_id"], a["size"], dest)
        except SystemExit as e:
            print("  ABORT (disk guard): {}".format(e), flush=True)
            break

        print("  extracting band-power cube (chunked read of the full recording)...", flush=True)
        try:
            build_cache(dest, sid, CACHE_DIR)
        except Exception as e:  # noqa: BLE001
            print("  cache extraction FAILED ({}): {}".format(type(e).__name__, e), flush=True)

        try:
            sz = os.path.getsize(dest) / 1e9
            os.remove(dest)
            print("  deleted raw file ({:.2f} GB freed)".format(sz), flush=True)
        except OSError as e:
            print("  WARNING: could not delete {}: {}".format(dest, e), flush=True)

        print("  sub-{} done in {:.1f} min".format(sid, (time.time() - t0) / 60.0), flush=True)

    print("\nAll requested subjects processed.", flush=True)


if __name__ == "__main__":
    main()
