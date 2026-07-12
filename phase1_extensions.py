"""Run all Phase 1 extension experiments (movement window, CEBRA, AAL, multi-subject).

  python phase1_extensions.py --file path\\to\\sub-01_ses-3_behavior+ecephys.nwb
"""

import argparse
import os
import subprocess
import sys

NWB_DEFAULT = r"C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb"
PY = sys.executable
SCRIPT = os.path.join(os.path.dirname(__file__), "phase1_resolution.py")


def run_cmd(extra, out_dir):
    cmd = [PY, SCRIPT, "--out-dir", out_dir] + extra
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=NWB_DEFAULT)
    ap.add_argument("--skip-cebra", action="store_true", help="skip CEBRA (slow)")
    ap.add_argument("--dur-min", type=float, default=30.0)
    args = ap.parse_args()
    f = ["--file", args.file, "--dur-min", str(args.dur_min)]

    # 1. Movement-rich window — speed labels (no reaches in this span)
    run_cmd(f + ["--anchor", "movement", "--label", "speed_median"],
            "phase1_out_movement")

    # 2. AAL channel selection (fixed atlas)
    run_cmd(f + ["--channel-method", "aal"], "phase1_out_aal")

    # 3. Coord-box baseline for comparison
    run_cmd(f + ["--channel-method", "box"], "phase1_out_box")

    # 4. CEBRA vs band on same reach-dense span
    if not args.skip_cebra:
        run_cmd(f + ["--anchor", "reach", "--features", "both",
                       "--cebra-iter", "1500"], "phase1_out_cebra")

    # 5. Multi-subject loop (uses every large .nwb found)
    run_cmd(["--files", args.file, "--anchor", "reach"], "phase1_out_multisub")

    print("\nAll extension runs complete. See phase1_out_*/ directories.")


if __name__ == "__main__":
    main()
