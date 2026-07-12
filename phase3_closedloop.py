"""Biomarker -> closed-loop control simulation (Scangos-style).

Takes the α/β arousal biomarker (the sleep-vs-active LDA axis from phase3_biomarker) and
asks the closed-loop-DBS question: if this coordinate drove an implant, how well would it
trigger? We stream the biomarker in true time order and fire a trigger with hysteresis
(two thresholds, to avoid chattering), then measure the quantities a device engineer cares
about: detection latency at state onset, sensitivity over the target episode, and the
false-trigger rate during the non-target period.

This is a feasibility simulation on labeled data (sleep = the detectable target state), not
a therapy claim -- it shows the biomarker *could* serve as a closed-loop readout, and with
what latency / false-alarm trade-off.

Run (dbs-ml env):
  python phase3_closedloop.py --out-dir phase3_manifold
"""

import argparse
import os

import numpy as np

from nwb_dataset import good_channel_indices
from phase3_eval import build_epoch_dataset, find_default_file
from phase3_manifold import STATE_BANDS, T5_LABELS, drop_outliers
from phase3_biomarker import fit_biomarker


def hysteresis_trigger(x, hi, lo):
    """Two-threshold state machine: ON when x rises above hi, OFF when it falls below lo."""
    out = np.zeros(len(x), dtype=int)
    state = 0
    for i, v in enumerate(x):
        if state == 0 and v >= hi:
            state = 1
        elif state == 1 and v <= lo:
            state = 0
        out[i] = state
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--out-dir", default="phase3_manifold")
    ap.add_argument("--epoch-window-sec", type=float, default=10.0)
    ap.add_argument("--epoch-max-per-label", type=int, default=150)
    ap.add_argument("--margin", type=float, default=0.5,
                    help="hysteresis half-width as a fraction of the active→sleep gap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = args.nwb or find_default_file()
    os.makedirs(args.out_dir, exist_ok=True)
    with __import__("h5py").File(path, "r") as f:
        good = good_channel_indices(f)

    X, y, t0s = build_epoch_dataset(path, good, STATE_BANDS, window_sec=args.epoch_window_sec,
                                    max_per_label=args.epoch_max_per_label,
                                    label_set=set(T5_LABELS), seed=args.seed, verbose=False)
    keep = drop_outliers(X)
    X, y, t0s = X[keep], np.asarray(y)[keep], np.asarray(t0s)[keep]
    auc, proj, y_bin, _, _ = fit_biomarker(X, y, len(good), STATE_BANDS, args.seed)

    # time order
    order = np.argsort(t0s)
    proj, y_bin, t = proj[order], y_bin[order], t0s[order] / 3600.0  # hours

    # thresholds from the two class means (device would calibrate similarly)
    m_sleep, m_active = proj[y_bin == 1].mean(), proj[y_bin == 0].mean()
    mid = 0.5 * (m_sleep + m_active)
    half = args.margin * (m_sleep - m_active)
    hi, lo = mid + half, mid - half
    trig = hysteresis_trigger(proj, hi, lo)

    # metrics
    sleep_idx = np.where(y_bin == 1)[0]
    active_idx = np.where(y_bin == 0)[0]
    sensitivity = float(trig[sleep_idx].mean()) if len(sleep_idx) else float("nan")
    specificity = float(1 - trig[active_idx].mean()) if len(active_idx) else float("nan")
    # detection latency: time from first sleep window to first trigger during sleep block
    latency = float("nan")
    if len(sleep_idx):
        onset_t = t[sleep_idx[0]]
        fired = np.where((trig == 1) & (t >= onset_t))[0]
        if len(fired):
            latency = float((t[fired[0]] - onset_t) * 60.0)  # minutes
    # false triggers per hour during active span
    active_span_h = float(t[active_idx].max() - t[active_idx].min()) if len(active_idx) else 0.0
    false_on = int(trig[active_idx].sum())
    fa_per_h = false_on / active_span_h if active_span_h > 0 else float("nan")

    print("\n================ CLOSED-LOOP FEASIBILITY ================")
    print("biomarker AUC (sleep vs active)        {:.3f}".format(auc))
    print("trigger thresholds  hi={:.2f}  lo={:.2f}  (hysteresis)".format(hi, lo))
    print("sensitivity (sleep flagged)            {:.1%}".format(sensitivity))
    print("specificity (active correctly quiet)   {:.1%}".format(specificity))
    print("detection latency at sleep onset       {:.1f} min".format(latency))
    print("false triggers during active           {} ({:.2f} / hour)".format(false_on, fa_per_h))
    print("========================================================\n")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 4.4))
        ax.axhspan(lo, hi, color="#cccccc", alpha=.3, label="hysteresis band")
        ax.scatter(t[y_bin == 0], proj[y_bin == 0], s=10, c="#4c78c8", alpha=.5, label="active")
        ax.scatter(t[y_bin == 1], proj[y_bin == 1], s=10, c="#59a14f", alpha=.65, label="Sleep/rest (target)")
        # shade where the device would be stimulating
        on = trig == 1
        if on.any():
            ax.fill_between(t, proj.min() - 1, proj.max() + 1, where=on, color="#d1495b",
                            alpha=.12, step="mid", label="trigger ON (stim)")
        ax.axhline(hi, color="#b0752a", ls="--", lw=1)
        ax.axhline(lo, color="#b0752a", ls="--", lw=1)
        ax.set_xlabel("time into recording (hours)")
        ax.set_ylabel("biomarker value\n(high = sleep-like)")
        ax.set_title("Closed-loop simulation: the biomarker triggering stimulation "
                     "(sens {:.0%}, {:.1f}/h false, {:.0f} min latency)".format(
                         sensitivity, fa_per_h, latency))
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=.25)
        fig.tight_layout()
        out = os.path.join(args.out_dir, "closedloop_sim.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
