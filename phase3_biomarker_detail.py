"""Biomarker detail: tensor peek, brain-overlay (glass brain), and the multi-state view.

Answers three questions:
  1. What is the biomarker's tensor shape, with a sample to inspect.
  2. Where does it live in the brain, overlaid on an MNI glass brain (nilearn).
  3. Why sleep-vs-active only -- show ALL states on the arousal axis and a multi-class
     view, so the binary choice is justified by the data rather than assumed.

Run (dbs-ml env):
  python phase3_biomarker_detail.py --nwb <sub-01 path> --out-dir phase3_manifold
"""

import argparse
import os

import numpy as np

from nwb_dataset import good_channel_indices, electrode_coords
from phase3_eval import build_epoch_dataset, find_default_file
from phase3_manifold import STATE_BANDS, T5_LABELS, drop_outliers
from phase3_biomarker import fit_biomarker


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--out-dir", default="phase3_manifold")
    ap.add_argument("--epoch-window-sec", type=float, default=10.0)
    ap.add_argument("--epoch-max-per-label", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = args.nwb or find_default_file()
    os.makedirs(args.out_dir, exist_ok=True)
    sid = os.path.basename(path).split("_")[0].replace("sub-", "")
    with __import__("h5py").File(path, "r") as f:
        good = good_channel_indices(f)
    n_ch, n_bd = len(good), len(STATE_BANDS)

    X, y, t0s = build_epoch_dataset(path, good, STATE_BANDS, window_sec=args.epoch_window_sec,
                                    max_per_label=args.epoch_max_per_label,
                                    label_set=set(T5_LABELS), seed=args.seed, verbose=False)
    keep = drop_outliers(X)
    X, y, t0s = X[keep], np.asarray(y)[keep], np.asarray(t0s)[keep]

    # ---- 1. tensor peek ---------------------------------------------------- #
    print("\n================ BIOMARKER TENSOR ================")
    print("feature matrix X       : {}  (windows x [channels*bands])".format(X.shape))
    print("  = interpretable as   : ({}, {}, {})  (windows, {} channels, {} bands)".format(
        X.shape[0], n_ch, n_bd, n_ch, n_bd))
    print("  bands (columns)      : {}".format(list(STATE_BANDS)))
    print("labels y               : {}  e.g. {}".format(y.shape, list(y[:4])))
    print("window times t0s       : {}  (seconds)".format(t0s.shape))
    print("\nsample -- X[0] reshaped to (channel, band), first 4 channels (log band-power):")
    x0 = X[0].reshape(n_ch, n_bd)
    for ci in range(4):
        print("  ch{:<3d} ".format(int(good[ci])) + "  ".join(
            "{}={:+.2f}".format(b[:5], x0[ci, bi]) for bi, b in enumerate(STATE_BANDS)))

    auc, proj, y_bin, w_el, band_frac = fit_biomarker(X, y, n_ch, STATE_BANDS, args.seed)
    W = None
    # recover full LDA weights (per channel*band) for the peek
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    Xz = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis().fit(Xz, (y == "Sleep/rest").astype(int))
    W = lda.coef_.ravel()
    print("\nLDA weight vector      : {}  (one weight per channel*band)".format(W.shape))
    print("biomarker value 'proj' : {}  (one number per window)  sample: {}".format(
        proj.shape, np.round(proj[:6], 2)))
    top = np.argsort(-np.abs(W))[:5]
    print("top-5 weighted features (channel, band, weight):")
    for k in top:
        print("  ch{:<3d} {:11s} w={:+.2f}".format(int(good[k // n_bd]), STATE_BANDS[k % n_bd], W[k]))
    print("biomarker AUC (sleep vs active): {:.3f}".format(auc))
    print("==================================================\n")

    # ---- 2. brain overlay (glass brain) ------------------------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        from nilearn import plotting
        xyz, _ = electrode_coords(path)
        coords = xyz[good]
        vals = w_el / (np.abs(w_el).max() + 1e-9)
        disp = plotting.plot_markers(
            vals, coords, node_size=40, node_cmap="magma", display_mode="ortho",
            title="sub-{}: biomarker weight on the brain (sleep-vs-active)".format(sid),
            colorbar=True)
        out = os.path.join(args.out_dir, "biomarker_glassbrain_sub{}.png".format(sid))
        disp.savefig(out, dpi=150)
        disp.close()
        print("wrote", out)
    except Exception as e:  # noqa: BLE001
        print("glass-brain skipped:", type(e).__name__, e)

    # ---- 3. all states on the arousal axis + multiclass -------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = sorted(set(y))
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
        hrs = t0s / 3600.0
        for i, l in enumerate(labels):
            m = y == l
            ax[0].scatter(hrs[m], proj[m], s=12, alpha=.6, color=plt.cm.tab10(i / 10), label=l)
        ax[0].set_xlabel("time into recording (hours)")
        ax[0].set_ylabel("arousal-axis biomarker\n(high = sleep-like)")
        ax[0].set_title("All behavioral states on the arousal axis")
        ax[0].legend(fontsize=7, ncol=2)
        ax[0].grid(alpha=.25)

        # multiclass LDA 2D: does anything beyond sleep separate?
        from sklearn.preprocessing import StandardScaler as SS
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA2
        Z = LDA2(n_components=2).fit(SS().fit_transform(X), y).transform(SS().fit_transform(X))
        for i, l in enumerate(labels):
            m = y == l
            ax[1].scatter(Z[m, 0], Z[m, 1], s=12, alpha=.6, color=plt.cm.tab10(i / 10), label=l)
        ax[1].set_xlabel("LD1 (mostly sleep-vs-active)")
        ax[1].set_ylabel("LD2 (next-best separation)")
        ax[1].set_title("Multi-state discriminant space")
        ax[1].legend(fontsize=7, ncol=2)
        ax[1].grid(alpha=.25)
        fig.tight_layout()
        out = os.path.join(args.out_dir, "biomarker_multistate_sub{}.png".format(sid))
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)
    except Exception as e:  # noqa: BLE001
        print("multistate plot skipped:", type(e).__name__, e)


if __name__ == "__main__":
    main()
