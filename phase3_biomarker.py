"""Make the state biomarker concrete: what it is, where it lives in the brain, what it
looks like when it activates, and whether it is the same signal across patients.

Produces
--------
1. biomarker_timeseries.png -- the biomarker value across the whole 24 h recording,
   colored by behavioral state. Answers "what does it look like when it activates?" and
   probes whether it is merely binary sleep/wake or a graded arousal/circadian signal.
2. biomarker_brain.png -- each subject's electrodes plotted in MNI brain coordinates,
   colored by how much they drive the sleep-vs-active axis. Answers "what coordinates
   does it live in the brain, and does it look the same across patients?"
3. biomarker_crosssubject.csv -- per-subject AUC, dominant band, dominant region.

Run (dbs-ml env):
  python phase3_biomarker.py --out-dir phase3_manifold
"""

import argparse
import csv
import os

import numpy as np

from nwb_dataset import good_channel_indices, electrode_coords, mni_to_aal, SENSORIMOTOR_AAL
from phase3_eval import build_epoch_dataset, find_default_file
from phase3_crosssubject import find_subject_files
from phase3_manifold import STATE_BANDS, T5_LABELS, drop_outliers


def fit_biomarker(X, y, n_ch, bands, seed=0):
    """LDA sleep(=1)-vs-active(=0). Returns auc, signed projection (sleep high),
    per-electrode importance (|weight| summed over bands), per-band fraction."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    y_bin = (np.asarray(y) == "Sleep/rest").astype(int)
    Xz = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xz))
    cut = int(0.7 * len(order))
    tr, te = order[:cut], order[cut:]
    lda = LinearDiscriminantAnalysis().fit(Xz[tr], y_bin[tr])
    auc = float(roc_auc_score(y_bin[te], lda.decision_function(Xz[te])))
    proj = lda.transform(Xz).ravel()
    if proj[y_bin == 1].mean() < proj[y_bin == 0].mean():  # orient so sleep is high
        proj = -proj
    w = np.abs(lda.coef_.ravel())
    nb = len(bands)
    w_el = np.array([w[ci * nb:(ci + 1) * nb].sum() for ci in range(n_ch)])
    band_frac = np.array([w[bi::nb].sum() for bi in range(nb)])
    band_frac = band_frac / (band_frac.sum() + 1e-12)
    return auc, proj, y_bin, w_el, band_frac


def region_of(path, good_ch):
    xyz, _ = electrode_coords(path)
    try:
        names = mni_to_aal(xyz)
    except Exception:  # noqa: BLE001
        names = ["unknown"] * len(xyz)
    return xyz[good_ch], [names[i] for i in good_ch]


def plot_timeseries(proj, t0s, y_bin, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hrs = np.asarray(t0s) / 3600.0
    order = np.argsort(hrs)
    hrs, proj, y_bin = hrs[order], proj[order], y_bin[order]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.scatter(hrs[y_bin == 0], proj[y_bin == 0], s=10, c="#4c78c8", alpha=.5, label="active")
    ax.scatter(hrs[y_bin == 1], proj[y_bin == 1], s=10, c="#59a14f", alpha=.6, label="Sleep/rest")
    # rolling median to expose slow (circadian) structure
    if len(hrs) > 20:
        idx = np.argsort(hrs)
        win = max(5, len(hrs) // 40)
        rm = np.convolve(proj[idx], np.ones(win) / win, mode="same")
        ax.plot(hrs[idx], rm, color="#b0752a", lw=1.6, alpha=.9, label="rolling mean")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("time into recording (hours)")
    ax.set_ylabel("biomarker value\n(high = sleep-like)")
    ax.set_title("The arousal/state biomarker across the recording")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_brains(per_sub, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subs = list(per_sub)
    n = len(subs)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.2), squeeze=False)
    for ax, sid in zip(axes[0], subs):
        xyz, w = per_sub[sid]["xyz"], per_sub[sid]["w"]
        wn = w / (w.max() + 1e-9)
        # axial top-down view: x (L-R) vs y (post-ant); color+size by importance
        p = ax.scatter(xyz[:, 0], xyz[:, 1], c=wn, s=20 + 200 * wn, cmap="magma",
                       vmin=0, vmax=1, edgecolors="k", linewidths=.3, alpha=.85)
        ax.set_title("sub-{}  (AUC {:.2f})".format(sid, per_sub[sid]["auc"]), fontsize=11)
        ax.set_xlabel("x  (L → R, mm)"); ax.set_ylabel("y  (post → ant, mm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
    fig.suptitle("Where the sleep-vs-active biomarker lives (electrode weight, axial view)", fontsize=12)
    fig.colorbar(p, ax=axes[0].tolist(), shrink=.7, label="normalized weight")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--out-dir", default="phase3_manifold")
    ap.add_argument("--extra-root", default="ajile12-nwb-data")
    ap.add_argument("--epoch-window-sec", type=float, default=10.0)
    ap.add_argument("--epoch-max-per-label", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    primary = args.nwb or find_default_file()
    os.makedirs(args.out_dir, exist_ok=True)
    files = find_subject_files(args.extra_root, primary)
    print("subjects:", list(files))

    per_sub, rows = {}, []
    for sid, path in files.items():
        print("\n[biomarker] sub-{}".format(sid))
        try:
            with __import__("h5py").File(path, "r") as f:
                good = good_channel_indices(f)
            X, y, t0s = build_epoch_dataset(path, good, STATE_BANDS,
                                            window_sec=args.epoch_window_sec,
                                            max_per_label=args.epoch_max_per_label,
                                            label_set=set(T5_LABELS), seed=args.seed, verbose=False)
        except Exception as e:  # noqa: BLE001
            print("  skip ({})".format(type(e).__name__)); continue
        if (np.asarray(y) == "Sleep/rest").sum() < 10 or (np.asarray(y) != "Sleep/rest").sum() < 10:
            print("  not enough sleep/active windows; skip"); continue
        keep = drop_outliers(X)
        X, y, t0s = X[keep], np.asarray(y)[keep], np.asarray(t0s)[keep]
        auc, proj, y_bin, w_el, band_frac = fit_biomarker(X, y, len(good), STATE_BANDS, args.seed)
        xyz, regions = region_of(path, good)
        # dominant region by summed electrode weight
        reg_w = {}
        for r, wv in zip(regions, w_el):
            base = r.replace("_L", "").replace("_R", "")
            reg_w[base] = reg_w.get(base, 0.0) + wv
        top_region = max(reg_w, key=reg_w.get) if reg_w else "n/a"
        top_band = STATE_BANDS[int(np.argmax(band_frac))]
        per_sub[sid] = {"xyz": xyz, "w": w_el, "auc": auc}
        rows.append({"subject": sid, "auc": round(auc, 3), "n_windows": len(y),
                    "top_band": top_band, "top_region": top_region,
                    "sensorimotor_frac": round(float(np.mean([
                        any(s.lower() in r.lower() for s in SENSORIMOTOR_AAL) for r in regions])), 2)})
        print("  AUC={:.3f}  top band={}  top region={}".format(auc, top_band, top_region))
        if sid == list(files)[0] or (primary and os.path.basename(path) == os.path.basename(primary)):
            plot_timeseries(proj, t0s, y_bin, os.path.join(args.out_dir, "biomarker_timeseries.png"))

    if per_sub:
        plot_brains(per_sub, os.path.join(args.out_dir, "biomarker_brain.png"))
    with open(os.path.join(args.out_dir, "biomarker_crosssubject.csv"), "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nper-subject biomarker:")
    for r in rows:
        print("  sub-{}: AUC {:.3f}, {}-led, {}".format(r["subject"], r["auc"], r["top_band"], r["top_region"]))
    print("\nDone. Outputs in", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
