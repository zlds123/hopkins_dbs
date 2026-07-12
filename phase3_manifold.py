"""Phase 4 (pilot): the manifold + biomarker view of naturalistic human ECoG.

Reframes Phase 1-3 from "can we decode?" to "what is the geometry of the low-dimensional
manifold on which naturalistic behavior is written in cortex, and can a coordinate on it
serve as a state biomarker?" -- the Gallego/Miller/Solla neural-manifold lens, extended
toward the psychiatric closed-loop-DBS biomarker question (Scangos/Provenza).

Analyses
--------
1. STATE manifold (whole-file epoch windows): PCA embedding colored by behavioral state,
   with intrinsic dimensionality (participation ratio) and state separability (silhouette).
   This is the biomarker-relevant object -- states are the slow variable a psychiatric
   implant would track.
2. MOVEMENT manifold (reach-dense continuous stream): PCA colored by reach / wrist speed --
   shows whether discrete events occupy structured regions.
3. BIOMARKER axis: a linear discriminant separating sleep vs. active, reported as an
   interpretable "which band drives it" readout (the object Scangos et al. found by hand).
4. LINGUA-FRANCA probe: is the *relative geometry* of behavioral states shared across
   subjects? Procrustes-align each subject's state-centroid configuration to a reference
   and report disparity -- the "shared concept space beneath different languages" test,
   for brains. This is the experiment the raw cross-subject CKA (~0) could not address.

Run (dbs-ml env):
  python phase3_manifold.py --stage core --out-dir phase3_manifold
  python phase3_manifold.py --stage full --out-dir phase3_manifold   # + cross-subject
"""

import argparse
import csv
import os

import numpy as np

from nwb_dataset import good_channel_indices, BANDS
from phase1_resolution import select_channels, pick_window
from phase3_eval import get_primary_stream, build_epoch_dataset, find_default_file
from phase3_crosssubject import find_subject_files

STATE_BANDS = ("theta", "alpha", "beta", "low_gamma", "high_gamma")
T5_LABELS = ("Sleep/rest", "Talk", "TV", "Talk, TV", "Computer/phone", "Inactive")


# --------------------------------------------------------------------------- #
# Geometry metrics
# --------------------------------------------------------------------------- #
def participation_ratio(explained_variance):
    """Intrinsic dimensionality: (Σλ)^2 / Σλ^2. ~1 => one axis dominates; ~D => isotropic."""
    lam = np.asarray(explained_variance, dtype=float)
    return float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12))


def separability(Z, labels, max_n=3000, seed=0):
    from sklearn.metrics import silhouette_score
    idx = np.arange(len(Z))
    if len(idx) > max_n:
        idx = np.random.default_rng(seed).choice(idx, max_n, replace=False)
    try:
        return float(silhouette_score(Z[idx], np.asarray(labels)[idx]))
    except Exception:  # noqa: BLE001
        return float("nan")


# --------------------------------------------------------------------------- #
# 1 + 2: build manifolds
# --------------------------------------------------------------------------- #
def state_manifold(path, good_ch, bands, window_sec, max_per_label, seed):
    X, y, t0s = build_epoch_dataset(path, good_ch, bands, window_sec=window_sec,
                                    max_per_label=max_per_label, label_set=set(T5_LABELS),
                                    seed=seed, verbose=True)
    return X, y, t0s


def drop_outliers(X, z=8.0):
    """Keep-mask dropping rows whose robust distance from the median is extreme.

    One artifact epoch window can otherwise dominate the whole PCA/UMAP (it stretched
    PC1 to ~-350 in the first pass), so we remove windows whose L2 distance from the
    feature-median is > z robust-SDs before embedding."""
    med = np.median(X, axis=0)
    norm = np.linalg.norm(X - med, axis=1)
    nmed = np.median(norm)
    nmad = np.median(np.abs(norm - nmed)) + 1e-9
    return (np.abs(norm - nmed) / (1.4826 * nmad)) < z


def robust_scale(X, clip=8.0):
    from sklearn.preprocessing import RobustScaler
    Xz = RobustScaler().fit_transform(X)
    return np.clip(Xz, -clip, clip)


def fit_pca(Xz, k=10):
    """PCA on already-scaled features (kept only for the dimensionality metric)."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(k, Xz.shape[1]), random_state=0).fit(Xz)
    return p, p.transform(Xz)


def umap_2d(Xz, seed=0, n_neighbors=15, min_dist=0.1, max_n=15000):
    """2-D UMAP embedding (nonlinear); PCA fallback if umap-learn is unavailable.
    Returns (coords, sub_idx, method) -- sub_idx maps coords back to input rows."""
    idx = np.arange(len(Xz))
    if len(idx) > max_n:
        idx = np.random.default_rng(seed).choice(idx, max_n, replace=False)
    try:
        import umap
        coords = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                           random_state=seed).fit_transform(Xz[idx])
        return coords, idx, "UMAP"
    except Exception as e:  # noqa: BLE001
        from sklearn.decomposition import PCA
        print("  UMAP unavailable ({}); PCA fallback".format(type(e).__name__))
        return PCA(n_components=2, random_state=seed).fit_transform(Xz[idx]), idx, "PCA"


# --------------------------------------------------------------------------- #
# 3: biomarker axis (sleep vs active)
# --------------------------------------------------------------------------- #
def biomarker_axis(X, y, good_ch, bands, seed=0):
    """LDA sleep-vs-active axis; return AUC, the projection, and band-aggregated loadings."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    y_bin = (np.asarray(y) == "Sleep/rest").astype(int)  # 1 = sleep/rest, 0 = active
    sc = StandardScaler().fit(X)
    Xz = sc.transform(X)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xz))
    cut = int(0.7 * len(order))
    tr, te = order[:cut], order[cut:]

    lda = LinearDiscriminantAnalysis().fit(Xz[tr], y_bin[tr])
    proj = lda.transform(Xz).ravel()
    auc = float(roc_auc_score(y_bin[te], lda.decision_function(Xz[te])))

    # map |weight| back to (channel, band) then aggregate by band
    w = np.abs(lda.coef_.ravel())
    nb = len(bands)
    band_load = {b: 0.0 for b in bands}
    for ci in range(len(good_ch)):
        for bi, b in enumerate(bands):
            band_load[b] += w[ci * nb + bi]
    tot = sum(band_load.values()) + 1e-12
    band_frac = {b: band_load[b] / tot for b in bands}
    return auc, proj, y_bin, band_frac


# --------------------------------------------------------------------------- #
# 4: lingua-franca probe (shared state geometry across subjects)
# --------------------------------------------------------------------------- #
def state_centroids(X, y, states, k=6):
    """Per-state centroid in a k-dim PCA space -> (n_states, k) config for Procrustes."""
    _, Z = fit_pca(robust_scale(X), k=k)
    cents = []
    for s in states:
        m = np.asarray(y) == s
        if m.sum() == 0:
            return None
        cents.append(Z[m].mean(axis=0))
    return np.vstack(cents)


def lingua_franca(subject_epoch, seed=0, k=6):
    """Procrustes-align each subject's state-centroid config to a reference; low disparity
    => the relative arrangement of behavioral states is shared (a 'lingua franca')."""
    from scipy.spatial import procrustes

    subs = list(subject_epoch)
    # common states present (>= a few windows) in every subject
    common = None
    for s in subs:
        present = {lab for lab in set(subject_epoch[s][1])
                   if np.sum(np.asarray(subject_epoch[s][1]) == lab) >= 5}
        common = present if common is None else (common & present)
    common = [s for s in T5_LABELS if s in (common or set())]
    if len(common) < 3:
        print("  <3 states shared across all subjects; lingua-franca test not possible")
        return [], common

    configs = {}
    for s in subs:
        X, y, _ = subject_epoch[s]
        c = state_centroids(X, y, common, k=min(k, len(common) - 1, X.shape[1]))
        if c is not None:
            configs[s] = c
    ref = subs[0]
    rows = []
    for s in subs:
        if s == ref or s not in configs or ref not in configs:
            continue
        # procrustes needs same shape; both are (len(common), k)
        _, _, disparity = procrustes(configs[ref], configs[s])
        rows.append({"reference": ref, "subject": s, "n_states": len(common),
                    "procrustes_disparity": float(disparity)})
        print("  Procrustes disparity  {} vs {}: {:.3f}  (0=identical geometry, 1=unrelated)".format(
            ref, s, disparity))
    return rows, common


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_state_manifold(coords, y, out_path, biomarker_proj=None, method="UMAP"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y = np.asarray(y)
        labels = sorted(set(y))
        ncol = 2 if biomarker_proj is not None else 1
        fig, axes = plt.subplots(1, ncol, figsize=(6.6 * ncol, 5.4), squeeze=False)

        ax = axes[0][0]
        for i, l in enumerate(labels):
            m = y == l
            ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.7,
                       color=plt.cm.tab10(i / 10), label=l, edgecolors="none")
        ax.set_title("State manifold ({} of band-power)".format(method))
        ax.set_xlabel("{}-1".format(method)); ax.set_ylabel("{}-2".format(method))
        ax.legend(fontsize=8, loc="best", framealpha=.9)

        if biomarker_proj is not None:
            ax2 = axes[0][1]
            active = np.array([l != "Sleep/rest" for l in y])
            ax2.hist(biomarker_proj[active], bins=40, alpha=0.65, label="active",
                     color="#4c78c8")
            ax2.hist(biomarker_proj[~active], bins=40, alpha=0.65, label="Sleep/rest",
                     color="#59a14f")
            ax2.set_title("Biomarker axis (sleep-vs-active LDA)")
            ax2.set_xlabel("biomarker value"); ax2.set_ylabel("# windows")
            ax2.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print("wrote", out_path)
    except Exception as e:  # noqa: BLE001
        print("state manifold plot skipped:", e)


def plot_movement_manifold(coords, reach, speed, out_path, method="UMAP"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sp = np.clip(speed, 0, np.percentile(speed, 99) + 1e-9)
        # draw non-reach first, then reach on top so events are visible
        order = np.argsort(reach)
        specs = [("reach (red = reach)", reach, "coolwarm"), ("wrist speed", sp, "viridis")]
        for a, (title, c, cmap) in zip(ax, specs):
            p = a.scatter(coords[order, 0], coords[order, 1], c=np.asarray(c)[order],
                          cmap=cmap, s=5, alpha=0.5, edgecolors="none")
            a.set_title("Movement manifold — {}".format(title))
            a.set_xlabel("{}-1".format(method)); a.set_ylabel("{}-2".format(method))
            fig.colorbar(p, ax=a, shrink=0.7)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print("wrote", out_path)
    except Exception as e:  # noqa: BLE001
        print("movement manifold plot skipped:", e)


def save_csv(rows, path):
    if not rows:
        print("(nothing to write for {})".format(path))
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--out-dir", default="phase3_manifold")
    ap.add_argument("--stage", choices=["smoke", "core", "full"], default="core")
    ap.add_argument("--dur-min", type=float, default=45.0)
    ap.add_argument("--anchor", choices=["reach", "movement"], default="reach")
    ap.add_argument("--bands", default="beta,high_gamma")
    ap.add_argument("--out-rate", type=float, default=30.0)
    ap.add_argument("--smooth-hz", type=float, default=6.0)
    ap.add_argument("--epoch-window-sec", type=float, default=10.0)
    ap.add_argument("--epoch-max-per-label", type=int, default=150)
    ap.add_argument("--extra-root", default="ajile12-nwb-data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = args.nwb or find_default_file()
    if not path or not os.path.exists(path):
        raise SystemExit("No NWB file found. Pass --nwb explicitly.")
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    print("file:", path)

    mpl = 40 if args.stage == "smoke" else args.epoch_max_per_label
    geom_rows, bio_rows, summary = [], [], []

    # ---- 1 + 3: STATE manifold + biomarker (all good channels, broadband) ------ #
    print("\n=== STATE manifold (whole-file epochs) ===")
    import h5py
    with h5py.File(path, "r") as f:
        good_ch = good_channel_indices(f)
    state_bands = STATE_BANDS
    Xs, ys, t0s = state_manifold(path, good_ch, state_bands, args.epoch_window_sec, mpl, args.seed)
    keep = drop_outliers(Xs)
    n_drop = int((~keep).sum())
    Xs, ys, t0s = Xs[keep], np.asarray(ys)[keep], np.asarray(t0s)[keep]
    print("  dropped {} outlier window(s); {} remain".format(n_drop, len(ys)))
    Xsz = robust_scale(Xs)
    p_state, _ = fit_pca(Xsz, k=10)
    pr = participation_ratio(p_state.explained_variance_)
    coords_s, sub_s, method_s = umap_2d(Xsz, seed=args.seed)
    sil = separability(coords_s, ys[sub_s])
    geom_rows.append({"manifold": "state", "n_samples": len(ys), "n_features": Xs.shape[1],
                     "outliers_dropped": n_drop, "participation_ratio": round(pr, 2),
                     "embed": method_s, "silhouette_embed": round(sil, 3),
                     "n_categories": len(set(ys))})
    print("  participation ratio={:.2f}  {} silhouette={:.3f}".format(pr, method_s, sil))

    auc_bio, proj, y_bin, band_frac = biomarker_axis(Xs, ys, good_ch, state_bands, args.seed)
    print("  biomarker (sleep-vs-active) AUC={:.3f}; band contributions:".format(auc_bio))
    for b, fr in sorted(band_frac.items(), key=lambda kv: -kv[1]):
        print("    {:11s} {:.1%}".format(b, fr))
        bio_rows.append({"axis": "sleep_vs_active", "band": b, "loading_fraction": round(fr, 4)})
    bio_rows.append({"axis": "sleep_vs_active", "band": "__AUC__", "loading_fraction": round(auc_bio, 4)})
    plot_state_manifold(coords_s, ys[sub_s], os.path.join(args.out_dir, "state_manifold.png"),
                        biomarker_proj=proj[sub_s], method=method_s)

    # ---- 2: MOVEMENT manifold (reach-dense continuous stream) ------------------- #
    print("\n=== MOVEMENT manifold (reach-dense stream) ===")
    channels, ch_method = select_channels(path, mode="sensorimotor")
    bands = tuple(b.strip() for b in args.bands.split(",") if b.strip())
    dur_min = 6.0 if args.stage == "smoke" else args.dur_min
    t0, t1, _ = pick_window(path, dur_min * 60.0, args.anchor, -1)
    stream = get_primary_stream(path, t0, t1, args.out_rate, bands, channels, args.smooth_hz, cache_dir)
    Xm = np.asarray(stream["X"], dtype=np.float64)
    reach = np.asarray(stream["reach"]).astype(int)
    speed = np.asarray(stream["speed"])[:, 0] if np.asarray(stream["speed"]).shape[1] else np.zeros(len(Xm))
    keep_m = drop_outliers(Xm)
    Xm, reach, speed = Xm[keep_m], reach[keep_m], speed[keep_m]
    Xmz = robust_scale(Xm)
    p_move, _ = fit_pca(Xmz, k=10)
    pr_m = participation_ratio(p_move.explained_variance_)
    coords_m, sub_m, method_m = umap_2d(Xmz, seed=args.seed)
    sil_m = separability(coords_m, reach[sub_m])
    geom_rows.append({"manifold": "movement", "n_samples": len(Xm), "n_features": Xm.shape[1],
                     "outliers_dropped": int((~keep_m).sum()), "participation_ratio": round(pr_m, 2),
                     "embed": method_m, "silhouette_embed": round(sil_m, 3), "n_categories": 2})
    print("  participation ratio={:.2f}  reach {} silhouette={:.3f}".format(pr_m, method_m, sil_m))
    plot_movement_manifold(coords_m, reach[sub_m], speed[sub_m],
                           os.path.join(args.out_dir, "movement_manifold.png"), method=method_m)

    save_csv(geom_rows, os.path.join(args.out_dir, "manifold_geometry.csv"))
    save_csv(bio_rows, os.path.join(args.out_dir, "biomarker_axis.csv"))

    # ---- 4: LINGUA-FRANCA probe (shared state geometry across subjects) --------- #
    lf_rows = []
    if args.stage == "full":
        print("\n=== LINGUA-FRANCA probe (cross-subject state geometry) ===")
        files = find_subject_files(args.extra_root, path)
        subject_epoch = {}
        for sid, p in files.items():
            print("[state epochs] sub-{}".format(sid))
            try:
                with h5py.File(p, "r") as f:
                    gch = good_channel_indices(f)
                Xk, yk, tk = state_manifold(p, gch, state_bands, args.epoch_window_sec, mpl, args.seed)
                if len(set(yk)) >= 3:
                    subject_epoch[sid] = (Xk, yk, tk)
            except Exception as e:  # noqa: BLE001
                print("  skip sub-{}: {}".format(sid, type(e).__name__))
        if len(subject_epoch) >= 2:
            lf_rows, common = lingua_franca(subject_epoch, args.seed)
            save_csv(lf_rows, os.path.join(args.out_dir, "linguafranca_procrustes.csv"))
        else:
            print("  <2 subjects with usable state epochs; skipping")

    # ---- hypotheses / findings / conclusions ---------------------------------- #
    print("\n================ MANIFOLD / BIOMARKER SUMMARY ================")
    print("H1 (structured state manifold): behavioral states occupy separable regions.")
    print("   -> silhouette(state, top3 PCs) = {:.3f}; PR = {:.2f} of {} dims  [{}]".format(
        sil, pr, Xs.shape[1], "SUPPORTED" if sil > 0.1 else "WEAK"))
    print("H2 (low-D biomarker of state): a linear axis separates sleep vs active.")
    print("   -> LDA AUC = {:.3f}, dominated by {} band  [{}]".format(
        auc_bio, max(band_frac, key=band_frac.get), "SUPPORTED" if auc_bio > 0.8 else "WEAK"))
    print("H3 (event structure in movement manifold): reach forms a region.")
    print("   -> reach silhouette = {:.3f}  [{}]".format(
        sil_m, "SUPPORTED" if sil_m > 0.03 else "WEAK/DIFFUSE"))
    if lf_rows:
        md = float(np.mean([r["procrustes_disparity"] for r in lf_rows]))
        print("H4 (neural lingua franca): state geometry is shared across subjects.")
        print("   -> mean Procrustes disparity = {:.3f}  [{}]".format(
            md, "SHARED" if md < 0.4 else "SUBJECT-SPECIFIC" if md > 0.7 else "PARTIAL"))
    print("=============================================================")
    print("\nDone. Outputs in", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
