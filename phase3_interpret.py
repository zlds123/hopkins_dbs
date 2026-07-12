"""Phase 3 interpretability: what does the latent space actually encode?

Directly addresses the mentor's note #5 -- "find the latent embedding space, understand
what's embedded, project neural data into behavioral categories (or continuous), what
aspects of behavior is encoded." This is representation *characterization*, not a decode
horse-race: accuracy is used here only as a probe of information content, per note #3
("accuracy as a standalone metric may be misguided").

Four analyses, each on the M0-M3 representations from ``phase3_eval.build_all_embeddings``:

  A. Per-variable linear/kNN probe -- how much of each behavioral variable (reach, movement,
     speed, velocity, coarse state) is recoverable from each representation. "What aspects
     of behavior are encoded", made quantitative, with a shuffled-label null for each.
  B. Dimension-wise probe -- for a learned embedding, how much each single latent dimension
     alone carries about each variable. Localizes "which dimension tracks what".
  C. Project into behavioral categories -- mean embedding per coarse-behavior/reach category
     and their separability (silhouette + nearest-centroid accuracy). "Project neural data
     into behavioral categories."
  D. RSA -- does the geometry of the neural embedding mirror the geometry of behavior
     (representational similarity between neural-latent and pose-feature spaces)?

Plus 2-D embedding scatter plots colored by each behavioral variable (visual inspection).

Reuses the exact stream/embedding machinery of ``phase3_eval.py`` (same 70/30 split, same
train-only fitting convention) so results are comparable to the decode tables.

Run (dbs-ml env):
  python phase3_interpret.py --stage core  --out-dir phase3_interpret
  python phase3_interpret.py --stage full  --out-dir phase3_interpret
"""

import argparse
import csv
import os

import numpy as np

from phase1_resolution import select_channels, pick_window, speed_from_threshold
from phase3_eval import (find_default_file, get_primary_stream, build_all_embeddings,
                        build_targets_from_stream, representation_for_target)


# --------------------------------------------------------------------------- #
# A. Per-variable probe (with null)
# --------------------------------------------------------------------------- #
def probe_variable(Z, y, keep, task, train_idx, test_idx, raw, seed=0):
    """Recoverability of one behavioral variable from representation Z, plus a
    shuffled-label null. Returns (score, null_score)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.metrics import roc_auc_score, r2_score

    ttr = train_idx[keep[train_idx]]
    tte = test_idx[keep[test_idx]]
    rng = np.random.default_rng(seed)

    def fit_score(ytr, yte):
        if task == "binary":
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                return float("nan")
            clf = (LogisticRegression(max_iter=1000, class_weight="balanced") if raw
                  else KNeighborsClassifier(n_neighbors=25))
            clf.fit(Z[ttr], ytr)
            return float(roc_auc_score(yte, clf.predict_proba(Z[tte])[:, 1]))
        model = Ridge(alpha=1.0) if raw else KNeighborsRegressor(n_neighbors=25)
        model.fit(Z[ttr], ytr)
        return float(r2_score(yte, model.predict(Z[tte]), multioutput="uniform_average"))

    score = fit_score(y[ttr], y[tte])
    null = fit_score(rng.permutation(y[ttr]), y[tte])
    return score, null


def run_variable_probe(embeddings, targets, train_idx, test_idx, models, seed=0):
    rows = []
    for tname, (y, keep, task) in targets.items():
        metric = "auc" if task == "binary" else "r2"
        print("\n[probe] {} ({})".format(tname, task))
        for m in models:
            if m not in embeddings:
                continue
            Z, raw = representation_for_target(embeddings, m)
            score, null = probe_variable(Z, y, keep, task, train_idx, test_idx, raw, seed)
            rows.append({"variable": tname, "model": m, "metric": metric,
                        "score": score, "null": null, "above_null": score - null})
            print("    {:3s} {}={:.3f}  (null={:.3f}, Δ={:+.3f})".format(
                m, metric, score, null, score - null))
    return rows


# --------------------------------------------------------------------------- #
# B. Dimension-wise probe (which latent dim carries which variable)
# --------------------------------------------------------------------------- #
def run_dimension_probe(embeddings, targets, train_idx, test_idx, models, seed=0):
    """Univariate probe: score each single latent dimension alone against each variable."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score, r2_score

    rows = []
    for m in models:
        if m not in embeddings or m == "M0":  # only learned latents have meaningful dims
            continue
        Z, _ = representation_for_target(embeddings, m)
        d = Z.shape[1]
        for tname, (y, keep, task) in targets.items():
            ttr = train_idx[keep[train_idx]]
            tte = test_idx[keep[test_idx]]
            best_dim, best_score = -1, -np.inf
            for j in range(d):
                zj = Z[:, j:j + 1]
                if task == "binary":
                    if len(np.unique(y[ttr])) < 2 or len(np.unique(y[tte])) < 2:
                        continue
                    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(zj[ttr], y[ttr])
                    s = roc_auc_score(y[tte], clf.predict_proba(zj[tte])[:, 1])
                    s = max(s, 1 - s)  # a dim can be anti-correlated; report |separation|
                else:
                    rg = Ridge(alpha=1.0).fit(zj[ttr], y[ttr])
                    s = r2_score(y[tte], rg.predict(zj[tte]), multioutput="uniform_average")
                rows.append({"model": m, "variable": tname, "dim": j, "score": float(s)})
                if s > best_score:
                    best_dim, best_score = j, s
            print("  [{}] {}: best single dim = {} ({:.3f})".format(m, tname, best_dim, best_score))
    return rows


# --------------------------------------------------------------------------- #
# C. Project into behavioral categories (separability)
# --------------------------------------------------------------------------- #
def run_category_projection(embeddings, stream, train_idx, test_idx, models):
    """Separability of coarse-behavior categories in each representation."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestCentroid
    from sklearn.metrics import accuracy_score

    beh = np.array([str(b) for b in stream["behavior"]])
    keep = np.array([b not in ("", "nan") and "Blocklist" not in b and "break" not in b.lower()
                    for b in beh])
    if keep.sum() < 50 or len(set(beh[keep])) < 2:
        print("  (not enough labeled coarse-behavior samples in this window; skipping)")
        return []

    rows = []
    ttr = train_idx[keep[train_idx]]
    tte = test_idx[keep[test_idx]]
    for m in models:
        if m not in embeddings:
            continue
        Z, _ = representation_for_target(embeddings, m)
        # subsample for silhouette (O(n^2))
        idx = np.where(keep)[0]
        sub = idx if len(idx) <= 3000 else np.random.default_rng(0).choice(idx, 3000, replace=False)
        try:
            sil = float(silhouette_score(Z[sub], beh[sub]))
        except Exception:  # noqa: BLE001
            sil = float("nan")
        nc = NearestCentroid().fit(Z[ttr], beh[ttr])
        acc = float(accuracy_score(beh[tte], nc.predict(Z[tte])))
        chance = float(max(np.mean(beh[tte] == c) for c in set(beh[tte])))
        rows.append({"model": m, "silhouette": sil, "nc_accuracy": acc, "chance": chance,
                    "n_categories": len(set(beh[keep]))})
        print("  [{}] silhouette={:.3f}  nearest-centroid acc={:.3f} (chance={:.3f})".format(
            m, sil, acc, chance))
    return rows


# --------------------------------------------------------------------------- #
# D. RSA (neural-latent geometry vs. behavior geometry)
# --------------------------------------------------------------------------- #
def _rdm(X, n=800, seed=0):
    """Condensed representational dissimilarity (1 - correlation) over a subsample."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xs = X[idx]
    Xs = Xs - Xs.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12
    corr = (Xs @ Xs.T) / (norm @ norm.T)
    iu = np.triu_indices(len(idx), k=1)
    return (1.0 - corr)[iu], idx


def run_rsa(embeddings, behavior_features, models, seed=0):
    """Spearman correlation between each representation's RDM and the behavior-space RDM."""
    from scipy.stats import spearmanr

    beh_rdm, idx = _rdm(behavior_features, seed=seed)
    rows = []
    for m in models:
        if m not in embeddings:
            continue
        Z, _ = representation_for_target(embeddings, m)
        Zs = Z[idx]
        Zs = Zs - Zs.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-12
        corr = (Zs @ Zs.T) / (norm @ norm.T)
        iu = np.triu_indices(len(idx), k=1)
        neural_rdm = (1.0 - corr)[iu]
        rho = float(spearmanr(neural_rdm, beh_rdm).correlation)
        rows.append({"model": m, "rsa_spearman": rho})
        print("  [{}] RSA (neural-latent vs behavior geometry) rho = {:.3f}".format(m, rho))
    return rows


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_embedding_colored(embeddings, stream, targets, out_path, models):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        color_specs = []
        if "T1" in targets:
            color_specs.append(("reach", targets["T1"][0].astype(float), "coolwarm"))
        if "T3" in targets:
            s = targets["T3"][0]
            color_specs.append(("speed", np.clip(s, 0, np.percentile(s, 99)), "viridis"))
        learned = [m for m in models if m in embeddings and m != "M0"]
        if not learned or not color_specs:
            return
        rng = np.random.default_rng(0)
        n = len(stream["X"])
        sub = rng.choice(n, size=min(6000, n), replace=False)

        fig, axes = plt.subplots(len(learned), len(color_specs),
                                figsize=(4.2 * len(color_specs), 3.8 * len(learned)), squeeze=False)
        for r, m in enumerate(learned):
            Z, _ = representation_for_target(embeddings, m)
            Z2 = PCA(n_components=2, random_state=0).fit_transform(Z[sub]) if Z.shape[1] > 2 else Z[sub, :2]
            for c, (cname, cvals, cmap) in enumerate(color_specs):
                ax = axes[r][c]
                p = ax.scatter(Z2[:, 0], Z2[:, 1], c=cvals[sub], cmap=cmap, s=3, alpha=0.5)
                ax.set_title("{} — {}".format(m, cname), fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(p, ax=ax, shrink=0.7)
        fig.suptitle("Latent space (2-D PCA) colored by behavior")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print("wrote", out_path)
    except Exception as e:  # noqa: BLE001
        print("embedding plot skipped:", e)


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
    ap.add_argument("--out-dir", default="phase3_interpret")
    ap.add_argument("--stage", choices=["smoke", "core", "full"], default="core")
    ap.add_argument("--start", type=float, default=-1)
    ap.add_argument("--dur-min", type=float, default=45.0)
    ap.add_argument("--anchor", choices=["reach", "movement"], default="reach")
    ap.add_argument("--channel-method", choices=["sensorimotor", "aal", "box", "good"],
                    default="sensorimotor")
    ap.add_argument("--bands", default="beta,high_gamma")
    ap.add_argument("--out-rate", type=float, default=30.0)
    ap.add_argument("--smooth-hz", type=float, default=6.0)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--models", default="M0,M1,M2,M3")
    ap.add_argument("--targets", default="T1,T2,T3,T4")
    ap.add_argument("--cebra-iter", type=int, default=1500)
    ap.add_argument("--cebra-time-offsets", type=int, default=10)
    ap.add_argument("--tt-iter", type=int, default=1500)
    ap.add_argument("--tt-time-offset", type=int, default=0)
    ap.add_argument("--tt-temperature", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = args.nwb or find_default_file()
    if not path or not os.path.exists(path):
        raise SystemExit("No NWB file found. Pass --nwb explicitly.")
    print("file:", path)
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    dur_min, cebra_iter, tt_iter = args.dur_min, args.cebra_iter, args.tt_iter
    if args.stage == "smoke":
        dur_min, cebra_iter, tt_iter = 6.0, 200, 200

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    targets_req = [t.strip() for t in args.targets.split(",") if t.strip()]

    channels, ch_method = select_channels(path, mode=args.channel_method)
    bands = tuple(b.strip() for b in args.bands.split(",") if b.strip())
    t0, t1, win_note = pick_window(path, dur_min * 60.0, args.anchor, args.start)
    print("window:", win_note, "channels:", ch_method, "({})".format(len(channels)))

    stream = get_primary_stream(path, t0, t1, args.out_rate, bands, channels,
                                args.smooth_hz, cache_dir)
    stream["X"] = np.asarray(stream["X"], dtype=np.float32)
    T = len(stream["X"])
    k = int(T * 0.7)
    train_idx, test_idx = np.arange(k), np.arange(k, T)
    targets = {kk: v for kk, v in build_targets_from_stream(stream).items() if kk in targets_req}

    print("\n=== building M0-M3 embeddings (dim={}) ===".format(args.dim))
    embeddings = build_all_embeddings(stream, train_idx, args.dim, models, cebra_iter, tt_iter,
                                      args.tt_time_offset, args.tt_temperature,
                                      args.cebra_time_offsets, cache_dir, args.seed,
                                      tag="interpret", verbose=True)

    print("\n=== A. per-variable probe (what is encoded, vs null) ===")
    probe_rows = run_variable_probe(embeddings, targets, train_idx, test_idx, models, args.seed)
    save_csv(probe_rows, os.path.join(args.out_dir, "interpret_variable_probe.csv"))

    print("\n=== C. project into behavioral categories ===")
    cat_rows = run_category_projection(embeddings, stream, train_idx, test_idx, models)
    save_csv(cat_rows, os.path.join(args.out_dir, "interpret_category_projection.csv"))

    plot_embedding_colored(embeddings, stream, targets,
                           os.path.join(args.out_dir, "interpret_embedding_pca.png"), models)

    if args.stage == "full":
        print("\n=== B. dimension-wise probe (which dim tracks what) ===")
        dim_rows = run_dimension_probe(embeddings, targets, train_idx, test_idx, models, args.seed)
        save_csv(dim_rows, os.path.join(args.out_dir, "interpret_dimension_probe.csv"))

        print("\n=== D. RSA (neural geometry vs behavior geometry) ===")
        import two_tower as tt
        beh_feats = tt.build_behavior_matrix(stream)
        rsa_rows = run_rsa(embeddings, beh_feats, models, args.seed)
        save_csv(rsa_rows, os.path.join(args.out_dir, "interpret_rsa.csv"))

    print("\nDone. Outputs in", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
