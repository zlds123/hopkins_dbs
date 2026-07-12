"""Apply CEBRA to AJILE12 ECoG with **continuous wrist velocity** as the auxiliary variable.

Pipeline
--------
1. Find an active ~1 h window (most reach onsets) and build a continuous, time-aligned
   stream (high-gamma envelope @ 30 Hz + wrist velocity + reach/behavior labels).
   Cached to ``.npz`` so re-runs are instant.
2. Train **CEBRA-Time** (discovery, no labels) and **CEBRA-Behavior**
   (``conditional='time_delta'``, conditioned on wrist velocity).
3. Plot 3-D embeddings colored by reach / behavior / wrist speed.
4. Decode reach from the embedding (kNN) on a held-out *time* split and compare to a
   logistic-regression baseline trained directly on the high-gamma features.

Run inside the ``dbs-ml`` env:
    python cebra_ajile.py --minutes 60 --max-iter 3000 --dim 3 --out cebra_out
"""

import argparse
import os

import numpy as np

NWB = r"C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def get_stream(nwb, minutes, cache, window="movement", channels="sensorimotor",
               out_rate=30.0, bands=("high_gamma",), verbose=True):
    if cache and os.path.exists(cache):
        print("Loading cached stream:", cache)
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    from nwb_dataset import (build_continuous_stream, find_active_window,
                             find_movement_window, sensorimotor_channels)

    # Channel selection
    if channels == "sensorimotor":
        ch = sensorimotor_channels(nwb)
        if len(ch) == 0:
            print("  no sensorimotor channels found; falling back to good.")
            ch = "good"
    else:
        ch = "good"

    # Window selection
    if window == "movement":
        t0, t1, score = find_movement_window(nwb, dur_sec=minutes * 60.0)
        print("Movement window: {:.0f}-{:.0f}s (fast-frame frac={:.3f})".format(
            t0, t1, score))
    else:
        t0, t1, n = find_active_window(nwb, dur_sec=minutes * 60.0)
        print("Reach window: {:.0f}-{:.0f}s ({} reaches)".format(t0, t1, n))

    s = build_continuous_stream(nwb, t0, t1, out_rate=out_rate, bands=bands,
                                ecog_channels=ch, verbose=verbose)
    if cache:
        save = {k: s[k] for k in ("t", "X", "vel", "speed", "pos", "reach",
                                  "behavior", "channels", "feature_names")}
        np.savez_compressed(cache, **save)
        print("Cached ->", cache)
    return s


# --------------------------------------------------------------------------- #
# CEBRA
# --------------------------------------------------------------------------- #
def fit_cebra(X, aux, conditional, dim, arch, time_offsets, max_iter, train_idx):
    from cebra import CEBRA

    model = CEBRA(model_architecture=arch, conditional=conditional,
                  time_offsets=time_offsets, output_dimension=dim,
                  max_iterations=max_iter, batch_size=512, distance="cosine",
                  learning_rate=3e-4, temperature=1.0, device="cuda_if_available",
                  verbose=True)
    if aux is None:
        model.fit(X[train_idx])
    else:
        model.fit(X[train_idx], aux[train_idx])
    return model, model.transform(X)


def decode_reach(emb, reach, k):
    """kNN decode reach on a time split; return test ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier

    tr = slice(0, k)
    te = slice(k, len(emb))
    if len(np.unique(reach[tr])) < 2 or len(np.unique(reach[te])) < 2:
        return float("nan")
    clf = KNeighborsClassifier(n_neighbors=25)
    clf.fit(emb[tr], reach[tr])
    proba = clf.predict_proba(emb[te])[:, 1]
    return roc_auc_score(reach[te], proba)


def baseline_reach(X, reach, k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    tr = slice(0, k)
    te = slice(k, len(X))
    if len(np.unique(reach[tr])) < 2 or len(np.unique(reach[te])) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[tr], reach[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(reach[te], proba)


def move_auc(Z, speed, k, raw=False):
    """AUC for moving-vs-rest (top-25% wrist speed = moving). Balanced-ish target."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier

    tr, te = slice(0, k), slice(k, len(Z))
    thr = np.percentile(speed[tr], 75)
    y = (speed > thr).astype(int)
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return float("nan")
    clf = (LogisticRegression(max_iter=1000, class_weight="balanced") if raw
           else KNeighborsClassifier(n_neighbors=25))
    clf.fit(Z[tr], y[tr])
    return roc_auc_score(y[te], clf.predict_proba(Z[te])[:, 1])


def speed_r2(Z, speed, k, raw=False):
    """Held-out R^2 for regressing continuous wrist speed from the representation."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.neighbors import KNeighborsRegressor

    tr, te = slice(0, k), slice(k, len(Z))
    model = Ridge(alpha=1.0) if raw else KNeighborsRegressor(n_neighbors=25)
    model.fit(Z[tr], speed[tr])
    return r2_score(speed[te], model.predict(Z[te]))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_embeddings(emb, s, title, path, max_pts=12000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(emb)
    idx = np.random.default_rng(0).choice(n, size=min(max_pts, n), replace=False)
    e = emb[idx]
    reach = s["reach"][idx]
    speed = s["speed"][idx, 0] if s["speed"].shape[1] else np.zeros(len(idx))
    beh = np.array([str(b) for b in s["behavior"]])[idx]
    labels = sorted(set(beh))
    code = np.array([labels.index(b) for b in beh])

    fig = plt.figure(figsize=(15, 4.6))
    specs = [("reach (0/1)", reach, "coolwarm", None),
             ("wrist speed", np.clip(speed, 0, np.percentile(speed, 99) + 1e-9), "viridis", None),
             ("behavior", code, "tab10", labels)]
    for i, (sub, c, cmap, leg) in enumerate(specs, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        p = ax.scatter(e[:, 0], e[:, 1], e[:, 2], c=c, cmap=cmap, s=2, alpha=0.5)
        ax.set_title(sub, fontsize=10)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        if leg is None:
            fig.colorbar(p, ax=ax, shrink=0.5, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("  wrote", path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwb", default=NWB)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--out", default="cebra_out")
    ap.add_argument("--out-rate", type=float, default=30.0)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--arch", default="offset10-model")
    ap.add_argument("--time-offsets", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--window", choices=["movement", "reach"], default="movement")
    ap.add_argument("--channels", choices=["sensorimotor", "good"],
                    default="sensorimotor")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "stream_{}_{}_{:.0f}min_{:.0f}hz.npz".format(
        args.window, args.channels, args.minutes, args.out_rate))
    s = get_stream(args.nwb, args.minutes, cache, window=args.window,
                   channels=args.channels, out_rate=args.out_rate)

    X = np.asarray(s["X"], dtype=np.float32)
    vel = np.asarray(s["vel"], dtype=np.float32)
    reach = np.asarray(s["reach"]).astype(int)
    n = len(X)
    k = int(n * 0.7)
    print("Stream: T={}  X={}  vel={}  reach_frac={:.3f}".format(
        n, X.shape, vel.shape, reach.mean()))

    train_idx = np.arange(k)

    print("\n[CEBRA-Time] fitting ...")
    _, emb_time = fit_cebra(X, None, "time", args.dim, args.arch,
                            args.time_offsets, args.max_iter, train_idx)
    print("[CEBRA-Behavior] fitting (aux = wrist velocity) ...")
    _, emb_beh = fit_cebra(X, vel, "time_delta", args.dim, args.arch,
                           args.time_offsets, args.max_iter, train_idx)

    np.savez_compressed(os.path.join(args.out, "embeddings.npz"),
                        emb_time=emb_time, emb_beh=emb_beh,
                        reach=reach, speed=s["speed"], behavior=s["behavior"])

    plot_embeddings(emb_time, s, "CEBRA-Time (discovery)",
                    os.path.join(args.out, "emb_time.png"))
    plot_embeddings(emb_beh, s, "CEBRA-Behavior (wrist velocity)",
                    os.path.join(args.out, "emb_behavior.png"))

    spd = s["speed"][:, 0].astype(float)
    reps = [("raw high-gamma", X, True),
            ("CEBRA-Time", emb_time, False),
            ("CEBRA-Behavior", emb_beh, False)]

    print("\n=== Held-out decoding (last 30% of time) ===")
    print("reach_frac={:.3f}  X={}  T={}".format(reach.mean(), X.shape, n))
    header = "{:16s} | reach AUC | move AUC | speed R^2".format("representation")
    print(header); print("-" * len(header))
    lines = []
    for name, Z, raw in reps:
        ra = baseline_reach(Z, reach, k) if raw else decode_reach(Z, reach, k)
        ma = move_auc(Z, spd, k, raw=raw)
        r2 = speed_r2(Z, spd, k, raw=raw)
        print("{:16s} |   {:.3f}   |  {:.3f}   |  {:+.3f}".format(name, ra, ma, r2))
        lines.append("{}: reach_AUC={:.4f} move_AUC={:.4f} speed_R2={:.4f}".format(
            name, ra, ma, r2))
    with open(os.path.join(args.out, "results.txt"), "w") as fh:
        fh.write("window={} channels={} T={} reach_frac={:.4f}\n".format(
            args.window, args.channels, n, reach.mean()))
        fh.write("\n".join(lines) + "\n")
    print("\nDone. Outputs in", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
