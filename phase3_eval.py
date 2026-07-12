"""Phase 3: unified eval harness for M0-M3 on AJILE12 (multimodal alignment / transfer).

Tests whether two-tower InfoNCE alignment (M3, ``two_tower.py``) beats raw band-power
(M0) and single-tower CEBRA (M1 = CEBRA-Time, M2 = CEBRA-Behavior) -- and under what
conditions (label scarcity, coarse vs. fine targets, within-subject cross-span
consistency). See ``two_tower.py`` for the M3 model itself.

Targets
-------
T1 reach (binary), T2 movement (binary, speed-thresholded), T3 speed (regression),
T4 velocity (regression) -- all four read directly off one continuous stream
(``build_continuous_stream``) over a single reach-dense or movement-rich window.
T5 coarse behavior epoch (multiclass: Sleep/rest, Talk, TV, ...), T6 sleep-vs-active
(binary) -- built separately from ``windows_from_epochs`` scanned over the *whole*
24h file, since AJILE12 epoch labels occupy disjoint multi-hour stretches of the day
and cannot appear inside a 45-90 min continuous window.

Fitting convention: M1-M3 are fit on the first 70% of time (``train_idx``) only, and
every headline metric is reported on the held-out last 30% -- see ``two_tower.py``'s
docstring for why (their inputs double as T2-T4 decode targets).

Eval axes (gated by ``--stage``)
---------------------------------
  core     - E2 target sweep: blocked/held-out decode of M0-M3 on requested targets.
  extended - + E1 dim sweep {8,16,32}, E3 label efficiency (T1/T2), E4 bidirectional
             decode (H3.1: does z_b decode independently, or is alignment one-way?).
  full     - + E-CKA cross-span consistency (within-subject substitute for the N=1-
             blocked cross-subject H3.3), T5/T6 coarse-behavior targets.
  smoke    - tiny duration/iterations, core stage only, pipeline sanity check.

Run (in the dbs-ml env):
  python phase3_eval.py --stage smoke --out-dir phase3_smoke
  python phase3_eval.py --stage core --out-dir phase3_out
  python phase3_eval.py --stage full --out-dir phase3_out
"""

import argparse
import csv
import glob
import hashlib
import os

import numpy as np

from phase1_resolution import (select_channels, pick_window, speed_from_threshold,
                               blocked_auc, blocked_r2)

FRACTIONS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
SEEDS = (0, 1, 2, 3, 4)
T5_LABELS = ("Sleep/rest", "Talk", "TV", "Talk, TV", "Computer/phone", "Inactive")


# --------------------------------------------------------------------------- #
# Data / caching
# --------------------------------------------------------------------------- #
def find_default_file():
    cands = glob.glob("*.nwb") + glob.glob(os.path.join("ajile12-nwb-data", "**", "*.nwb"),
                                           recursive=True)
    cands = [p for p in cands if os.path.getsize(p) > 1e9]
    return max(cands, key=os.path.getsize) if cands else None


def get_primary_stream(path, t0, t1, out_rate, bands, channels, smooth_hz, cache_dir):
    from nwb_dataset import build_continuous_stream

    key = "{}|{:.0f}|{:.0f}|{:.0f}|{}|ch{}|{:.0f}".format(
        os.path.basename(path), t0, t1, out_rate, ",".join(bands), len(channels), smooth_hz)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    cache = os.path.join(cache_dir, "primary_stream_{}.npz".format(h))
    if os.path.exists(cache):
        print("loading cached primary stream:", cache)
        d = np.load(cache, allow_pickle=True)
        return {"X": d["X"], "vel": d["vel"], "speed": d["speed"], "pos": d["pos"],
                "reach": d["reach"], "behavior": d["behavior"], "t": d["t"],
                "scaler": (d["scaler_mu"], d["scaler_sd"])}
    print("building primary stream (chunked read of the 15 GB file)...")
    s = build_continuous_stream(path, t0, t1, out_rate=out_rate, bands=bands,
                                ecog_channels=channels, zscore=True,
                                smooth_hz=smooth_hz, verbose=True)
    mu, sd = s["scaler"]
    np.savez_compressed(cache, X=s["X"], vel=s["vel"], speed=s["speed"], pos=s["pos"],
                        reach=s["reach"], behavior=s["behavior"], t=s["t"],
                        scaler_mu=mu, scaler_sd=sd)
    return {"X": s["X"], "vel": s["vel"], "speed": s["speed"], "pos": s["pos"],
            "reach": s["reach"], "behavior": s["behavior"], "t": s["t"], "scaler": (mu, sd)}


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def build_targets_from_stream(stream):
    """T1 reach, T2 movement, T3 speed, T4 velocity -> {name: (y, valid_mask, task)}."""
    reach = np.asarray(stream["reach"]).astype(int)
    T = len(reach)
    targets = {"T1": (reach, np.ones(T, dtype=bool), "binary")}

    speed = np.asarray(stream["speed"])
    if speed.ndim == 2 and speed.shape[1]:
        speed0 = speed[:, 0].astype(np.float64)
        y2, keep2 = speed_from_threshold(speed0)
        targets["T2"] = (np.clip(y2, 0, 1).astype(int), keep2, "binary")
        cap = np.nanpercentile(speed0, 99)
        y3 = np.log1p(np.clip(speed0, 0, cap))
        targets["T3"] = (y3, np.isfinite(y3), "regression")

    vel = np.asarray(stream["vel"])
    if vel.ndim == 2 and vel.shape[1] >= 2:
        y4 = vel[:, :2].astype(np.float64)
        targets["T4"] = (y4, np.all(np.isfinite(y4), axis=1), "regression")

    return targets


# --------------------------------------------------------------------------- #
# Model embeddings (M0-M3), uniform API
# --------------------------------------------------------------------------- #
def build_all_embeddings(stream, train_idx, dim, models, cebra_iter, tt_iter,
                         tt_time_offset, tt_temperature, cebra_time_offsets,
                         cache_dir, seed, tag="", verbose=True):
    X = np.asarray(stream["X"], dtype=np.float32)
    vel = np.asarray(stream["vel"], dtype=np.float32)
    out = {"M0": {"X": X}}

    if "M1" in models:
        from cebra_ajile import fit_cebra
        if verbose:
            print("[M1 CEBRA-Time] fitting (dim={})...".format(dim))
        model, emb = fit_cebra(X, None, "time", dim, "offset10-model",
                               cebra_time_offsets, cebra_iter, train_idx)
        out["M1"] = {"Z": emb.astype(np.float64), "model": model}

    if "M2" in models:
        if vel.shape[1] < 2:
            print("  skipping M2 (no wrist velocity in stream)")
        else:
            from cebra_ajile import fit_cebra
            if verbose:
                print("[M2 CEBRA-Behavior] fitting (dim={})...".format(dim))
            model, emb = fit_cebra(X, vel[:, :2], "time_delta", dim, "offset10-model",
                                   cebra_time_offsets, cebra_iter, train_idx)
            out["M2"] = {"Z": emb.astype(np.float64), "model": model}

    if "M3" in models:
        import two_tower as tt
        B = tt.build_behavior_matrix(stream)
        if verbose:
            print("[M3 two-tower] fitting (dim={})...".format(dim))
        res = tt.get_two_tower(X, B, cache_dir, dim, train_idx,
                               time_offset=tt_time_offset, temperature=tt_temperature,
                               batch_size=512, max_iter=tt_iter, seed=seed,
                               tag="{}_d{}".format(tag, dim), verbose=verbose)
        out["M3"] = {"Z_n": res["z_n"], "Z_b": res["z_b"], "model": res["model"]}

    return out


def representation_for_target(embeddings, model_name, direction="forward"):
    if model_name == "M0":
        return embeddings["M0"]["X"], True
    if model_name in ("M1", "M2"):
        return embeddings[model_name]["Z"], False
    if model_name == "M3":
        key = "Z_n" if direction == "forward" else "Z_b"
        return embeddings["M3"][key], False
    raise ValueError(model_name)


# --------------------------------------------------------------------------- #
# Decoders
# --------------------------------------------------------------------------- #
def decode_binary(Z, y, train_idx, test_idx, raw):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        return float("nan")
    clf = (LogisticRegression(max_iter=1000, class_weight="balanced") if raw
          else KNeighborsClassifier(n_neighbors=25))
    clf.fit(Z[train_idx], y[train_idx])
    p = clf.predict_proba(Z[test_idx])[:, 1]
    return float(roc_auc_score(y[test_idx], p))


def decode_regression(Z, y, train_idx, test_idx, raw):
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score
    model = Ridge(alpha=1.0) if raw else KNeighborsRegressor(n_neighbors=25)
    model.fit(Z[train_idx], y[train_idx])
    pred = model.predict(Z[test_idx])
    return float(r2_score(y[test_idx], pred, multioutput="uniform_average"))


# --------------------------------------------------------------------------- #
# E2: target sweep
# --------------------------------------------------------------------------- #
def run_target_sweep(embeddings, targets, train_idx, test_idx, models):
    rows = []
    for tname, (y, keep, task) in targets.items():
        metric = "auc" if task == "binary" else "r2"
        ttr = train_idx[keep[train_idx]]
        tte = test_idx[keep[test_idx]]
        print("\n[target sweep] {} (task={}, n_train={}, n_test={})".format(
            tname, task, len(ttr), len(tte)))
        for m in models:
            if m not in embeddings:
                continue
            Z, raw = representation_for_target(embeddings, m)
            score = (decode_binary(Z, y, ttr, tte, raw) if task == "binary"
                    else decode_regression(Z, y, ttr, tte, raw))
            rows.append({"target": tname, "model": m, "metric": metric, "value": score})
            print("    {:3s} {}={:.3f}".format(m, metric, score))
        Xv = embeddings["M0"]["X"]
        bcv = (blocked_auc(Xv[keep], y[keep], k=5) if task == "binary"
              else blocked_r2(Xv[keep], y[keep], k=5))
        rows.append({"target": tname, "model": "M0_blockedcv5", "metric": metric, "value": bcv})
        print("    M0_blockedcv5 {}={:.3f}  (supplementary 5-fold; no fitting step, no leakage)".format(
            metric, bcv))
    return rows


# --------------------------------------------------------------------------- #
# E1: dimension sweep (T1/T2 only, reduced iterations)
# --------------------------------------------------------------------------- #
def run_dim_sweep(stream, train_idx, test_idx, targets, dims, models,
                  cebra_iter, tt_iter, tt_time_offset, tt_temperature,
                  cebra_time_offsets, cache_dir, seed):
    rows = []
    sweep_models = [m for m in models if m in ("M1", "M2", "M3")]
    for dim in dims:
        print("\n[dim sweep] dim={}".format(dim))
        emb = build_all_embeddings(stream, train_idx, dim, sweep_models, cebra_iter, tt_iter,
                                   tt_time_offset, tt_temperature, cebra_time_offsets,
                                   cache_dir, seed, tag="dimsweep", verbose=False)
        for tname in ("T1", "T2"):
            if tname not in targets:
                continue
            y, keep, task = targets[tname]
            ttr = train_idx[keep[train_idx]]
            tte = test_idx[keep[test_idx]]
            for m in sweep_models:
                if m not in emb:
                    continue
                Z, raw = representation_for_target(emb, m)
                score = decode_binary(Z, y, ttr, tte, raw)
                rows.append({"target": tname, "model": m, "dim": dim, "auc": score})
                print("    {} {} dim={} auc={:.3f}".format(tname, m, dim, score))
    return rows


# --------------------------------------------------------------------------- #
# E3: label efficiency (T1/T2 only)
# --------------------------------------------------------------------------- #
def stratified_subsample(y, frac, rng):
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        m = max(1, int(round(len(ci) * frac)))
        idx.append(rng.choice(ci, size=min(m, len(ci)), replace=False))
    return np.concatenate(idx)


def label_efficiency_curve(Z, y, keep, train_idx, test_idx, raw):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score

    ttr = train_idx[keep[train_idx]]
    tte = test_idx[keep[test_idx]]
    if len(np.unique(y[tte])) < 2:
        return None
    n_labels, aucs = [], []
    for frac in FRACTIONS:
        seed_auc, n_used = [], 0
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            sub = ttr[stratified_subsample(y[ttr], frac, rng)]
            if len(np.unique(y[sub])) < 2:
                continue
            clf = (LogisticRegression(max_iter=1000, class_weight="balanced") if raw
                  else KNeighborsClassifier(n_neighbors=min(25, max(1, len(sub) // 2))))
            clf.fit(Z[sub], y[sub])
            seed_auc.append(roc_auc_score(y[tte], clf.predict_proba(Z[tte])[:, 1]))
            n_used = len(sub)
        if seed_auc:
            n_labels.append(n_used)
            aucs.append(float(np.mean(seed_auc)))
    return np.array(n_labels), np.array(aucs)


def run_label_efficiency(embeddings, targets, train_idx, test_idx, models):
    rows, plot_data = [], {}
    for tname in ("T1", "T2"):
        if tname not in targets:
            continue
        y, keep, task = targets[tname]
        if task != "binary":
            continue
        print("\n[label efficiency] {}".format(tname))
        plot_data[tname] = []
        for m in models:
            if m not in embeddings:
                continue
            Z, raw = representation_for_target(embeddings, m)
            res = label_efficiency_curve(Z, y, keep, train_idx, test_idx, raw)
            if res is None:
                continue
            nl, au = res
            plot_data[tname].append((m, nl, au))
            for a, b in zip(nl, au):
                rows.append({"target": tname, "model": m, "n_labels": int(a), "test_auc": float(b)})
            print("    {}: {}".format(m, ", ".join("n={} auc={:.3f}".format(a, b) for a, b in zip(nl, au))))
    return rows, plot_data


# --------------------------------------------------------------------------- #
# E4: bidirectional decode (H3.1)
# --------------------------------------------------------------------------- #
def run_bidirectional(embeddings, targets, train_idx, test_idx, seed=0):
    rows = []
    if "M3" not in embeddings:
        return rows
    Z_b = embeddings["M3"]["Z_b"]

    if "T1" in targets:
        y, keep, _ = targets["T1"]
        ttr = train_idx[keep[train_idx]]
        tte = test_idx[keep[test_idx]]
        auc = decode_binary(Z_b, y, ttr, tte, raw=False)
        rows.append({"direction": "z_b->T1(reach)", "metric": "auc", "value": auc})
        print("  z_b -> T1 (reach) AUC = {:.3f}".format(auc))

    from sklearn.decomposition import PCA
    X = embeddings["M0"]["X"]
    k = min(8, X.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(X[train_idx])
    Y_pca = pca.transform(X)
    r2 = decode_regression(Z_b, Y_pca, train_idx, test_idx, raw=False)
    rows.append({"direction": "z_b->neuralPCA(k={})".format(k), "metric": "r2", "value": r2})
    print("  z_b -> neural-PCA(top-{}) R2 = {:.3f}  (z_b->speed/velocity excluded: circular, "
         "B_t is built from those values)".format(k, r2))
    return rows


# --------------------------------------------------------------------------- #
# E-CKA: cross-span consistency (within-subject substitute for cross-subject H3.3)
# --------------------------------------------------------------------------- #
def linear_cka(A, B):
    """Linear CKA (Kornblith et al. 2019) via the Gram-trick -- O(T*d^2), not O(T^2)."""
    A = np.asarray(A, dtype=np.float64) - np.mean(A, axis=0, keepdims=True)
    B = np.asarray(B, dtype=np.float64) - np.mean(B, axis=0, keepdims=True)
    hsic = np.linalg.norm(B.T @ A, ord="fro") ** 2
    norm_a = np.linalg.norm(A.T @ A, ord="fro")
    norm_b = np.linalg.norm(B.T @ B, ord="fro")
    return float(hsic / (norm_a * norm_b + 1e-12))


def behavior_matched_pairs(stream_a, stream_b, n_bins=10, min_per_bin=20, n_pairs=2000, seed=0):
    """Pair samples across two spans by (reach flag, within-span speed decile) bins.

    Simplification of a full nearest-neighbor speed match: same-bin random pairing,
    which is enough to give CKA a comparable, non-circular pairing without needing a
    per-sample NN search over two long streams.
    """
    def bin_ids(reach, speed):
        order = np.argsort(np.argsort(speed))
        dec = np.clip((order / max(1, len(speed)) * n_bins).astype(int), 0, n_bins - 1)
        return reach.astype(int) * n_bins + dec

    reach_a = np.asarray(stream_a["reach"]).astype(int)
    reach_b = np.asarray(stream_b["reach"]).astype(int)
    speed_a = np.asarray(stream_a["speed"])[:, 0]
    speed_b = np.asarray(stream_b["speed"])[:, 0]
    bins_a = bin_ids(reach_a, speed_a)
    bins_b = bin_ids(reach_b, speed_b)

    rng = np.random.default_rng(seed)
    common = sorted(set(np.unique(bins_a)) & set(np.unique(bins_b)))
    per_bin = max(1, n_pairs // max(1, len(common)))
    idx_a_all, idx_b_all = [], []
    for bidx in common:
        ia = np.where(bins_a == bidx)[0]
        ib = np.where(bins_b == bidx)[0]
        if len(ia) < min_per_bin or len(ib) < min_per_bin:
            continue
        m = min(per_bin, len(ia), len(ib))
        idx_a_all.append(rng.choice(ia, size=m, replace=False))
        idx_b_all.append(rng.choice(ib, size=m, replace=False))
    if not idx_a_all:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(idx_a_all), np.concatenate(idx_b_all)


def run_cka(path, channels, bands, out_rate, smooth_hz, dim, cebra_iter, tt_iter,
           tt_time_offset, tt_temperature, cebra_time_offsets, cka_dur_min,
           cache_dir, seed, verbose=True):
    from nwb_dataset import build_continuous_stream, find_active_window, find_movement_window

    dur = cka_dur_min * 60.0
    t0a, t1a, _ = find_active_window(path, dur_sec=dur, step_sec=120.0)
    t0b, t1b, _ = find_movement_window(path, dur_sec=dur, step_sec=60.0)
    print("CKA span A (reach-dense): {:.0f}-{:.0f}s".format(t0a, t1a))
    print("CKA span B (movement-rich): {:.0f}-{:.0f}s".format(t0b, t1b))
    stream_a = build_continuous_stream(path, t0a, t1a, out_rate=out_rate, bands=bands,
                                       ecog_channels=channels, zscore=True,
                                       smooth_hz=smooth_hz, verbose=False)
    stream_b = build_continuous_stream(path, t0b, t1b, out_rate=out_rate, bands=bands,
                                       ecog_channels=channels, zscore=True,
                                       smooth_hz=smooth_hz, verbose=False)
    train_a = np.arange(int(len(stream_a["X"]) * 0.7))
    train_b = np.arange(int(len(stream_b["X"]) * 0.7))
    emb_a = build_all_embeddings(stream_a, train_a, dim, ("M1", "M2", "M3"), cebra_iter, tt_iter,
                                 tt_time_offset, tt_temperature, cebra_time_offsets,
                                 cache_dir, seed, tag="ckaA", verbose=verbose)
    emb_b = build_all_embeddings(stream_b, train_b, dim, ("M1", "M2", "M3"), cebra_iter, tt_iter,
                                 tt_time_offset, tt_temperature, cebra_time_offsets,
                                 cache_dir, seed, tag="ckaB", verbose=verbose)

    idx_a, idx_b = behavior_matched_pairs(stream_a, stream_b, seed=seed)
    rows = []
    if len(idx_a) < 20:
        print("  too few matched cross-span pairs ({}) -- skipping CKA".format(len(idx_a)))
        return rows

    rows.append({"model": "M0", "cka": linear_cka(stream_a["X"][idx_a], stream_b["X"][idx_b])})
    for m, key in (("M1", "Z"), ("M2", "Z"), ("M3", "Z_n")):
        if m not in emb_a or m not in emb_b:
            continue
        rows.append({"model": m, "cka": linear_cka(emb_a[m][key][idx_a], emb_b[m][key][idx_b])})
    for r in rows:
        print("  CKA[{}] = {:.3f}".format(r["model"], r["cka"]))
    return rows


# --------------------------------------------------------------------------- #
# T5/T6: coarse behavior epochs (whole-file scan)
# --------------------------------------------------------------------------- #
def build_epoch_dataset(path, channels, bands, window_sec=10.0, max_per_label=150,
                        label_set=None, seed=0, verbose=True):
    from nwb_dataset import windows_from_epochs, WindowedNWBDataset, extract_features, BANDS

    windows = windows_from_epochs(path, window_sec=window_sec, max_per_label=max_per_label, seed=seed)
    if label_set is not None:
        windows = [w for w in windows if str(w["label"]) in label_set]
    if not windows:
        raise ValueError("no epoch windows matched label_set={}".format(label_set))

    ds = WindowedNWBDataset(path, windows, window_sec, ecog_channels=channels, pose_keypoints=None)
    band_dict = {b: BANDS[b] for b in bands}
    X, y, t0s = [], [], []
    n = len(ds)
    for i in range(n):
        s = ds[i]
        if not np.all(np.isfinite(s["ecog"])):
            continue
        vec, _ = extract_features(s, s["fs"], bands=band_dict, include_pose=False)
        if not np.all(np.isfinite(vec)):
            continue
        X.append(vec)
        y.append(str(s["label"]))
        t0s.append(s["t0"])
        if verbose and (i + 1) % 200 == 0:
            print("  epoch windows processed {}/{}".format(i + 1, n))
    return np.asarray(X), np.asarray(y), np.asarray(t0s)


def group_split(t0s, seed=0, test_frac=0.3, bucket_sec=300.0):
    """Assign whole 5-min time buckets to train/test so nearby (correlated) epoch
    windows don't get split across the boundary -- a random per-window split would
    leak, since many 10s windows tile the same multi-minute annotated interval."""
    bucket_ids = (np.asarray(t0s) // bucket_sec).astype(int)
    rng = np.random.default_rng(seed)
    buckets = np.unique(bucket_ids)
    rng.shuffle(buckets)
    n_test = max(1, int(round(len(buckets) * test_frac)))
    test_buckets = set(buckets[:n_test].tolist())
    is_test = np.array([b in test_buckets for b in bucket_ids])
    return np.where(~is_test)[0], np.where(is_test)[0]


def run_epoch_targets(path, channels, bands, scaler, primary_models, epoch_embeddings,
                      window_sec, max_per_label, seed, verbose=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score

    X, y, t0s = build_epoch_dataset(path, channels, bands, window_sec=window_sec,
                                    max_per_label=max_per_label, label_set=set(T5_LABELS),
                                    seed=seed, verbose=verbose)
    tr, te = group_split(t0s, seed=seed)
    print("epoch dataset: {} windows, {} labels, n_train={} n_test={}".format(
        len(y), len(set(y)), len(tr), len(te)))

    reps = {"M0": (X, True)}
    if epoch_embeddings and primary_models is not None and scaler is not None:
        mu, sd = scaler
        Xz = ((X - mu) / sd).astype(np.float32)
        if "M1" in primary_models:
            reps["M1"] = (primary_models["M1"]["model"].transform(Xz), False)
        if "M2" in primary_models:
            reps["M2"] = (primary_models["M2"]["model"].transform(Xz), False)
        if "M3" in primary_models:
            import two_tower as tt
            zn, _ = tt.transform(primary_models["M3"]["model"], X=Xz)
            reps["M3"] = (zn, False)

    le = LabelEncoder().fit(y)
    yi = le.transform(y)
    y6 = (y == "Sleep/rest").astype(int)

    rows = []
    for m, (Z, raw) in reps.items():
        if len(np.unique(yi[tr])) >= 2 and len(np.unique(yi[te])) >= 2:
            clf = (LogisticRegression(max_iter=1000, class_weight="balanced") if raw
                  else KNeighborsClassifier(n_neighbors=15))
            clf.fit(Z[tr], yi[tr])
            proba = clf.predict_proba(Z[te])
            try:
                auc = float(roc_auc_score(yi[te], proba, multi_class="ovr", average="macro",
                                          labels=np.arange(len(le.classes_))))
            except ValueError:
                auc = float("nan")
            rows.append({"target": "T5", "model": m, "metric": "macro_auc_ovr", "value": auc})
            print("  T5 {} macro-AUC(ovr) = {:.3f}".format(m, auc))

        if len(np.unique(y6[tr])) >= 2 and len(np.unique(y6[te])) >= 2:
            auc = decode_binary(Z, y6, tr, te, raw)
            rows.append({"target": "T6", "model": m, "metric": "auc", "value": auc})
            print("  T6 {} AUC = {:.3f}".format(m, auc))
    return rows


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def save_csv(rows, path, fieldnames=None):
    if not rows:
        print("(nothing to write for {})".format(path))
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def plot_target_sweep(rows, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        targets = sorted(set(r["target"] for r in rows))
        fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.2), squeeze=False)
        for ax, tname in zip(axes[0], targets):
            sub = [r for r in rows if r["target"] == tname]
            models = [r["model"] for r in sub]
            values = [r["value"] for r in sub]
            colors = ["C3" if m == "M0_blockedcv5" else "C0" for m in models]
            ax.bar(range(len(models)), values, color=colors)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
            ax.set_title(tname)
            ax.set_ylabel(sub[0]["metric"])
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle("Phase 3 target sweep: M0-M3 decode performance")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print("wrote", out_path)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


def plot_dim_sweep(rows, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        targets = sorted(set(r["target"] for r in rows))
        fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4.2), squeeze=False)
        for ax, tname in zip(axes[0], targets):
            sub = [r for r in rows if r["target"] == tname]
            for m in sorted(set(r["model"] for r in sub)):
                msub = sorted([r for r in sub if r["model"] == m], key=lambda r: r["dim"])
                ax.plot([r["dim"] for r in msub], [r["auc"] for r in msub], "-o", label=m)
            ax.set_xlabel("embedding dim")
            ax.set_ylabel("AUC")
            ax.set_title(tname)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle("Phase 3 dimension sweep")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print("wrote", out_path)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


def plot_label_efficiency(plot_data, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tnames = list(plot_data.keys())
        if not tnames:
            return
        fig, axes = plt.subplots(1, len(tnames), figsize=(6 * len(tnames), 4.6), squeeze=False)
        for ax, tname in zip(axes[0], tnames):
            for m, nl, au in plot_data[tname]:
                ax.plot(nl, au, "-o", label=m)
            ax.set_xscale("log")
            ax.set_xlabel("# labeled training samples")
            ax.set_ylabel("test AUC")
            ax.set_title(tname)
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle("Phase 3 label efficiency (test = held-out last 30% of time)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print("wrote", out_path)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


def plot_cka(rows, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        models = [r["model"] for r in rows]
        vals = [r["cka"] for r in rows]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(range(len(models)), vals, color="C2")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_ylabel("linear CKA")
        ax.set_title("Cross-span consistency (within-subject pilot; N=1 subject)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print("wrote", out_path)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


def save_target_sweep(rows, out_dir):
    save_csv(rows, os.path.join(out_dir, "phase3_target_sweep.csv"))
    plot_target_sweep(rows, os.path.join(out_dir, "phase3_target_sweep.png"))


def save_dim_sweep(rows, out_dir):
    save_csv(rows, os.path.join(out_dir, "phase3_dim_sweep.csv"))
    plot_dim_sweep(rows, os.path.join(out_dir, "phase3_dim_sweep.png"))


def save_label_efficiency(rows, plot_data, out_dir):
    save_csv(rows, os.path.join(out_dir, "phase3_label_efficiency.csv"))
    plot_label_efficiency(plot_data, os.path.join(out_dir, "phase3_label_efficiency.png"))


def save_bidirectional(rows, out_dir):
    save_csv(rows, os.path.join(out_dir, "phase3_bidirectional.csv"))


def save_cka(rows, out_dir):
    save_csv(rows, os.path.join(out_dir, "phase3_cka.csv"))
    plot_cka(rows, os.path.join(out_dir, "phase3_cka.png"))


def save_epoch_targets(rows, out_dir):
    save_csv(rows, os.path.join(out_dir, "phase3_epoch_targets.csv"))


# --------------------------------------------------------------------------- #
# Summary verdict
# --------------------------------------------------------------------------- #
def print_summary(target_rows, bidir_rows, label_eff_rows, cka_rows):
    print("\n================ PHASE 3 SUMMARY ================")

    fwd = next((r["value"] for r in target_rows if r["target"] == "T1" and r["model"] == "M3"), None)
    rev = next((r["value"] for r in bidir_rows if r["direction"] == "z_b->T1(reach)"), None)
    if fwd is not None and rev is not None:
        fwd_ok, rev_ok = fwd > 0.55, rev > 0.55
        if fwd_ok and rev_ok:
            h31 = "SYMMETRIC (both z_n->behavior and z_b->reach decodable)"
        elif fwd_ok:
            h31 = "ONE-WAY (neural tower rich; behavior tower not independently decodable)"
        elif rev_ok:
            h31 = "ONE-WAY (unusual: only z_b->reach decodable)"
        else:
            h31 = "NEITHER direction clearly decodable"
        print("H3.1 bidirectionality: {}  (z_n->T1 AUC={:.3f}, z_b->T1 AUC={:.3f})".format(h31, fwd, rev))
    else:
        print("H3.1 bidirectionality: not evaluated (need --stage extended/full)")

    if label_eff_rows:
        for tname in ("T1", "T2"):
            sub = [r for r in label_eff_rows if r["target"] == tname]
            if not sub:
                continue
            min_n = min(r["n_labels"] for r in sub)
            at_min = [r for r in sub if r["n_labels"] == min_n]
            best = max(at_min, key=lambda r: r["test_auc"])
            m0 = next((r for r in at_min if r["model"] == "M0"), None)
            if m0 and best["model"] != "M0" and best["test_auc"] > m0["test_auc"] + 0.03:
                print("H3.2 label efficiency [{}]: {} beats M0 at n={} labels ({:.3f} vs {:.3f})".format(
                    tname, best["model"], min_n, best["test_auc"], m0["test_auc"]))
            else:
                print("H3.2 label efficiency [{}]: no clear win over raw features at low labels".format(tname))
    else:
        print("H3.2 label efficiency: not evaluated (need --stage extended/full)")

    if cka_rows:
        best = max(cka_rows, key=lambda r: r["cka"])
        print("H3.3 cross-span consistency (within-subject pilot; cross-subject UNTESTED, N=1 file): "
             "highest CKA = {} ({:.3f})".format(best["model"], best["cka"]))
    else:
        print("H3.3 cross-span consistency: not evaluated (need --stage full)")
    print("===================================================\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", default=None, help="NWB file (default: largest .nwb found)")
    ap.add_argument("--out-dir", default="phase3_out")
    ap.add_argument("--stage", choices=["smoke", "core", "extended", "full"], default="core")
    ap.add_argument("--start", type=float, default=-1)
    ap.add_argument("--dur-min", type=float, default=45.0)
    ap.add_argument("--anchor", choices=["reach", "movement"], default="reach")
    ap.add_argument("--channel-method", choices=["sensorimotor", "aal", "box", "good"],
                    default="sensorimotor")
    ap.add_argument("--bands", default="beta,high_gamma")
    ap.add_argument("--out-rate", type=float, default=30.0)
    ap.add_argument("--smooth-hz", type=float, default=6.0)
    ap.add_argument("--dim", type=int, default=16, help="embedding dim for all non-sweep experiments")
    ap.add_argument("--dims", default="8,16,32", help="dims for the E1 dimension sweep")
    ap.add_argument("--targets", default="T1,T2,T3,T4")
    ap.add_argument("--models", default="M0,M1,M2,M3")
    ap.add_argument("--cebra-iter", type=int, default=1500)
    ap.add_argument("--cebra-time-offsets", type=int, default=10)
    ap.add_argument("--tt-iter", type=int, default=1500)
    ap.add_argument("--tt-time-offset", type=int, default=0)
    ap.add_argument("--tt-temperature", type=float, default=0.1)
    ap.add_argument("--sweep-iter-frac", type=float, default=0.3,
                    help="fraction of --cebra-iter/--tt-iter used for E1 dim sweep and E-CKA fits")
    ap.add_argument("--epoch-embeddings", action="store_true",
                    help="apply primary-stream-trained M1-M3 encoders to T5/T6 (off=M0 only)")
    ap.add_argument("--epoch-window-sec", type=float, default=10.0)
    ap.add_argument("--epoch-max-per-label", type=int, default=150)
    ap.add_argument("--cka-dur-min", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = args.nwb or find_default_file()
    if not path or not os.path.exists(path):
        raise SystemExit("No NWB file found. Pass --nwb explicitly.")
    print("file:", path)
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    stage = args.stage
    dur_min, cebra_iter, tt_iter = args.dur_min, args.cebra_iter, args.tt_iter
    if stage == "smoke":
        dur_min, cebra_iter, tt_iter = 6.0, 200, 200
        print("[smoke stage] overriding dur-min={} cebra-iter={} tt-iter={}".format(
            dur_min, cebra_iter, tt_iter))

    targets_req = [t.strip() for t in args.targets.split(",") if t.strip()]
    models_req = [m.strip() for m in args.models.split(",") if m.strip()]
    dims = [int(d) for d in args.dims.split(",") if d.strip()]

    channels, ch_method = select_channels(path, mode=args.channel_method)
    bands = tuple(b.strip() for b in args.bands.split(",") if b.strip())
    t0, t1, win_note = pick_window(path, dur_min * 60.0, args.anchor, args.start)
    print("window:", win_note, "{:.0f}-{:.0f}s  channels:".format(t0, t1), ch_method,
         "({})".format(len(channels)))

    stream = get_primary_stream(path, t0, t1, args.out_rate, bands, channels,
                                args.smooth_hz, cache_dir)
    stream["X"] = np.asarray(stream["X"], dtype=np.float32)
    T = len(stream["X"])
    k_split = int(T * 0.7)
    train_idx, test_idx = np.arange(k_split), np.arange(k_split, T)
    print("stream T={}  train={}  test={}".format(T, len(train_idx), len(test_idx)))

    targets = {k: v for k, v in build_targets_from_stream(stream).items() if k in targets_req}

    print("\n=== building M0-M3 embeddings (dim={}) ===".format(args.dim))
    embeddings = build_all_embeddings(stream, train_idx, args.dim, models_req, cebra_iter, tt_iter,
                                      args.tt_time_offset, args.tt_temperature,
                                      args.cebra_time_offsets, cache_dir, args.seed,
                                      tag="primary", verbose=True)

    print("\n=== E2: target sweep ===")
    target_rows = run_target_sweep(embeddings, targets, train_idx, test_idx, models_req)
    save_target_sweep(target_rows, args.out_dir)

    label_eff_rows, cka_rows, bidir_rows = [], [], []
    sweep_cebra_iter = max(200, int(cebra_iter * args.sweep_iter_frac))
    sweep_tt_iter = max(200, int(tt_iter * args.sweep_iter_frac))

    if stage in ("extended", "full"):
        print("\n=== E1: dim sweep ===")
        dim_rows = run_dim_sweep(stream, train_idx, test_idx, targets, dims, models_req,
                                 sweep_cebra_iter, sweep_tt_iter, args.tt_time_offset,
                                 args.tt_temperature, args.cebra_time_offsets, cache_dir, args.seed)
        save_dim_sweep(dim_rows, args.out_dir)

        print("\n=== E3: label efficiency ===")
        label_eff_rows, plot_data = run_label_efficiency(embeddings, targets, train_idx, test_idx, models_req)
        save_label_efficiency(label_eff_rows, plot_data, args.out_dir)

        print("\n=== E4: bidirectional decode ===")
        bidir_rows = run_bidirectional(embeddings, targets, train_idx, test_idx, seed=args.seed)
        save_bidirectional(bidir_rows, args.out_dir)

    if stage == "full":
        print("\n=== E-CKA: cross-span consistency ===")
        cka_rows = run_cka(path, channels, bands, args.out_rate, args.smooth_hz, args.dim,
                           sweep_cebra_iter, sweep_tt_iter, args.tt_time_offset, args.tt_temperature,
                           args.cebra_time_offsets, args.cka_dur_min, cache_dir, args.seed)
        save_cka(cka_rows, args.out_dir)

        print("\n=== T5/T6: coarse behavior (epoch windows, whole 24h file) ===")
        epoch_rows = run_epoch_targets(path, channels, bands, stream.get("scaler"),
                                       embeddings if args.epoch_embeddings else None,
                                       args.epoch_embeddings, args.epoch_window_sec,
                                       args.epoch_max_per_label, args.seed)
        save_epoch_targets(epoch_rows, args.out_dir)

    print_summary(target_rows, bidir_rows, label_eff_rows, cka_rows)
    print("Done. Outputs in", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
