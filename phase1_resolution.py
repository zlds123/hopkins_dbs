"""Phase 1: temporal-resolution sweep for neural -> behavior decoding (AJILE12).

Tests H1.1 (window length), H1.2 (lag), H1.3 (causal cost) on band-power or CEBRA
embeddings, with reach-dense or movement-rich spans and AAL or coord-box channels.

Run:
  python phase1_resolution.py --file path\\to\\sub-01_ses-3_behavior+ecephys.nwb
  python phase1_resolution.py --anchor movement --features cebra --out-dir phase1_out_move_cebra
  python phase1_resolution.py --files sub-01.nwb,sub-02.nwb --out-dir phase1_multisub
"""

import argparse
import glob
import hashlib
import os

import numpy as np

from nwb_dataset import (build_continuous_stream, find_active_window,
                         find_movement_window, sensorimotor_channels,
                         good_channel_indices, electrode_coords, mni_to_aal,
                         SENSORIMOTOR_AAL)


# --------------------------------------------------------------------------- #
# Data / caching
# --------------------------------------------------------------------------- #
def find_default_file():
    cands = (glob.glob("*.nwb")
             + glob.glob(os.path.join("..", "*.nwb"))
             + glob.glob(os.path.join("ajile12-nwb-data", "**", "*.nwb"), recursive=True))
    cands = [p for p in cands if os.path.getsize(p) > 1e9]  # AJILE12-scale only
    return max(cands, key=os.path.getsize) if cands else None


def find_nwb_files(pattern=None):
    if pattern:
        return sorted(glob.glob(pattern))
    cands = (glob.glob("*.nwb")
             + glob.glob(os.path.join("..", "*.nwb"))
             + glob.glob(os.path.join("ajile12-nwb-data", "**", "*.nwb"), recursive=True))
    return sorted({p for p in cands if os.path.isfile(p) and os.path.getsize(p) > 1e9})


def select_channels(path, mode="sensorimotor", verbose=True):
    """mode: good | sensorimotor | aal | box"""
    if mode == "good":
        with __import__("h5py").File(path, "r") as f:
            return good_channel_indices(f), "good"
    if mode == "box":
        xyz, good = electrode_coords(path)
        pool = np.where(good)[0]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        box = (np.abs(x) >= 25) & (y >= -40) & (y <= 15) & (z >= 15)
        ch = np.array([i for i in pool if box[i]], dtype=int)
        if verbose:
            print("coord-box channels: {}/{}".format(len(ch), len(pool)))
        return ch, "coord_box"
    if mode == "aal":
        xyz, good = electrode_coords(path)
        pool = np.where(good)[0]
        names = mni_to_aal(xyz)
        ch = [i for i in pool
              if any(r.lower() in names[i].lower() for r in SENSORIMOTOR_AAL)]
        method = "AAL"
        if not ch:
            ch, method = select_channels(path, mode="box", verbose=verbose)
            method = "AAL-empty+" + method
        else:
            if verbose:
                print("AAL sensorimotor channels: {}/{}".format(len(ch), len(pool)))
        return np.array(sorted(set(ch)), dtype=int), method
    # default sensorimotor (AAL + box fallback inside helper)
    ch = sensorimotor_channels(path, verbose=verbose)
    return ch, "sensorimotor"


def pick_window(path, dur_sec, anchor, start=-1):
    if start is not None and start >= 0:
        return float(start), float(start) + dur_sec, "manual"
    if anchor == "movement":
        t0, t1, score = find_movement_window(path, dur_sec=dur_sec, step_sec=120.0)
        return t0, t1, "movement(score={:.3f})".format(score)
    t0, t1, n = find_active_window(path, dur_sec=dur_sec, step_sec=300.0)
    return t0, t1, "reach(n={})".format(n)


def get_stream(path, t0, t1, out_rate, bands, channels, smooth_hz, out_dir, tag=""):
    ch_tag = channels if isinstance(channels, str) else "ch{}".format(len(channels))
    key = "{}|{:.0f}|{:.0f}|{:.0f}|{}|{}|{:.0f}|{}".format(
        os.path.basename(path), t0, t1, out_rate, ",".join(bands), ch_tag, smooth_hz, tag)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    cache = os.path.join(out_dir, "stream_{}.npz".format(h))
    if os.path.exists(cache):
        print("loading cached stream:", cache)
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print("building continuous stream (chunked)...")
    s = build_continuous_stream(path, t0, t1, out_rate=out_rate, bands=bands,
                                ecog_channels=channels, zscore=False,
                                smooth_hz=smooth_hz, verbose=False)
    np.savez_compressed(
        cache, X=s["X"], speed=s["speed"], reach=s["reach"],
        vel=s["vel"], t=s["t"], out_rate=s["out_rate"])
    return {"X": s["X"], "speed": s["speed"], "reach": s["reach"],
            "vel": s["vel"], "t": s["t"], "out_rate": np.asarray(s["out_rate"])}


def build_cebra_matrix(stream, dim=8, max_iter=2000, arch="offset10-model"):
    """Self-supervised CEBRA-Behavior embedding (wrist velocity auxiliary)."""
    from cebra import CEBRA
    X = stream["X"].astype(np.float32)
    vel = np.asarray(stream["vel"], dtype=np.float32)
    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError("CEBRA needs wrist velocity in stream['vel']")
    aux = vel[:, :2]
    print("fitting CEBRA (dim={}, max_iter={})...".format(dim, max_iter))
    model = CEBRA(
        model_architecture=arch, conditional="time_delta",
        time_offsets=10, output_dimension=dim, max_iterations=max_iter,
        batch_size=512, distance="cosine", learning_rate=3e-4,
        device="cpu", verbose=False,
    )
    model.fit(X, aux)
    Z = model.transform(X).astype(np.float64)
    print("CEBRA embedding:", Z.shape)
    return Z


# --------------------------------------------------------------------------- #
# Windowed features + scorers
# --------------------------------------------------------------------------- #
def windowed_features(csum, pred_idx, L_samp, lag_samp, causal):
    T = csum.shape[0] - 1
    ref = pred_idx - lag_samp
    if causal:
        a = ref - L_samp + 1
        b = ref + 1
    else:
        a = ref - L_samp // 2
        b = a + L_samp
    a = np.clip(a, 0, T)
    b = np.clip(b, 0, T)
    width = np.maximum(b - a, 1)[:, None]
    return (csum[b] - csum[a]) / width


def blocked_auc(Xf, y, k=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    n = len(y)
    folds = np.array_split(np.arange(n), k)
    scores = []
    for j in range(k):
        te = folds[j]
        tr = np.concatenate([folds[m] for m in range(k) if m != j])
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(Xf[tr])
        clf = LogisticRegression(max_iter=400, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(Xf[tr]), y[tr])
        p = clf.predict_proba(sc.transform(Xf[te]))[:, 1]
        scores.append(roc_auc_score(y[te], p))
    return float(np.mean(scores)) if scores else np.nan


def blocked_r2(Xf, y, k=5):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    n = len(y)
    folds = np.array_split(np.arange(n), k)
    scores = []
    for j in range(k):
        te = folds[j]
        tr = np.concatenate([folds[m] for m in range(k) if m != j])
        sc = StandardScaler().fit(Xf[tr])
        rg = Ridge(alpha=10.0)
        rg.fit(sc.transform(Xf[tr]), y[tr])
        scores.append(r2_score(y[te], rg.predict(sc.transform(Xf[te]))))
    return float(np.mean(scores)) if scores else np.nan


def speed_from_threshold(speed_raw, lo_pct=40, hi_pct=60):
    """Balanced movement-vs-rest from continuous speed (for movement-rich spans)."""
    s = np.asarray(speed_raw, dtype=float)
    finite = s[np.isfinite(s) & (s > 0)]
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    y = np.full(s.shape[0], -1, dtype=int)
    y[s <= lo] = 0
    y[s >= hi] = 1
    return y, y >= 0


def prepare_targets(stream, label_mode, fs, pred_idx):
    reach = stream["reach"].astype(int)
    speed_raw = stream["speed"]
    if speed_raw.ndim == 2 and speed_raw.shape[1]:
        speed_raw = speed_raw[:, 0].astype(np.float64)
    else:
        speed_raw = None

    if label_mode == "speed_balanced" and speed_raw is not None:
        y_all, keep = speed_from_threshold(speed_raw)
        y_move = y_all[pred_idx]
        move_keep = keep[pred_idx]
    elif label_mode == "speed_median" and speed_raw is not None:
        s = speed_raw[pred_idx]
        y_move = (s > np.nanmedian(s)).astype(int)
        move_keep = np.ones(len(pred_idx), dtype=bool)
    else:
        y_move = reach[pred_idx]
        move_keep = np.ones(len(pred_idx), dtype=bool)

    if speed_raw is not None:
        cap = np.nanpercentile(speed_raw, 99)
        y_speed = np.log1p(np.clip(speed_raw, 0, cap))[pred_idx]
    else:
        y_speed = None
    return y_move, y_speed, move_keep, speed_raw


# --------------------------------------------------------------------------- #
# Core sweep
# --------------------------------------------------------------------------- #
def run_sweep(X, fs, pred_idx, y_move, y_speed, args):
    T, C = X.shape
    csum = np.vstack([np.zeros((1, C)), np.cumsum(X, axis=0)])

    win_list = [float(v) for v in args.windows.split(",")]
    lag_list = [round(v, 3) for v in np.arange(args.lag_min, args.lag_max + 1e-9, args.lag_step)]
    causal_L = [float(v) for v in args.causal_windows.split(",")]

    def score(L, lag, causal):
        Ls = max(1, int(round(L * fs)))
        lg = int(round(lag * fs))
        Xf = windowed_features(csum, pred_idx, Ls, lg, causal)
        auc = blocked_auc(Xf, y_move, args.cv)
        r2 = blocked_r2(Xf, y_speed, args.cv) if y_speed is not None else np.nan
        return auc, r2

    win_rows, lag_rows, cz_rows = [], [], []
    for L in win_list:
        auc, r2 = score(L, 0.0, True)
        win_rows.append((L, auc, r2))
    for lag in lag_list:
        auc, r2 = score(args.lag_window, lag, False)
        lag_rows.append((lag, auc, r2))
    for L in causal_L:
        a_c, r_c = score(L, 0.0, True)
        a_a, r_a = score(L, 0.0, False)
        cz_rows.append((L, a_c, a_a, r_c, r_a))

    return np.array(win_rows), np.array(lag_rows), np.array(cz_rows)


def summarize(win_rows, lag_rows, cz_rows):
    Ls, aucs = win_rows[:, 0], win_rows[:, 1]
    r2s = win_rows[:, 2]
    bi = int(np.nanargmax(aucs)) if np.any(np.isfinite(aucs)) else int(np.nanargmax(r2s))
    lags, laucs = lag_rows[:, 0], lag_rows[:, 1]
    lr2 = lag_rows[:, 2]
    li = int(np.nanargmax(laucs)) if np.any(np.isfinite(laucs)) else int(np.nanargmax(lr2))
    cost = float(np.nanmean(cz_rows[:, 2] - cz_rows[:, 1]))
    return {
        "best_win_s": float(Ls[bi]),
        "best_win_auc": float(aucs[bi]) if np.isfinite(aucs[bi]) else float("nan"),
        "best_lag_s": float(lags[li]),
        "best_lag_auc": float(laucs[li]) if np.isfinite(laucs[li]) else float("nan"),
        "mean_causal_cost": cost,
        "best_win_r2": float(r2s[bi]),
        "best_lag_r2": float(lr2[li]),
    }


def save_results(win_rows, lag_rows, cz_rows, out_dir, title_suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "phase1_window.csv"), win_rows,
               delimiter=",", header="window_s,auc_move,r2_speed", comments="")
    np.savetxt(os.path.join(out_dir, "phase1_lag.csv"), lag_rows,
               delimiter=",", header="lag_s,auc_move,r2_speed", comments="")
    np.savetxt(os.path.join(out_dir, "phase1_causal.csv"), cz_rows,
               delimiter=",",
               header="window_s,auc_causal,auc_acausal,r2_causal,r2_acausal", comments="")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        ax[0].plot(win_rows[:, 0], win_rows[:, 1], "-o", color="C0")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("window (s)")
        ax[0].set_ylabel("movement AUC")
        ax[0].set_title("H1.1 window")
        ax[0].grid(alpha=0.3)
        ax[1].plot(lag_rows[:, 0], lag_rows[:, 1], "-o", color="C0")
        ax[1].axvline(lag_rows[np.nanargmax(lag_rows[:, 1]), 0], color="C2", ls="--")
        ax[1].set_xlabel("lag (s)")
        ax[1].set_ylabel("movement AUC")
        ax[1].set_title("H1.2 lag")
        ax[1].grid(alpha=0.3)
        w = 0.35
        xpos = np.arange(len(cz_rows))
        ax[2].bar(xpos - w / 2, cz_rows[:, 1], w, label="causal")
        ax[2].bar(xpos + w / 2, cz_rows[:, 2], w, label="acausal")
        ax[2].set_xticks(xpos)
        ax[2].set_xticklabels(["{:.2f}".format(v) for v in cz_rows[:, 0]])
        ax[2].legend(fontsize=8)
        ax[2].set_title("H1.3 causal vs acausal")
        fig.suptitle("Phase 1 sweep" + (" · " + title_suffix if title_suffix else ""))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "phase1_resolution.png"), dpi=130)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


def run_one(path, args, out_dir=None):
    out_dir = out_dir or args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("file:", path)

    dur = args.dur_min * 60.0
    t0, t1, win_note = pick_window(path, dur, args.anchor, args.start)
    print("window:", win_note, "{:.0f}-{:.0f}s".format(t0, t1))

    channels, ch_method = select_channels(path, mode=args.channel_method)
    bands = tuple(b.strip() for b in args.bands.split(",") if b.strip())
    tag = "{}|{}|{}".format(args.anchor, args.features, ch_method)
    stream = get_stream(path, t0, t1, args.out_rate, bands, channels,
                        args.smooth_hz, out_dir, tag=tag)

    feature_sets = []
    if args.features in ("band", "both"):
        feature_sets.append(("band", stream["X"].astype(np.float64)))
    if args.features in ("cebra", "both"):
        feature_sets.append(("cebra", build_cebra_matrix(
            stream, dim=args.cebra_dim, max_iter=args.cebra_iter)))

    results = {}
    for feat_name, X in feature_sets:
        fs = float(stream["out_rate"])
        T = X.shape[0]

        win_list = [float(v) for v in args.windows.split(",")]
        causal_L = [float(v) for v in args.causal_windows.split(",")]
        max_half = max(max(win_list), max(causal_L)) / 2.0
        margin = int(round((max_half + max(abs(args.lag_min), abs(args.lag_max))) * fs)) + 1
        stride = max(1, int(round(args.stride_sec * fs)))
        pred_idx = np.arange(margin, T - margin, stride)

        y_move, y_speed, _, speed_raw = prepare_targets(stream, args.label, fs, pred_idx)
        if speed_raw is not None:
            print("  speed std={:.3f} (raw px/s)".format(float(np.nanstd(speed_raw))))

        print("[{}] grid={} x {} feat; label={}; balance {:.3f}".format(
            feat_name, len(pred_idx), X.shape[1], args.label, y_move.mean()))

        sub_out = out_dir if feat_name == "band" and args.features != "both" else os.path.join(
            out_dir, feat_name)
        win_rows, lag_rows, cz_rows = run_sweep(
            X, fs, pred_idx, y_move, y_speed, args)
        summ = summarize(win_rows, lag_rows, cz_rows)
        save_results(win_rows, lag_rows, cz_rows, sub_out,
                     title_suffix="{} · {} · {}".format(feat_name, args.anchor, ch_method))
        results[feat_name] = summ
        print("  best win {:.2f}s AUC={:.3f} | best lag {:+.2f}s AUC={:.3f} | causal cost {:+.3f} | R2={:+.3f}".format(
            summ["best_win_s"], summ["best_win_auc"], summ["best_lag_s"],
            summ["best_lag_auc"], summ["mean_causal_cost"], summ["best_win_r2"]))
    return {"file": path, "window": win_note, "channels": ch_method, **results}


def run(args):
    if args.files:
        paths = [p.strip() for p in args.files.split(",") if p.strip()]
    else:
        path = args.file or find_default_file()
        paths = [path] if path else []

    if not paths:
        raise SystemExit("No NWB files found.")

    rows = []
    for path in paths:
        if not os.path.exists(path):
            print("skip missing:", path)
            continue
        sub_out = args.out_dir
        if len(paths) > 1:
            sub_out = os.path.join(args.out_dir, os.path.splitext(os.path.basename(path))[0])
        res = run_one(path, args, out_dir=sub_out)
        for feat in ("band", "cebra"):
            if feat in res and isinstance(res[feat], dict):
                row = {"file": os.path.basename(path), "features": feat,
                       "window": res["window"], "channels": res["channels"]}
                row.update(res[feat])
                rows.append(row)

    if rows:
        import csv
        csv_path = os.path.join(args.out_dir, "phase1_multisubject_summary.csv")
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("\nwrote", csv_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default=None)
    ap.add_argument("--files", default=None, help="comma-separated NWB paths for multi-subject")
    ap.add_argument("--start", type=float, default=-1)
    ap.add_argument("--dur-min", type=float, default=45.0)
    ap.add_argument("--anchor", choices=["reach", "movement"], default="reach")
    ap.add_argument("--out-rate", type=float, default=50.0)
    ap.add_argument("--smooth-hz", type=float, default=15.0)
    ap.add_argument("--channel-method", choices=["sensorimotor", "aal", "box", "good"],
                    default="sensorimotor")
    ap.add_argument("--features", choices=["band", "cebra", "both"], default="band")
    ap.add_argument("--cebra-dim", type=int, default=8)
    ap.add_argument("--cebra-iter", type=int, default=2000)
    ap.add_argument("--label", choices=["reach", "speed_balanced", "speed_median"], default="reach",
                    help="reach annotation, speed threshold (drops middle), or speed median split")
    ap.add_argument("--bands", default="beta,high_gamma")
    ap.add_argument("--windows", default="0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0,3.0,4.0")
    ap.add_argument("--lag-window", type=float, default=0.5)
    ap.add_argument("--lag-min", type=float, default=-1.0)
    ap.add_argument("--lag-max", type=float, default=1.0)
    ap.add_argument("--lag-step", type=float, default=0.1)
    ap.add_argument("--causal-windows", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--stride-sec", type=float, default=0.4)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--out-dir", default="phase1_out")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
