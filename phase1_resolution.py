"""Phase 1: temporal-resolution sweep for neural -> behavior decoding (AJILE12).

This probes *how the time axis is cut* changes how well ECoG predicts behavior,
testing the three Phase-1 hypotheses on a single AJILE12 file:

  H1.1  Decoding accuracy depends non-monotonically on the feature window length
        (too short = noisy, too long = blurs the event).      -> window sweep
  H1.2  Accuracy peaks at a non-zero neural-to-behavioral lag
        (neural activity leads/lags the behavior).             -> lag sweep
  H1.3  Causal (past-only) windows cost accuracy vs acausal
        (centered) windows that may peek at the future.        -> causal sweep

Method
------
We build ONE continuous band-power stream over a reach-dense span (chunked, so the
15 GB file is never loaded whole), then derive windowed features cheaply by
averaging the per-sample band-power envelope over a window via a cumulative sum.
For each (window length L, lag, causal flag) we decode two targets:

  * movement-vs-rest  (curated /intervals/reaches) -> Logistic regression, AUC
  * wrist speed       (R_Wrist |velocity|)         -> Ridge regression, R^2

All scoring uses *blocked* time-series cross-validation (contiguous folds), so no
future sample helps predict a past one. A fixed margin is reserved at both ends so
every configuration is evaluated on the exact same set of prediction times, making
the curves directly comparable.

Run (dbs-ml env):
  python phase1_resolution.py --file "C:\\path\\to\\sub-01_ses-3_behavior+ecephys.nwb"
"""

import argparse
import glob
import hashlib
import os

import numpy as np

from nwb_dataset import (build_continuous_stream, find_active_window,
                         sensorimotor_channels)


# --------------------------------------------------------------------------- #
# Data / caching
# --------------------------------------------------------------------------- #
def find_default_file():
    cands = (glob.glob("*.nwb")
             + glob.glob(os.path.join("..", "*.nwb"))
             + glob.glob(os.path.join("ajile12-nwb-data", "**", "*.nwb"), recursive=True))
    return max(cands, key=os.path.getsize) if cands else None


def get_stream(path, t0, t1, out_rate, bands, channels, smooth_hz, out_dir):
    """Build (and cache) a continuous, *un*-z-scored band-power stream."""
    ch_tag = channels if isinstance(channels, str) else "sm{}".format(len(channels))
    key = "{}|{:.0f}|{:.0f}|{:.0f}|{}|{}|{:.0f}".format(
        os.path.basename(path), t0, t1, out_rate, ",".join(bands), ch_tag, smooth_hz)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    cache = os.path.join(out_dir, "stream_{}.npz".format(h))
    if os.path.exists(cache):
        print("loading cached stream:", cache)
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print("building continuous stream (chunked read of the large file)...")
    s = build_continuous_stream(path, t0, t1, out_rate=out_rate, bands=bands,
                                ecog_channels=channels, zscore=False,
                                smooth_hz=smooth_hz, verbose=False)
    np.savez_compressed(
        cache, X=s["X"], speed=s["speed"], reach=s["reach"],
        t=s["t"], out_rate=s["out_rate"])
    return {"X": s["X"], "speed": s["speed"], "reach": s["reach"],
            "t": s["t"], "out_rate": np.asarray(s["out_rate"])}


# --------------------------------------------------------------------------- #
# Windowed-feature builder (cumulative-sum trick) + scorers
# --------------------------------------------------------------------------- #
def windowed_features(csum, pred_idx, L_samp, lag_samp, causal):
    """Mean band-power over a window for each prediction index.

    csum : (T+1, C) cumulative sum of X (csum[0]=0), so mean over [a,b) is fast.
    Window is placed around a neural reference index ``ref = p - lag_samp``
    (lag>0 => neural leads behavior, i.e. an earlier neural window).
      causal   : trailing window [ref-L+1, ref+1)  (past-only)
      acausal  : centered window [ref-L//2, ref-L//2+L)
    Returns (X_feat (n, C)).
    """
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


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run(args):
    path = args.file or find_default_file()
    if not path or not os.path.exists(path):
        raise SystemExit("No NWB file found. Pass --file explicitly.")
    os.makedirs(args.out_dir, exist_ok=True)
    print("file:", path)

    dur = args.dur_min * 60.0
    if args.start is not None and args.start >= 0:
        t0, t1 = float(args.start), float(args.start) + dur
    else:
        t0, t1, nrch = find_active_window(path, dur_sec=dur, step_sec=300.0)
        print("reach-dense window: {:.0f}-{:.0f}s ({} reach onsets)".format(t0, t1, nrch))

    channels = "good"
    if args.channels == "sensorimotor":
        try:
            channels = sensorimotor_channels(path)
        except Exception as e:  # noqa: BLE001
            print("sensorimotor selection failed ({}); using good".format(e))
            channels = "good"

    bands = tuple(b.strip() for b in args.bands.split(",") if b.strip())
    stream = get_stream(path, t0, t1, args.out_rate, bands, channels,
                        args.smooth_hz, args.out_dir)
    X = stream["X"].astype(np.float64)
    fs = float(stream["out_rate"])
    reach = stream["reach"].astype(int)
    speed = stream["speed"]
    if speed.ndim == 2 and speed.shape[1]:
        speed = speed[:, 0].astype(np.float64)
        # robustify: occlusion-interpolation produces rare huge velocity spikes that
        # otherwise dominate the variance and make R^2 meaningless. Clip then log.
        cap = np.nanpercentile(speed, 99)
        speed = np.log1p(np.clip(speed, 0, cap))
    else:
        speed = None
    T, C = X.shape
    print("stream: {} samples x {} feat @ {:.0f} Hz; reach prevalence {:.3f}".format(
        T, C, fs, reach.mean()))

    csum = np.vstack([np.zeros((1, C)), np.cumsum(X, axis=0)])

    # parse sweeps
    win_list = [float(v) for v in args.windows.split(",")]
    lag_list = [round(v, 3) for v in np.arange(args.lag_min, args.lag_max + 1e-9, args.lag_step)]
    causal_L = [float(v) for v in args.causal_windows.split(",")]

    # common margin so every config shares one prediction set
    max_half = max(max(win_list), max(causal_L)) / 2.0
    margin = int(round((max_half + max(abs(args.lag_min), abs(args.lag_max))) * fs)) + 1
    stride = max(1, int(round(args.stride_sec * fs)))
    pred_idx = np.arange(margin, T - margin, stride)
    y_move = reach[pred_idx]
    y_speed = speed[pred_idx] if speed is not None else None
    print("prediction grid: {} windows (stride {:.2f}s); movement balance {:.3f}".format(
        len(pred_idx), args.stride_sec, y_move.mean()))

    def score(L, lag, causal):
        Ls = max(1, int(round(L * fs)))
        lg = int(round(lag * fs))
        Xf = windowed_features(csum, pred_idx, Ls, lg, causal)
        auc = blocked_auc(Xf, y_move, args.cv)
        r2 = blocked_r2(Xf, y_speed, args.cv) if y_speed is not None else np.nan
        return auc, r2

    # ---- H1.1 window-length sweep (causal, lag 0) ------------------------- #
    print("\n[H1.1] window-length sweep (causal, lag=0)")
    win_rows = []
    for L in win_list:
        auc, r2 = score(L, 0.0, True)
        win_rows.append((L, auc, r2))
        print("  L={:5.2f}s  AUC={:.3f}  R2={:+.3f}".format(L, auc, r2))
    win_rows = np.array(win_rows)

    # ---- H1.2 lag sweep (centered, fixed L) ------------------------------- #
    print("\n[H1.2] lag sweep (centered window L={:.2f}s)".format(args.lag_window))
    lag_rows = []
    for lag in lag_list:
        auc, r2 = score(args.lag_window, lag, False)
        lag_rows.append((lag, auc, r2))
        print("  lag={:+.2f}s  AUC={:.3f}  R2={:+.3f}".format(lag, auc, r2))
    lag_rows = np.array(lag_rows)

    # ---- H1.3 causal vs acausal (lag 0) ----------------------------------- #
    print("\n[H1.3] causal vs acausal (lag=0)")
    cz_rows = []
    for L in causal_L:
        a_c, r_c = score(L, 0.0, True)
        a_a, r_a = score(L, 0.0, False)
        cz_rows.append((L, a_c, a_a, r_c, r_a))
        print("  L={:5.2f}s  AUC causal={:.3f} acausal={:.3f} (cost {:+.3f})  "
              "R2 causal={:+.3f} acausal={:+.3f}".format(L, a_c, a_a, a_a - a_c, r_c, r_a))
    cz_rows = np.array(cz_rows)

    _summary(win_rows, lag_rows, cz_rows, args)
    _save(win_rows, lag_rows, cz_rows, args)


def _summary(win_rows, lag_rows, cz_rows, args):
    print("\n================ PHASE 1 SUMMARY ================")
    # H1.1: interior peak?
    Ls, aucs = win_rows[:, 0], win_rows[:, 1]
    bi = int(np.nanargmax(aucs))
    interior = 0 < bi < len(Ls) - 1
    print("H1.1 window length : best AUC {:.3f} at L={:.2f}s  -> {}".format(
        aucs[bi], Ls[bi], "INTERIOR peak (non-monotonic)" if interior
        else "peak at an extreme (monotonic in this range)"))
    # H1.2: non-zero lag peak?
    lags, laucs = lag_rows[:, 0], lag_rows[:, 1]
    li = int(np.nanargmax(laucs))
    best_lag = lags[li]
    print("H1.2 neural lead   : best AUC {:.3f} at lag={:+.2f}s  -> {}".format(
        laucs[li], best_lag,
        "non-zero lag (neural {} behavior)".format("leads" if best_lag > 0 else "lags")
        if abs(best_lag) > 1e-6 else "peak at lag 0"))
    # H1.3: causal cost
    cost = np.nanmean(cz_rows[:, 2] - cz_rows[:, 1])
    print("H1.3 causal cost   : mean acausal-minus-causal AUC = {:+.3f}  -> {}".format(
        cost, "causal costs accuracy" if cost > 0.005 else "no meaningful cost"))
    print("=================================================\n")


def _save(win_rows, lag_rows, cz_rows, args):
    np.savetxt(os.path.join(args.out_dir, "phase1_window.csv"), win_rows,
               delimiter=",", header="window_s,auc_move,r2_speed", comments="")
    np.savetxt(os.path.join(args.out_dir, "phase1_lag.csv"), lag_rows,
               delimiter=",", header="lag_s,auc_move,r2_speed", comments="")
    np.savetxt(os.path.join(args.out_dir, "phase1_causal.csv"), cz_rows,
               delimiter=",",
               header="window_s,auc_causal,auc_acausal,r2_causal,r2_acausal", comments="")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

        ax[0].plot(win_rows[:, 0], win_rows[:, 1], "-o", color="C0", label="movement AUC")
        ax[0].axhline(0.5, color="gray", ls=":", lw=1)
        ax[0].set_xscale("log")
        ax[0].set_xlabel("feature window length (s, log)")
        ax[0].set_ylabel("movement AUC", color="C0")
        ax0b = ax[0].twinx()
        ax0b.plot(win_rows[:, 0], win_rows[:, 2], "-s", color="C3", label="speed R2")
        ax0b.set_ylabel("wrist-speed R$^2$", color="C3")
        ax[0].set_title("H1.1  window length (causal, lag 0)")
        ax[0].grid(alpha=0.3)

        ax[1].plot(lag_rows[:, 0], lag_rows[:, 1], "-o", color="C0")
        ax[1].axvline(0.0, color="gray", ls=":", lw=1)
        bi = int(np.nanargmax(lag_rows[:, 1]))
        ax[1].axvline(lag_rows[bi, 0], color="C2", ls="--", lw=1,
                      label="peak {:+.2f}s".format(lag_rows[bi, 0]))
        ax[1].set_xlabel("neural-to-behavior lag (s)  [+ = neural leads]")
        ax[1].set_ylabel("movement AUC")
        ax[1].set_title("H1.2  lag (centered L={:.2f}s)".format(args.lag_window))
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)

        w = 0.35
        xpos = np.arange(len(cz_rows))
        ax[2].bar(xpos - w / 2, cz_rows[:, 1], w, label="causal", color="C0")
        ax[2].bar(xpos + w / 2, cz_rows[:, 2], w, label="acausal", color="C1")
        ax[2].set_xticks(xpos)
        ax[2].set_xticklabels(["{:.2f}".format(v) for v in cz_rows[:, 0]])
        ax[2].set_ylim(0.5, max(0.6, np.nanmax(cz_rows[:, 1:3]) + 0.02))
        ax[2].set_xlabel("window length (s)")
        ax[2].set_ylabel("movement AUC")
        ax[2].set_title("H1.3  causal vs acausal")
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3, axis="y")

        fig.suptitle("Phase 1 temporal-resolution sweep (AJILE12, bands={})".format(args.bands))
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, "phase1_resolution.png")
        fig.savefig(out_png, dpi=130)
        print("wrote", out_png)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default=None)
    ap.add_argument("--start", type=float, default=-1)
    ap.add_argument("--dur-min", type=float, default=45.0)
    ap.add_argument("--out-rate", type=float, default=50.0, help="base feature rate (Hz)")
    ap.add_argument("--smooth-hz", type=float, default=15.0,
                    help="envelope low-pass; high => window length, not pre-smoothing, drives integration")
    ap.add_argument("--channels", choices=["good", "sensorimotor"], default="sensorimotor")
    ap.add_argument("--bands", default="beta,high_gamma")
    ap.add_argument("--windows", default="0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0,3.0,4.0",
                    help="window lengths (s) for H1.1")
    ap.add_argument("--lag-window", type=float, default=0.5, help="fixed window for the lag sweep (s)")
    ap.add_argument("--lag-min", type=float, default=-1.0)
    ap.add_argument("--lag-max", type=float, default=1.0)
    ap.add_argument("--lag-step", type=float, default=0.1)
    ap.add_argument("--causal-windows", default="0.25,0.5,1.0,2.0",
                    help="window lengths (s) for the H1.3 causal/acausal comparison")
    ap.add_argument("--stride-sec", type=float, default=0.4, help="spacing of prediction times (s)")
    ap.add_argument("--cv", type=int, default=5, help="blocked CV folds")
    ap.add_argument("--out-dir", default="phase1_out")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
