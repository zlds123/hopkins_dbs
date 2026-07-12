"""Phase 2 feasibility probe: does AJILE12 support causal online adaptation?

The Phase 2 claim ("an adaptive decoder beats a static one under drift") is only
meaningful if the data actually drifts. This script tests both preconditions on a
single AJILE12 file, with a simple, rigorous *prequential* (test-then-train)
streaming protocol on a movement-vs-rest target that is already known to be
ECoG-decodable.

Three decoders are streamed in true time order over sequential blocks:

  1. static       - logistic regression fit once on a warm-up span, then frozen.
  2. online-SGD   - logistic SGD that updates its weights after each block
                    (partial_fit); normalization fixed from warm-up, so weight
                    adaptation alone must absorb the drift.
  3. sliding-refit- logistic regression fully re-fit (weights + normalization)
                    on the most recent K blocks before each evaluation.

For every block (after warm-up) each decoder is first *evaluated* (AUC) and only
then allowed to learn from that block's labels -- i.e. it never sees the future.

Feasibility verdict (printed + plotted):
  * Drift exists  if the static decoder's block AUC declines over time and the
    feature distribution shifts away from the warm-up mean.
  * Adaptation helps if online-SGD and/or sliding-refit keep higher mean AUC than
    static over the streamed period.

Run (in the dbs-ml env):
  python phase2_feasibility.py --file path\to\sub-XX_ses-Y_..._ecephys.nwb
"""

import argparse
import glob
import os

import numpy as np

from nwb_dataset import (build_continuous_stream, find_active_window,
                         find_movement_window, sensorimotor_channels)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_default_file():
    """Pick the largest .nwb in the current folder as a convenience default."""
    cands = glob.glob("*.nwb") + glob.glob(os.path.join("ajile12-nwb-data", "**", "*.nwb"),
                                           recursive=True)
    if not cands:
        return None
    return max(cands, key=lambda p: os.path.getsize(p))


def movement_label(stream, kind="reach", lo_pct=40, hi_pct=60):
    """Binary movement-vs-rest target.

    kind="reach": use the curated /intervals/reaches annotation (1 inside a reach).
    kind="speed": threshold R_Wrist speed, dropping the ambiguous middle band.

    Returns (y, keep) where keep masks the samples used.
    """
    if kind == "reach" or stream["speed"].shape[1] == 0:
        y = stream["reach"].astype(int)
        return y, np.ones_like(y, dtype=bool)
    s = stream["speed"][:, 0]  # R_Wrist is the first requested keypoint
    finite = s[np.isfinite(s) & (s > 0)]
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    y = np.full(s.shape[0], -1, dtype=int)
    y[s <= lo] = 0
    y[s >= hi] = 1
    keep = y >= 0
    return y, keep


def block_auc(model, scaler, Xb, yb):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(yb)) < 2:
        return np.nan
    p = model.predict_proba(scaler.transform(Xb))[:, 1]
    return float(roc_auc_score(yb, p))


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def run(args):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, SGDClassifier

    path = args.file or find_default_file()
    if not path or not os.path.exists(path):
        raise SystemExit("No NWB file found. Pass --file explicitly.")
    print("file:", path)

    # ---- choose a long, active span so the target is decodable ------------ #
    dur = args.dur_min * 60.0
    if args.start is not None and args.start >= 0:
        t0, t1 = float(args.start), float(args.start) + dur
    elif args.anchor == "reach":
        t0, t1, nrch = find_active_window(path, dur_sec=dur, step_sec=300.0)
        print("auto reach-dense window: {:.0f}-{:.0f}s ({} reach onsets)".format(t0, t1, nrch))
    else:
        t0, t1, score = find_movement_window(path, dur_sec=dur, step_sec=120.0)
        print("auto movement window: {:.0f}-{:.0f}s (movement score {:.2f})".format(t0, t1, score))

    channels = "good"
    if args.channels == "sensorimotor":
        try:
            channels = sensorimotor_channels(path)
        except Exception as e:  # noqa: BLE001
            print("sensorimotor selection failed ({}); using good channels".format(e))
            channels = "good"

    # zscore=False on purpose: global z-scoring would use *future* statistics and
    # partially erase the very drift we want to measure. We normalize causally.
    print("building continuous stream (this reads the 15 GB file in chunks)...")
    bands = tuple(b.strip() for b in args.bands.split(",") if b.strip())
    stream = build_continuous_stream(
        path, t0, t1, out_rate=args.out_rate, bands=bands,
        ecog_channels=channels, zscore=False, verbose=False,
    )
    X = stream["X"]
    fs_out = stream["out_rate"]
    y, keep = movement_label(stream, kind=args.label)
    X, y = X[keep], y[keep]
    t = stream["t"][keep]
    print("stream: {} samples x {} features @ {:.0f} Hz; class balance {:.2f}".format(
        X.shape[0], X.shape[1], fs_out, y.mean()))

    # ---- partition into sequential time blocks ---------------------------- #
    block = int(round(args.block_min * 60 * fs_out))
    n_blocks = X.shape[0] // block
    if n_blocks < 6:
        raise SystemExit("Too few blocks ({}). Increase --dur-min or --out-rate.".format(n_blocks))
    warm = max(1, int(round(args.warmup_min / args.block_min)))
    print("{} blocks of {:.0f} min; {} warm-up blocks; sliding window K={}".format(
        n_blocks, args.block_min, warm, args.sliding_k))

    def blk(i):
        sl = slice(i * block, (i + 1) * block)
        return X[sl], y[sl]

    # ---- warm-up: fit static + scaler + seed online model ----------------- #
    Xw = np.vstack([blk(i)[0] for i in range(warm)])
    yw = np.concatenate([blk(i)[1] for i in range(warm)])
    scaler = StandardScaler().fit(Xw)
    static = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(
        scaler.transform(Xw), yw)

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=yw)
    online = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="optimal",
                           class_weight={0: cw[0], 1: cw[1]})
    online.partial_fit(scaler.transform(Xw), yw, classes=np.array([0, 1]))

    # ---- optional induced-drift positive control -------------------------- #
    # A progressive random *rotation* in standardized feature space, growing from
    # identity (warm-up) to a full orthonormal rotation by the end. This conserves
    # the information (a re-fit decoder can fully recover) but reorganizes the
    # representation, so a decoder with frozen weights is scrambled. This is the
    # correct positive control: it shows adaptation matters *when* drift reorganizes
    # the signal -- unlike a gain/offset, which is largely AUC-preserving for a
    # linear model. strength=0 -> observed (uninduced) drift only.
    rng2 = np.random.default_rng(1)
    d = X.shape[1]
    strength = float(args.induce_drift)
    eye = np.eye(d, dtype=np.float32)
    Rrot, _ = np.linalg.qr(rng2.normal(size=(d, d)))
    Rrot = Rrot.astype(np.float32)
    mu_w = scaler.mean_.astype(np.float32)
    sd_w = scaler.scale_.astype(np.float32)

    def apply_drift(Xb, frac):
        if strength <= 0:
            return Xb
        f = min(1.0, strength * frac)
        M = (1.0 - f) * eye + f * Rrot
        z = (Xb - mu_w) / sd_w
        return ((z @ M) * sd_w + mu_w).astype(np.float32)

    # ---- stream evaluation (test-then-train) ------------------------------ #
    rows, buf_X, buf_y = [], [blk(i)[0] for i in range(warm)], [blk(i)[1] for i in range(warm)]
    n_eval = n_blocks - warm
    for i in range(warm, n_blocks):
        Xb_raw, yb = blk(i)
        frac = (i - warm + 1) / n_eval
        Xb = apply_drift(Xb_raw, frac)

        auc_static = block_auc(static, scaler, Xb, yb)
        auc_online = block_auc(online, scaler, Xb, yb)

        # sliding-refit: full retrain (own scaler) on last K (drifted) blocks
        kX = np.vstack(buf_X[-args.sliding_k:])
        kY = np.concatenate(buf_y[-args.sliding_k:])
        auc_slide = np.nan
        if len(np.unique(kY)) == 2:
            sc_k = StandardScaler().fit(kX)
            slide = LogisticRegression(max_iter=1000, C=1.0,
                                       class_weight="balanced").fit(sc_k.transform(kX), kY)
            auc_slide = block_auc(slide, sc_k, Xb, yb)

        # standardized drift: mean per-feature shift from warm-up, in SD units
        zshift = (Xb.mean(axis=0) - scaler.mean_) / scaler.scale_
        drift = float(np.sqrt(np.mean(zshift ** 2)))
        rows.append((i * args.block_min, auc_static, auc_online, auc_slide, drift))

        # now (and only now) learn from this (drifted) block
        if len(np.unique(yb)) == 2:
            online.partial_fit(scaler.transform(Xb), yb)
        buf_X.append(Xb)
        buf_y.append(yb)

    rows = np.array(rows, dtype=float)
    tmin, a_stat, a_onl, a_sld, drift = rows.T

    # ---- shuffled-label null (sanity) ------------------------------------- #
    rng = np.random.default_rng(0)
    ysh = rng.permutation(yw)
    null = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(Xw), ysh)
    null_auc = np.nanmean([block_auc(null, scaler, *blk(i)) for i in range(warm, n_blocks)])

    # ---- verdict ---------------------------------------------------------- #
    def m(a):
        return float(np.nanmean(a))

    # drift: does static decline? slope of static AUC vs time, + drift growth
    slope = np.polyfit(tmin, np.nan_to_num(a_stat, nan=np.nanmean(a_stat)), 1)[0]
    early = m(a_stat[: len(a_stat) // 2])
    late = m(a_stat[len(a_stat) // 2:])

    print("\n================ FEASIBILITY SUMMARY ================")
    print("mean block AUC   static={:.3f}  online-SGD={:.3f}  sliding-refit={:.3f}  (null={:.3f})".format(
        m(a_stat), m(a_onl), m(a_sld), null_auc))
    print("static AUC  first-half={:.3f}  second-half={:.3f}  slope={:+.4f}/min".format(early, late, slope))
    print("feature drift  start={:.2f}  end={:.2f} SD  (mean per-feature shift vs warm-up)".format(
        drift[0], drift[-1]))
    mode = "INDUCED drift (strength {:.1f})".format(strength) if strength > 0 else "OBSERVED drift only"
    print("mode: {}".format(mode))
    # judge decodability by the best *adaptive* decoder: the static one is exactly
    # what drift is expected to break, so it must not gate "is there signal here".
    best_adaptive = max(m(a_onl), m(a_sld))
    decodable = best_adaptive > null_auc + 0.05
    drift_hurts_static = (late < early - 0.02) or (m(a_stat) < best_adaptive - 0.03) or (drift[-1] > 0.30)
    adapt_recovers = (m(a_sld) > m(a_stat) + 0.02) or (m(a_onl) > m(a_stat) + 0.02)
    print("target decodable:   {}".format("YES" if decodable else "NO (decoder ~ null)"))
    print("static decays/drift:{}".format("YES" if drift_hurts_static else "weak/none"))
    print("adaptation recovers:{}".format("YES" if adapt_recovers else "no"))
    if not decodable:
        verdict = "NEEDS A BETTER TARGET/WINDOW"
    elif drift_hurts_static and adapt_recovers:
        verdict = "YES"
    elif not drift_hurts_static:
        verdict = "DECODABLE BUT NO DRIFT IN THIS SPAN (try cross-session or --induce-drift)"
    else:
        verdict = "INCONCLUSIVE"
    print("PHASE 2 FEASIBLE:   {}".format(verdict))
    print("====================================================\n")

    # ---- plot + csv ------------------------------------------------------- #
    os.makedirs(args.out_dir, exist_ok=True)
    np.savetxt(os.path.join(args.out_dir, "phase2_feasibility.csv"), rows,
               delimiter=",", header="t_min,auc_static,auc_online,auc_sliding,feature_drift",
               comments="")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
        ax[0].plot(tmin, a_stat, "-o", ms=3, label="static (frozen)")
        ax[0].plot(tmin, a_onl, "-o", ms=3, label="online-SGD (adaptive)")
        ax[0].plot(tmin, a_sld, "-o", ms=3, label="sliding-refit (K={})".format(args.sliding_k))
        ax[0].axhline(0.5, color="gray", ls=":", lw=1, label="chance")
        ax[0].axhline(null_auc, color="crimson", ls="--", lw=1, label="shuffled null")
        ax[0].set_ylabel("block AUC (movement vs rest)")
        ax[0].set_title("Phase 2 feasibility: streaming decoders on AJILE12  [{}]".format(mode))
        ax[0].legend(loc="lower left", fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].plot(tmin, drift, "-o", ms=3, color="purple")
        ax[1].set_ylabel("feature drift")
        ax[1].set_xlabel("time into recording (min)")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, "phase2_feasibility.png")
        fig.savefig(out_png, dpi=130)
        print("wrote", out_png)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default=None, help="NWB file (default: largest .nwb found)")
    ap.add_argument("--start", type=float, default=-1, help="span start (s); <0 = auto movement-rich")
    ap.add_argument("--dur-min", type=float, default=90.0, help="span length (minutes)")
    ap.add_argument("--out-rate", type=float, default=20.0, help="feature/grid rate (Hz)")
    ap.add_argument("--label", choices=["reach", "speed"], default="reach",
                    help="target: curated reach annotation or balanced wrist-speed threshold")
    ap.add_argument("--anchor", choices=["reach", "movement"], default="reach",
                    help="auto window selection: reach-dense or wrist-movement-rich")
    ap.add_argument("--channels", choices=["good", "sensorimotor"], default="good")
    ap.add_argument("--bands", default="beta,high_gamma",
                    help="comma-separated bands for features (e.g. 'beta,high_gamma')")
    ap.add_argument("--induce-drift", type=float, default=0.0,
                    help="positive-control non-stationarity strength (0=off, ~1=strong)")
    ap.add_argument("--block-min", type=float, default=3.0, help="evaluation block length (minutes)")
    ap.add_argument("--warmup-min", type=float, default=15.0, help="initial training span (minutes)")
    ap.add_argument("--sliding-k", type=int, default=5, help="blocks in the sliding-refit window")
    ap.add_argument("--out-dir", default="phase2_out")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
