"""Decompose the cross-patient transfer story: try to kill the global-vs-local explanation.

Per-advisor feedback ("global v. local figures to prove that, how to refute that, precise"),
this runs the ablations that would falsify the claim "a channel-agnostic spectral summary
transfers because it captures GLOBAL arousal, not LOCAL/spatial activity identity":

  A. Behavioral-granularity curve: cross-patient AUC as targets go from coarse (sleep vs
     active) to fine (6-way state). If the global-arousal account is right, AUC should fall
     off systematically with granularity.
  B. Spatial-information ablation: electrode-identity (naive index alignment, no anatomy) vs
     whole-brain anatomical ROI aggregation vs fully channel-agnostic spectral summary, on
     the SAME binary arousal target. Tests whether spatial information helps or hurts
     transfer.
  C. Frequency-band ablation: repeat the channel-agnostic transfer restricted to one band at
     a time (and combinations), to check the "alpha/beta biomarker" claim is not just an
     artifact of LDA weights but actually carries the transferable signal.
  D. Electrode-subsampling robustness: randomly subsample K of each patient's channels before
     building the channel-agnostic summary, repeated with resampling, to see whether transfer
     survives large reductions in montage size/coverage.
  E. Awake-rest control: Sleep vs Inactive (both motionless) isolates vigilance from gross
     motor confounds; Inactive vs Engaged isolates engagement from vigilance.
  F. Simple-baseline control: a single classic theta/beta "vigilance ratio" feature vs the
     35-feature spectral summary, on the same LOSO protocol -- tests whether the transferable
     signal is genuinely new or just classic vigilance physiology.

FEASIBILITY NOTE on a full 12-subject LOSO: AJILE12 is ~850 GB across 12 subjects x ~5
sessions; only 3 subjects (~26 GB) are downloaded, and only ~54 GB of disk is free (of a
953 GB volume already at 95% use) -- not enough headroom to add the remaining subjects. All
experiments below therefore run on the 3 locally available patients (sub-01, sub-06, sub-07)
as a bounded pilot. The script is written so more subjects are a one-line addition later
(drop more .nwb files into ajile12-nwb-data/ and rerun with --cache).

Because re-reading each 15+ GB file is the expensive step, this script caches ONE thing per
patient -- a (windows x channels x bands) log band-power cube, labels, window start times,
and electrode MNI coordinates -- and derives every ablation above from that cache in seconds.

Run (dbs-ml env):
  python phase4_decompose_transfer.py --cache        # one-time per-patient extraction (~10 min/patient)
  python phase4_decompose_transfer.py                 # reruns all ablations from cache (~seconds)
"""

import argparse
import csv
import os

import numpy as np

from nwb_dataset import good_channel_indices, electrode_coords, mni_to_aal
from phase3_eval import build_epoch_dataset, find_default_file
from phase3_crosssubject import find_subject_files
from phase3_manifold import STATE_BANDS, T5_LABELS, drop_outliers

N_BANDS = len(STATE_BANDS)  # (theta, alpha, beta, low_gamma, high_gamma)
PCTS = (10, 25, 50, 75, 90)

# Fixed, patient-independent macro-region vocabulary (whole-brain, not just sensorimotor).
# Every patient's electrodes are bucketed into this SAME set of region x hemisphere labels,
# so the ROI feature vector has identical layout across patients regardless of montage.
MACRO_KEYS = [
    ("Precentral", "Central"), ("Postcentral", "Central"), ("Rolandic", "Central"),
    ("Paracentral", "Central"), ("Supp_Motor", "Central"),
    ("Frontal", "Frontal"), ("Temporal", "Temporal"), ("Parietal", "Parietal"),
    ("Occipital", "Occipital"), ("Insula", "Insula"), ("Cingulum", "Cingulate"),
]
MACRO_REGIONS = sorted({"{}_{}".format(lab, h) for _, lab in MACRO_KEYS for h in ("L", "R")}
                       | {"Other_L", "Other_R", "Other_U"})


def macro_region(aal_name):
    hemi = "R" if aal_name.endswith("_R") else ("L" if aal_name.endswith("_L") else "U")
    base = aal_name.replace("_L", "").replace("_R", "")
    for key, label in MACRO_KEYS:
        if key.lower() in base.lower():
            return "{}_{}".format(label, hemi)
    return "Other_{}".format(hemi)


# --------------------------------------------------------------------------- #
# Caching (the one expensive step)
# --------------------------------------------------------------------------- #
def cache_path(cache_dir, sid):
    return os.path.join(cache_dir, "cube_sub{}.npz".format(sid))


def build_cache(path, sid, cache_dir, window_sec=10.0, max_per_label=150, seed=0):
    with __import__("h5py").File(path, "r") as f:
        good = good_channel_indices(f)
    X, y, t0s = build_epoch_dataset(path, good, STATE_BANDS, window_sec=window_sec,
                                    max_per_label=max_per_label, label_set=set(T5_LABELS),
                                    seed=seed, verbose=False)
    keep = drop_outliers(X)
    X, y, t0s = X[keep], np.asarray(y)[keep], np.asarray(t0s)[keep]
    cube = X.reshape(len(X), len(good), N_BANDS).astype(np.float32)

    xyz, _ = electrode_coords(path)
    xyz = xyz[good]
    try:
        aal_names = mni_to_aal(xyz)
    except Exception as e:  # noqa: BLE001
        print("  AAL mapping failed for sub-{} ({}); ROI ablation will be skipped".format(
            sid, type(e).__name__))
        aal_names = ["unknown"] * len(good)
    macro = np.array([macro_region(n) for n in aal_names])

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path(cache_dir, sid), cube=cube, y=y, t0s=t0s, macro=macro,
                        n_ch=len(good))
    print("  cached sub-{}: cube {}  states {}".format(sid, cube.shape, sorted(set(y))))


def load_cache(cache_dir, sid):
    d = np.load(cache_path(cache_dir, sid), allow_pickle=True)
    return d["cube"], d["y"], d["t0s"], d["macro"]


# --------------------------------------------------------------------------- #
# Feature builders (all derived from the cached cube, no NWB access)
# --------------------------------------------------------------------------- #
def spectral_summary(cube, band_idx=None):
    """Channel-agnostic: per-band mean/std/percentiles across channels."""
    band_idx = band_idx if band_idx is not None else range(N_BANDS)
    feats = []
    for bi in band_idx:
        band = cube[:, :, bi]
        feats.append(band.mean(axis=1)); feats.append(band.std(axis=1))
        for p in PCTS:
            feats.append(np.percentile(band, p, axis=1))
    return np.column_stack(feats)


def naive_identity_features(cube, k):
    """Electrode-IDENTITY features: the first k channel columns, index-aligned across
    patients with no anatomical correspondence (channel #3 of patient A treated as the
    'same feature' as channel #3 of patient B). A deliberately naive spatial baseline."""
    k = min(k, cube.shape[1])
    return cube[:, :k, :].reshape(len(cube), -1)


def roi_features(cube, macro, regions=MACRO_REGIONS):
    """Anatomically-grounded ROI aggregation: mean band power per macro-region x hemisphere,
    NaN for regions this patient has no coverage in (imputed later at the pooling step)."""
    T = cube.shape[0]
    feat = np.full((T, len(regions) * N_BANDS), np.nan, dtype=np.float32)
    for ri, r in enumerate(regions):
        m = macro == r
        if m.any():
            block = cube[:, m, :].mean(axis=1)  # (T, N_BANDS)
            feat[:, ri * N_BANDS:(ri + 1) * N_BANDS] = block
    return feat


def impute_pool(train_list, test):
    """Column-wise mean impute using the pooled training patients' statistics."""
    pool = np.vstack(train_list)
    mu = np.nanmean(pool, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    train_imp = [np.where(np.isnan(t), mu, t) for t in train_list]
    test_imp = np.where(np.isnan(test), mu, test)
    return train_imp, test_imp


def vigilance_ratio(cube):
    """Classic single-feature vigilance index: mean theta power - mean beta power across
    channels (log power, so this is a log theta/beta ratio). Higher = drowsier."""
    theta_i, beta_i = STATE_BANDS.index("theta"), STATE_BANDS.index("beta")
    return (cube[:, :, theta_i].mean(axis=1) - cube[:, :, beta_i].mean(axis=1))[:, None]


# --------------------------------------------------------------------------- #
# Regrouping labels for the granularity curve
# --------------------------------------------------------------------------- #
def regroup(y, level):
    if level == "L1_binary":
        return np.where(y == "Sleep/rest", "Rest", "Active")
    if level == "L2_3class":
        m = {"Sleep/rest": "Sleep", "Inactive": "InactiveAwake"}
        return np.array([m.get(v, "Engaged") for v in y])
    if level == "L3_4class":
        m = {"Sleep/rest": "Sleep", "Inactive": "InactiveAwake",
            "TV": "Passive", "Computer/phone": "Passive",
            "Talk": "Communicative", "Talk, TV": "Communicative"}
        return np.array([m.get(v, "Other") for v in y])
    if level == "L4_full":
        return np.asarray(y)
    raise ValueError(level)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score(train_F, train_y, test_F, test_y):
    """Binary -> AUC; multi-class (>=3) -> one-vs-rest macro-AUC. NaN if a class is
    missing from either side (transfer undefined for that class set)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    classes = sorted(set(train_y) & set(test_y))
    if len(classes) < 2:
        return float("nan")
    mtr, mte = np.isin(train_y, classes), np.isin(test_y, classes)
    if len(set(train_y[mtr])) < len(classes) or len(set(test_y[mte])) < len(classes):
        return float("nan")
    sc = StandardScaler().fit(train_F[mtr])
    lda = LinearDiscriminantAnalysis().fit(sc.transform(train_F[mtr]), train_y[mtr])
    if len(classes) == 2:
        proba = lda.predict_proba(sc.transform(test_F[mte]))
        pos = list(lda.classes_).index(classes[1]) if classes[1] in lda.classes_ else 1
        return float(roc_auc_score((test_y[mte] == classes[1]).astype(int), proba[:, pos]))
    order = list(lda.classes_)
    proba = lda.predict_proba(sc.transform(test_F[mte]))
    proba = proba[:, [order.index(c) for c in classes]]
    return float(roc_auc_score(test_y[mte], proba, multi_class="ovr", average="macro",
                               labels=classes))


MIN_CLASS_N = 10  # a patient needs >=this many windows of a class to usefully teach it


def usable_subjects(y_by_sub, min_n=MIN_CLASS_N):
    """Subjects whose windows cover >=2 classes with >=min_n examples each. Excluding
    single-class patients from the pool matters: a patient with e.g. zero Sleep windows
    (sub-04, whose ECoG recording ends before its sleep period) still gets pooled into
    every OTHER patient's training set if not filtered, which skews the pooled
    StandardScaler/LDA fit toward that patient's one-sided class and measurably changes
    the transfer AUC -- this was caught by comparing against the original 3-patient
    result and is the reason this filter exists."""
    usable = {}
    for s, y in y_by_sub.items():
        vals, counts = np.unique(y, return_counts=True)
        if (counts >= min_n).sum() >= 2:
            usable[s] = True
    return list(usable)


def loso(feat_by_sub, y_by_sub, impute=False, min_n=MIN_CLASS_N):
    """Leave-one-patient-out mean transfer score, restricted to subjects with usable
    class coverage for this target (see usable_subjects) -- both as held-out test
    patients and as training-pool contributors."""
    subs = usable_subjects(y_by_sub, min_n=min_n)
    dropped = [s for s in feat_by_sub if s not in subs]
    aucs = {}
    for held in subs:
        train_subs = [s for s in subs if s != held]
        if not train_subs:
            aucs[held] = float("nan")
            continue
        Ftr_list = [feat_by_sub[s] for s in train_subs]
        Fte = feat_by_sub[held]
        if impute:
            Ftr_list, Fte = impute_pool(Ftr_list, Fte)
        Ftr = np.vstack(Ftr_list)
        ytr = np.concatenate([y_by_sub[s] for s in train_subs])
        aucs[held] = score(Ftr, ytr, Fte, y_by_sub[held])
    for s in dropped:
        aucs.setdefault(s, float("nan"))
    return aucs, dropped


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def _mean(aucs):
    vals = [v for v in aucs.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def within_patient_ceiling(feat, y, seed=0):
    """Same-feature, same-target within-patient upper bound: random 70/30 split of one
    patient's own windows, scored with the identical score() protocol used for LOSO. This
    isolates the cross-patient transfer gap from any change in feature representation --
    i.e. it answers "is the bottleneck cross-patient generalization, or decodability at
    all," using the exact channel-agnostic feature the transfer test itself uses."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    return score(feat[tr], y[tr], feat[te], y[te])


def exp_granularity(cubes, ys):
    print("\n=== A. Behavioral-granularity curve (cross-patient LOSO vs. same-feature within-patient ceiling) ===")
    rows = []
    for level in ("L1_binary", "L2_3class", "L3_4class", "L4_full"):
        feat = {s: spectral_summary(cubes[s]) for s in cubes}
        yg = {s: regroup(ys[s], level) for s in ys}
        n_classes = len(set.union(*[set(yg[s]) for s in yg]))
        aucs, dropped = loso(feat, yg)
        mean_auc = _mean(aucs)

        usable = usable_subjects(yg)
        within_aucs = {s: within_patient_ceiling(feat[s], yg[s]) for s in usable}
        mean_within = _mean(within_aucs)

        print("  {:12s} (~{} classes): per-patient LOSO {}  mean LOSO = {:.3f}  |  mean within-patient (same feature) = {:.3f}{}".format(
            level, n_classes, {k: round(v, 3) for k, v in aucs.items() if np.isfinite(v)},
            mean_auc, mean_within, "  [excluded: {}]".format(dropped) if dropped else ""))
        rows.append({"level": level, "n_classes": n_classes, "mean_loso_auc": round(mean_auc, 4),
                    "mean_within_auc": round(mean_within, 4) if np.isfinite(mean_within) else "",
                    "excluded": ",".join(dropped),
                    **{"sub{}".format(k): round(v, 4) if np.isfinite(v) else "" for k, v in aucs.items()}})
    return rows


def exp_spatial(cubes, ys, macros):
    print("\n=== B. Spatial-information ablation (binary Sleep vs Active) ===")
    yb = {s: regroup(ys[s], "L1_binary") for s in ys}
    usable = usable_subjects(yb)
    min_ch = min(cubes[s].shape[1] for s in usable)
    rows = []

    feat_id = {s: naive_identity_features(cubes[s], min_ch) for s in cubes}
    a_id, d_id = loso(feat_id, yb)
    print("  electrode-identity (naive index align, k={} ch): {}  mean={:.3f}{}".format(
        min_ch, {k: round(v, 3) for k, v in a_id.items() if np.isfinite(v)}, _mean(a_id),
        "  [excluded: {}]".format(d_id) if d_id else ""))
    rows.append({"representation": "electrode_identity_naive", "n_features": min_ch * N_BANDS,
                "mean_loso_auc": round(_mean(a_id), 4), "excluded": ",".join(d_id)})

    feat_roi = {s: roi_features(cubes[s], macros[s]) for s in cubes}
    a_roi, d_roi = loso(feat_roi, yb, impute=True)
    print("  anatomical ROI aggregation ({} regions): {}  mean={:.3f}{}".format(
        len(MACRO_REGIONS), {k: round(v, 3) for k, v in a_roi.items() if np.isfinite(v)}, _mean(a_roi),
        "  [excluded: {}]".format(d_roi) if d_roi else ""))
    rows.append({"representation": "anatomical_roi", "n_features": len(MACRO_REGIONS) * N_BANDS,
                "mean_loso_auc": round(_mean(a_roi), 4), "excluded": ",".join(d_roi)})

    feat_ch = {s: spectral_summary(cubes[s]) for s in cubes}
    a_ch, d_ch = loso(feat_ch, yb)
    print("  channel-agnostic spectral summary: {}  mean={:.3f}{}".format(
        {k: round(v, 3) for k, v in a_ch.items() if np.isfinite(v)}, _mean(a_ch),
        "  [excluded: {}]".format(d_ch) if d_ch else ""))
    rows.append({"representation": "channel_agnostic", "n_features": 35,
                "mean_loso_auc": round(_mean(a_ch), 4), "excluded": ",".join(d_ch)})
    return rows


def exp_frequency(cubes, ys):
    print("\n=== C. Frequency-band ablation (binary Sleep vs Active, channel-agnostic) ===")
    yb = {s: regroup(ys[s], "L1_binary") for s in ys}
    rows = []
    combos = {b: (i,) for i, b in enumerate(STATE_BANDS)}
    combos["alpha+beta"] = (STATE_BANDS.index("alpha"), STATE_BANDS.index("beta"))
    combos["all_5_bands"] = tuple(range(N_BANDS))
    for name, idx in combos.items():
        feat = {s: spectral_summary(cubes[s], band_idx=idx) for s in cubes}
        aucs, dropped = loso(feat, yb)
        m = _mean(aucs)
        print("  {:14s}: mean LOSO AUC = {:.3f}".format(name, m))
        rows.append({"bands": name, "mean_loso_auc": round(m, 4)})
    return rows


def exp_subsampling(cubes, ys, seed=0, ks=(10, 20, 30, 45, 60), n_repeat=5):
    print("\n=== D. Electrode-subsampling robustness (binary, channel-agnostic) ===")
    yb = {s: regroup(ys[s], "L1_binary") for s in ys}
    usable = usable_subjects(yb)
    rng = np.random.default_rng(seed)
    rows = []
    max_k = min(cubes[s].shape[1] for s in usable)
    for k in [k for k in ks if k <= max_k] + [max_k]:
        rep_means = []
        for r in range(n_repeat):
            feat = {}
            for s, cube in cubes.items():
                idx = rng.choice(cube.shape[1], size=k, replace=False)
                feat[s] = spectral_summary(cube[:, idx, :])
            aucs, _dropped = loso(feat, yb)
            rep_means.append(_mean(aucs))
        print("  k={:3d} channels: mean LOSO AUC = {:.3f} +/- {:.3f}  (over {} resamples)".format(
            k, np.mean(rep_means), np.std(rep_means), n_repeat))
        rows.append({"n_channels": k, "mean_loso_auc": round(float(np.mean(rep_means)), 4),
                    "std_loso_auc": round(float(np.std(rep_means)), 4)})
    return rows


def exp_awake_rest(cubes, ys):
    print("\n=== E. Awake-rest control (isolate vigilance from motion / engagement) ===")
    rows = []
    for name, keep_labels, pos in [
        ("Sleep_vs_Inactive (both motionless)", {"Sleep/rest", "Inactive"}, "Sleep/rest"),
        ("Inactive_vs_Engaged (both awake)", {"Inactive", "Talk", "TV", "Computer/phone", "Talk, TV"}, "Inactive"),
    ]:
        yfilt, feat = {}, {}
        for s in cubes:
            mask = np.isin(ys[s], list(keep_labels))
            if mask.sum() < 10:
                continue
            lab = np.where(ys[s][mask] == pos, "A", "B")
            if len(set(lab)) < 2:
                continue
            yfilt[s] = lab
            feat[s] = spectral_summary(cubes[s][mask])
        if len(yfilt) < 3:
            print("  {}: insufficient patients with both classes, skipped".format(name))
            continue
        aucs, dropped = loso(feat, yfilt)
        m = _mean(aucs)
        print("  {}: per-patient {}  mean LOSO AUC = {:.3f}{}".format(
            name, {k: round(v, 3) for k, v in aucs.items() if np.isfinite(v)}, m,
            "  [excluded: {}]".format(dropped) if dropped else ""))
        rows.append({"contrast": name, "mean_loso_auc": round(m, 4)})
    return rows


def exp_simple_baseline(cubes, ys):
    print("\n=== F. Simple-baseline control (binary Sleep vs Active) ===")
    yb = {s: regroup(ys[s], "L1_binary") for s in ys}
    feat_simple = {s: vigilance_ratio(cubes[s]) for s in cubes}
    a_simple, d_s = loso(feat_simple, yb)
    feat_full = {s: spectral_summary(cubes[s]) for s in cubes}
    a_full, d_f = loso(feat_full, yb)
    m_s, m_f = _mean(a_simple), _mean(a_full)
    print("  theta/beta vigilance ratio (1 feature): mean LOSO AUC = {:.3f}  ({})".format(
        m_s, {k: round(v, 3) for k, v in a_simple.items() if np.isfinite(v)}))
    print("  full 35-feature spectral summary       : mean LOSO AUC = {:.3f}  ({})".format(
        m_f, {k: round(v, 3) for k, v in a_full.items() if np.isfinite(v)}))
    print("  delta (full - simple) = {:+.3f}".format(m_f - m_s))
    return [{"model": "theta_beta_ratio_1feat", "mean_loso_auc": round(m_s, 4)},
            {"model": "spectral_summary_35feat", "mean_loso_auc": round(m_f, 4)}]


def save_csv(rows, path):
    if not rows:
        print("(nothing to write for {})".format(path)); return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--extra-root", default="ajile12-nwb-data")
    ap.add_argument("--cache-dir", default="phase4_cache")
    ap.add_argument("--out-dir", default="phase4_decompose")
    ap.add_argument("--cache", action="store_true", help="(re)build the per-patient cube cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = find_subject_files(args.extra_root, args.nwb or find_default_file())
    print("subjects found:", list(files))

    if args.cache:
        for sid, path in files.items():
            print("[cache] sub-{}".format(sid))
            build_cache(path, sid, args.cache_dir, seed=args.seed)

    # Load from cache for every subject that HAS a cache file, regardless of whether the raw
    # NWB path was rediscovered this run (find_default_file() does not search the parent
    # Downloads folder where the primary sub-01 file lives, so relying on `files` alone would
    # silently drop it once --nwb is omitted after the initial --cache pass).
    import glob
    cached_ids = sorted({os.path.basename(p).split("cube_sub")[1].split(".npz")[0]
                         for p in glob.glob(os.path.join(args.cache_dir, "cube_sub*.npz"))})
    all_ids = sorted(set(files) | set(cached_ids))
    print("cached on disk:", cached_ids, " | union with discovered NWB files:", all_ids)

    cubes, ys, t0s_by, macros = {}, {}, {}, {}
    for sid in all_ids:
        if not os.path.exists(cache_path(args.cache_dir, sid)):
            print("no cache for sub-{}; skipping (run with --cache first)".format(sid))
            continue
        c, y, t0s, m = load_cache(args.cache_dir, sid)
        cubes[sid], ys[sid], t0s_by[sid], macros[sid] = c, y, t0s, m
    if len(cubes) < 3:
        raise SystemExit("Need >=3 cached patients.")
    print("loaded cached cubes for:", list(cubes), "shapes:", {s: cubes[s].shape for s in cubes})

    save_csv(exp_granularity(cubes, ys), os.path.join(args.out_dir, "granularity_curve.csv"))
    save_csv(exp_spatial(cubes, ys, macros), os.path.join(args.out_dir, "spatial_ablation.csv"))
    save_csv(exp_frequency(cubes, ys), os.path.join(args.out_dir, "frequency_ablation.csv"))
    save_csv(exp_subsampling(cubes, ys, seed=args.seed), os.path.join(args.out_dir, "electrode_subsampling.csv"))
    save_csv(exp_awake_rest(cubes, ys), os.path.join(args.out_dir, "awake_rest_control.csv"))
    save_csv(exp_simple_baseline(cubes, ys), os.path.join(args.out_dir, "simple_baseline.csv"))

    print("\nDone. Outputs in", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
