"""Phase 3 cross-subject transfer (H3.3) -- leave-one-subject-out decoding.

Addresses mentor note #4 ("more patients") and the handoff's blocking gap. Unlike the
within-subject cross-span CKA in ``phase3_eval.py``, this trains on a *pool* of subjects
and tests on a *held-out* subject -- the real cross-patient generalization question.

The hard part is that every subject has a different electrode layout, so channel-space
features are not comparable across people. Following the HTNet idea (Peterson, Rao &
Brunton 2021 -- see PRIOR_WORK_AJILE12.md), we project each subject's electrodes onto a
*fixed set of anatomical ROIs* (AAL sensorimotor regions) and average band-power within
each ROI. That yields an identical-dimension feature vector per subject regardless of how
many electrodes they have or where. A subject with no coverage in an ROI gets a missing
column, imputed to the training pool's mean (and such subjects legitimately transfer
poorly -- that's a finding about electrode placement, not a bug).

Models:
  M0  region band-power (linear decode)               -- the cross-subject baseline.
  M3  two-tower InfoNCE on the pooled region features  -- optional (--with-twotower);
      the towers have fixed input dim thanks to ROI projection, so a single encoder can
      train across subjects and be applied to the held-out one.

Run (dbs-ml env), after downloading extra subjects into ajile12-nwb-data/:
  python phase3_crosssubject.py --stage smoke
  python phase3_crosssubject.py --stage core --with-twotower
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

from nwb_dataset import (build_continuous_stream, find_active_window, find_movement_window,
                        good_channel_indices, electrode_coords, mni_to_aal, BANDS)

# Base AAL sensorimotor region names (hemisphere-collapsed) -> fixed feature layout.
REGIONS = ("Precentral", "Postcentral", "Rolandic_Oper", "Supp_Motor_Area", "Paracentral_Lobule")


# --------------------------------------------------------------------------- #
# Subject discovery
# --------------------------------------------------------------------------- #
def find_subject_files(extra_root="ajile12-nwb-data", primary=None):
    """Map subject-id -> nwb path. One session per subject (smallest if several)."""
    cands = glob.glob(os.path.join(extra_root, "*.nwb"))
    if primary and os.path.exists(primary):
        cands.append(primary)
    cands = [p for p in cands if os.path.getsize(p) > 1e9]  # AJILE12-scale only

    by_sub = {}
    for p in cands:
        m = re.search(r"sub-([A-Za-z0-9]+)", os.path.basename(p))
        sid = m.group(1) if m else os.path.basename(p)
        if sid not in by_sub or os.path.getsize(p) < os.path.getsize(by_sub[sid]):
            by_sub[sid] = p
    return dict(sorted(by_sub.items()))


# --------------------------------------------------------------------------- #
# ROI-projected features (fixed dim across subjects)
# --------------------------------------------------------------------------- #
def region_features(path, dur_min, anchor, bands, out_rate, smooth_hz, regions=REGIONS,
                    verbose=True):
    """Build a continuous stream on good channels, then average band-power within each
    fixed AAL ROI -> (T, n_regions*n_bands) with a per-region coverage count."""
    dur = dur_min * 60.0
    if anchor == "movement":
        t0, t1, _ = find_movement_window(path, dur_sec=dur, step_sec=120.0)
    else:
        t0, t1, _ = find_active_window(path, dur_sec=dur, step_sec=300.0)

    with __import__("h5py").File(path, "r") as f:
        good = good_channel_indices(f)
    xyz, _ = electrode_coords(path)
    try:
        names = mni_to_aal(xyz)
    except Exception as e:  # noqa: BLE001
        print("  AAL mapping failed ({}); this subject cannot be ROI-projected".format(
            type(e).__name__))
        return None

    # channel -> region index (base-name match), restricted to good channels
    ch_region = {}
    for ci in good:
        nm = names[ci].lower()
        for ri, r in enumerate(regions):
            if r.lower() in nm:
                ch_region[int(ci)] = ri
                break
    used_channels = np.array(sorted(ch_region), dtype=int)
    if len(used_channels) == 0:
        if verbose:
            print("  no electrodes in any sensorimotor ROI -> all-missing features")

    s = build_continuous_stream(path, t0, t1, out_rate=out_rate, bands=bands,
                                ecog_channels=(used_channels if len(used_channels) else good),
                                zscore=True, smooth_hz=smooth_hz, verbose=False)
    X = np.asarray(s["X"], dtype=np.float32)  # (T, n_used_ch * n_bands), z-scored
    nb = len(bands)
    T = X.shape[0]
    feat = np.full((T, len(regions) * nb), np.nan, dtype=np.float32)
    coverage = np.zeros(len(regions), dtype=int)

    if len(used_channels):
        # columns of X are laid out [ch0_b0, ch0_b1, ..., ch1_b0, ...]
        for local_ci, ci in enumerate(used_channels):
            ri = ch_region[int(ci)]
            coverage[ri] += 1
            for bi in range(nb):
                col = local_ci * nb + bi
                tgt = ri * nb + bi
                cur = feat[:, tgt]
                feat[:, tgt] = X[:, col] if np.all(np.isnan(cur)) else cur + X[:, col]
        # average within each region that had >1 channel
        for ri in range(len(regions)):
            if coverage[ri] > 1:
                for bi in range(nb):
                    feat[:, ri * nb + bi] /= coverage[ri]

    if verbose:
        print("  {}: T={}, ROI coverage {}/{} regions, {} good ch in ROIs".format(
            os.path.basename(path), T, int((coverage > 0).sum()), len(regions), len(used_channels)))
    return {"feat": feat, "reach": np.asarray(s["reach"]).astype(int),
            "speed": np.asarray(s["speed"])[:, 0] if s["speed"].shape[1] else np.zeros(T),
            "coverage": coverage}


def impute(train_feat, test_feat):
    """Fill NaN columns (ROIs a subject lacks) with the training-pool column mean."""
    mu = np.nanmean(train_feat, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    tr = np.where(np.isnan(train_feat), mu, train_feat)
    te = np.where(np.isnan(test_feat), mu, test_feat)
    return tr, te


# --------------------------------------------------------------------------- #
# CKA (import the same implementation used within-subject)
# --------------------------------------------------------------------------- #
def linear_cka(A, B):
    from phase3_eval import linear_cka as _cka
    return _cka(A, B)


# --------------------------------------------------------------------------- #
# LOSO
# --------------------------------------------------------------------------- #
def movement_label(speed):
    from phase1_resolution import speed_from_threshold
    y, keep = speed_from_threshold(speed)
    return np.clip(y, 0, 1).astype(int), keep


def loso_decode(subject_feats, target, with_twotower, dim, tt_iter, seed, verbose=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    subs = list(subject_feats)
    rows = []
    for held in subs:
        train_subs = [s for s in subs if s != held]

        def stack(names, which):
            Xs, ys = [], []
            for s in names:
                d = subject_feats[s]
                if target == "reach":
                    y, keep = d["reach"], np.ones(len(d["reach"]), bool)
                else:
                    y, keep = movement_label(d["speed"])
                Xs.append(d["feat"][keep])
                ys.append(y[keep])
            return np.vstack(Xs), np.concatenate(ys)

        Xtr, ytr = stack(train_subs, target)
        Xte, yte = stack([held], target)
        Xtr, Xte = impute(Xtr, Xte)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        auc_m0 = float("nan")
        if len(np.unique(ytr)) == 2 and len(np.unique(yte)) == 2:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
            auc_m0 = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        rows.append({"held_out": held, "model": "M0", "target": target, "auc": auc_m0})
        if verbose:
            print("  LOSO test={} M0 {} AUC = {:.3f}".format(held, target, auc_m0))

        if with_twotower:
            auc_m3 = loso_twotower(subject_feats, train_subs, held, target, dim, tt_iter, seed)
            rows.append({"held_out": held, "model": "M3", "target": target, "auc": auc_m3})
            if verbose:
                print("  LOSO test={} M3 {} AUC = {:.3f}".format(held, target, auc_m3))
    return rows


def loso_twotower(subject_feats, train_subs, held, target, dim, tt_iter, seed):
    """Train a two-tower encoder on pooled ROI features (neural) vs. speed (behavior),
    then decode the held-out subject from z_n. ROI projection makes input dim shared."""
    import two_tower as tt
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score

    Xtr = np.vstack([subject_feats[s]["feat"] for s in train_subs])
    Btr = np.concatenate([subject_feats[s]["speed"] for s in train_subs])[:, None]
    Xte = subject_feats[held]["feat"]
    Xtr, Xte = impute(Xtr, Xte)

    model, Z_tr, _ = tt.fit_two_tower(Xtr, Btr.astype(np.float32), dim,
                                      np.arange(len(Xtr)), max_iter=tt_iter, seed=seed, verbose=False)
    Z_te, _ = tt.transform(model, X=Xte)

    if target == "reach":
        ytr = np.concatenate([subject_feats[s]["reach"] for s in train_subs])
        yte, keep_te = subject_feats[held]["reach"], np.ones(len(Xte), bool)
        keep_tr = np.ones(len(Z_tr), bool)
    else:
        ytr, keep_tr = movement_label(np.concatenate([subject_feats[s]["speed"] for s in train_subs]))
        yte, keep_te = movement_label(subject_feats[held]["speed"])
    if len(np.unique(ytr[keep_tr])) < 2 or len(np.unique(yte[keep_te])) < 2:
        return float("nan")
    clf = KNeighborsClassifier(n_neighbors=25).fit(Z_tr[keep_tr], ytr[keep_tr])
    return float(roc_auc_score(yte[keep_te], clf.predict_proba(Z_te[keep_te])[:, 1]))


def cross_subject_cka(subject_feats, seed=0, n=1500):
    """CKA between every subject pair on behavior-matched (reach/rest) ROI features."""
    from phase3_eval import behavior_matched_pairs
    subs = list(subject_feats)
    rows = []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            a, b = subs[i], subs[j]
            sa = {"reach": subject_feats[a]["reach"], "speed": subject_feats[a]["speed"][:, None]}
            sb = {"reach": subject_feats[b]["reach"], "speed": subject_feats[b]["speed"][:, None]}
            ia, ib = behavior_matched_pairs(sa, sb, seed=seed)
            if len(ia) < 20:
                continue
            fa, fb = impute(subject_feats[a]["feat"][ia], subject_feats[b]["feat"][ib])
            rows.append({"subject_a": a, "subject_b": b, "cka": linear_cka(fa, fb)})
            print("  CKA[{} vs {}] = {:.3f}".format(a, b, rows[-1]["cka"]))
    return rows


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
    ap.add_argument("--extra-root", default="ajile12-nwb-data")
    ap.add_argument("--primary", default=r"C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb")
    ap.add_argument("--out-dir", default="phase3_crosssubject")
    ap.add_argument("--stage", choices=["smoke", "core", "full"], default="core")
    ap.add_argument("--dur-min", type=float, default=30.0)
    ap.add_argument("--anchor", choices=["reach", "movement"], default="reach")
    ap.add_argument("--bands", default="beta,high_gamma")
    ap.add_argument("--out-rate", type=float, default=30.0)
    ap.add_argument("--smooth-hz", type=float, default=6.0)
    ap.add_argument("--targets", default="reach,movement")
    ap.add_argument("--with-twotower", action="store_true")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--tt-iter", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dur_min, tt_iter = args.dur_min, args.tt_iter
    if args.stage == "smoke":
        dur_min, tt_iter = 6.0, 200

    os.makedirs(args.out_dir, exist_ok=True)
    bands = tuple(b.strip() for b in args.bands.split(",") if b.strip())
    for b in bands:
        if b not in BANDS:
            raise SystemExit("unknown band {}".format(b))

    files = find_subject_files(args.extra_root, args.primary)
    print("subjects found:", list(files))
    if len(files) < 3:
        print("WARNING: LOSO needs >=3 subjects for a meaningful held-out test; found {}.".format(
            len(files)))

    subject_feats = {}
    for sid, path in files.items():
        print("\n[features] sub-{}".format(sid))
        rf = region_features(path, dur_min, args.anchor, bands, args.out_rate, args.smooth_hz)
        if rf is not None:
            subject_feats[sid] = rf
    if len(subject_feats) < 3:
        raise SystemExit("Fewer than 3 ROI-projectable subjects; cannot run LOSO.")

    all_rows = []
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        print("\n=== LOSO decode: target={} ===".format(target))
        all_rows += loso_decode(subject_feats, target, args.with_twotower, args.dim,
                                tt_iter, args.seed)
    save_csv(all_rows, os.path.join(args.out_dir, "crosssubject_loso.csv"))

    if args.stage in ("core", "full"):
        print("\n=== cross-subject CKA (ROI features) ===")
        cka_rows = cross_subject_cka(subject_feats, args.seed)
        save_csv(cka_rows, os.path.join(args.out_dir, "crosssubject_cka.csv"))

    # summary
    print("\n================ CROSS-SUBJECT SUMMARY ================")
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        for model in (["M0", "M3"] if args.with_twotower else ["M0"]):
            aucs = [r["auc"] for r in all_rows
                    if r["target"] == target and r["model"] == model and np.isfinite(r["auc"])]
            if aucs:
                print("  {} {}: mean LOSO AUC = {:.3f} across {} held-out subjects (range {:.3f}-{:.3f})".format(
                    target, model, np.mean(aucs), len(aucs), min(aucs), max(aucs)))
    print("  (AUC ~0.5 => cross-subject transfer fails for that target/model; electrode")
    print("   placement + limited N are the usual causes -- see PRIOR_WORK_AJILE12.md / HTNet.)")
    print("======================================================")
    print("\nDone. Outputs in", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
