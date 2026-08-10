import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase4_decompose")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_transfer_decomposition.png")


def read_csv(name):
    with open(os.path.join(D, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


gran = read_csv("granularity_curve.csv")
freq = read_csv("frequency_ablation.csv")
awake = read_csv("awake_rest_control.csv")
spat = read_csv("spatial_ablation.csv")

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))

# A. granularity curve: cross-patient LOSO vs. same-feature within-patient ceiling
levels = ["L1_binary", "L2_3class", "L3_4class", "L4_full"]
labels = ["Binary\n(rest/active)", "3-class", "4-class", "6-class\n(full)"]
vals = [float(next(r["mean_loso_auc"] for r in gran if r["level"] == l)) for l in levels]
wvals = [float(next(r["mean_within_auc"] for r in gran if r["level"] == l)) for l in levels]
a = ax[0, 0]
a.plot(range(4), wvals, "-s", color="#3d8b6b", lw=2, ms=6, label="within-patient ceiling")
a.plot(range(4), vals, "-o", color="#2b6cb0", lw=2, ms=6, label="cross-patient (LOSO)")
a.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=8)
a.set_ylabel("AUC (same feature)")
a.set_title("A. Target granularity", fontsize=10)
a.set_ylim(0.4, 1.0); a.legend(fontsize=6.5, loc="lower left"); a.grid(alpha=.25)

# B. frequency ablation (subset: single bands + best combo + all)
fbands = ["theta", "alpha", "beta", "alpha+beta", "all_5_bands"]
flabels = ["theta", "alpha", "beta", "alpha+\nbeta", "all 5\nbands"]
fvals = [float(next(r["mean_loso_auc"] for r in freq if r["bands"] == b)) for b in fbands]
a = ax[0, 1]
colors = ["#8899aa", "#8899aa", "#8899aa", "#c9a227", "#2b6cb0"]
a.bar(range(len(fbands)), fvals, color=colors)
a.axhline(0.5, color="gray", ls=":", lw=1)
a.set_xticks(range(len(fbands))); a.set_xticklabels(flabels, fontsize=8)
a.set_ylabel("Zero-shot LOSO AUC")
a.set_title("B. Frequency-band content", fontsize=10)
a.set_ylim(0, 1.0); a.grid(alpha=.25)

# C. awake-rest control
names = [r["contrast"] for r in awake]
nvals = [float(r["mean_loso_auc"]) for r in awake]
a = ax[1, 0]
colors = ["#3d8b6b", "#e07a5f"]
a.bar(range(len(names)), nvals, color=colors[:len(names)])
a.axhline(0.5, color="gray", ls=":", lw=1)
a.set_xticks(range(len(names)))
a.set_xticklabels(["Rest vs.\nInactive\n(both still)", "Inactive vs.\nEngaged\n(both awake)"], fontsize=8)
a.set_ylabel("Zero-shot LOSO AUC")
a.set_title("C. Motor-confound control", fontsize=10)
a.set_ylim(0, 1.0); a.grid(alpha=.25)

# D. spatial-information ablation
reps = ["electrode_identity_naive", "anatomical_roi", "channel_agnostic"]
rlabels = ["Electrode\nidentity\n(naive)", "Anatomical\nROI", "Channel-\nagnostic"]
rvals = [float(next(r["mean_loso_auc"] for r in spat if r["representation"] == x)) for x in reps]
a = ax[1, 1]
a.bar(range(3), rvals, color=["#e07a5f", "#3d8b6b", "#2b6cb0"])
a.axhline(0.5, color="gray", ls=":", lw=1)
a.set_xticks(range(3)); a.set_xticklabels(rlabels, fontsize=8)
a.set_ylabel("Zero-shot LOSO AUC")
a.set_title("D. Spatial representation", fontsize=10)
a.set_ylim(0, 1.0); a.grid(alpha=.25)

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
