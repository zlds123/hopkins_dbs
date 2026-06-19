"""Memory-safe windowed Dataset + feature engineering for AJILE12-style NWB files.

Designed for the large AJILE12 file (``sub-XX_ses-Y_behavior+ecephys.nwb``), which
holds ~15 GB of ECoG that must never be loaded whole. Everything here reads only
short, lazily-sliced windows via h5py.

Core pieces
-----------
- ``WindowedNWBDataset`` : indexable object that returns one aligned window
  (ECoG + optional physiology + pose) per item. Works standalone or can be
  wrapped by a torch ``DataLoader`` (single-process).
- ``hilbert_bandpower`` : the standard ECoG feature -- per-channel, per-band log
  amplitude (Butterworth bandpass -> Hilbert envelope -> average over window).
- ``windows_from_reaches`` / ``windows_from_epochs`` : build labelled windows for
  (a) reach-vs-baseline movement detection or (b) coarse-behavior classification.
- ``build_feature_matrix`` : iterate a dataset and stack features into ``(X, y)``
  ready for scikit-learn.

Signal-processing choices follow the AJILE12 / Peterson-Brunton lineage and the
naturalistic-ECoG decoding literature: common-median reference (already applied
in the file), 5 canonical bands with high-gamma (70-110 Hz) as the key motor
band, Hilbert envelope as the instantaneous amplitude estimate, and simple
amplitude-based artifact flagging.
"""

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt, hilbert

# Canonical frequency bands (Hz). high_gamma is the dominant movement-related band.
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "low_gamma": (30, 55),
    "high_gamma": (70, 110),
}


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _decode(v):
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def _series_rate(f, path, default):
    st = f.get(path + "/starting_time")
    if st is not None and "rate" in st.attrs:
        return float(st.attrs["rate"]), float(st[()])
    return float(default), 0.0


def good_channel_indices(f):
    """Indices of electrodes flagged ``good`` (falls back to all channels)."""
    el = f.get("general/extracellular_ephys/electrodes")
    if el is not None and "good" in el:
        return np.where(el["good"][:].astype(bool))[0]
    es = f.get("acquisition/ElectricalSeries/data")
    n = es.shape[1] if es is not None and es.ndim > 1 else 0
    return np.arange(n)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def hilbert_bandpower(ecog, fs, bands=BANDS, log=True, order=4):
    """Per-channel, per-band amplitude via Butterworth bandpass + Hilbert envelope.

    Parameters
    ----------
    ecog : ndarray (n_samples, n_channels)
    fs   : sampling rate (Hz)

    Returns
    -------
    feats : ndarray (n_channels, n_bands)  -- mean (log) envelope per channel/band
    names : list[str] band names (columns)
    """
    ecog = np.asarray(ecog, dtype=float)
    ecog = ecog - ecog.mean(axis=0, keepdims=True)  # remove per-window DC
    nyq = fs / 2.0
    n_ch = ecog.shape[1]
    band_names = list(bands.keys())
    out = np.empty((n_ch, len(band_names)), dtype=float)
    for j, name in enumerate(band_names):
        lo, hi = bands[name]
        hi = min(hi, nyq - 1)
        sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
        filt = sosfiltfilt(sos, ecog, axis=0)
        env = np.abs(hilbert(filt, axis=0))      # instantaneous amplitude
        power = env.mean(axis=0)                 # average over the window
        out[:, j] = np.log(power + 1e-12) if log else power
    return out, band_names


def extract_features(sample, fs, bands=BANDS, include_pose=True):
    """Turn one dataset sample (dict) into a flat feature vector + names."""
    feats, band_names = hilbert_bandpower(sample["ecog"], fs, bands=bands)
    names = [f"ch{c}_{b}" for c in sample["channels"] for b in band_names]
    vec = feats.reshape(-1)

    if include_pose and sample.get("pose"):
        for kp, arr in sample["pose"].items():
            a = np.asarray(arr, dtype=float)
            if a.size == 0:
                continue
            # speed magnitude between consecutive frames (robust to NaNs)
            d = np.diff(a, axis=0)
            speed = np.sqrt(np.nansum(d ** 2, axis=1))
            vec = np.concatenate([vec, [np.nanmean(speed), np.nanmax(speed),
                                        np.nanstd(speed)]])
            names += [f"{kp}_speed_mean", f"{kp}_speed_max", f"{kp}_speed_std"]
    return vec, names


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class WindowedNWBDataset:
    """Indexable, lazily-sliced windows of aligned ECoG / physiology / pose.

    Parameters
    ----------
    path : str
        Path to the NWB (HDF5) file.
    windows : list[dict]
        Each dict must have ``t0`` (window start, seconds) and ``label``.
    window_sec : float
        Window length in seconds (fixed for all items).
    ecog_channels : "good" | "all" | sequence[int]
    pose_keypoints : sequence[str] | None
        Keypoints to return (None -> none). Ignored if file has no pose.
    include_physio : bool
        Also return ECG/EOG channels if present.
    """

    def __init__(self, path, windows, window_sec, ecog_channels="good",
                 pose_keypoints=("L_Wrist", "R_Wrist"), include_physio=False):
        self.path = path
        self.windows = list(windows)
        self.window_sec = float(window_sec)
        self.include_physio = include_physio
        self._f = None

        with h5py.File(path, "r") as f:
            self.ecog_rate, self.ecog_t0 = _series_rate(f, "acquisition/ElectricalSeries", 500.0)
            self.ecog_conv = float(f["acquisition/ElectricalSeries/data"].attrs.get("conversion", 1.0))
            self.n_ecog = f["acquisition/ElectricalSeries/data"].shape[0]
            if ecog_channels == "good":
                self.channels = good_channel_indices(f)
            elif ecog_channels == "all":
                self.channels = np.arange(f["acquisition/ElectricalSeries/data"].shape[1])
            else:
                self.channels = np.asarray(ecog_channels, dtype=int)
            # h5py fancy indexing needs sorted unique indices
            self.channels = np.unique(self.channels)

            pos = f.get("processing/behavior/Position")
            if pos is not None and pose_keypoints:
                self.pose_keypoints = [k for k in pose_keypoints if k in pos]
                self.pose_rate, self.pose_t0 = _series_rate(
                    f, "processing/behavior/Position/" + self.pose_keypoints[0], 30.0
                ) if self.pose_keypoints else (None, 0.0)
            else:
                self.pose_keypoints, self.pose_rate, self.pose_t0 = [], None, 0.0

            self.physio_names = [n for n in ("ECGL", "ECGR", "EOGL", "EOGR")
                                 if include_physio and f.get("acquisition/" + n) is not None]

    # -- file handle management (lazy, reused) ------------------------------ #
    def _h5(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        t0 = float(w["t0"])
        f = self._h5()

        # ECoG slice (only window x selected channels is read)
        s0 = int(round((t0 - self.ecog_t0) * self.ecog_rate))
        s1 = s0 + int(round(self.window_sec * self.ecog_rate))
        s0 = max(0, min(s0, self.n_ecog - 1))
        s1 = max(s0 + 1, min(s1, self.n_ecog))
        ecog = f["acquisition/ElectricalSeries/data"][s0:s1, self.channels].astype(np.float32)
        ecog = ecog * self.ecog_conv  # -> volts

        out = {"ecog": ecog, "channels": self.channels, "label": w.get("label"),
               "t0": t0, "fs": self.ecog_rate}

        # Physiology (ECG / EOG), same rate as ECoG
        if self.physio_names:
            phys = {}
            for n in self.physio_names:
                d = f["acquisition/" + n + "/data"]
                phys[n] = d[s0:s1].astype(np.float32)
            out["physio"] = phys

        # Pose (own rate); align by time, return per-keypoint (n, 2)
        if self.pose_keypoints:
            p0 = int(round((t0 - self.pose_t0) * self.pose_rate))
            p1 = p0 + int(round(self.window_sec * self.pose_rate))
            pose = {}
            for kp in self.pose_keypoints:
                d = f["processing/behavior/Position/" + kp + "/data"]
                a0 = max(0, min(p0, d.shape[0] - 1))
                a1 = max(a0 + 1, min(p1, d.shape[0]))
                pose[kp] = d[a0:a1, :].astype(np.float32)
            out["pose"] = pose

        return out

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Window builders
# --------------------------------------------------------------------------- #
def _read_intervals(f, name):
    g = f.get("intervals/" + name)
    if g is None:
        return None
    cols = {}
    for k, obj in g.items():
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and not k.endswith("_index"):
            arr = obj[:]
            cols[k] = np.array([_decode(x) for x in arr], dtype=object) if arr.dtype == object else arr
    return cols


def windows_from_reaches(path, window_sec=2.0, pre=0.5, n_baseline=None,
                         min_gap=4.0, avoid_blocklist=True, seed=0):
    """Reach-vs-baseline movement-detection windows.

    Positive windows are anchored ``pre`` seconds before each reach onset.
    Negative (baseline) windows are sampled away from any reach onset and away
    from blocklisted epochs (data breaks).
    """
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        reaches = _read_intervals(f, "reaches")
        if reaches is None:
            raise ValueError("No /intervals/reaches table in this file.")
        onsets = np.asarray(reaches["start_time"], dtype=float)
        fs, _ = _series_rate(f, "acquisition/ElectricalSeries", 500.0)
        T = f["acquisition/ElectricalSeries/data"].shape[0] / fs

        bad = []
        if avoid_blocklist:
            ep = _read_intervals(f, "epochs")
            if ep is not None and "labels" in ep:
                for st, sp, lab in zip(ep["start_time"], ep["stop_time"], ep["labels"]):
                    if "Blocklist" in str(lab) or "break" in str(lab).lower():
                        bad.append((float(st), float(sp)))

    def in_bad(t):
        return any(a <= t <= b for a, b in bad)

    windows = [{"t0": float(o - pre), "label": 1, "onset": float(o)} for o in onsets
               if 0 <= o - pre and (o - pre + window_sec) <= T]

    n_baseline = n_baseline or len(windows)
    onset_sorted = np.sort(onsets)
    neg, tries = [], 0
    while len(neg) < n_baseline and tries < n_baseline * 100:
        tries += 1
        t0 = float(rng.uniform(0, max(1.0, T - window_sec)))
        center = t0 + window_sec / 2
        # reject if near any reach onset
        idx = np.searchsorted(onset_sorted, center)
        near = False
        for k in (idx - 1, idx):
            if 0 <= k < len(onset_sorted) and abs(onset_sorted[k] - center) < (window_sec / 2 + min_gap):
                near = True
                break
        if near or in_bad(center) or in_bad(t0) or in_bad(t0 + window_sec):
            continue
        neg.append({"t0": t0, "label": 0, "onset": None})

    windows += neg
    rng.shuffle(windows)
    return windows


def windows_from_epochs(path, window_sec=10.0, step_sec=None, include=None,
                        exclude_substr=("Blocklist", "break"), max_per_label=None,
                        seed=0):
    """Sliding windows tiled inside coarse-behavior epochs; label = epoch label."""
    rng = np.random.default_rng(seed)
    step_sec = step_sec or window_sec
    with h5py.File(path, "r") as f:
        ep = _read_intervals(f, "epochs")
        if ep is None or "labels" not in ep:
            raise ValueError("No /intervals/epochs with labels in this file.")
        rows = list(zip(ep["start_time"], ep["stop_time"], ep["labels"]))

    windows, counts = [], {}
    for st, sp, lab in rows:
        lab = str(lab)
        if any(s.lower() in lab.lower() for s in exclude_substr):
            continue
        if include is not None and lab not in include:
            continue
        t = float(st)
        while t + window_sec <= float(sp):
            if max_per_label is None or counts.get(lab, 0) < max_per_label:
                windows.append({"t0": t, "label": lab})
                counts[lab] = counts.get(lab, 0) + 1
            t += step_sec
    rng.shuffle(windows)
    return windows


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #
def build_feature_matrix(dataset, bands=BANDS, include_pose=True, verbose=True):
    """Iterate a ``WindowedNWBDataset`` -> (X, y, feature_names).

    Skips windows with non-finite ECoG (basic artifact rejection).
    """
    X, y, names = [], [], None
    n = len(dataset)
    for i in range(n):
        sample = dataset[i]
        if not np.all(np.isfinite(sample["ecog"])):
            continue
        vec, nm = extract_features(sample, sample["fs"], bands=bands, include_pose=include_pose)
        if not np.all(np.isfinite(vec)):
            continue
        X.append(vec)
        y.append(sample["label"])
        names = nm
        if verbose and (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{n} windows")
    return np.asarray(X), np.asarray(y), names
