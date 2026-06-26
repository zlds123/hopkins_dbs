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


# --------------------------------------------------------------------------- #
# Continuous feature stream (for CEBRA and other latent-variable models)
# --------------------------------------------------------------------------- #
def _interp_nan(a):
    """Linearly interpolate NaNs along axis 0 of a (T, k) array (per column)."""
    a = np.asarray(a, dtype=float).copy()
    if a.ndim == 1:
        a = a[:, None]
    t = np.arange(a.shape[0])
    for c in range(a.shape[1]):
        col = a[:, c]
        m = np.isfinite(col)
        if m.sum() == 0:
            col[:] = 0.0
        elif m.sum() < col.size:
            col[~m] = np.interp(t[~m], t[m], col[m])
        a[:, c] = col
    return a


def find_active_window(path, dur_sec=3600.0, step_sec=300.0):
    """Find the ``dur_sec`` window containing the most reach onsets.

    Returns ``(t_start, t_stop, n_reaches)``. Sleep/blocklist-heavy spans are
    naturally avoided because they contain no reaches.
    """
    with h5py.File(path, "r") as f:
        reaches = _read_intervals(f, "reaches")
        if reaches is None:
            raise ValueError("No /intervals/reaches in this file.")
        fs, _ = _series_rate(f, "acquisition/ElectricalSeries", 500.0)
        T = f["acquisition/ElectricalSeries/data"].shape[0] / fs
    onsets = np.sort(np.asarray(reaches["start_time"], dtype=float))

    best_t, best_n = 0.0, -1
    t = 0.0
    while t + dur_sec <= T:
        n = int(np.sum((onsets >= t) & (onsets < t + dur_sec)))
        if n > best_n:
            best_t, best_n = t, n
        t += step_sec
    return best_t, best_t + dur_sec, best_n


def find_movement_window(path, dur_sec=1800.0, step_sec=120.0, keypoint="R_Wrist",
                         speed_pct=80, min_valid=0.5):
    """Find the ``dur_sec`` window richest in *actual wrist movement*.

    Scores each candidate by the fraction of **valid** (non-occluded) frames whose
    wrist speed exceeds a global high-speed threshold (``speed_pct`` percentile).
    Windows with poor pose coverage (< ``min_valid``) are skipped, so we don't pick
    a span that merely looks fast because of occlusion-interpolation jumps.

    Returns ``(t_start, t_stop, score)``.
    """
    with h5py.File(path, "r") as f:
        prate, pt0 = _series_rate(f, "processing/behavior/Position/" + keypoint, 30.0)
        d = f["processing/behavior/Position/" + keypoint + "/data"][:]
    valid = np.isfinite(d).all(axis=1)
    xy = _interp_nan(d)
    v = np.gradient(xy, 1.0 / prate, axis=0)
    speed = np.sqrt((v ** 2).sum(axis=1))
    speed[~valid] = np.nan
    thr = np.nanpercentile(speed, speed_pct)

    win = int(round(dur_sec * prate))
    step = int(round(step_sec * prate))
    best_t, best_score = 0.0, -1.0
    for s0 in range(0, max(1, len(speed) - win), step):
        seg = speed[s0:s0 + win]
        m = np.isfinite(seg)
        if m.mean() < min_valid:
            continue
        score = float((seg[m] > thr).mean())
        if score > best_score:
            best_t, best_score = pt0 + s0 / prate, score
    return best_t, best_t + dur_sec, best_score


# --------------------------------------------------------------------------- #
# Electrode anatomy -> sensorimotor channel selection
# --------------------------------------------------------------------------- #
# AAL regions that make up sensorimotor cortex (hand/arm + face motor + SMA).
SENSORIMOTOR_AAL = ("Precentral", "Postcentral", "Rolandic_Oper",
                    "Supp_Motor_Area", "Paracentral_Lobule")


def electrode_coords(path):
    """Return ``(xyz (n,3) MNI, good_mask (n,))`` for all electrodes."""
    with h5py.File(path, "r") as f:
        el = f["general/extracellular_ephys/electrodes"]
        xyz = np.column_stack([el["x"][:], el["y"][:], el["z"][:]]).astype(float)
        good = el["good"][:].astype(bool) if "good" in el else np.ones(len(xyz), bool)
    return xyz, good


def mni_to_aal(coords):
    """Map ``(n,3)`` MNI coordinates -> list of AAL region-name strings.

    Requires ``nilearn`` (one-time atlas download). SSL verification is relaxed
    because the atlas host often fails strict verification on Windows.
    """
    import os
    import ssl
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass
    ssl._create_default_https_context = ssl._create_unverified_context

    from nilearn import datasets, image

    aal = datasets.fetch_atlas_aal()
    img = image.load_img(aal.maps)
    vol = np.asarray(img.get_fdata())
    inv = np.linalg.inv(img.affine)
    labels = list(aal.labels)
    indices = [str(i) for i in aal.indices]

    names = []
    for x, y, z in coords:
        vox = inv.dot([x, y, z, 1.0])[:3]
        i, j, k = np.round(vox).astype(int)
        name = "unknown"
        if (0 <= i < vol.shape[0] and 0 <= j < vol.shape[1] and 0 <= k < vol.shape[2]):
            val = int(vol[i, j, k])
            if val != 0 and str(val) in indices:
                name = labels[indices.index(str(val))]
        names.append(name)
    return names


def sensorimotor_channels(path, good_only=True, regions=SENSORIMOTOR_AAL,
                          fallback_box=True, verbose=True):
    """Indices of (good) electrodes in sensorimotor cortex.

    Primary: AAL region labels from MNI coords. Fallback (if nilearn/atlas fails):
    a coordinate box around the central sulcus (lateral, peri-central, dorsal-ish).
    """
    xyz, good = electrode_coords(path)
    pool = np.where(good)[0] if good_only else np.arange(len(xyz))

    sel, method = [], ""
    try:
        names = mni_to_aal(xyz)
        sel = [i for i in pool
               if any(r.lower() in names[i].lower() for r in regions)]
        method = "AAL"
    except Exception as e:  # noqa: BLE001
        method = "AAL-failed({})".format(type(e).__name__)

    if not sel and fallback_box:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        box = (np.abs(x) >= 25) & (y >= -40) & (y <= 15) & (z >= 15)
        sel = [i for i in pool if box[i]]
        method += "+coord_box"

    sel = np.array(sorted(set(sel)), dtype=int)
    if verbose:
        print("sensorimotor channels: {}/{} selected via {}".format(
            len(sel), len(pool), method))
    return sel


def build_continuous_stream(path, t_start, t_stop, out_rate=30.0,
                            bands=("high_gamma",), ecog_channels="good",
                            pose_keypoints=("R_Wrist", "L_Wrist"),
                            zscore=True, smooth_hz=6.0, block_sec=60.0,
                            pad_sec=2.0, order=4, verbose=True):
    """Build time-aligned continuous arrays over ``[t_start, t_stop)`` for CEBRA.

    The ECoG band-power envelope is computed *chunked* (overlap-and-discard) so
    the 15 GB file is never loaded whole, then resampled to ``out_rate`` and put
    on a common time grid with wrist kinematics and behavior labels.

    Returns a dict with:
        t          (T,)            time vector (s)
        X          (T, C)          z-scored band-power envelope (C = n_ch*n_bands)
        vel        (T, 2*K)        wrist velocity (dx,dy per keypoint) -- CEBRA aux
        speed      (T, K)          wrist speed magnitude per keypoint
        pos        (T, 2*K)        wrist position (interpolated over occlusions)
        reach      (T,)  int       1 inside a reach interval, else 0
        behavior   (T,)  object    coarse-behavior label per sample ('' if none)
        channels, feature_names, keypoints, fs, out_rate
    """
    bands = tuple(bands)
    for b in bands:
        if b not in BANDS:
            raise ValueError("Unknown band '{}'. Options: {}".format(b, list(BANDS)))

    t_grid = np.arange(t_start, t_stop, 1.0 / out_rate)
    T = t_grid.size

    with h5py.File(path, "r") as f:
        fs, ecog_t0 = _series_rate(f, "acquisition/ElectricalSeries", 500.0)
        conv = float(f["acquisition/ElectricalSeries/data"].attrs.get("conversion", 1.0))
        n_samp = f["acquisition/ElectricalSeries/data"].shape[0]
        if isinstance(ecog_channels, str) and ecog_channels == "good":
            channels = good_channel_indices(f)
        elif isinstance(ecog_channels, str) and ecog_channels == "all":
            channels = np.arange(f["acquisition/ElectricalSeries/data"].shape[1])
        else:
            channels = np.unique(np.asarray(ecog_channels, dtype=int))
        nyq = fs / 2.0

        # Pre-build band SOS filters.
        sos_by_band = {}
        for b in bands:
            lo, hi = BANDS[b]
            hi = min(hi, nyq - 1)
            sos_by_band[b] = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
        sos_smooth = butter(2, min(smooth_hz, nyq - 1) / nyq, btype="low", output="sos")

        C = len(channels) * len(bands)
        X = np.full((T, C), np.nan, dtype=np.float32)

        # ---- chunked envelope extraction (overlap-and-discard) ----------- #
        pad = int(round(pad_sec * fs))
        cur = t_start
        while cur < t_stop:
            ba, bb = cur, min(cur + block_sec, t_stop)
            s0 = max(0, int(round((ba - ecog_t0) * fs)) - pad)
            s1 = min(n_samp, int(round((bb - ecog_t0) * fs)) + pad)
            if s1 - s0 < 10:
                cur = bb
                continue
            raw = f["acquisition/ElectricalSeries/data"][s0:s1, channels].astype(np.float64)
            raw = raw * conv
            raw = raw - raw.mean(axis=0, keepdims=True)
            local_t = ecog_t0 + np.arange(s0, s1) / fs
            mask = (t_grid >= ba) & (t_grid < bb)
            tgt = t_grid[mask]
            if tgt.size == 0:
                cur = bb
                continue
            for bi, b in enumerate(bands):
                filt = sosfiltfilt(sos_by_band[b], raw, axis=0)
                env = np.abs(hilbert(filt, axis=0))
                env = sosfiltfilt(sos_smooth, env, axis=0)
                env = np.clip(env, 0, None)
                for ci in range(len(channels)):
                    col = ci * len(bands) + bi
                    X[mask, col] = np.interp(tgt, local_t, env[:, ci]).astype(np.float32)
            if verbose:
                print("  ecog block {:.0f}-{:.0f}s done".format(ba, bb))
            cur = bb

        # ---- pose -> position / velocity on the common grid ------------- #
        pos_parts, vel_parts, speed_parts, kps = [], [], [], []
        posg = f.get("processing/behavior/Position")
        if posg is not None:
            for kp in pose_keypoints:
                if kp not in posg:
                    continue
                prate, pt0 = _series_rate(f, "processing/behavior/Position/" + kp, 30.0)
                d = posg[kp + "/data"]
                p0 = max(0, int(round((t_start - pt0) * prate)) - 1)
                p1 = min(d.shape[0], int(round((t_stop - pt0) * prate)) + 2)
                arr = _interp_nan(d[p0:p1, :])
                ptimes = pt0 + np.arange(p0, p1) / prate
                xy = np.column_stack([np.interp(t_grid, ptimes, arr[:, k])
                                      for k in range(arr.shape[1])])
                v = np.gradient(xy, 1.0 / out_rate, axis=0)
                pos_parts.append(xy)
                vel_parts.append(v)
                speed_parts.append(np.sqrt((v ** 2).sum(axis=1)))
                kps.append(kp)

        # ---- labels: reach mask + coarse behavior per sample ------------ #
        reach = np.zeros(T, dtype=np.int64)
        reaches = _read_intervals(f, "reaches")
        if reaches is not None and "stop_time" in reaches:
            for s, e in zip(reaches["start_time"], reaches["stop_time"]):
                reach[(t_grid >= float(s)) & (t_grid < float(e))] = 1

        behavior = np.array([""] * T, dtype=object)
        ep = _read_intervals(f, "epochs")
        if ep is not None and "labels" in ep:
            for s, e, lab in zip(ep["start_time"], ep["stop_time"], ep["labels"]):
                behavior[(t_grid >= float(s)) & (t_grid < float(e))] = str(lab)

    # ---- finalize ------------------------------------------------------- #
    X = _interp_nan(X)  # fill any grid points missed at block seams
    scaler = None
    if zscore:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mu) / sd
        scaler = (mu.squeeze(0), sd.squeeze(0))

    feature_names = ["ch{}_{}".format(int(c), b) for c in channels for b in bands]
    pos = np.concatenate(pos_parts, axis=1) if pos_parts else np.zeros((T, 0))
    vel = np.concatenate(vel_parts, axis=1) if vel_parts else np.zeros((T, 0))
    speed = np.column_stack(speed_parts) if speed_parts else np.zeros((T, 0))

    return {
        "t": t_grid, "X": X.astype(np.float32),
        "vel": vel.astype(np.float32), "speed": speed.astype(np.float32),
        "pos": pos.astype(np.float32), "reach": reach, "behavior": behavior,
        "channels": np.asarray(channels), "feature_names": feature_names,
        "keypoints": kps, "fs": fs, "out_rate": float(out_rate),
        "bands": list(bands), "scaler": scaler,
    }
