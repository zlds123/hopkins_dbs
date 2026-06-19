"""Sonify AJILE12 ECoG -- turn neural windows into *audible* WAV clips, labeled by behavior.

Why this works
--------------
ECoG is just a voltage wiggling at ~500 Hz, which is right at the edge of human
hearing. If we play the samples back *faster* (pitch-shift up), the spectral
structure lands squarely in the audible range:

    output_rate = ecog_rate * SPEEDUP            (e.g. 500 Hz * 12 = 6000 Hz)

With SPEEDUP=12 the bands map to:
    beta  (12-30 Hz)  -> ~144-360 Hz   (the resting "idle hum")
    high-gamma (70-110 Hz) -> ~840-1320 Hz   (the movement "crackle")

So at rest you hear a steady hum; during movement the hum ducks (beta
desynchronization) and a broadband crackle rides on top (high-gamma burst).

Everything is read *lazily* through ``WindowedNWBDataset`` -- only a few short
windows of a single channel are ever pulled into memory, so this is safe on the
15 GB file.

Outputs (written to ``--out`` / ``sonified/`` by default)
    *.wav            one clip per behavior + a reach-active vs rest contrast
    *.png            matching spectrogram (0-150 Hz) so you can *see* it too
    index.html       a one-page player: label + play button + spectrogram
    manifest.csv     machine-readable index of every clip

Run
    python sonify.py                          # uses the default AJILE12 path
    python sonify.py --nwb PATH --out DIR --speedup 12 --window-sec 24
"""

import argparse
import csv
import html
import os

import h5py
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, spectrogram

from nwb_dataset import (
    BANDS,
    WindowedNWBDataset,
    _read_intervals,
    _series_rate,
    good_channel_indices,
    hilbert_bandpower,
)

DEFAULT_NWB = r"C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb"

# Each band drives one voice, tuned to a consonant C-major stack so the chord
# always sounds harmonious. high_gamma (movement) is the highest, brightest note.
BAND_PITCH = {
    "theta": 130.81,      # C3  -- low drone
    "alpha": 164.81,      # E3
    "beta": 196.00,       # G3  -- resting "idle" voice
    "low_gamma": 261.63,  # C4
    "high_gamma": 329.63,  # E4  -- movement "sparkle"
}
# Plucked-note layer (triggered by high-gamma bursts) uses C-major pentatonic.
PLUCK_SCALE = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33]


# --------------------------------------------------------------------------- #
# Audio rendering
# --------------------------------------------------------------------------- #
def _to_audio(x, fs, speedup=12, hp_cut=2.0, fade_ms=15.0, scale=None):
    """Pitch-shift a 1-D neural trace into an int16 WAV array.

    Returns ``(audio_int16, out_rate)``. ``scale`` (a peak amplitude in the same
    units as ``x``) is used to normalize; pass a *shared* scale across clips so
    loud neural activity stays loud relative to quiet clips.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean()

    # Drop slow drift so the hum/crackle dominate instead of DC wander.
    nyq = fs / 2.0
    if hp_cut and hp_cut < nyq:
        sos = butter(2, hp_cut / nyq, btype="high", output="sos")
        x = sosfiltfilt(sos, x)

    if scale is None:
        scale = np.percentile(np.abs(x), 99.7) or 1.0
    x = np.clip(x / (scale + 1e-12), -1.0, 1.0)

    out_rate = int(round(fs * speedup))
    n_fade = int(out_rate * fade_ms / 1000.0)
    if n_fade > 0 and 2 * n_fade < x.size:
        ramp = np.linspace(0.0, 1.0, n_fade)
        x[:n_fade] *= ramp
        x[-n_fade:] *= ramp[::-1]

    return (x * 32767.0).astype(np.int16), out_rate


def _spectrogram_png(x, fs, path, title):
    """Save a 0-150 Hz spectrogram of the *original* (un-shifted) trace."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    nperseg = min(256, x.size)
    noverlap = int(nperseg * 0.875)
    fxx, txx, sxx = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    keep = fxx <= 150
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.pcolormesh(txx, fxx[keep], 10 * np.log10(sxx[keep] + 1e-20),
                  shading="auto", cmap="magma")
    for lo, hi in (BANDS["beta"], BANDS["high_gamma"]):
        ax.axhline(lo, color="cyan", lw=0.4, alpha=0.5)
        ax.axhline(hi, color="cyan", lw=0.4, alpha=0.5)
    ax.set_ylabel("Hz")
    ax.set_xlabel("seconds (real time)")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 150)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Musification (parameter-mapping sonification)
# --------------------------------------------------------------------------- #
def _band_envelope(x, fs, lo, hi, smooth_hz=6.0, order=4):
    """Instantaneous amplitude of one band (bandpass -> Hilbert -> smooth)."""
    from scipy.signal import hilbert

    nyq = fs / 2.0
    hi = min(hi, nyq - 1)
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    if smooth_hz and smooth_hz < nyq:
        sos_s = butter(2, smooth_hz / nyq, btype="low", output="sos")
        env = sosfiltfilt(sos_s, env)
    return np.clip(env, 0, None)


def _band_envelopes(x, fs):
    """All band envelopes for one trace (dict band -> envelope at sample rate fs)."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    x = x - x.mean()
    return {b: _band_envelope(x, fs, lo, hi) for b, (lo, hi) in BANDS.items()}


def _voice(freq, amp_env, sr):
    """A sustained tone (fundamental + a couple of harmonics) shaped by amp_env."""
    n = amp_env.size
    t = np.arange(n) / sr
    tone = (np.sin(2 * np.pi * freq * t)
            + 0.35 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.15 * np.sin(2 * np.pi * 3 * freq * t))
    return tone * amp_env


def _pluck(freq, sr, dur=0.32, gain=1.0):
    """Short decaying plucked note (sine + octave) for the rhythm layer."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    decay = np.exp(-t / (dur * 0.35))
    wave = (np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * 2 * freq * t))
    return gain * decay * wave


def musify_clip(env_dict, fs, window_sec, band_scales, audio_sec=10.0,
                out_rate=22050, fade_ms=20.0):
    """Render one clip's band envelopes into a musical waveform (float, [-1,1]).

    Time is *stretched* (window_sec real -> audio_sec audio) independently of
    pitch: pitches come from BAND_PITCH, not from the playback rate.
    """
    from scipy.signal import find_peaks

    n = int(round(audio_sec * out_rate))
    src_t = np.linspace(0.0, 1.0, num=next(iter(env_dict.values())).size)
    dst_t = np.linspace(0.0, 1.0, num=n)

    mix = np.zeros(n, dtype=np.float64)
    for band, env in env_dict.items():
        amp = np.interp(dst_t, src_t, env) / (band_scales[band] + 1e-12)
        amp = np.clip(amp, 0.0, 1.3)
        mix += _voice(BAND_PITCH[band], amp, out_rate)

    # Rhythm layer: high-gamma bursts -> plucked notes. Threshold is *absolute*
    # (shared across clips) so quiet windows stay sparse and active windows
    # ring out -- the note density itself encodes how active the cortex is.
    hg = env_dict["high_gamma"]
    hg_scale = band_scales["high_gamma"]
    thr = 0.6 * hg_scale
    peaks, _ = find_peaks(hg, height=thr, distance=int(fs * 0.4))
    for k, p in enumerate(peaks):
        pos = int((p / hg.size) * n)
        strength = float(np.clip((hg[p] - thr) / (hg_scale + 1e-12), 0.2, 1.5))
        freq = PLUCK_SCALE[k % len(PLUCK_SCALE)]
        note = _pluck(freq, out_rate, gain=0.9 * strength)
        end = min(n, pos + note.size)
        mix[pos:end] += note[: end - pos]

    peak = np.percentile(np.abs(mix), 99.9) or 1.0
    mix = np.clip(mix / peak, -1.0, 1.0)
    n_fade = int(out_rate * fade_ms / 1000.0)
    if n_fade > 0 and 2 * n_fade < mix.size:
        ramp = np.linspace(0, 1, n_fade)
        mix[:n_fade] *= ramp
        mix[-n_fade:] *= ramp[::-1]
    return mix, out_rate, len(peaks)


# --------------------------------------------------------------------------- #
# Window selection
# --------------------------------------------------------------------------- #
def _reach_active_and_quiet(path, window_sec, min_gap=2.0):
    """Return ``(t_active, n_reaches, t_quiet)`` time offsets.

    Active = the window containing the most reach onsets.
    Quiet  = a window with zero reach onsets, away from blocklisted breaks.
    """
    with h5py.File(path, "r") as f:
        reaches = _read_intervals(f, "reaches")
        fs, _ = _series_rate(f, "acquisition/ElectricalSeries", 500.0)
        T = f["acquisition/ElectricalSeries/data"].shape[0] / fs
        bad = []
        ep = _read_intervals(f, "epochs")
        if ep is not None and "labels" in ep:
            for st, sp, lab in zip(ep["start_time"], ep["stop_time"], ep["labels"]):
                if "blocklist" in str(lab).lower() or "break" in str(lab).lower():
                    bad.append((float(st), float(sp)))

    onsets = np.sort(np.asarray(reaches["start_time"], dtype=float))

    # Active: slide candidate starts at each onset, count onsets within window.
    best_t, best_n = 0.0, -1
    for o in onsets:
        t0 = max(0.0, o - 1.0)
        if t0 + window_sec > T:
            continue
        n = int(np.sum((onsets >= t0) & (onsets < t0 + window_sec)))
        if n > best_n:
            best_t, best_n = t0, n

    def in_bad(a, b):
        return any(not (b < s or a > e) for s, e in bad)

    # Quiet: scan a grid for a window with no onsets and no blocklist overlap.
    t_quiet = None
    for t0 in np.arange(0.0, max(1.0, T - window_sec), window_sec / 2.0):
        a, b = t0, t0 + window_sec
        if np.any((onsets >= a - min_gap) & (onsets < b + min_gap)):
            continue
        if in_bad(a, b):
            continue
        t_quiet = float(t0)
        break

    return best_t, best_n, t_quiet


def _pick_channel(path, t_active, t_quiet, window_sec):
    """Channel (column index, electrode id, region) with the biggest high-gamma
    increase from quiet -> active. Falls back to highest-variance channel."""
    windows = [{"t0": t_active, "label": "active"}]
    if t_quiet is not None:
        windows.append({"t0": t_quiet, "label": "quiet"})
    ds = WindowedNWBDataset(path, windows, window_sec, ecog_channels="good",
                            pose_keypoints=None)
    hg = list(BANDS).index("high_gamma")
    act = ds[0]
    act_hg, _ = hilbert_bandpower(act["ecog"], act["fs"])
    if t_quiet is not None:
        qui = ds[1]
        qui_hg, _ = hilbert_bandpower(qui["ecog"], qui["fs"])
        score = act_hg[:, hg] - qui_hg[:, hg]
    else:
        score = act_hg[:, hg]
    col = int(np.argmax(score))
    chan_id = int(ds.channels[col])

    region = ""
    with h5py.File(path, "r") as f:
        el = f.get("general/extracellular_ephys/electrodes")
        if el is not None and "location" in el:
            loc = el["location"][chan_id]
            region = loc.decode() if isinstance(loc, bytes) else str(loc)
        # This file stores 'unknown' for every location; fall back to MNI coords.
        if (not region or region.lower() == "unknown") and el is not None \
                and all(k in el for k in ("x", "y", "z")):
            region = "MNI ({:.0f},{:.0f},{:.0f})".format(
                float(el["x"][chan_id]), float(el["y"][chan_id]), float(el["z"][chan_id]))
    ds.close()
    return col, chan_id, region


# --------------------------------------------------------------------------- #
# Clip builder
# --------------------------------------------------------------------------- #
def _gather_clips(path, window_sec, channel_col, max_behaviors=None):
    """Collect raw 1-channel traces for: reach-active, rest, and each behavior.

    Returns a list of dicts: name, label, desc, x (1-D float), fs.
    """
    from nwb_dataset import windows_from_epochs

    clips = []

    t_active, n_reaches, t_quiet = _reach_active_and_quiet(path, window_sec)
    contrast_windows = [{"t0": t_active, "label": "reach_active"}]
    descs = {"reach_active": "{} reach onsets in this window -- listen for the hum "
                             "ducking with repeated crackles".format(n_reaches)}
    if t_quiet is not None:
        contrast_windows.append({"t0": t_quiet, "label": "rest_quiet"})
        descs["rest_quiet"] = "no reaches -- steady resting hum (beta idle rhythm)"

    ds = WindowedNWBDataset(path, contrast_windows, window_sec,
                            ecog_channels="good", pose_keypoints=None)
    for i, w in enumerate(contrast_windows):
        s = ds[i]
        clips.append({"name": "contrast__" + w["label"], "label": w["label"],
                      "desc": descs.get(w["label"], ""),
                      "x": s["ecog"][:, channel_col].astype(np.float64),
                      "fs": s["fs"]})
    ds.close()

    # One window per coarse-behavior label.
    bw = windows_from_epochs(path, window_sec=window_sec, max_per_label=1, seed=0)
    seen = set()
    bw_unique = []
    for w in bw:
        lab = str(w["label"]).strip()
        if not lab or lab in seen:
            continue
        seen.add(lab)
        bw_unique.append(w)
    if max_behaviors:
        bw_unique = bw_unique[:max_behaviors]
    if bw_unique:
        dsb = WindowedNWBDataset(path, bw_unique, window_sec,
                                 ecog_channels="good", pose_keypoints=None)
        for i, w in enumerate(bw_unique):
            s = dsb[i]
            safe = "".join(c if c.isalnum() else "_" for c in str(w["label"]))
            clips.append({"name": "behavior__" + safe, "label": str(w["label"]),
                          "desc": "coarse-behavior epoch: " + str(w["label"]),
                          "x": s["ecog"][:, channel_col].astype(np.float64),
                          "fs": s["fs"]})
        dsb.close()

    return clips


# --------------------------------------------------------------------------- #
# HTML player
# --------------------------------------------------------------------------- #
def _write_index_html(out_dir, rows, meta):
    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>ECoG sonification</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:860px;margin:30px auto;color:#1a1a1a}
 h1{font-size:22px} .meta{color:#555;font-size:13px;margin-bottom:24px}
 .clip{border:1px solid #e2e2e2;border-radius:10px;padding:14px 16px;margin:14px 0}
 .label{font-weight:600;font-size:16px} .desc{color:#555;font-size:13px;margin:4px 0 10px}
 audio{width:100%;margin:6px 0} img{width:100%;border-radius:6px;margin-top:8px}
 .tag{display:inline-block;background:#eef;border-radius:6px;padding:1px 8px;font-size:12px;margin-left:8px}
</style></head><body>
<h1>What the brain sounds like when it moves a hand</h1>"""]
    parts.append('<div class="meta">' + html.escape(meta) + "</div>")
    for r in rows:
        parts.append('<div class="clip">')
        parts.append('<div class="label">' + html.escape(r["label"]) +
                     '<span class="tag">' + html.escape(r["kind"]) + "</span></div>")
        if r["desc"]:
            parts.append('<div class="desc">' + html.escape(r["desc"]) + "</div>")
        parts.append('<audio controls preload="none" src="' + r["wav"] + '"></audio>')
        parts.append('<img loading="lazy" src="' + r["png"] + '" alt="spectrogram">')
        parts.append("</div>")
    parts.append("</body></html>")
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def sonify_file(nwb_path, out_dir="sonified", window_sec=24.0, speedup=12,
                max_behaviors=None, mode="music", audio_sec=10.0, out_rate=22050,
                verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print("Locating reach-active / quiet windows ...")
    t_active, n_reaches, t_quiet = _reach_active_and_quiet(nwb_path, window_sec)
    if verbose:
        print("  active @ {:.1f}s ({} reaches), quiet @ {}".format(
            t_active, n_reaches, "n/a" if t_quiet is None else "{:.1f}s".format(t_quiet)))

    col, chan_id, region = _pick_channel(nwb_path, t_active, t_quiet, window_sec)
    if verbose:
        print("Most movement-modulated channel: column {}, electrode id {}, region '{}'"
              .format(col, chan_id, region))

    clips = _gather_clips(nwb_path, window_sec, col, max_behaviors=max_behaviors)

    # Precompute per-band envelopes + shared per-band scales (music mode). Sharing
    # scales across clips preserves behavior differences while balancing the 1/f
    # power imbalance so the high-gamma voice stays audible.
    band_scales = None
    if mode == "music":
        for c in clips:
            c["env"] = _band_envelopes(c["x"], c["fs"])
        band_scales = {}
        for b in BANDS:
            allv = np.concatenate([c["env"][b] for c in clips])
            band_scales[b] = np.percentile(allv, 99.0) or 1.0
    else:
        all_abs = np.concatenate([np.abs(c["x"] - c["x"].mean()) for c in clips])
        scale = np.percentile(all_abs, 99.7) or 1.0

    rows = []
    for c in clips:
        wav_name = c["name"] + ".wav"
        png_name = c["name"] + ".png"
        if mode == "music":
            audio_f, sr, n_notes = musify_clip(
                c["env"], c["fs"], window_sec, band_scales,
                audio_sec=audio_sec, out_rate=out_rate)
            audio = (audio_f * 32767.0).astype(np.int16)
        else:
            audio, sr = _to_audio(c["x"], c["fs"], speedup=speedup, scale=scale)
            n_notes = None
        wavfile.write(os.path.join(out_dir, wav_name), sr, audio)
        title = "{}  (ch {} / {})".format(c["label"], chan_id, region or "?")
        _spectrogram_png(c["x"], c["fs"], os.path.join(out_dir, png_name), title)
        kind = "reach contrast" if c["name"].startswith("contrast") else "behavior"
        desc = c["desc"]
        if mode == "music" and n_notes is not None:
            desc = (desc + " | " if desc else "") + "{} burst-notes".format(n_notes)
        rows.append({"label": c["label"], "kind": kind, "desc": desc,
                     "wav": wav_name, "png": png_name,
                     "real_sec": round(len(c["x"]) / c["fs"], 1),
                     "audio_sec": round(len(audio) / sr, 2),
                     "out_rate": sr})
        if verbose:
            print("  wrote {}  ({:.1f}s real -> {:.2f}s audio @ {} Hz){}".format(
                wav_name, rows[-1]["real_sec"], rows[-1]["audio_sec"], sr,
                "" if n_notes is None else "  [{} notes]".format(n_notes)))

    if mode == "music":
        meta = ("Source: {} | channel {} ({}) | MODE: music (parameter-mapping) | "
                "5 bands -> C-major chord voices, high-gamma bursts -> plucked notes | "
                "{:.0f}s brain -> {:.0f}s audio".format(
                    os.path.basename(nwb_path), chan_id, region or "?",
                    window_sec, audio_sec))
    else:
        meta = ("Source: {} | channel {} ({}) | MODE: raw audification | {}x speed-up "
                "(beta -> ~{:.0f}-{:.0f} Hz, high-gamma -> ~{:.0f}-{:.0f} Hz) | "
                "{:.0f}s windows".format(
                    os.path.basename(nwb_path), chan_id, region or "?", speedup,
                    BANDS["beta"][0] * speedup, BANDS["beta"][1] * speedup,
                    BANDS["high_gamma"][0] * speedup, BANDS["high_gamma"][1] * speedup,
                    window_sec))

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["label", "kind", "wav", "png",
                                            "real_sec", "audio_sec", "out_rate", "desc"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    _write_index_html(out_dir, rows, meta)
    if verbose:
        print("\nDone. Open: " + os.path.abspath(os.path.join(out_dir, "index.html")))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Sonify AJILE12 ECoG by behavior.")
    ap.add_argument("--nwb", default=DEFAULT_NWB, help="Path to the NWB file.")
    ap.add_argument("--out", default="sonified", help="Output directory.")
    ap.add_argument("--window-sec", type=float, default=24.0,
                    help="Neural window length (s). audio_len = window/speedup.")
    ap.add_argument("--speedup", type=int, default=12,
                    help="Pitch-shift factor (raw audification mode only).")
    ap.add_argument("--mode", choices=["music", "audio"], default="music",
                    help="'music' = parameter-mapping synthesis; 'audio' = raw replay.")
    ap.add_argument("--audio-sec", type=float, default=10.0,
                    help="Output clip length in seconds (music mode).")
    ap.add_argument("--max-behaviors", type=int, default=None,
                    help="Limit number of behavior clips.")
    args = ap.parse_args()
    sonify_file(args.nwb, out_dir=args.out, window_sec=args.window_sec,
                speedup=args.speedup, mode=args.mode, audio_sec=args.audio_sec,
                max_behaviors=args.max_behaviors)


if __name__ == "__main__":
    main()
