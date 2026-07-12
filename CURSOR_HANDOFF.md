# Cursor Handoff — hopkins_dbs

**Session transcript:** `C:\Users\Zhiyu\.cursor\projects\c-Users-Zhiyu-Downloads-hopkins-dbs\agent-transcripts\f8d879c1-0912-4502-a49a-56dc833b8454\f8d879c1-0912-4502-a49a-56dc833b8454.jsonl`

---

## Original goal and current status

**Goal:** Build a reproducible AJILE12 (human ECoG + pose) analysis stack and execute a three-phase research program on **closed-loop neural decoding**: temporal resolution (Phase 1) → online adaptation under drift (Phase 2) → multimodal alignment / transfer (Phase 3), with translation framing to psychiatric DBS (slow state, sparse labels).

**Status:**

| Phase | Status | Summary |
|-------|--------|---------|
| **Env + data profiling** | Done | Conda `dbs` (py3.7) + `dbs-ml` (py3.10); memory-safe NWB pipeline |
| **CEBRA (one-tower)** | Done | Trained/evaluated; raw band-power wins at full supervision |
| **Phase 1** | Done (N=1) | H1.1–H1.3 supported on sub-01; extensions run; mentor notebooks executed |
| **Phase 2 feasibility** | Done (N=1) | Within-session drift weak; rotation positive control validates adaptation machinery |
| **Phase 3** | **Designed only** | Experiment matrix + interpretation guide written; **no code** (`two_tower.py`, `phase3_eval.py` missing) |
| **Literature / proposal** | Done | Canvas + academic writeup; Phase 3 conclusion contextualization (neuro / biomarker / control) |

**Blocking gap:** Only one full AJILE12 NWB locally (`sub-01_ses-3`). Cross-subject claims (H3.3) cannot be tested yet.

---

## Requirements and constraints

- **Data:** `C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb` (~15 GB). Never load whole ECoG; slice via h5py windows only. `*.nwb` gitignored.
- **Environments:**
  - `conda activate dbs` — Python 3.7, pinned 2020-era stack (original notebooks, `brunton_lab_to_nwb`).
  - `conda activate dbs-ml` — Python 3.10, CEBRA 0.6.1, torch 2.x (CPU), sklearn, h5py. **Use for all phase/CEBRA scripts.**
- **Run pattern:** `conda run -n dbs-ml python <script>.py ...` from `C:\Users\Zhiyu\Downloads\hopkins_dbs`.
- **Commits:** User commits manually; do not commit unless asked. Most new files are still untracked.
- **Translation framing:** AJILE12 = fast motor sandbox; psychiatric DBS claims need RAM (stim-logged) / DABI (psychiatric) data later. Do not over-claim motor → depression transfer.
- **Evaluation hygiene:** Time-blocked CV (not random); pose as **target** not input for honest neural decoding; prequential test-then-train for Phase 2 streaming.

---

## Architecture and decisions

### Data layer (`nwb_dataset.py`)
- **`WindowedNWBDataset`**: lazy window reads for PyTorch/sklearn.
- **`hilbert_bandpower`**: 5 bands (θ, α, β, low-γ, **high-γ 70–110 Hz**); log envelope; common-median ref already in file.
- **`build_continuous_stream`**: aligned 30 Hz grid — `X` (band power), `vel`, `speed`, `reach`, `epoch` labels; cached `.npz` streams.
- **Window anchors:** `find_active_window` (reach-dense), `find_movement_window` (wrist-speed-rich).
- **Channel selection:** `good` (85) | `sensorimotor` (AAL → coord-box fallback) | `aal` | `box` (18 peri-central).
- **AAL:** `_fetch_aal_atlas()` prefers cached SPM12 atlas; SSL retry fallback.

### Phase 1 (`phase1_resolution.py`)
- Sweeps window length (H1.1), lag (H1.2), causal vs acausal cost (H1.3).
- Features: `band` | `cebra` | `both`; labels: `reach` | `speed_balanced` | `speed_median`.
- Blocked 5-fold CV; logistic regression; outputs CSV + PNG per sweep.
- Batch runner: `phase1_extensions.py`.

### CEBRA (`cebra_ajile.py`, `cebra_analyze.py`, `cebra_label_efficiency.py`)
- Input: high-γ @ 30 Hz (not raw 500 Hz ECoG).
- Models: CEBRA-Time + CEBRA-Behavior (wrist velocity aux).
- Decode: kNN on embeddings vs logistic on raw; label-efficiency sweep.
- **Decision:** Single-tower only here; two-tower deferred to Phase 3.

### Phase 2 (`phase2_feasibility.py`)
- Prequential streaming: static | online-SGD | sliding-refit (K recent blocks).
- Causal per-block normalization (no global z-score — masks drift).
- `--induce-drift`: progressive random **rotation** in standardized feature space (replaced gain+offset perturbation).
- Verdicts: `YES` | `DECODABLE BUT NO DRIFT IN THIS SPAN` | `INCONCLUSIVE`.

### Phase 3 (planned, not built)
- Models: M0 raw, M1 CEBRA-Time, M2 CEBRA-Behavior, M3 **two-tower InfoNCE**.
- Targets T1–T6: reach, movement, speed, velocity, coarse epochs, sleep vs active.
- Dims {8, 16, 32}; blocked CV; label efficiency; CKA cross-span/subject; bidirectional decode.

### Proposal artifacts (outside repo root)
- `C:\Users\Zhiyu\.cursor\projects\c-Users-Zhiyu-Downloads-hopkins-dbs\canvases\DBS-literature-landscape.canvas.tsx`
- `C:\Users\Zhiyu\.cursor\projects\c-Users-Zhiyu-Downloads-hopkins-dbs\canvases\phase2-feasibility-explainer.canvas.tsx`

---

## Files created or modified

### Core library
| File | Change |
|------|--------|
| `nwb_dataset.py` | **Created/extended.** Windowed dataset, Hilbert features, reach/epoch windows, `build_continuous_stream`, `find_movement_window`, `sensorimotor_channels`, AAL mapping with SSL/cache fix, electrode coords. |

### Phase 1
| File | Change |
|------|--------|
| `phase1_resolution.py` | **Created/extended.** Full H1.1–H1.3 sweeps; `--anchor`, `--features`, `--channel-method`, `--label`, multi-file, CEBRA embedding path, stream caching. |
| `phase1_extensions.py` | **Created.** Batch runner for movement/AAL/box/CEBRA/multisub experiments. |
| `phase1_report.ipynb` | **Created/executed.** Mentor report for baseline 45-min reach-dense run. |
| `phase1_extensions_report.ipynb` | **Created/executed.** Extension results notebook. |
| `phase1_out/`, `phase1_smoke/` | Output dirs: window/lag/causal CSVs, PNGs, cached streams. |
| `phase1_out_movement/`, `phase1_out_aal/`, `phase1_out_box/`, `phase1_out_cebra/`, `phase1_out_multisub/` | Extension outputs. |

### CEBRA
| File | Change |
|------|--------|
| `cebra_ajile.py` | **Created.** Train CEBRA-Time/Behavior, plots, decode comparison. |
| `cebra_analyze.py` | **Created.** Blocked CV vs single-split diagnostics on cached embeddings. |
| `cebra_label_efficiency.py` | **Created.** AUC vs #labels sweep. |
| `cebra_out/`, `cebra_out2/` | 60-min reach window + 30-min movement/sensorimotor runs; embeddings, plots, results. |

### Phase 2
| File | Change |
|------|--------|
| `phase2_feasibility.py` | **Created.** Streaming decoder feasibility + induced rotation control. |
| `phase2_feasibility_explainer.html` | **Created.** Static HTML explainer (user committed manually). |
| `phase2_out_observed/`, `phase2_out_induced/`, `phase2_out/` | Feasibility CSVs + PNGs. |

### Earlier session artifacts (still in repo)
| File | Role |
|------|------|
| `sonify.py`, `sonified/` | ECoG sonification demo |
| `notebook.ipynb`, `data_profiling.ipynb` (if present) | Exploration notebooks |
| `.gitignore` | Ignores `*.nwb`, `ajile12-nwb-data/`, caches |

### Modified tracked files (per git status snapshot)
- `nwb_dataset.py`, `phase1_resolution.py`, `phase1_report.ipynb` — modified vs last commit; many other files untracked.

### Not created (explicit gaps)
- `two_tower.py`, `phase3_eval.py`
- RAM ds005557 loader for real Phase 2 stim data
- BrainBERT feature path (H1.4 deferred)

---

## Key results (sub-01 only)

### Phase 1 — 45-min reach-dense, 18 sensorimotor ch, beta+high-γ (`phase1_out/`)
- **H1.1:** Peak movement AUC **~0.815 @ 0.2 s** window.
- **H1.2:** Peak AUC **~0.846 @ −0.5 s** lag (neural leads behavior).
- **H1.3:** Causal cost up to **~+0.20 AUC** at 2 s window vs acausal.
- **Speed R²:** Negative across sweeps (continuous kinematics not linearly decodable).

### Phase 1 extensions — 30-min spans
- **Movement-rich window + speed_median label:** AUC ~0.49–0.50 (chance); high movement variance does not yield decode.
- **AAL (5 ch):** Peak AUC ~0.67 vs **coord-box (18 ch) ~0.77** — denser motor coverage wins.
- **CEBRA 8D vs band (reach-dense):** Band peak ~0.77–0.78; CEBRA ~0.75–0.77 — **band ≥ CEBRA**.
- **Multisub:** Pipeline works; summary has **N=1** only.

### CEBRA — 60-min reach window (`cebra_out/`)
- Reach AUC: raw **0.675** > CEBRA-Time **0.638** > CEBRA-Behavior **0.578**.
- Wrist-speed R² ≈ 0 (blocked CV confirms weak signal, not just drift).
- Label efficiency: raw wins at full labels; CEBRA-Time slight edge only at ~1.5k labels.

### CEBRA — 30-min movement window (`cebra_out2/`)
- No reaches in span; move AUC ~0.50; speed R² strongly negative.

### Phase 2 — observed vs induced (`phase2_out_observed/`, `phase2_out_induced/`)
- **Observed:** static 0.626, online 0.631, sliding 0.649; static AUC **rose** (0.582→0.650) → **no within-session drift**.
- **Induced rotation (strength 1.0):** static 0.533 (decays), online 0.607, sliding 0.626 → **adaptation recovers under representational drift**.
- Verdict observed: `DECODABLE BUT NO DRIFT IN THIS SPAN`; induced: machinery validated.

---

## Bugs, errors, and failing tests

**No automated test suite.** Issues encountered and resolved:

| Issue | Resolution |
|-------|------------|
| `conda` not on PATH after Miniconda install | `conda init powershell`; user must restart terminal |
| SSL errors in `dbs` env pip | Use `conda run -n dbs pip ...` |
| AAL atlas download SSL (`gin.cnrs.fr`) | SPM12 cache + unverified SSL fallback in `_fetch_aal_atlas()` |
| CEBRA arch `offset10` invalid | Use `offset10-model` |
| Phase 2 false-positive `PHASE 2 FEASIBLE: YES` | Tightened verdict logic; balanced null model |
| Phase 2 SGD float32/float64 mismatch | Fixed dtype casting |
| Gain+offset induced drift AUC-preserving | Replaced with rotation perturbation |
| Movement extension: no reaches → NaN AUC | Added `--label speed_median`; `summarize()` NaN guard |
| Git commit blocked in agent sandbox | User commits manually |
| `phase1_multisubject_summary.csv` empty when no rows | Only written when multisub runs produce data |

**Currently open / not fixed:**
- Within-session AJILE12 drift too weak for Phase 2 headline claims on observed data alone.
- Only one subject — cross-subject pipeline untested.
- Phase 3 unimplemented.
- `cebra_out2` movement window: decoding at chance.

---

## Commands run (representative)

```powershell
# Environment
conda create -n dbs python=3.7 -y
conda create -n dbs-ml python=3.10 -y
conda run -n dbs-ml pip install cebra h5py scipy "numpy<2" matplotlib scikit-learn

# CEBRA
conda run -n dbs-ml python cebra_ajile.py --minutes 60 --dim 3 --out cebra_out
conda run -n dbs-ml python cebra_analyze.py --dir cebra_out
conda run -n dbs-ml python cebra_label_efficiency.py --dir cebra_out

# Phase 1
conda run -n dbs-ml python phase1_resolution.py --file "C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb" --dur-min 6 --channels good --out-dir phase1_smoke
conda run -n dbs-ml python phase1_resolution.py --file "C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb" --dur-min 45 --channels sensorimotor --out-dir phase1_out
conda run -n dbs-ml python phase1_extensions.py --file "C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb" --dur-min 30
# (CEBRA extension run separately — outputs in phase1_out_cebra/)

# Phase 2
conda run -n dbs-ml python phase2_feasibility.py --file "C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb" --out-dir phase2_out_observed
conda run -n dbs-ml python phase2_feasibility.py --file "C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb" --induce-drift 1.0 --out-dir phase2_out_induced
```

All completed without persistent script crashes after fixes above. Phase 1 full run ~7 min; Phase 2 ~3–5 min per run; CEBRA 60-min ~tens of minutes on CPU.

---

## Approaches attempted and rejected

| Approach | Why rejected |
|----------|--------------|
| Feed raw 500 Hz ECoG to CEBRA | Memory/scale; use 30 Hz high-γ envelope stream |
| Pose features as decoder **inputs** | Label leakage for reach detection; pose is target |
| Global z-score on full stream (Phase 2) | Removes drift signal being measured |
| Reach label for Phase 2 (1% positive) | Collapsed online-SGD; use balanced speed or reach with all good channels |
| Gain+offset induced drift | Linear decoder AUC-invariant; rotation used instead |
| Two-tower CEBRA in initial pass | Scope; reserved for Phase 3 |
| BrainBERT / H1.4 in Phase 1 | Deferred; band-power + CEBRA sufficient for pilot |
| Random (non-blocked) CV | Inflates scores under nonstationarity; blocked CV adopted |
| AAL-only channels without box fallback | Only 5 ch locally; box fallback or denser coverage needed |
| Claiming Phase 2 feasible from observed AJILE12 alone | Static decoder did not decay within session |

---

## Unresolved questions and risks

1. **Does two-tower alignment beat raw features** when labels are scarce or across subjects? (Phase 3 core question; untested.)
2. **Cross-subject transfer:** Need 3–6 AJILE12 sessions from [Dandiset 000055](https://dandiarchive.org/dandiset/000055).
3. **Real drift:** Within-session AJILE12 appears stationary; cross-day / cross-session drift unknown.
4. **Psychiatric translation:** RAM (ds005557) for logged stim + slow targets; DABI for psychiatric cohort — no loaders built.
5. **Continuous kinematics:** Speed R² ≤ 0 across representations — may limit fine motor BCI claims; event/state decoding may be the viable path.
6. **CEBRA hyperparameters:** Dim, iterations, window choice materially affect embeddings; no systematic sweep in Phase 1 extensions beyond 8D/1500 iter.
7. **Git hygiene:** Large output dirs and scripts mostly untracked; user manages commits.

---

## Recommended next steps (in order)

1. **Implement Phase 3 core**
   - `two_tower.py`: dual encoders (ECoG band-power tower + pose/behavior tower), InfoNCE alignment, dim sweep {8,16,32}.
   - `phase3_eval.py`: unified eval for T1–T6, blocked CV, label-efficiency curves, CKA cross-span consistency, bidirectional decode.
   - Reuse `build_continuous_stream` + cached `.npz`; mirror `cebra_ajile.py` caching pattern.

2. **Download more AJILE12 NWBs** into `ajile12-nwb-data/` (or parent `Downloads/`), rerun `phase1_extensions.py` multisub + Phase 3 cross-subject metrics.

3. **Run Phase 3 pilot on sub-01** before scaling — compare M0–M3 on T1 (reach) and T2 (movement) first; expand to T3–T6 if runtime acceptable.

4. **Update mentor artifacts** — fold Phase 1 extension + Phase 2 verdicts into `phase1_report.ipynb` or a single combined writeup; link `phase2_feasibility_explainer.html`.

5. **RAM Phase 2 loader** (when data available) — replace induced-drift argument with real stim-logged sessions for observed drift.

6. **Optional:** BrainBERT feature baseline (H1.4); only if mentor requests after Phase 3 pilot.

---

## Quick reference paths

| Resource | Path |
|----------|------|
| Primary NWB | `C:\Users\Zhiyu\Downloads\sub-01_ses-3_behavior+ecephys.nwb` |
| Project root | `C:\Users\Zhiyu\Downloads\hopkins_dbs` |
| ML env | `conda activate dbs-ml` |
| Phase 1 outputs | `phase1_out/`, `phase1_out_*/` |
| CEBRA outputs | `cebra_out/`, `cebra_out2/` |
| Phase 2 outputs | `phase2_out_observed/`, `phase2_out_induced/` |
| Literature canvas | `.cursor/projects/.../canvases/DBS-literature-landscape.canvas.tsx` |

---

*Handoff written 2026-07-11. No code changes beyond this file.*
