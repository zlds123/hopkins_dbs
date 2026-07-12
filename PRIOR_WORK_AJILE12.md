# Prior work on AJILE12 — what's been done, and where this project is new

Grounds mentor note #1 ("what have others done / not done with this data"). Citations
below were located via literature search on 2026-07-12; **entries marked ⚠ still need a
full-text read before citing in the writeup** — this is a map, not a substitute for
reading the papers.

---

## The dataset itself

- **Peterson, Brunton et al. (2022), *Scientific Data*** — "AJILE12: Long-term
  naturalistic human intracranial neural recordings and pose."
  [nature.com/articles/s41597-022-01280-y](https://www.nature.com/articles/s41597-022-01280-y)
  (bioRxiv 2021: [2021.07.26.453884](https://www.biorxiv.org/content/10.1101/2021.07.26.453884v1.full)).
  The data descriptor: 12 participants, 55 semi-continuous days, ECoG @ 500 Hz (≥64
  electrodes/participant, ~1280 h total), synchronized upper-body pose, thousands of
  wrist-movement events, 10 event-related movement features, coarse behavioral-state
  labels, and 14 electrode-level features. **This is the paper defining every field we
  read in `nwb_dataset.py`.** Must-cite as the data source.

## Movement decoding on AJILE (the main published use)

- **Wang, Farhadi et al. — "AJILE Movement Prediction: Multimodal Deep Learning for
  Natural Human Neural Recordings and Video."** ⚠
  [semanticscholar](https://www.semanticscholar.org/paper/6014ee834cf7abcaf33a23eccb6c54a9fb99aed2).
  CNN+LSTM over **ECoG + video**, detecting and *predicting* movement up to ~800 ms
  before onset. This is the closest prior art to our multimodal framing — but note it
  fuses raw video, not pose kinematics, and targets movement onset, not a shared
  interpretable latent. Differentiator for us: we align to **pose/kinematic** behavior
  and study the *representation*, not just prediction lead time.

- **Peterson, Singh, Wang, Rao, Brunton (2021), *eNeuro*** — "Behavioral and neural
  variability of naturalistic arm movements." ⚠ Characterizes the *variability* of
  naturalistic wrist movements and their neural correlates. Relevant to our Phase 1
  timing/variability results (it establishes that naturalistic movement is far more
  variable than task movement — context for why our speed R² is near zero).

## Cross-participant transfer (directly overlaps our H3.3)

- **Peterson, Rao, Brunton (2021), *J. Neural Eng.* — HTNet.** "Generalized neural
  decoders for transfer learning across participants and recording modalities."
  [doi:10.1088/1741-2552/abda0b](https://doi.org/10.1088/1741-2552/abda0b)
  (bioRxiv: [2020.10.30.362558](https://www.biorxiv.org/content/10.1101/2020.10.30.362558v1)).
  **The paper we must position against.** HTNet = a CNN decoder with (a) a Hilbert-transform
  layer computing spectral power at data-driven frequencies and (b) **a layer projecting
  electrode-level data onto predefined brain regions** — trained on pooled ECoG from 11/12
  participants and tested on the held-out participant (LOSO), even transferring to EEG.
  Fine-tuning reached tailored-decoder performance with as few as ~50 ECoG events.
  - **Why this matters for us:** HTNet already showed cross-participant movement decoding
    works *and* that **anatomical region-projection is the key trick** for a common
    feature space across differing electrode layouts. Our `phase3_crosssubject.py` LOSO
    driver deliberately reuses that idea (AAL-region band-power aggregation). We are **not**
    claiming to beat HTNet at movement transfer — our novelty is elsewhere (below).

## Behavioral-state decoding (directly overlaps our T5/T6)

- **(2024) *Front. Hum. Neurosci.* — "Consistent spectro-spatial features of human ECoG
  successfully decode naturalistic behavioral states."** ⚠
  [PMC11169785](https://pmc.ncbi.nlm.nih.gov/articles/PMC11169785/).
  Decodes AJILE12 coarse states (talking, TV, computer/phone vs. sleep/rest) and finds
  they are discriminable from **long-term mean shifts, variance shifts, and covariance
  structure** — i.e. hand-crafted spectro-spatial (band-power) features. **This both
  scoops a naive "we decode behavioral states" claim AND independently supports our
  finding that band-power (M0) is a strong baseline.** Our T5/T6 must cite this and frame
  as: reproducing their band-power result, then asking whether a *learned aligned latent*
  adds interpretability/transfer on top.

## The method we're importing

- **Schneider, Lee, Mathis (2023), *Nature* — CEBRA.** "Learnable latent embeddings for
  joint behavioural and neural analysis." [doi:10.1038/s41586-023-06031-6](https://www.nature.com/articles/s41586-023-06031-6)
  ([arXiv:2204.00673](https://arxiv.org/abs/2204.00673), [code](https://github.com/AdaptiveMotorControlLab/CEBRA)).
  Contrastive latents conditioned on time and/or behavior; emphasizes **consistent,
  interpretable** embeddings. We use CEBRA-Time (M1) / CEBRA-Behavior (M2) as baselines
  and extend to a symmetric **two-tower** (M3). CEBRA has been applied to primate/rodent
  ephys and calcium; **application to human naturalistic ECoG + the two-tower symmetric
  variant appears open.**

---

## The gap this project targets (what is NOT done)

Putting the above together, the following appear unclaimed on AJILE12 and are where our
contribution should be staked — note that **none of these is "beat band-power on movement
AUC"** (HTNet + the 2024 states paper show band-power/region features already decode both
movement and state well):

1. **Symmetric multimodal contrastive alignment (two-tower InfoNCE) on human ECoG+pose.**
   Wang/Farhadi fused ECoG+video for prediction; nobody (found) builds a *queryable shared
   neural↔pose latent* here. → our M3 + bidirectional decode (H3.1).
2. **Latent-space interpretability** — *what* aspects of behavior each embedding dimension
   encodes, and whether the manifold is behavior-organized. This is mentor note #5 and the
   real thesis; `phase3_interpret.py` addresses it. HTNet/CNN work is decode-accuracy-first,
   not representation-first.
3. **Label-efficiency of alignment** under clinically-realistic sparse labels (H3.2). HTNet
   showed fine-tuning efficiency for a *supervised* CNN; the question of whether
   *self-supervised alignment* buys label efficiency is different.
4. **Timescale × representation interaction** — Phase 1 characterized decoding vs. window/lag
   for band-power; whether the optimal timescale differs for learned latents is open.
5. **Translational framing to psychiatric DBS** (slow state, sparse labels, cross-day drift)
   — orthogonal to the motor-BCI framing of all prior AJILE work.

**Honest positioning:** movement decoding and state decoding on AJILE12 are *solved-ish*
with hand features (HTNet; the 2024 states paper). Our defensible novelty is
representation-level (what's encoded, does alignment help when labels are scarce or across
patients, and does it transfer) — exactly the reframe the mentors are pushing.
