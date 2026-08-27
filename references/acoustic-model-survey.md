# Survey: pretrained models for vessel identification from hydrophone audio

**Retrieved 2026-08-27.** The question asked: *is there any publicly downloadable model that
identifies vessels from underwater audio, so we do not have to train one?* Any task counted --
presence, type, distance, count.

This is the evidence behind decisions 0009 and 0010 and behind dropping VTUAD from the plan.
**Be ruthless about "code released" vs "WEIGHTS released"** -- most of the field releases only code.

## The answer

**Yes, one, and it is an unusually good fit.** Ocean Networks Canada released a model trained on
ONC hydrophone spectrograms, and `ICLISTENHF1266` -- our device -- is in its training/eval set.

| | |
|---|---|
| Code | https://github.com/OceanNetworksCanada/selfsupervision_anomalies_onc -- PyTorch, **MIT** |
| Weights | Hugging Face `merileo/*`, MIT-tagged. CPU-viable: `cnn_baseline/cnn_best.pt`; also `ssamba_finetune_small/`. Full SSAMBA finetune `merileo/finetune-amba-base-...-noexclude` (~1.1 GB, GPU only) |
| Labelled eval data | HF `merileo/different_locations_incl_backgroundpipelinenormals_multilabel` (+ `_SMALL.h5`) |
| Classes | `Anomaly, Data Gap, Dropout, Engine Noise, Rain, Sensitivity, Tonal, Unknown Feature` |
| Input | ONC `.mat` spectrogram `[F, T]`, expected `[854, 1000]` -> percentile clip -> `log` -> min-max -> `cv2.resize` to 512x512. `dataset_mean 51.506817`, `dataset_std 13.638703`. **Not WAV.** |
| Devices | `ICLISTENHF1253, HF1266, HF1354, HF6324, HF6094, HF6093, HF6020, HF1951, HF6095, HF1252` + 3 JASCO AMARs |
| Reported for HF1266 | P 0.96 / R 0.88 / AUC 0.99 on 149 samples -- **epoch-1 figures from `result.csv`, unverified** |

Caveats: SSAMBA/Mamba needs Linux + NVIDIA + `mamba_ssm` (we have no GPU -- use the CNN baseline).
Model cards are a bare `license: mit` with **no documented class ordering** -- read `args.pkl` or
the h5 `label_names`. Our product is `.fft.gz` at 1200x512, **not** the expected 854x1000; the
reconciliation is the B0 gate in `acoustics_plan_v2.md`.

## Runners-up

**UATR-CMoE** -- https://github.com/YuanX9/UATR-CMoE, real 135 MB release asset
`best_model.ckpt`, arXiv 2402.11919. ResNet18 mixture-of-experts, input `[B,1,300,300]`.
Classes: `Dredger, Fishboat, Motorboat, Musselboat, Naturalnoise, Oceanliner, Passengers, RORO,
Sailboat` -- **the only released model naming our target population**. Rejected because: **no
licence declared** (legally all-rights-reserved), no feature-extraction script shipped (only
pre-extracted `.npy`), and ShipsEar's discriminative content is heavily sub-1 kHz where our floor
bites.

**PANNs / AudioSet, zero-shot.** AudioSet's 527 classes include vessel classes outright:

| idx | mid | name |
|---|---|---|
| 301 | /m/019jd | Boat, Water vehicle |
| 302 | /m/0hsrw | Sailboat, sailing ship |
| 303 | /m/056ks2 | Rowboat, canoe, kayak |
| **304** | /m/02rlv9 | **Motorboat, speedboat** |
| 305 | /m/06q74 | Ship |

So any AudioSet-trained tagger emits a vessel score with zero training. PANNs CNN14 weights:
https://zenodo.org/records/3576403 (MIT code, CC-BY weights), 32 kHz, mel fmin 50 / fmax 14000.
YAMNet is 16 kHz, fmin 125, and **521** classes not 527 -- use its own `yamnet_class_map.csv`.
CLAP-LAION (https://huggingface.co/laion/clap-htsat-unfused, Apache-2.0) is text-queryable, so
prompts can name the underwater domain.

**Nobody has published how any of these performs on underwater vessel audio.** The nearest number
is a linear-probing benchmark (https://arxiv.org/html/2601.08358): BEATs frozen embeddings + linear
probe reach **74% on ShipsEar / 65.4% on DeepShip**, with the paper's own warning that the
embedding spaces are "dominated by recording-specific artifacts rather than ship-type structure."
That absence is informative -- if raw class 304 worked underwater, someone would have said so.
Treat it as a one-hour experiment, not a plan.

## Ruled out: code released, weights NOT released

* **Domingos et al. 2022** (IEEE Access, VTUAD's author) -- https://github.com/lucascesarfd/underwater_snd
  and https://github.com/lucascesarfd/onc_dataset, both **MIT, both code-only, zero checkpoints**.
  Published VTUAD benchmark to reproduce: 94.95 / 94.45 / 93.11 % per scenario, 84.13 % combined.
  Preprocessing: 32 kHz, 1 s clips, `FREQ_BINS=95`, `N_FFT=2048`, `HOP=256`, **`FMIN=18`,
  `FMAX=4186`** -- note that is *below* most of our support.
  `onc_dataset` is the ONC->AIS->WAV labelling pipeline, and confirms **VTUAD labels are AIS ship-type
  codes** (`generate_metadata.py`), contradicting a secondary source that called it AIS-independent.
* **Decrop, Deneudt, Parcerisas, Schall, Debusschere 2025**, IEEE JSTARS
  `10.1109/JSTARS.2025.3593779`, full text https://www.vliz.be/imisdocs/publications/416272.pdf.
  Dataset open, CC-BY, `10.14284/723`, ~26.5k x 10 s clips, 116 days, Belgian North Sea, 48 kHz.
  **No code, no weights -- no data/code availability statement anywhere in the paper.**
  Task: 11 x 1 km distance bins to nearest AIS vessel. Base model BioLingual/CLAP.
  Pleasure craft: **75 occurrences**, merged into "other". **They already ran our experiment:**
  applied to DeepShip (Strait of Georgia BC) performance "dropped notably... does not transfer
  seamlessly across environments with differing sound propagation" -- confusion matrix only, no
  numbers.
* **Renaud, Aravindan, Spadon 2026**, `arXiv:2607.13840`, OCEANS'26. AIS-aligned PAM labelling with
  continuous range, CPA, and a no-/single-/two-contact window taxonomy -- conceptually exactly the
  label design we want. **Method only: no data, no code, JASCO AMARs not ONC, no vessel length, and
  non-AIS small craft explicitly excluded.** Good methodology citation, not a source.
* **CATFISH** https://github.com/Jotels/CATFISH (arXiv 2505.23964) -- releases endpoint returns
  `[]`. Reports the Domingos confusion structure: 14% of background predicted as tug, 6% tug as
  background, which CATFISH reduces to 2.2 / 1.9%.
* Scanned with empty results for weight files: `HLAST_DeepShip_ParameterEfficient` (release tag with
  empty assets), `PANN_Models_DeepShip`, `NEHD_UATR`, `ADMIS-TONGJI/UATR-benchmark`,
  `gmlfks/Deepship_Underwater`, `Sunset-Shen/shipsear-resnet18`, and others.
* **Zenodo and Hugging Face:** no vessel-model weight records beyond the `merileo` family. One
  unlabelled lead, `hildehummel/Conformer_UATR` (1.33 GB `.ckpt`, no README, no licence, no code,
  no card) -- not usable without contacting the uploader.

## Ruled out: the marine-acoustics ecosystem ships whale models, not vessel models

Ketos/MERIDIAN (GPLv3; only the North Atlantic Right Whale upcall detector), AISdb, PAMGuard
(Noise Band / Noise Monitor are *measurement* modules), NOAA NEFSC/PIFSC/NCEI (cetacean detectors
and hybrid-millidecade sound-level products), Triton/HARP/Scripps, Orcasound (`orca-noise-model` is
a placeholder; their vessel classifier is *visual*), OpenSoundscape / koogu / animal-spot,
Perch / AVES / BioLingual, ECHO/JASCO (commercial), MBARI / OOI / JONAS / JOMOPANS / QUIETMED
(standards and metrics, not classifiers). **AquaSignal** (arXiv 2505.14285, trained on DeepShip +
ONC, classification *and* novelty detection) is the one tantalising case with no released code or
weights -- worth an email, not worth planning around.

## The non-ML alternative, which is the community standard

If no model survives, the published method is **band-level SPL against a rolling percentile
ambient**: decidecade (third-octave / hybrid-millidecade) band levels, vessel passage declared when
bands exceed the background by ~3-6 dB for a minimum duration, confirmed by the Lloyd's-mirror
hyperbolic pattern and the broadband cavitation hump.

Reference implementations, **all PSD-domain, so all run on `.fft.gz` without touching WAV**:

| Tool | URL |
|---|---|
| PyPAM | https://github.com/lifewatch/pypam |
| MBARI `pbp` | https://github.com/mbari-org/pbp |
| MHKiT-Python (IEC TS 62600-40; **explicitly supports icListen**) | https://github.com/MHKiT-Software/MHKiT-Python |
| IOOS SoundCoop notebooks | https://github.com/ioos/soundcoop |

Standards and precedent: JOMOPANS terminology
(https://northsearegion.eu/media/17741/jomopans_wp3-standard-terminology_final.pdf); ADEON /
hybrid-millidecade; ISO 17208-1 / ANSI-ASA S12.64 are *measurement* standards, **not** detection
algorithms -- do not cite them as such. DEMON (Ferguson et al.,
https://ieeexplore.ieee.org/document/7943488/) is the classical small-boat propeller detector but
**needs waveforms and will not run on PSD**. Closest applied precedent to our use case -- small
non-AIS vessels, single bottom-mounted hydrophone, MPA compliance:
https://www.sciencedirect.com/science/article/pii/S0308597X19309005 and
https://link.springer.com/article/10.1007/s10661-024-12497-2.

## Loose end worth one email

**VTUAD v2** (Domingos, Brinkworth, Santos, Sammut 2025, `arXiv:2512.11165`) describes ten classes
including **`pleasure craft`, `sailing` and `fishing`** -- 1,111 recordings, 479 ships, ~116 h. It
is the only corpus found anywhere containing our target population. **No release location exists**
-- not DataPort, not Zenodo, not GitHub. The author's address is in the paper. Not pursued this
week; recorded here so it is not rediscovered from scratch.
