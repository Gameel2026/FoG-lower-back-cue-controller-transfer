# Laboratory-to-home transfer of a lightweight temporal cue controller for freezing of gait

Code, registered analysis plan and results for the manuscript
**"Laboratory-to-home transfer of a lightweight temporal cue controller for freezing of gait: an external-validation study with a single lower-back accelerometer"** (submitted to *Biomedical Engineering Letters*).

## 1. Question

Can a cue controller developed on **laboratory** recordings from a single lower-back accelerometer be transferred,
**without retraining or re-tuning**, to **home** recordings and to independent laboratory datasets, while keeping its
early-activation and false-alarm behaviour?

The controller is an *anticipatory cue controller*: it is trained to switch a cue on during the 4 s before freezing
(Pre-FoG), during freezing and during the 3 s after it, and it is evaluated with two event-level outcomes:

| Outcome | Meaning |
|---|---|
| **Timely activation** | cue active within the 4 s before onset (eligible episodes: ≥ 4 s without FoG before onset) — *anticipation* |
| **Episode coverage** | cue active at any time during the episode, including after onset — called *detection* in the registered plan |
| False alarms per hour | activation runs not overlapping any interval from 4 s before onset to 3 s after offset |
| Cue specificity | proportion of windows not requiring a cue in which the cue was off |

Every result is compared with a **circular-shift surrogate** (chance reference).

## 2. Data (all public; not redistributed here)

| Role | Dataset | Subjects | Source |
|---|---|---|---|
| Development (all of it) | tdcsfog (laboratory) | 62 | FoG contest data, Zenodo https://doi.org/10.5281/zenodo.10959560 (`train/tdcsfog`) |
| **Primary test** | **defog (home)** | **45** | same resource (`train/defog`, `train/notype`); no subject shared with tdcsfog |
| Secondary test | FoG-STAR | 22 | lower-back sensor, converted with `fogstar_convert.py` |
| Secondary test | DAPHNET | 10 | trunk sensor, https://doi.org/10.24432/C56K78 |

All signals: lower-back accelerometer, 64 Hz, milli-g, axes ordered forward/vertical/lateral where documented;
FoG segments separated by < 2 s merged into one episode.

## 3. What was registered, and when

The registration covers the **transfer evaluation**, not the whole study. Chronology:

| Step | What was done | Data used |
|---|---|---|
| 1 | Conversion, annotation statistics, choice of the 2 s merging rule (`kaggle_merge_check.py`) | labels of tdcsfog and defog, no model output |
| 2 | Development (`bmel_dev.py`, `bmel_dev2.py`); the initial hypothesis (LSTM better than RF) was **not** supported | tdcsfog only |
| 3 | LSTM chosen as primary controller (efficiency); operating-point rule; final training length (`bmel_freeze.py`) | tdcsfog only |
| 4 | `PREREGISTRATION_BMEL.md` + `frozen_config.json` committed (**48e0c87**, 6 Oct 2026 00:12 GMT+3); `bmel_transfer.py` committed (**5bce1f6**, 00:20) | none |
| 5 | Frozen controllers applied to defog, FoG-STAR, DAPHNET (`bmel_transfer.py`) | test data |
| 6 | Exploratory, sensitivity and robustness analyses — **labelled post hoc** | test data |

`PREREGISTRATION_BMEL.md` and `frozen_config.json` are kept exactly as committed in step 4.

## 4. Scripts (run in this order)

| # | Script | Purpose | Output |
|---|---|---|---|
| 1 | `kaggle_convert.py` | convert tdcsfog/defog to the DAPHNET text format (lower back, 64 Hz, mg, Valid & Task, 2 s merging) | `daphnet_format/` |
| 2 | `kaggle_merge_check.py` | annotation statistics for the merging rule (raw labels) | screen |
| 3 | `bmel_dev.py` | development: 8 controllers, 10 subject-grouped folds, surrogate, cost | `result/results_bmel_dev/` |
| 4 | `bmel_dev2.py` | trade-off curves, LSTM ≥ 5 epochs | `result/results_bmel_dev2/` |
| 5 | `bmel_freeze.py` | fixes operating points and training length from development data only | `frozen_config.json` |
| 6 | `bmel_transfer.py` | **registered transfer evaluation** | `result/results_bmel_transfer/` |
| 7 | `bmel_posthoc.py` | post hoc P1–P4: within-dataset retraining, home false alarms by task, local recalibration, axis check | `result/results_bmel_posthoc/` |
| 8 | `bmel_posthoc_axes.py` | post hoc P5: FoG-STAR after aligning the vertical axis | `result/results_bmel_posthoc/P5_*.csv` |
| 9 | `bmel_oracle.py` | post hoc P6: oracle activity context (cue off in balance tasks; upper bound) | `result/results_bmel_posthoc/P6_*.csv` |
| 10 | `bmel_sensitivity.py` | sensitivity: 2,000 shifts, task-preserving surrogate, deployment cost | `result/results_bmel_sensitivity/` |
| 11 | `bmel_robustness.py` | robustness: patient as unit of inference, merging rules 1 s and 3 s | `result/results_bmel_robustness/` |

The scripts reuse shared pipeline modules (windowing, features, hysteresis, event metrics, LSTM training), which are
included in the folder `pipeline/` (`fog_pipeline.py`, `fog_study.py`, `fog_improve.py`, `fog_fix3.py`, `fog_deep.py`,
`fog_lstm_controller.py`, `fog_floor_sensitivity.py`, `fog_stats.py`, `paths.py`). Copy them next to the scripts before
running. The scripts write their outputs to `results_bmel_*/` in the working folder; the files from the study are stored
in `result/`. Paths to the data are set at the top of each script and in `paths.py`.

Requirements: see `requirements.txt` (Python 3.10+, numpy, pandas, scipy, scikit-learn, PyTorch, joblib, matplotlib).

## 5. Main results

| | Timely activation (chance), p | Episode coverage (chance), p | False alarms/h |
|---|---|---|---|
| **defog, home (primary)** | 17.7% (19.9%), 0.86 | 78.6% (77.9%), 0.31 | 150.7 |
| FoG-STAR | 20.8% (19.3%), 0.39 | 65.1% (64.6%), 0.56 | 103.7 |
| DAPHNET | 34.9% (13.2%), 0.005 | 92.8% (53.4%), 0.005 | 79.8 |

Frozen LSTM controller (θon 0.65, θoff 0.40), 200 shifts as registered. Neither primary hypothesis was supported.
The conclusions were unchanged with 2,000 shifts, a task-preserving surrogate, the patient as the unit of inference,
and merging rules of 1 s and 3 s. Deployment cost (laptop): LSTM 0.28 MB, 8.3 ms processing per 0.5 s window;
RF reference 228 MB, 33 ms.

## 6. Notes

* Large intermediate files (`*.pkl`, `*.npy`) are not included; they are recreated by the scripts.
* **Episode counts.** All event metrics count episodes within continuous recording segments (`fog_study.event_metrics`);
  for defog this gives **625 episodes (401 eligible)**, the number reported in the manuscript. The descriptive table
  `results_bmel_transfer/T5_datasets.csv` (611) counted contiguous FoG runs without separating recordings, so two
  episodes at the end of one recording and the start of the next were counted once; this affected only that descriptive
  count, not any metric, threshold or registered result.
* **Terminology.** The outcome called *detection* in `PREREGISTRATION_BMEL.md` and in the script outputs is reported as
  *episode coverage* in the manuscript (cue active at any time during the episode, including after onset).
* **Patient-level analysis.** The patient-specific surrogate analysis (`bmel_robustness.py`) was added after registration
  and is reported as supporting evidence; the registered primary analysis is the pooled circular-shift test.
* Analyses after step 5 are post hoc and explanatory; they do not change the registered result.

## 7. Citation

Please cite the article once published, or this repository:
Soliman AM, Nashaat Gamil DY, Hammad MS, Hassan MA. FoG lower-back cue controller transfer (code and registered
analysis plan). GitHub, 2026. https://github.com/Gameel2026/FoG-lower-back-cue-controller-transfer

## 8. Licence

Code: MIT (see `LICENSE`). The datasets remain under their original licences.
