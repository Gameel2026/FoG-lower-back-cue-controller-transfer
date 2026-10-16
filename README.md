# Laboratory-to-home transfer of a lightweight temporal cue controller for freezing of gait

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23418062.svg)](https://doi.org/10.5281/zenodo.23418062)

Code, registered analysis plan and results for the manuscript
**"Laboratory-to-home transfer of a lightweight temporal cue controller for freezing of gait: a multi-dataset validation study with a single lower-back accelerometer"** (submitted to *Biomedical Engineering Letters*).

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
| 12 | `conf2_step1_analysis.py` | completed calibration analysis (plan `CONF2_PLAN.md`); provides the home-trained population thresholds (C-pop); needs `conf_step1_analysis.py` | `results_conference2/` |
| 13 | `bmel_posthoc2.py` | post hoc P7–P8 (`POSTHOC_ADDENDUM.md`): home-trained controller, full threshold sweep at home | `results_bmel_posthoc2/` |
| 14 | `bmel_posthoc3.py` | post hoc P9–P10 (`POSTHOC_ADDENDUM_2.md`): medication state, cue-activation burden | `results_bmel_posthoc3/` |
| 15 | `bmel_posthoc4.py` | post hoc (`POSTHOC_ADDENDUM_3.md`): timing of activation relative to onset, ten further training seeds, training-free baselines (freeze index, movement power) | `results_bmel_posthoc4/` |
| 16 | `bmel_posthoc5.py` | post hoc (`POSTHOC_ADDENDUM_4.md`): subject overlap tdcsfog/defog, timing by protocol task, temporal-context ablation and compact random forest, patient bootstrap intervals, software and hardware | `results_bmel_posthoc5/` |
| 17 | `bmel_posthoc6.py` | post hoc (`POSTHOC_ADDENDUM_5.md`): home-trained LSTM with early stopping and thresholds selected on held-out training subjects | `results_bmel_posthoc6/` |
| 18 | `bmel_posthoc7.py` | post hoc (`POSTHOC_ADDENDUM_6.md`): frozen controllers on defog with the forward axis inverted | `results_bmel_posthoc7/` |

### Paper labels and result files

| Paper label | What | Script | Result file |
|---|---|---|---|
| H1–H3 | registered transfer evaluation | `bmel_transfer.py` | `result/results_bmel_transfer/T1–T6` |
| supporting | patient level, 2,000 shifts, task-preserving surrogate, merging rules, deployment cost | `bmel_robustness.py`, `bmel_sensitivity.py` | `result/results_bmel_robustness/`, `result/results_bmel_sensitivity/` |
| P1–P4 | within-dataset retraining, home false alarms by task, threshold re-selection, axis check | `bmel_posthoc.py` | `result/results_bmel_posthoc/P1–P4` |
| P5, P6 | FoG-STAR axis alignment, oracle activity context | `bmel_posthoc_axes.py`, `bmel_oracle.py` | `result/results_bmel_posthoc/P5`, `P6` |
| P7, P8 | home-trained LSTM (C-pop thresholds), threshold grid | `bmel_posthoc2.py` | `results_bmel_posthoc2/Q1*`, `Q2*` |
| P9, P10 | medication state, cue-activation burden | `bmel_posthoc3.py` | `results_bmel_posthoc3/R1_medication.csv`, `R2_burden.csv` |
| P11–P13 | activation timing, training seeds, training-free baselines | `bmel_posthoc4.py` | `results_bmel_posthoc4/A1–A3` |
| B1–B5 (P15 = B3) | subject overlap, timing by task, current-window LSTM and depth-limited RF, patient bootstrap, environment | `bmel_posthoc5.py` | `results_bmel_posthoc5/B1–B5` |
| P14 | home training with early stopping | `bmel_posthoc6.py` | `results_bmel_posthoc6/H1*` |
| P16 | forward axis inverted at home | `bmel_posthoc7.py` | `results_bmel_posthoc7/F1_forward_axis.csv` |

File prefixes R1/R2 and H1 are reused in different folders: `results_bmel_posthoc3/R1_medication.csv` is P9 (not the
robustness file `result/results_bmel_robustness/R1_patient_level.csv`), and `results_bmel_posthoc6/H1_*` is P14 (not
hypothesis H1).

The scripts reuse shared pipeline modules (windowing, features, hysteresis, event metrics, LSTM training), which are
included in this repository (`fog_pipeline.py`, `fog_study.py`, `fog_improve.py`, `fog_fix3.py`, `fog_deep.py`,
`fog_lstm_controller.py`, `fog_floor_sensitivity.py`, `fog_stats.py`, `paths.py`). Copy them next to the scripts before
running. The scripts write their outputs to `results_bmel_*/` in the working folder; the files from the study are stored
in `result/`. Paths to the data are set at the top of each script and in `paths.py`.

Requirements: see `requirements.txt` (numpy, pandas, scipy, scikit-learn, PyTorch, joblib, matplotlib). The analyses were run
with Python 3.8.5, numpy 1.19.2, pandas 1.1.3, scikit-learn 1.3.2 and PyTorch 2.4.1 (CPU) on Windows 11 (build 26100).

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

Post hoc (P7–P10): with thresholds selected on training subjects, a home-trained LSTM reduced false alarms to 25.2/h
(lower in all 45 patients) and episode coverage exceeded chance (52.2% vs 49.1%, p = 0.003), but timely activation did
not; no threshold pair of the grid gave timely activation above chance for either model (smallest p = 0.16 among pairs meeting the laboratory constraints).

Post hoc (addenda 3 and 4): at home, the cue was often already active more than 4 s before onset (47.6% of eligible
episodes, chance 43.6%), mostly during turning; timely activation exceeded chance on DAPHNET for all ten further training
seeds and for none at home or on FoG-STAR; a movement-only baseline covered DAPHNET episodes above chance but did not
show timely activation above chance, and the freeze-index baseline exceeded chance for neither outcome in any dataset.
No contest subject code appeared in both tdcsfog and defog. With early stopping on home training subjects (addendum 5;
median two epochs, range 1–8), neither timely activation nor episode coverage exceeded chance at home.
With the forward axis of the home recordings inverted (addendum 6; mean forward acceleration +184 mg in tdcsfog,
−199 mg in defog), the frozen LSTM produced fewer false alarms (107/h instead of 151/h) and its pooled episode coverage
exceeded chance (71.2% vs 68.8%, p = 0.019, 2,000 shifts) but not at patient level (p = 0.41); timely activation
remained at chance (15.0% vs 16.9%, p = 0.88). The RF changed little (93/h).

## 6. Notes

* **Addendum 1 (P7–P8).** The pooled values of the home-trained controller (C-pop) were already available from a
  separate, completed calibration analysis (`results_conference2/defog_K1_summary.csv`, plan `CONF2_PLAN.md`), as the
  addendum states; addendum 1 added the patient-level and threshold-grid analyses. The first committed `bmel_posthoc2.py`
  selected rows labelled "C-pop", which do not exist in `defog_K4_thresholds.csv`; it was corrected to read the population
  pair (column `population_pair`) of the C-personal rows (commit 0808dfd). `CONF2_PLAN.md` was committed together with
  its results.
* **Related analyses.** `results_conference2/` and `conf*_analysis.py` belong to separate studies with the same model and
  home data (activity-specific thresholds; patient-specific calibration); neither improved home performance. They are
  included because P7 reuses their population thresholds.
* **Episode counts in the registered plan.** `PREREGISTRATION_BMEL.md` gives 949 tdcsfog episodes, counted on the
  sample-level labels; the window-level count used in all event metrics is 942.
* **Shared modules.** `fog_*.py` and `paths.py` come from the authors' earlier pipeline; functions not called by the
  scripts listed above (for example SMOTE or other datasets in `paths.py`) are not used in this study.
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
* Analyses after step 5 are post hoc and exploratory; they do not change the registered result.
* **Post hoc addenda.** `POSTHOC_ADDENDUM.md` to `POSTHOC_ADDENDUM_6.md` were each committed before the results of the corresponding script
  (`bmel_posthoc2.py` to `bmel_posthoc7.py`) were added (see the commit
  history); `PREREGISTRATION_BMEL.md`, `frozen_config.json` and `bmel_transfer.py` are unchanged.

## 7. Citation

Please cite the article once published, or this archived release:
Soliman AM, Nashaat Gamil DY, Hammad MS, Hassan MA. FoG lower-back cue controller transfer: code and registered
analysis plan (v1.4.0). Zenodo, 2026. https://doi.org/10.5281/zenodo.23418062
(v1.0.0, archived on the day of the registered transfer evaluation; it also contains P1–P6 and the supporting analyses:
https://doi.org/10.5281/zenodo.23184573)

## 8. Licence

Code: MIT (see `LICENSE`). The datasets remain under their original licences.
