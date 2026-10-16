# Analysis plan: patient-specific calibration of an on-demand FoG cue controller (conference paper 2)

**Written after the recording inventory (annotations only, `K0_runs_per_patient.csv`) and before any analysis
of this study was run.** Commit this file before running `conf2_step1_analysis.py`.

**Prior work disclosed:** the same LSTM, features and data were used in our earlier studies (BMEL transfer study;
conference study 1, whose pre-registered hypothesis on activity-specific thresholds was not supported). Thresholds
calibrated on the patient's own recordings have not been evaluated in any of them.

## Question
Does calibrating the two cue thresholds on other recordings of the same patient improve the cue controller on a new
recording of that patient, compared with thresholds calibrated on other patients, while the model itself stays the same?

## Data
- **defog (primary, home):** 45 patients, 137 recordings, 20.7 h, 625 episodes; 37 patients have FoG in >= 2 recordings.
- **tdcsfog (replication, laboratory):** 62 patients, 833 recordings, 15.3 h, 942 episodes; 40 patients have FoG in >= 2 recordings.
- Lower-back accelerometer, 0.5 s windows, 72 features, cue-needed target (pre-FoG, FoG, post-FoG), same conversion
  and 2 s merging rule as in our previous work. A "recording" is one converted file (subject x run).

## Model (fixed, never fitted on the evaluated patient)
2-layer LSTM of our previous work (4 s of features, training length from frozen_config.json), subject-grouped
10-fold CV within each dataset (defog: the same folds and test probabilities as bmel_posthoc.py; tdcsfog: the same
GroupKFold split computed here). Inner subject-grouped 3-fold out-of-fold probabilities for the training patients.

## Controllers (hysteresis; grid θon 0.30–0.95 step 0.05, θoff 0.10–θon step 0.10; rule = max window-level F1 of the
cue-needed target, ties: first pair in grid order)
- **C-pop:** pair selected on the training patients' out-of-fold probabilities (all windows).
- **C-personal (leave-one-recording-out):** for each recording r of a held-out patient, the pair is selected with
  the same rule on that patient's **other** recordings (probabilities of the same held-out model) and applied to r.
  If the other recordings contain no cue-needed window, C-pop's pair is used.
- **C-cal-2, C-cal-5, C-cal-10 (calibration curve):** as C-personal, but only the first 2, 5 or 10 minutes of the
  patient's other recordings are used (recordings taken in run-number order, windows in time order: a
  label-independent choice that mimics a short calibration session). Same fallback; the percentage of recordings
  that fall back is reported for every controller.
Patients with a single recording use C-pop's pair in all personal controllers.

## Outcomes
- **Primary:** per-patient window-level cue F1 on the evaluated recordings, C-personal vs C-pop, two-sided Wilcoxon
  signed-rank test (patients with at least one cue-needed window; defog).
- **Secondary (both datasets):** per-patient false alarms per hour and episode coverage, C-personal vs C-pop
  (Wilcoxon, Holm across these two); for each calibration budget, per-patient F1 vs C-pop (Wilcoxon, Holm across the
  three budgets) and false alarms/h and coverage vs C-pop (Holm across these two); pooled specificity, false alarms/h,
  coverage and timely activation of every controller against a circular-shift surrogate (2,000 shifts);
  per-patient change in F1 against the amount of calibration data (minutes, Spearman, exploratory).

## Criteria fixed in advance
- **Supported (defog):** C-personal has a higher per-patient F1 than C-pop with p < 0.05.
- **Replicated (tdcsfog):** same direction, p reported.
- **Minimum calibration:** the smallest budget (2, 5, 10 min) whose per-patient F1 is higher than C-pop's with
  Holm-adjusted p < 0.05 and which is clinically favourable (below); "none" if no budget qualifies.
- **Clinical reading:** the improvement is called clinically favourable only if pooled false alarms per hour are
  not higher than C-pop's by more than 10% **and** pooled coverage is not lower by more than 5 percentage points.
- All results are reported whatever their direction.
