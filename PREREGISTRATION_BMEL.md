# Pre-registered analysis plan: laboratory-to-home transfer of a lightweight temporal cue controller for freezing of gait

**Status:** committed before any controller was applied to the test datasets (defog, FoG-STAR lower back, DAPHNET trunk).
The analysis code (`bmel_transfer.py`) and the frozen settings (`frozen_config.json`) are committed together with this file.

## 1. Question
Can a cue controller developed on laboratory recordings from a single lower-back accelerometer be transferred,
without retraining or re-tuning, to home recordings and to independent cohorts, while keeping its
early-activation, detection and false-alarm behaviour?

## 2. What happened during development (tdcsfog only)
Development used only the labelled tdcsfog recordings of the Kaggle/Zenodo FoG competition data
(62 subjects, 15.3 h, 949 episodes after merging), with 10 subject-grouped folds.
1. The initial development hypothesis was that an LSTM controller would reduce false alarms without loss of
   timely activation compared with a random-forest (RF) controller. **It was not supported:** at matched false-alarm
   rates the two models produced almost the same timely activation (trade-off curves, `results_bmel_dev2/`).
   The pre-specified development rule therefore selected the RF controller; calibration, fusion and longer LSTM
   training did not change this.
2. The LSTM matched the RF trade-off with about 38 times fewer parameters and about 40 times faster inference
   (`D6_compute.csv`). The question of this study was therefore changed to the one in Section 1, with the LSTM as
   the primary controller (efficiency) and the RF as reference.
3. Episode-merging rule (gaps < 2 s merged) was chosen from annotation statistics only (no model output). These
   statistics included the defog labels (fragmented annotations: median segment 0.7 s); no model was applied to defog.

## 3. Data (all lower-back accelerometer, 64 Hz, mg, episodes separated by < 2 s merged)
| Role | Dataset | Subjects | Notes |
|---|---|---|---|
| Training (all of it) | tdcsfog (laboratory) | 62 | Kaggle/Zenodo `train/tdcsfog`, 128 Hz -> 64 Hz, m/s^2 -> mg |
| **Primary test** | **defog (home)** | **45** | `train/defog` + `train/notype`; only Valid & Task samples; 100 Hz -> 64 Hz, g -> mg |
| Secondary test | FoG-STAR | 22 | lower-back sensor as converted by `fogstar_convert.py` (60 Hz -> 64 Hz) |
| Secondary test | DAPHNET | 10 | trunk (lower-back) sensor, original 64 Hz files |
No subject is shared between tdcsfog and defog. Axis order: forward, vertical, lateral where documented
(Kaggle, DAPHNET); FoG-STAR axes are used as converted, without remapping.
**Prior use of the test data by the authors:** FoG-STAR (ankle sensor) and DAPHNET (trunk sensor, random-forest
controller with a different operating rule) were analysed in a separate study; no LSTM controller, no lower-back
FoG-STAR analysis and no model trained on tdcsfog has been applied to any test dataset.

## 4. Frozen controllers (from `frozen_config.json`)
* Features: 72 hand-crafted features per 0.5 s window (causal 0.5-20 Hz Butterworth filter); target: cue needed
  (Pre-FoG 4 s, FoG, Post-FoG 3 s).
* LSTM (primary): 2 x 64 units on the last 4 s (8 windows) of standardised features; trained once on all tdcsfog
  for **2 epochs** (median of the 10 development folds); standardisation statistics from tdcsfog.
* RF (reference): 100 trees, 23 features ranked by mutual information on tdcsfog.
* Operating points (hysteresis), chosen on development out-of-fold probabilities by the rule: maximum timely
  activation subject to <= 50 false alarms per hour and specificity >= 0.70:
  * **LSTM: theta_on = 0.65, theta_off = 0.40** (development: timely 20.5%, detection 73.8%, 48.6 false alarms/h, specificity 0.877)
  * **RF: theta_on = 0.80, theta_off = 0.20** (development: timely 16.2%, detection 71.4%, 42.3 false alarms/h, specificity 0.886)
* No retraining, recalibration or threshold change on any test dataset.

## 5. Outcomes
* Timely activation: cue active within the 4 s before onset, on **eligible episodes** (>= 4 s FoG-free recording before onset).
* Detection (cue active during the episode), false alarms per hour (activation runs that touch no interval from 4 s
  before onset to 3 s after offset), cue specificity, median cue-off delay, median lead time.
* Chance reference: circular shift of the controller output within each recording, 200 shifts, one-sided p = (1 + #null >= observed)/(1 + 200).

## 6. Hypotheses
Primary (defog, LSTM controller), Holm correction across H1 and H2:
* **H1** Timely activation exceeds the surrogate level.
* **H2** Detection exceeds the surrogate level.
* **H3 (descriptive)** False alarms per hour at home compared with the laboratory budget of 50/h.
Secondary (reported regardless of outcome):
* H1-H3 on FoG-STAR and DAPHNET.
* LSTM versus RF per subject (Wilcoxon, Holm across metrics; descriptive).
* Reference: both models retrained within each test dataset (subject-grouped CV, same training length and frozen
  operating points), to separate the effect of new data from the effect of transfer.

## 7. Reporting
All results are reported as obtained, including hypotheses that are not supported. Any deviation from this plan
will be stated explicitly and labelled as post hoc.
