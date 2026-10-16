# Post hoc addendum 2: medication state and cue-activation burden at home (defog)

**Written after the analyses of POSTHOC_ADDENDUM.md were obtained and before the analyses below were run.**
Registered files and earlier results are unchanged. Everything below is post hoc and will be reported as such.
Inputs are the same saved probabilities and thresholds as in POSTHOC_ADDENDUM.md, plus the medication state of each
defog recording from `kaggle_subject_mapping.csv` (written by kaggle_convert.py from the contest metadata).
Controllers: frozen laboratory LSTM (θon/θoff 0.65/0.40) and home-trained LSTM with population thresholds (C-pop).

## R1 Medication state
For the recordings in the ON and in the OFF medication state separately: false alarms/h, specificity, timely
activation and episode coverage, each against a circular-shift surrogate (2,000 shifts) within that subset, and the
patient-level excess over chance (500 shifts per patient; median, patient bootstrap 95% CI, one-sided Wilcoxon).
Descriptive; no correction across subsets; recordings without a medication label are excluded from this analysis.

## R2 Cue-activation burden
For both controllers on all defog recordings: proportion of time with the cue on outside the target intervals
(from 4 s before onset to 3 s after offset), minutes of unnecessary cue per hour of recording, and the duration
distribution (median, interquartile range, 90th percentile) of false-alarm activation runs and of all activation runs.
Descriptive.

All results are reported whatever their direction.
