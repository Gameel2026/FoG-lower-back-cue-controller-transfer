# Post hoc addendum 6: forward-axis sign in the home recordings

**Written after the analyses of POSTHOC_ADDENDUM_5.md were obtained and before the analysis below was run.**
Registered files and earlier results are unchanged. The analysis is post hoc and will be reported as such, whatever
its direction. Script: `bmel_posthoc7.py`; outputs: `results_bmel_posthoc7/`.

## Question
The mean forward acceleration differed in sign between tdcsfog (+184 mg) and defog (−199 mg) (P4,
`result/results_bmel_posthoc/P4_axis_check.csv`), whereas the vertical and lateral means agreed. This analysis checks
whether the sign of the forward axis explains the home result.

## F1 Frozen controllers on defog with the forward axis inverted
The converted defog recordings are read again, the forward-axis signal is multiplied by −1, and windows and features are
recomputed with the unchanged pipeline. As a check, the features recomputed without inversion must reproduce the
registered LSTM probabilities. The frozen laboratory LSTM and RF (models, standardisation and thresholds of
`frozen_config.json`, no retraining) are applied to the inverted input. Outcomes: timely activation and episode
coverage against the circular-shift surrogate (2,000 shifts), false alarms per hour, specificity and patient-specific
excess (500 shifts per patient), compared with the registered input. Descriptive; no correction for multiplicity.
