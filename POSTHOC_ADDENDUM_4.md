# Post hoc addendum 4: dataset independence, timing by task, temporal context, compact forest

**Written after the analyses of POSTHOC_ADDENDUM_3.md were obtained and before the analyses below were run.**
Registered files and earlier results are unchanged. Everything below is post hoc and will be reported as such,
whatever its direction. Script: `bmel_posthoc5.py`; outputs: `results_bmel_posthoc5/`.

## B1 Subject overlap between tdcsfog and defog
The contest subject codes of the tdcsfog and defog recordings (`kaggle_subject_mapping.csv`) are compared. If any code
appears in both, the primary home analysis of the frozen laboratory LSTM is repeated without the shared subjects
(2,000 shifts; patient-specific excess).

## B2 Timing categories by protocol task
The four timing categories of POSTHOC_ADDENDUM_3.md (A1) are tabulated for defog by the protocol task at episode
onset (task annotations of the contest data), each against the circular-shift surrogate (500 shifts). Descriptive.

## B3 Temporal context and a compact random forest
Three controllers are developed on tdcsfog with one procedure: subject-grouped ten-fold out-of-fold probabilities, the
frozen operating-point rule applied to these probabilities (highest timely activation with ≤ 50 false alarms per hour
and specificity ≥ 0.70), and a final model trained on all tdcsfog:
(a) the LSTM with 4 s history (replication of the frozen configuration, two epochs);
(b) the same LSTM receiving the current window only (temporal-context ablation);
(c) a random forest with 100 trees of maximum depth 10 on the 23 mutual-information features (compact reference).
Each is applied unchanged to defog, FoG-STAR and DAPHNET (2,000 shifts; patient-specific excess) and compared per
patient with the frozen laboratory LSTM (two-sided Wilcoxon). Model size and single-window inference time of (c) are
measured on the same laptop.

## B4 Uncertainty of absolute performance
Per-patient median timely activation, episode coverage and false alarms per hour of the frozen laboratory LSTM, with
patient bootstrap 95% confidence intervals (5,000 resamples).

## B5 Software and hardware
Python and library versions, operating system and processor of the computer used for the analyses and timings.

All analyses are descriptive; no correction for multiplicity.
