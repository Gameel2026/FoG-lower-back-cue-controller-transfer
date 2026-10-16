# Post hoc addendum: home-trained controller and full threshold sweep at home (defog)

**Written after all registered, supporting and exploratory analyses of this repository were obtained, and before
the analyses below were run.** PREREGISTRATION_BMEL.md, frozen_config.json and bmel_transfer.py are unchanged; the
registered results are unchanged. Everything below is post hoc and will be reported as such.

**Inputs (already saved, no model is trained):**
- frozen laboratory LSTM probabilities on defog: `results_bmel_transfer/T_probs_defog.pkl` (frozen θon/θoff 0.65/0.40);
- home-trained LSTM probabilities on defog, subject-grouped 10-fold out-of-fold (each patient predicted by a model not
  trained on that patient): `results_bmel_posthoc/ref_lstm_defog.npy` (used in P1);
- population thresholds of the home-trained controller selected by window F1 on the training patients of each fold
  ("C-pop" of a separate, already completed calibration analysis; its plan `CONF2_PLAN.md`, script
  `conf2_step1_analysis.py` and results `results_conference2/defog_K1–K4` are added to this repository unchanged,
  together with this addendum): `results_conference2/defog_K4_thresholds.csv`.

## Q1 Home-trained controller (C-pop): robustness
- Pooled timely activation, episode coverage, false alarms/h and specificity; circular-shift surrogate with 2,000 shifts
  (check: values must equal `results_conference2/defog_K1_summary.csv`).
- Patient-level excess over chance (500 shifts per patient), median with patient bootstrap 95% CI (5,000 resamples),
  patients above chance, one-sided Wilcoxon — the same procedure as the supporting analysis of the paper (Section 2.7).
- Per-patient comparison with the frozen laboratory LSTM on the same patients: false alarms/h, episode coverage and
  timely activation; two-sided Wilcoxon signed-rank, Holm correction across the three metrics.

## Q2 Full threshold sweep at home
For the frozen laboratory LSTM and for the home-trained LSTM, every threshold pair of the grid used in the paper
(θon 0.30–0.95 step 0.05, θoff 0.10–θon step 0.10; 84 pairs) is applied to the defog probabilities: false alarms/h,
specificity, timely activation and episode coverage, each with a 200-shift circular-shift surrogate.
- **Question:** does any pair with false alarms ≤ 50/h and specificity ≥ 0.70 (the laboratory operating-point
  constraints) give timely activation above chance? Holm correction across the pairs meeting the constraints, per model;
  the smallest uncorrected p-value is also reported. Pairs with p < 0.05 are re-tested with 2,000 shifts.
- Descriptive: best timely activation at ≤ 25, ≤ 50 and ≤ 100 false alarms/h (as in Fig. 2 of the paper), and the same
  for episode coverage.

All results are reported whatever their direction.
