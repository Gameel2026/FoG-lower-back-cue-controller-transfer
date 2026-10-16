# Post hoc addendum 5: home training with early stopping

**Written after the analyses of POSTHOC_ADDENDUM_4.md were obtained and before the analysis below was run.**
Registered files and earlier results are unchanged. The analysis is post hoc and will be reported as such, whatever
its direction. Script: `bmel_posthoc6.py`; outputs: `results_bmel_posthoc6/`.

## Question
The home-trained LSTMs reported so far (P1, P7) used the training length fixed during development (two epochs).
This analysis checks whether a training length chosen on home data changes the home result.

## H1 Home-trained LSTM with early stopping
On defog, subject-grouped ten-fold cross-validation. In each fold, using the training subjects only:
1. 20% of the training subjects are held out (subject-grouped); the LSTM is trained on the remaining 80% with early
   stopping on the balanced accuracy of the held-out subjects (patience six epochs, maximum 30), as in development;
2. a model trained for the selected number of epochs on the 80% predicts the held-out training subjects, and the
   hysteresis thresholds are selected on these predictions with (a) the frozen rule (highest timely activation with
   ≤ 50 false alarms per hour and specificity ≥ 0.70) and (b) window F1 with specificity ≥ 0.70;
3. a model trained on all training subjects for the selected number of epochs predicts the test subjects.
Same architecture, features and standardisation as the frozen controller. Outcomes on the pooled test predictions:
timely activation and episode coverage against the circular-shift surrogate (2,000 shifts), false alarms per hour,
specificity and patient-specific excess (500 shifts per patient). The selected number of epochs per fold is reported.
Descriptive; no correction for multiplicity.
