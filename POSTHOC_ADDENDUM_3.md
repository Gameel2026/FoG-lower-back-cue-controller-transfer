# Post hoc addendum 3: timing of cue activation, non-learned baselines and seed variability

**Written after the analyses of POSTHOC_ADDENDUM_2.md were obtained and before the analyses below were run.**
Registered files and earlier results are unchanged. Everything below is post hoc and will be reported as such,
whatever its direction. Script: `bmel_posthoc4.py`; outputs: `results_bmel_posthoc4/`.
Inputs: the saved transfer probabilities (`results_bmel_transfer/T_probs_*.pkl`), the tdcsfog feature cache
(`results_bmel_dev/cache_tdcsfog_back.pkl`), `frozen_config.json`, and for A1 the home-trained probabilities and
thresholds of POSTHOC_ADDENDUM.md.

## A1 Decomposition of timely activation
For every eligible episode in defog, FoG-STAR and DAPHNET, the frozen laboratory LSTM output (θon/θoff 0.65/0.40) is
classified as: (i) cue switched on within the 4 s before onset and active at onset (timely activation as registered);
(ii) cue already active for more than 4 s at onset; (iii) cue off at onset but switched on during the episode, with
the latency from onset; (iv) cue never active during the episode. Each proportion is compared with a circular-shift
surrogate (500 shifts within recording segments). Repeated for the home-trained LSTM with training-subject thresholds
on defog. Descriptive.

## A2 Seed variability of the frozen controller
The frozen LSTM configuration (all tdcsfog, two epochs, frozen thresholds) is retrained with ten additional random
seeds (1–10); seed 42 is retrained as a reproducibility check. Each model is applied without retraining to the three
test datasets, with timely activation, episode coverage, false alarms per hour and specificity, each against the
circular-shift surrogate (2,000 shifts). Reported as median and range across seeds and the number of seeds with
p < 0.05. Descriptive.

## A3 Non-learned baselines
Two training-free controllers computed from the same 0.5 s windows: (a) the freeze index averaged over the three axes,
(b) the total 0.5–8 Hz power summed over the three axes (a movement-only controller). Both are orientation-independent.
Each score is mapped to [0, 1] by its empirical distribution in tdcsfog, and the hysteresis pair is selected on tdcsfog
with the frozen rule (highest timely activation with ≤ 50 false alarms per hour and specificity ≥ 0.70; otherwise the
pair with the fewest false alarms). The baselines are then applied unchanged to the three test datasets and evaluated
as above (2,000 shifts; patient-specific excess with 500 shifts per patient), and compared per patient with the frozen
LSTM (two-sided Wilcoxon signed-rank tests on timely activation, episode coverage and false alarms per hour).
Descriptive; no correction for multiplicity.
