import json
import pickle
import numpy as np
import pandas as pd

DEV = "results_bmel_dev2"
FA_BUDGET = 50.0
SPEC_FLOOR = 0.70
SOURCES = {"LSTM": "LSTM", "RF": "RF"}


def main():
    C = pd.read_csv(f"{DEV}/E3_tradeoff_curves.csv")
    ch = pd.read_csv(f"{DEV}/E5_choices.csv")
    cfg = {"fa_budget_per_hour": FA_BUDGET, "specificity_floor": SPEC_FLOOR, "development_data": "tdcsfog (62 subjects), 10 subject-grouped folds",
           "operating_points": {}}
    for name, src in SOURCES.items():
        g = C[(C.source == src) & (C.false_alarms_per_hour <= FA_BUDGET) & (C.cue_specificity >= SPEC_FLOOR)].copy()
        g = g.sort_values(["timely_eligible_%", "detected_%", "false_alarms_per_hour"], ascending=[False, False, True])
        r = g.iloc[0]
        cfg["operating_points"][name] = {k: float(r[k]) for k in ["theta_on", "theta_off", "timely_eligible_%",
                                                                  "detected_%", "false_alarms_per_hour", "cue_specificity"]}
    ep = ch.drop_duplicates("fold")["lstm_epochs"].astype(int)
    cfg["lstm_final_epochs"] = int(np.median(ep)); cfg["lstm_fold_epochs"] = ep.tolist()
    json.dump(cfg, open("frozen_config.json", "w"), indent=2)
    print(json.dumps(cfg, indent=2))
    print("\nSaved frozen_config.json - commit it with PREREGISTRATION_BMEL.md before any test data are analysed.")


if __name__ == "__main__":
    main()