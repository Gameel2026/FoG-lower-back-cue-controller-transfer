import os
import json
import pickle
import numpy as np
import pandas as pd

import bmel_dev as bd
import bmel_transfer as bt
import bmel_posthoc as bp
from bmel_sensitivity import FastMetrics, surrogate_segments, recording_segments
from fog_pipeline import Config, FOG, NOFOG
from fog_study import apply_hysteresis, event_metrics, binary_window_metrics, CUE_ON

N_SHIFTS = 2000


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=100)
    fz = json.load(open("frozen_config.json")); th = (fz["operating_points"]["LSTM"]["theta_on"], fz["operating_points"]["LSTM"]["theta_off"])
    path, _ = bt.TESTS["defog"]
    X, meta, y4, probs = pickle.load(open(os.path.join(bt.OUT, "T_probs_defog.pkl"), "rb"))
    y2 = meta["y2"].to_numpy(); allm = np.ones(len(y4), bool); onsets = bd.eligible_onsets(y2, meta, cfg)
    mp = pd.read_csv(bp.MAPPING); mp = mp[mp.folder == "defog_back"]; ids = dict(zip(mp["S"], mp["Id"]))
    tasks = pd.read_csv(bp.TASKS); tt = bp.window_times(path, meta)
    key = np.array([ids.get(f"S{int(s):02d}R{int(r):02d}") for s, r in zip(meta.subject, meta.run)], dtype=object)
    balance = np.zeros(len(meta), bool)
    for Id, grp in tasks[tasks.Task.str.startswith("MB")].groupby("Id"):
        idx = np.flatnonzero(key == Id)
        for _, r in grp.iterrows():
            balance[idx[(tt[idx] >= r.Begin) & (tt[idx] < r.End)]] = True
    on = apply_hysteresis(probs["LSTM"], meta, allm, *th)
    on_oracle = on & ~balance                                         
    fm = FastMetrics(meta, cfg, onsets); rec = recording_segments(meta); rows = []
    for name, o in [("Frozen LSTM (as registered)", on), ("Oracle: cue off during balance tasks", on_oracle)]:
        pred = np.where(o, FOG, NOFOG)
        rows.append({"analysis": name, "balance_windows_%": 100 * balance.mean(),
                     "cue_specificity": binary_window_metrics(y4, pred)["cue_specificity"],
                     "false_alarms_per_hour": event_metrics(y2, pred, meta, cfg)["false_alarms_per_hour"],
                     "FoG_windows_in_balance": int(np.sum(balance & (y2 == 2))),
                     **surrogate_segments(o, meta, cfg, onsets, rec, N_SHIFTS, fm=fm)})
    R = pd.DataFrame(rows); os.makedirs(bp.OUT, exist_ok=True); R.round(4).to_csv(f"{bp.OUT}/P6_oracle_context.csv", index=False)
    cols = ["analysis", "balance_windows_%", "cue_specificity", "false_alarms_per_hour", "timely_observed", "timely_surrogate_mean",
            "timely_p", "detection_observed", "detection_surrogate_mean", "detection_p", "FoG_windows_in_balance"]
    print("\nPOST HOC P6 - oracle activity context (upper bound)\n", R[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()