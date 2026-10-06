import os
import json
import pickle
import numpy as np
import pandas as pd

import bmel_dev as bd
import bmel_transfer as bt
import kaggle_convert as kc
from fog_pipeline import Config, load_daphnet, segment_windows, extract_features, relabel_four_class
from fog_deep import past_index

OUT = "results_bmel_posthoc"


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=100)
    fz = json.load(open("frozen_config.json"))
    th = {k: (v["theta_on"], v["theta_off"]) for k, v in fz["operating_points"].items()}
    M = pickle.load(open(os.path.join(bt.OUT, "T_models_lab.pkl"), "rb"))
    path, _ = bt.TESTS["FoG-STAR"]
    raw = load_daphnet(path)
    raw[["tr_x", "tr_y"]] = raw[["tr_y", "tr_x"]].to_numpy()                
    print("axis means after swap (mg):", raw.loc[raw.label != 0, ["tr_x", "tr_y", "tr_z"]].mean().round(1).to_dict())
    for _, idx in raw.groupby(["subject", "run"]).groups.items():
        raw.loc[idx, "label"] = kc.merge_gaps(raw.loc[idx, "label"].to_numpy().astype(int))
    windows, meta = segment_windows(raw, cfg)
    X = extract_features(windows, cfg); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    F = np.nan_to_num(X.to_numpy(np.float64)); P8 = past_index(meta, 8)
    probs = bt.predict(M, X, F, P8); rows = []
    for ctrl in ["LSTM", "RF"]:
        s, g, _ = bt.evaluate(probs[ctrl], th[ctrl], X, meta, y4, cfg, bd.N_SHIFTS)
        rows.append({"dataset": "FoG-STAR (axes x<->y, post hoc)", "controller": ctrl, **s,
                     "surrogate_timely": g.loc["timely_eligible_%", "surrogate_mean"], "p_timely": g.loc["timely_eligible_%", "p_one_sided"],
                     "surrogate_detection": g.loc["detected_%", "surrogate_mean"], "p_detection": g.loc["detected_%", "p_one_sided"]})
    R = pd.DataFrame(rows); R.round(4).to_csv(f"{OUT}/P5_fogstar_axes.csv", index=False)
    cols = ["controller", "cue_specificity", "timely_eligible_%", "surrogate_timely", "p_timely", "detected_%",
            "surrogate_detection", "p_detection", "false_alarms_per_hour"]
    print("\nPOST HOC P5 - FoG-STAR transfer after aligning the vertical axis\n", R[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()