import os
import sys
import pickle
import json
import warnings
import numpy as np
import pandas as pd

NEED_MOD = ["frozen_config.json", "bmel_transfer.py", "bmel_dev.py", "bmel_sensitivity.py", "bmel_robustness.py",
            "fog_pipeline.py", "fog_study.py", "fog_deep.py", "fog_lstm_controller.py"]
NEED_DATA = {"defog": ("results_bmel_transfer", "T_probs_defog.pkl"),
             "MODELS": ("results_bmel_transfer", "T_models_lab.pkl")}
OPTIONAL = {}
SKIP = {"$Recycle.Bin", "System Volume Information", "AppData", "anaconda3", "Anaconda3", ".git", "node_modules"}


def locate():
    starts = []
    try:
        starts.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    starts += [os.getcwd(), "D:\\New research fog work"]

    def has_mods(d):
        return all(os.path.exists(os.path.join(d, f)) for f in NEED_MOD)
    base = next((d for d in starts if os.path.isdir(d) and has_mods(d)), None)
    wanted = {**NEED_DATA, **OPTIONAL}
    hits = {k: [] for k in wanted}; mods = []
    if base is None or not all(os.path.exists(os.path.join(base, *v)) for v in wanted.values()):
        print("searching D:\\ and the user folder for the required files ...")
        for root in ["D:\\", os.path.expanduser("~")]:
            if not os.path.exists(root):
                continue
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d not in SKIP]
                if "frozen_config.json" in fn and "bmel_transfer.py" in fn and has_mods(dp):
                    mods.append(dp)
                for k, (sub, f) in wanted.items():
                    if f in fn:
                        hits[k].append(os.path.join(dp, f))
        if base is None:
            if not mods:
                raise FileNotFoundError("no folder contains all of: " + ", ".join(NEED_MOD))
            base = mods[0]
    paths = {}
    for k, (sub, f) in wanted.items():
        p = os.path.join(base, sub, f)
        if not os.path.exists(p):
            cand = [h for h in hits[k] if os.path.basename(os.path.dirname(h)) == sub] or hits[k]
            if not cand:
                if k in OPTIONAL:
                    print(f"  {f} not found: the analyses that need it are skipped")
                    continue
                if k == "DEV":
                    print("  cache_tdcsfog_back.pkl not found: tdcsfog will be rebuilt by bmel_dev.load_data")
                    continue
                raise FileNotFoundError(f"{f} not found")
            p = cand[0]
        paths[k] = p
    print("working folder:", base)
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return base, paths


BASE, PATHS = locate()
os.chdir(BASE); sys.path.insert(0, BASE)

import bmel_dev as bd
import bmel_robustness as br
import bmel_transfer as bt
import fog_lstm_controller as flc
from bmel_sensitivity import surrogate_segments, recording_segments
from fog_pipeline import Config, load_daphnet, segment_windows, extract_features
from fog_study import apply_hysteresis
from fog_deep import past_index, gather

OUT = "results_bmel_posthoc7"
N_SHIFTS = 2000
warnings.filterwarnings("ignore")


def evaluate(on, meta, y4, cfg):
    onsets = bd.eligible_onsets(meta["y2"].to_numpy(), meta, cfg)
    s = bd.summarise(y4, on, meta, cfg, onsets)
    sg = surrogate_segments(on, meta, cfg, onsets, recording_segments(meta), N_SHIFTS)
    out = {"specificity": s["cue_specificity"], "false_alarms_per_hour": s["false_alarms_per_hour"], "cue_on_%": 100 * on.mean(),
           "timely_%": s["timely_eligible_%"], "timely_chance": sg["timely_surrogate_mean"], "p_timely": sg["timely_p"],
           "coverage_%": s["detected_%"], "coverage_chance": sg["detection_surrogate_mean"], "p_coverage": sg["detection_p"]}
    P = br.patient_level(on, meta, cfg)
    for o in ["timely", "coverage"]:
        e = br.summarise_excess(P, f"{o}_obs", f"{o}_null")
        out[f"{o}_excess_median"] = e["median_excess"]; out[f"{o}_excess_CI"] = f"[{e['CI_low']:.1f}, {e['CI_high']:.1f}]"
        out[f"{o}_patients_above"] = f"{e['patients_above_chance']}/{e['n_patients']}"; out[f"{o}_excess_p"] = e["wilcoxon_p_one_sided"]
    return out


def features(raw, cfg):
    windows, meta = segment_windows(raw, cfg)
    return extract_features(windows, cfg), meta


def lstm_prob(M, X, meta):
    F = np.nan_to_num(X.to_numpy(np.float64))
    Z = gather(np.clip((F - M["mu"]) / M["sd"], -10, 10).astype(np.float32), past_index(meta, 8))
    return flc.prob(M["lstm"], Z)


def rf_prob(M, X):
    P, cl = M["rf"].predict_proba(X[M["feats"]]); return P[:, list(cl).index(1)]


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=100)
    fz = json.load(open("frozen_config.json"))
    th = {k: (v["theta_on"], v["theta_off"]) for k, v in fz["operating_points"].items()}
    X0, meta0, y4, probs = pickle.load(open(PATHS["defog"], "rb"))
    M = pickle.load(open(PATHS["MODELS"], "rb"))
    path = bt.TESTS["defog"][0]
    raw = load_daphnet(path)
    X1, meta1 = features(raw, cfg)
    assert len(meta1) == len(meta0) and (meta1[["subject", "run", "seg", "pos"]].to_numpy() == meta0[["subject", "run", "seg", "pos"]].to_numpy()).all()
    p_check = lstm_prob(M, X1, meta1)
    print("reproduction check, correlation with the registered probabilities:", round(float(np.corrcoef(p_check, probs["LSTM"])[0, 1]), 6))
    raw_f = raw.copy(); raw_f["tr_x"] = -raw_f["tr_x"]
    Xf, metaf = features(raw_f, cfg)
    print("mean forward acceleration (mg): original", round(float(raw["tr_x"][raw.label != 0].mean()), 1), "flipped", round(float(raw_f["tr_x"][raw_f.label != 0].mean()), 1))
    rows = []
    for name, Xs in [("as registered", X0), ("forward axis sign inverted", Xf)]:
        for ctrl, p in [("LSTM", probs["LSTM"] if name == "as registered" else lstm_prob(M, Xs, meta0)),
                        ("RF", probs["RF"] if name == "as registered" else rf_prob(M, Xs))]:
            on = apply_hysteresis(np.asarray(p), meta0, np.ones(len(meta0), bool), *th[ctrl])
            rows.append({"defog input": name, "controller": ctrl, **evaluate(on, meta0, y4, cfg)})
            print(f"  {name} / {ctrl}: done")
    R = pd.DataFrame(rows); R.round(4).to_csv(f"{OUT}/F1_forward_axis.csv", index=False)
    print(R.round(3).T.to_string(header=False))
    print(f"\nDone. Files in {OUT}/ (post hoc, POSTHOC_ADDENDUM_6.md)")


if __name__ == "__main__":
    main()