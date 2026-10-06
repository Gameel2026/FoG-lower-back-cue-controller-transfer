import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import bmel_dev as bd
import bmel_transfer as bt
from fog_pipeline import Config, load_daphnet, contiguous_runs, FOG, NOFOG
from fog_study import apply_hysteresis, event_metrics, sequences, CUE_ON
from fog_deep import past_index

TR = "results_bmel_transfer"
OUT = "results_bmel_posthoc"
KAGGLE = r"D:\New research fog work\kaggle_fog"
TASKS = os.path.join(KAGGLE, "Competition Dataset", "tasks.csv")
MAPPING = os.path.join(KAGGLE, "daphnet_format", "kaggle_subject_mapping.csv")
QUICK = False
warnings.filterwarnings("ignore")


def grid_select(p, y_on, meta, mask, onsets, cfg, budget=50.0, floor=0.70):
    y2 = meta["y2"].to_numpy(); t_ = y_on[mask].astype(bool); best = (None, (-1, -1, 0))
    for th in bd.GRID:
        o = apply_hysteresis(p, meta, mask, *th); spec = (~o[mask][~t_]).mean()
        e = event_metrics(y2, np.where(o, FOG, NOFOG), meta, cfg, mask)
        if spec >= floor and e["false_alarms_per_hour"] <= budget:
            v = (1, bd.timely_elig(o, onsets, cfg), e["detected_%"])
        else:
            v = (0, -e["false_alarms_per_hour"], 0)
        if v > best[1]:
            best = (th, v)
    return best[0]


def window_times(path, meta):
    raw = load_daphnet(path); t = np.zeros(len(meta)); W = 32
    starts = {}
    for (s, r), g in raw.groupby(["subject", "run"]):
        runs = [(a, b) for a, b in contiguous_runs(g["label"].to_numpy() != 0)]
        starts[(s, r)] = {i: a for i, (a, b) in enumerate(runs)}
    for i, (s, r, seg, pos) in enumerate(meta[["subject", "run", "seg", "pos"]].itertuples(index=False)):
        t[i] = (starts[(s, r)][seg] + pos * W) / 64.0
    return t


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=100)
    fz = json.load(open("frozen_config.json")); th = tuple(fz["operating_points"]["LSTM"][k] for k in ("theta_on", "theta_off"))
    epochs = fz["lstm_final_epochs"]; n_shifts = 20 if QUICK else bd.N_SHIFTS
    p1, p3, p4 = [], [], []
    raw = load_daphnet(bd.DATA); raw = raw[raw.label != 0]                       # development reference for P4
    p4.append({"dataset": "tdcsfog (training)", **{f"mean_{a}": raw[f"tr_{a}"].mean() for a in "xyz"},
               **{f"sd_{a}": raw[f"tr_{a}"].std() for a in "xyz"}})
    for name, (path, _) in bt.TESTS.items():
        X, meta, y4, probs = pickle.load(open(os.path.join(TR, f"T_probs_{name}.pkl"), "rb"))
        y2 = meta["y2"].to_numpy(); y_on = np.isin(y4, CUE_ON).astype(int); allm = np.ones(len(y4), bool)
        onsets = bd.eligible_onsets(y2, meta, cfg); g = meta.subject.to_numpy()
        # P1 reference LSTM (recomputed; the transfer run saved only its summary)
        ck = os.path.join(OUT, f"ref_lstm_{name}.npy")
        if os.path.exists(ck):
            p_ref = np.load(ck)
        else:
            F = np.nan_to_num(X.to_numpy(np.float64)); P8 = past_index(meta, 8); p_ref = np.zeros(len(y4))
            for tr_i, te_i in GroupKFold(min(10, len(np.unique(g)))).split(X, groups=g):
                M = bt.fit_models(X, F, P8, y_on, tr_i, epochs, cfg); p_ref[te_i] = bt.predict(M, X, F, P8)["LSTM"][te_i]
            np.save(ck, p_ref)
        on = apply_hysteresis(p_ref, meta, allm, *th)
        s = bd.summarise(y4, on, meta, cfg, onsets); sg = bd.surrogate(on, meta, cfg, onsets, n_shifts)
        for met in sg.index:
            p1.append({"dataset": name, "analysis": "within-dataset LSTM (post hoc surrogate)", "metric": met,
                       **sg.loc[met].to_dict(), "false_alarms_per_hour": s["false_alarms_per_hour"],
                       "cue_specificity": s["cue_specificity"]})
        # P3 local recalibration of the frozen laboratory LSTM
        on3 = np.zeros(len(y4), bool); chosen = []
        for tr_i, te_i in GroupKFold(min(10, len(np.unique(g)))).split(X, groups=g):
            tr = np.zeros(len(y4), bool); tr[tr_i] = True
            th_k = grid_select(probs["LSTM"], y_on, meta, tr, bd.eligible_onsets(y2, meta, cfg, tr), cfg)
            if th_k is None: th_k = th
            te = ~tr; on3[te] = apply_hysteresis(probs["LSTM"], meta, te, *th_k)[te]; chosen.append((float(th_k[0]), float(th_k[1])))
        s3 = bd.summarise(y4, on3, meta, cfg, onsets); sg3 = bd.surrogate(on3, meta, cfg, onsets, n_shifts)
        p3.append({"dataset": name, "analysis": "frozen lab LSTM, thresholds recalibrated on other subjects (post hoc)",
                   "thresholds_chosen": str(sorted(set(chosen))), **s3,
                   "p_timely": sg3.loc["timely_eligible_%", "p_one_sided"], "surrogate_timely": sg3.loc["timely_eligible_%", "surrogate_mean"],
                   "p_detection": sg3.loc["detected_%", "p_one_sided"], "surrogate_detection": sg3.loc["detected_%", "surrogate_mean"]})
        # P4 axis check on the raw lower-back signal
        raw = load_daphnet(path); raw = raw[raw.label != 0]
        p4.append({"dataset": name, **{f"mean_{a}": raw[f"tr_{a}"].mean() for a in "xyz"},
                   **{f"sd_{a}": raw[f"tr_{a}"].std() for a in "xyz"}})
        # P2 defog by task
        if name == "defog":
            mp = pd.read_csv(MAPPING); mp = mp[mp.folder == "defog_back"]
            ids = dict(zip(mp["S"], mp["Id"])); tasks = pd.read_csv(TASKS)
            tt = window_times(path, meta); task = np.array(["(no task)"] * len(meta), dtype=object)
            key = meta.apply(lambda m: ids.get(f"S{int(m.subject):02d}R{int(m.run):02d}"), axis=1).to_numpy()
            for Id, grp in tasks.groupby("Id"):
                idx = np.flatnonzero(key == Id)
                for _, r in grp.iterrows():
                    sel = idx[(tt[idx] >= r.Begin) & (tt[idx] < r.End)]; task[sel] = r.Task
            on_t = apply_hysteresis(probs["LSTM"], meta, allm, *th); pred = np.where(on_t, FOG, NOFOG)
            rows = []
            for tk in pd.unique(task):
                m = task == tk
                if m.sum() < 20: continue
                e = event_metrics(y2, pred, meta, cfg, m)
                rows.append({"task": tk, "hours": m.sum() * cfg.win_sec / 3600, "cue_needed_%": 100 * y_on[m].mean(),
                             "episodes": e["episodes"], "detected_%": e["detected_%"],
                             "false_alarms_per_hour": e["false_alarms_per_hour"], "cue_on_%": 100 * on_t[m].mean()})
            P2 = pd.DataFrame(rows).sort_values("hours", ascending=False)
            P2.round(3).to_csv(f"{OUT}/P2_defog_by_task.csv", index=False)
        print(f"  {name}: done")
    P1, P3, P4 = pd.DataFrame(p1), pd.DataFrame(p3), pd.DataFrame(p4)
    P1.round(4).to_csv(f"{OUT}/P1_reference_surrogate.csv", index=False)
    P3.round(4).to_csv(f"{OUT}/P3_recalibration.csv", index=False); P4.round(2).to_csv(f"{OUT}/P4_axis_check.csv", index=False)
    print("\nPOST HOC P1 - within-dataset LSTM versus chance\n", P1.round(3).to_string(index=False))
    print("\nPOST HOC P2 - defog by task (frozen laboratory LSTM)\n", P2.round(2).to_string(index=False))
    cols = ["dataset", "thresholds_chosen", "cue_specificity", "timely_eligible_%", "surrogate_timely", "p_timely",
            "detected_%", "surrogate_detection", "p_detection", "false_alarms_per_hour"]
    print("\nPOST HOC P3 - local recalibration of the operating point\n", P3[cols].round(3).to_string(index=False))
    print("\nPOST HOC P4 - lower-back axes (mg)\n", P4.round(1).to_string(index=False))
    print(f"\nDone. Files in {OUT}/  (all analyses are post hoc)")


if __name__ == "__main__":
    main()