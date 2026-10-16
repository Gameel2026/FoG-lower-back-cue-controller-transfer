import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

NEED_MOD = ["frozen_config.json", "bmel_dev.py", "bmel_sensitivity.py", "bmel_robustness.py",
            "fog_pipeline.py", "fog_study.py", "fog_deep.py", "fog_lstm_controller.py"]
NEED_DATA = {"defog": ("results_bmel_transfer", "T_probs_defog.pkl")}
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
import fog_lstm_controller as flc
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from bmel_sensitivity import surrogate_segments, recording_segments
from fog_pipeline import Config, FOG, NOFOG
from fog_study import apply_hysteresis, event_metrics, CUE_ON
from fog_deep import past_index

OUT = "results_bmel_posthoc6"
N_SHIFTS = 2000
MAX_EPOCHS = 30
FLOOR = 0.70
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


def select(p, y_on, meta, mask, cfg, rule):
    y2 = meta["y2"].to_numpy(); t_ = y_on[mask].astype(bool); onsets = bd.eligible_onsets(y2, meta, cfg, mask)
    best = (None, (-1, -1.0, -1.0))
    for th in bd.GRID:
        o = apply_hysteresis(p, meta, mask, *th); om = o[mask]; spec = (~om[~t_]).mean()
        if rule == "frozen rule":
            e = event_metrics(y2, np.where(o, FOG, NOFOG), meta, cfg, mask)
            v = (1, bd.timely_elig(o, onsets, cfg), e["detected_%"]) if spec >= FLOOR and e["false_alarms_per_hour"] <= 50 else (0, -e["false_alarms_per_hour"], 0)
        else:
            bacc = 0.5 * (om[t_].mean() + spec)
            v = (1, f1_score(t_, om), bacc) if spec >= FLOOR else (0, bacc, 0)
        if v > best[1]:
            best = (th, v)
    return tuple(float(x) for x in best[0])


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",))
    X, meta, y4, probs = pickle.load(open(PATHS["defog"], "rb"))
    y = np.isin(y4, CUE_ON).astype(int); g = meta.subject.to_numpy()
    F = np.nan_to_num(X.to_numpy(np.float64)); P8 = past_index(meta, 8)
    rules = ["frozen rule", "window F1 with specificity floor"]
    score = np.zeros(len(y)); on = {r: np.zeros(len(y), bool) for r in rules}; log = []
    for f, (tr, te) in enumerate(GroupKFold(min(10, len(np.unique(g)))).split(X, groups=g)):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed).split(tr, groups=g[tr]))
        fit, val = tr[a], tr[b]
        Z = flc.standardise(F, fit, P8)
        _, n_ep, vb = flc.train(Z, y, fit, MAX_EPOCHS, cfg.seed, val_idx=val)
        m, _, _ = flc.train(Z, y, fit, n_ep, cfg.seed)
        pv = np.zeros(len(y)); pv[val] = flc.prob(m, Z[val]); vm = np.zeros(len(y), bool); vm[val] = True
        th = {r: select(pv, y, meta, vm, cfg, r) for r in rules}
        Z = flc.standardise(F, tr, P8)
        m, _, _ = flc.train(Z, y, tr, n_ep, cfg.seed)
        tm = np.zeros(len(y), bool); tm[te] = True; score[te] = flc.prob(m, Z[te])
        for r in rules:
            on[r][te] = apply_hysteresis(score, meta, tm, *th[r])[te]
        log.append({"fold": f + 1, "early_stopped_epochs": n_ep, "validation_balanced_accuracy": vb,
                    **{f"theta {r}": str(th[r]) for r in rules}})
        print(f"  fold {f + 1}: epochs {n_ep}, thresholds {th}")
    L = pd.DataFrame(log); L.round(4).to_csv(f"{OUT}/H1_folds.csv", index=False)
    np.save(f"{OUT}/H1_scores.npy", score)
    R = pd.DataFrame([{"controller": f"home-trained LSTM, early stopping, {r}", **evaluate(on[r], meta, y4, cfg)} for r in rules])
    R.round(4).to_csv(f"{OUT}/H1_home_early_stopping.csv", index=False)
    print("\nEarly-stopped epochs per fold:", L.early_stopped_epochs.tolist(), "median", float(L.early_stopped_epochs.median()))
    print(R.round(3).T.to_string(header=False))
    print(f"\nDone. Files in {OUT}/ (post hoc, POSTHOC_ADDENDUM_5.md)")


if __name__ == "__main__":
    main()
