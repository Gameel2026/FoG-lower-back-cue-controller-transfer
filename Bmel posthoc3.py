import os
import sys
import ast
import json
import pickle
import warnings
import numpy as np
import pandas as pd

NEED_MOD = ["frozen_config.json", "bmel_transfer.py", "bmel_dev.py", "bmel_posthoc.py", "bmel_robustness.py",
            "bmel_sensitivity.py", "conf_step1_analysis.py", "fog_pipeline.py", "fog_study.py", "fog_stats.py"]
NEED_DATA = {"T": ("results_bmel_transfer", "T_probs_defog.pkl"), "REF": ("results_bmel_posthoc", "ref_lstm_defog.npy"),
             "K4": ("results_conference2", "defog_K4_thresholds.csv"), "MAP": ("daphnet_format", "kaggle_subject_mapping.csv")}
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
    hits = {k: [] for k in NEED_DATA}; mods = []
    if base is None or not all(os.path.exists(os.path.join(base, *v)) for v in NEED_DATA.values()):
        print("searching D:\\ and the user folder for the required files ...")
        for root in ["D:\\", os.path.expanduser("~")]:
            if not os.path.exists(root):
                continue
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d not in SKIP]
                if "frozen_config.json" in fn and "bmel_transfer.py" in fn and has_mods(dp):
                    mods.append(dp)
                for k, (sub, f) in NEED_DATA.items():
                    if f in fn:
                        hits[k].append(os.path.join(dp, f))
        if base is None:
            if not mods:
                raise FileNotFoundError("no folder contains all of: " + ", ".join(NEED_MOD))
            base = mods[0]
    paths = {}
    for k, (sub, f) in NEED_DATA.items():
        p = os.path.join(base, sub, f)
        if not os.path.exists(p):
            cand = [h for h in hits[k] if os.path.basename(os.path.dirname(h)) == sub] or hits[k]
            if not cand:
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
import conf_step1_analysis as c1
from bmel_sensitivity import FastMetrics, surrogate_segments, recording_segments
from fog_pipeline import Config
from fog_study import apply_hysteresis, sequences

OUT = "results_bmel_posthoc3"
N_SHIFTS = 2000
warnings.filterwarnings("ignore")


def controllers(cfg):
    th = tuple(json.load(open("frozen_config.json"))["operating_points"]["LSTM"][k] for k in ("theta_on", "theta_off"))
    X, meta, y4, probs = pickle.load(open(PATHS["T"], "rb"))
    p_lab = np.asarray(probs["LSTM"]); p_home = np.load(PATHS["REF"])
    assert len(p_lab) == len(p_home) == len(meta)
    N = len(meta); seqs = [idx for _, idx in sequences(meta)]
    on_lab = apply_hysteresis(p_lab, meta, np.ones(N, bool), *th)
    K4 = pd.read_csv(PATHS["K4"]); K4 = K4[K4.controller == "C-personal"].drop_duplicates(["subject", "run"])
    pair = {(int(r.subject), int(r.run)): ast.literal_eval(r.population_pair) for r in K4.itertuples()}
    keys = list(zip(meta.subject.astype(int), meta.run.astype(int)))
    a = np.array([pair[k][0] for k in keys]); b = np.array([pair[k][1] for k in keys])
    on_pop = c1.hysteresis_var(p_home, seqs, a, b)
    return meta, y4, {"frozen laboratory LSTM": on_lab, "home-trained LSTM, C-pop": on_pop}


def medication(meta):
    mp = pd.read_csv(PATHS["MAP"])
    mp = mp[mp.folder == "defog_back"].set_index("S")["Medication"]
    keys = [f"S{int(s):02d}R{int(r):02d}" for s, r in zip(meta.subject, meta.run)]
    return np.array([str(mp.get(k, "nan")).strip().lower() for k in keys])


def evaluate(on, y4, meta, cfg):
    onsets = bd.eligible_onsets(meta["y2"].to_numpy(), meta, cfg); fm = FastMetrics(meta, cfg, onsets)
    s = bd.summarise(y4, on, meta, cfg, onsets)
    sg = surrogate_segments(on, meta, cfg, onsets, recording_segments(meta), N_SHIFTS, fm=fm)
    P = br.patient_level(on, meta, cfg)
    ex_t = br.summarise_excess(P, "timely_obs", "timely_null"); ex_c = br.summarise_excess(P, "coverage_obs", "coverage_null")
    return {"patients": int(meta.subject.nunique()), "hours": len(meta) * cfg.win_sec / 3600, "episodes": int(s["episodes"]),
            "eligible": len(onsets), "false_alarms_per_hour": s["false_alarms_per_hour"], "specificity": s["cue_specificity"],
            "timely_%": s["timely_eligible_%"], "timely_surrogate": sg["timely_surrogate_mean"], "p_timely": sg["timely_p"],
            "coverage_%": s["detected_%"], "coverage_surrogate": sg["detection_surrogate_mean"], "p_coverage": sg["detection_p"],
            "timely_excess_median": ex_t["median_excess"], "timely_excess_CI": f"[{ex_t['CI_low']:.1f}, {ex_t['CI_high']:.1f}]",
            "timely_patients_above": f"{ex_t['patients_above_chance']}/{ex_t['n_patients']}", "timely_excess_p": ex_t["wilcoxon_p_one_sided"],
            "coverage_excess_median": ex_c["median_excess"], "coverage_excess_CI": f"[{ex_c['CI_low']:.1f}, {ex_c['CI_high']:.1f}]",
            "coverage_patients_above": f"{ex_c['patients_above_chance']}/{ex_c['n_patients']}", "coverage_excess_p": ex_c["wilcoxon_p_one_sided"]}


def burden(on, meta, cfg):
    y2 = meta["y2"].to_numpy(); w = cfg.win_sec; fa_len, all_len = [], []; outside_on = outside = 0
    for _, idx in sequences(meta):
        o = on[idx]; zone = np.zeros(len(idx), bool)
        for s, e in bd.contiguous_runs(y2[idx] == 2):
            zone[max(0, s - cfg.pre_w):min(len(idx), e + cfg.post_w)] = True
        outside += int((~zone).sum()); outside_on += int((o & ~zone).sum())
        for a, b in bd.contiguous_runs(o):
            all_len.append((b - a) * w)
            if not zone[a:b].any():
                fa_len.append((b - a) * w)
    hours = len(meta) * w / 3600; fa_len = np.asarray(fa_len); all_len = np.asarray(all_len)
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else np.nan
    return {"cue_on_outside_targets_%": 100 * outside_on / max(outside, 1),
            "unnecessary_cue_min_per_hour": outside_on * w / 60 / hours,
            "false_alarm_runs": len(fa_len), "fa_run_median_s": q(fa_len, 50), "fa_run_IQR_s": f"{q(fa_len, 25):.1f}-{q(fa_len, 75):.1f}",
            "fa_run_p90_s": q(fa_len, 90), "fa_runs_longer_10s_%": 100 * float((fa_len > 10).mean()) if len(fa_len) else np.nan,
            "all_runs": len(all_len), "run_median_s": q(all_len, 50), "run_p90_s": q(all_len, 90)}


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",))
    meta, y4, C = controllers(cfg); med = medication(meta)
    print("recordings by medication state:", pd.Series(med).groupby(
        pd.Series(meta.subject.astype(str) + "_" + meta.run.astype(str))).first().value_counts().to_dict())
    rows = []
    for state in sorted(set(med) - {"nan", "none", ""}):
        m = med == state; meta_s = meta[m].reset_index(drop=True); y4_s = y4[m]
        for name, on in C.items():
            rows.append({"medication": state, "controller": name, **evaluate(on[m], y4_s, meta_s, cfg)})
        print(f"  medication {state}: done")
    R1 = pd.DataFrame(rows)
    R2 = pd.DataFrame([{"controller": n, **burden(on, meta, cfg)} for n, on in C.items()])
    R1.round(4).to_csv(f"{OUT}/R1_medication.csv", index=False); R2.round(4).to_csv(f"{OUT}/R2_burden.csv", index=False)
    print("\nR1 medication state (2,000 shifts; patient-level excess in percentage points)\n", R1.round(3).T.to_string(header=False))
    print("\nR2 cue-activation burden\n", R2.round(3).T.to_string(header=False))
    print(f"\nDone. Files in {OUT}/ (post hoc, POSTHOC_ADDENDUM_2.md)")


if __name__ == "__main__":
    main()