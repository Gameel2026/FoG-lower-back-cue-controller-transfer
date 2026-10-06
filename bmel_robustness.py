import os
import sys
import json
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bmel_dev as bd
import bmel_transfer as bt
import kaggle_convert as kc
from bmel_sensitivity import FastMetrics, surrogate_segments, recording_segments
from fog_pipeline import Config, load_daphnet, segment_windows, relabel_four_class, FOG, NOFOG
from fog_study import apply_hysteresis, event_metrics, sequences

OUT = "results_bmel_robustness"
N_PATIENT_SHIFTS, N_BOOT, N_SHIFTS = 500, 5000, 2000
warnings.filterwarnings("ignore")


def patient_level(on, meta, cfg, seed=7):
    rng = np.random.default_rng(seed); rows = []
    for s in sorted(meta.subject.unique()):
        m = meta.subject.eq(s).to_numpy(); idx = np.flatnonzero(m)
        sub = meta.iloc[idx].reset_index(drop=True); o = on[idx]
        ons = bd.eligible_onsets(sub["y2"].to_numpy(), sub, cfg)
        n_ep = len(bd.contiguous_runs(sub["y2"].to_numpy() == 2))
        if n_ep == 0:
            continue
        fm = FastMetrics(sub, cfg, ons); obs_t, obs_c = fm(o)
        seg = recording_segments(sub); null = []
        lens = {g: np.flatnonzero(seg == g) for g in np.unique(seg)}
        for _ in range(N_PATIENT_SHIFTS):
            sh = o.copy()
            for g, w in lens.items():
                if len(w) > 1: sh[w] = np.roll(o[w], rng.integers(1, len(w)))
            null.append(fm(sh))
        null = np.array(null)
        rows.append({"subject": s, "episodes": n_ep, "eligible": len(ons),
                     "timely_obs": obs_t if len(ons) else np.nan, "timely_null": null[:, 0].mean() if len(ons) else np.nan,
                     "coverage_obs": obs_c, "coverage_null": null[:, 1].mean()})
    return pd.DataFrame(rows)


def summarise_excess(P, col_obs, col_null, seed=11):
    d = (P[col_obs] - P[col_null]).dropna().to_numpy(); rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(d, len(d))) for _ in range(N_BOOT)]
    p = wilcoxon(d, alternative="greater").pvalue if len(d) > 5 and np.any(d != 0) else np.nan
    return {"n_patients": len(d), "median_excess": np.median(d), "CI_low": np.percentile(boot, 2.5),
            "CI_high": np.percentile(boot, 97.5), "patients_above_chance": int((d > 0).sum()), "wilcoxon_p_one_sided": p}


def relabel_defog(gap):
    kc.MERGE_GAP_S = 0                                   
    mp = pd.read_csv(os.path.join(os.path.dirname(bt.TESTS["defog"][0]), "kaggle_subject_mapping.csv"))
    mp = mp[mp.folder == "defog_back"].set_index("S")
    raw = load_daphnet(bt.TESTS["defog"][0])
    for (s, r), idx in raw.groupby(["subject", "run"]).groups.items():
        row = mp.loc[f"S{int(s):02d}R{int(r):02d}"]
        f = glob.glob(os.path.join(kc.ROOT, "train", row["kind"], f"{row['Id']}.csv"))[0]
        _, lab = kc.convert_file(f, row["kind"])
        lab = kc.merge_gaps(lab, gap)                    
        if len(lab) != len(idx):
            raise ValueError(f"length mismatch for {row['Id']}")
        raw.loc[idx, "label"] = lab
    kc.MERGE_GAP_S = 2.0
    return raw


def relabel_other(path, gap):
    raw = load_daphnet(path)
    for _, idx in raw.groupby(["subject", "run"]).groups.items():
        raw.loc[idx, "label"] = kc.merge_gaps(raw.loc[idx, "label"].to_numpy().astype(int), gap)
    return raw


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=100)
    fz = json.load(open("frozen_config.json"))
    th = {k: (v["theta_on"], v["theta_off"]) for k, v in fz["operating_points"].items()}
    r1, r2 = [], []
    for name, (path, merge) in bt.TESTS.items():
        X, meta, y4, probs = pickle.load(open(os.path.join(bt.OUT, f"T_probs_{name}.pkl"), "rb"))
        allm = np.ones(len(meta), bool)
        for ctrl in ["LSTM", "RF"]:
            on = apply_hysteresis(probs[ctrl], meta, allm, *th[ctrl]); P = patient_level(on, meta, cfg)
            P.insert(0, "controller", ctrl); P.insert(0, "dataset", name)
            P.to_csv(f"{OUT}/R1_per_patient_{name}_{ctrl}.csv", index=False)
            for met, a, b in [("timely activation", "timely_obs", "timely_null"), ("episode coverage", "coverage_obs", "coverage_null")]:
                r1.append({"dataset": name, "controller": ctrl, "metric": met, **summarise_excess(P, a, b)})
        print(f"  {name}: patient-level done", flush=True)
        on = apply_hysteresis(probs["LSTM"], meta, allm, *th["LSTM"])
        for gap in (1.0, 2.0, 3.0):
            raw = relabel_defog(gap) if name == "defog" else relabel_other(path, gap)
            _, meta_g = segment_windows(raw, cfg)
            if len(meta_g) != len(meta):
                raise ValueError("window grid changed - labels must not change the valid stretches")
            ons = bd.eligible_onsets(meta_g["y2"].to_numpy(), meta_g, cfg); y2 = meta_g["y2"].to_numpy()
            e = event_metrics(y2, np.where(on, FOG, NOFOG), meta_g, cfg)
            sg = surrogate_segments(on, meta_g, cfg, ons, recording_segments(meta_g), N_SHIFTS)
            r2.append({"dataset": name, "merge_gap_s": gap, "episodes": e["episodes"], "eligible": len(ons),
                       "false_alarms_per_hour": e["false_alarms_per_hour"], **sg})
        print(f"  {name}: merging-rule sensitivity done", flush=True)
    A, B = pd.DataFrame(r1), pd.DataFrame(r2)
    A.round(4).to_csv(f"{OUT}/R1_patient_level.csv", index=False); B.round(4).to_csv(f"{OUT}/R2_merging_rule.csv", index=False)
    print("\nR1 - patient-level excess over chance (percentage points)\n", A.round(3).to_string(index=False))
    print("\nR2 - episode-merging rule (frozen LSTM)\n", B.round(4).to_string(index=False))
    print(f"\nDone. Files in {OUT}/ (post hoc robustness analyses)")


if __name__ == "__main__":
    main()