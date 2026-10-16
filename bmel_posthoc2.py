import os
import ast
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import bmel_dev as bd
import bmel_posthoc as bp
import bmel_transfer as bt
import bmel_robustness as br
import conf_step1_analysis as c1
from bmel_sensitivity import FastMetrics, surrogate_segments, recording_segments
from fog_pipeline import Config
from fog_study import apply_hysteresis, sequences, CUE_ON
from fog_stats import holm

OUT = "results_bmel_posthoc2"
N_POOLED, N_SWEEP, N_RETEST = 2000, 200, 2000
FA_MAX, SPEC_MIN = 50.0, 0.70
warnings.filterwarnings("ignore")

def metrics(on, y4, meta, cfg, onsets):
    s = bd.summarise(y4, on, meta, cfg, onsets)
    return {"false_alarms_per_hour": s["false_alarms_per_hour"], "specificity": s["cue_specificity"],
            "coverage_%": s["detected_%"], "timely_%": s["timely_eligible_%"]}

def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",))
    th_lab = tuple(json.load(open("frozen_config.json"))["operating_points"]["LSTM"][k] for k in ("theta_on", "theta_off"))
    X, meta, y4, probs = pickle.load(open(os.path.join(bt.OUT, "T_probs_defog.pkl"), "rb"))
    p_lab = np.asarray(probs["LSTM"]); p_home = np.load(os.path.join(bp.OUT, "ref_lstm_defog.npy"))
    assert len(p_lab) == len(p_home) == len(meta), "probability files do not match the defog windows"
    N = len(meta); allm = np.ones(N, bool); seqs = [idx for _, idx in sequences(meta)]
    onsets = bd.eligible_onsets(meta["y2"].to_numpy(), meta, cfg); fm = FastMetrics(meta, cfg, onsets)
    segs = recording_segments(meta)

    on_lab = apply_hysteresis(p_lab, meta, allm, *th_lab)
    K4 = pd.read_csv("results_conference2/defog_K4_thresholds.csv")
    K4 = K4[K4.controller == "C-pop"].drop_duplicates(["subject", "run"])
    pair = {(int(r.subject), int(r.run)): ast.literal_eval(r.population_pair) for r in K4.itertuples()}
    keys = list(zip(meta.subject.astype(int), meta.run.astype(int)))
    missing = sorted(set(keys) - set(pair))
    assert not missing, f"no population pair for recordings {missing[:5]}"
    a = np.array([pair[k][0] for k in keys]); b = np.array([pair[k][1] for k in keys])
    on_pop = c1.hysteresis_var(p_home, seqs, a, b)

    rows = []
    for name, on in [("frozen laboratory LSTM", on_lab), ("home-trained LSTM, C-pop", on_pop)]:
        sg = surrogate_segments(on, meta, cfg, onsets, segs, N_POOLED, fm=fm)
        rows.append({"controller": name, **metrics(on, y4, meta, cfg, onsets),
                     "timely_surrogate": sg["timely_surrogate_mean"], "p_timely": sg["timely_p"],
                     "coverage_surrogate": sg["detection_surrogate_mean"], "p_coverage": sg["detection_p"]})
    A = pd.DataFrame(rows)
    K1 = pd.read_csv("results_conference2/defog_K1_summary.csv").set_index("controller").loc["C-pop"]
    chk = A.iloc[1]
    ok = (abs(chk["false_alarms_per_hour"] - K1["false_alarms_per_hour"]) < 1e-2 and abs(chk["coverage_%"] - K1["coverage_%"]) < 1e-2
          and abs(chk["timely_%"] - K1["timely_%"]) < 1e-2)
    print(f"check C-pop reproduces defog_K1_summary.csv: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print(chk.to_dict(), K1.to_dict())

    rows = []
    for name, on in [("frozen laboratory LSTM", on_lab), ("home-trained LSTM, C-pop", on_pop)]:
        P = br.patient_level(on, meta, cfg)
        for met, x, y in [("timely activation", "timely_obs", "timely_null"), ("episode coverage", "coverage_obs", "coverage_null")]:
            rows.append({"controller": name, "metric": met, **br.summarise_excess(P, x, y)})
    B = pd.DataFrame(rows)

    L = c1.per_patient(on_lab, y4, meta, cfg).set_index("subject"); H = c1.per_patient(on_pop, y4, meta, cfg).set_index("subject")
    rows = []
    for met in ["false_alarms_per_hour", "coverage_%", "timely_%"]:
        x, y = H[met].to_numpy(float), L[met].to_numpy(float); okm = ~(np.isnan(x) | np.isnan(y)); d = x[okm] - y[okm]
        p = float(wilcoxon(x[okm], y[okm]).pvalue) if okm.sum() >= 5 and np.any(d != 0) else np.nan
        better = (d < 0) if met.startswith("false") else (d > 0)
        rows.append({"metric": met, "n_patients": int(okm.sum()), "median_home": float(np.median(x[okm])),
                     "median_laboratory": float(np.median(y[okm])), "median_difference": float(np.median(d)),
                     "patients_better_home": int(better.sum()), "patients_worse_home": int(((d > 0) if met.startswith("false") else (d < 0)).sum()),
                     "p_wilcoxon_two_sided": p})
    C = pd.DataFrame(rows); C["p_holm"] = np.nan; v = C.p_wilcoxon_two_sided.notna()
    if v.any(): C.loc[v, "p_holm"] = holm(C.loc[v, "p_wilcoxon_two_sided"].to_numpy())

    summ = []
    for mname, p in [("laboratory", p_lab), ("home", p_home)]:
        rows = []
        for th in bd.GRID:
            on = apply_hysteresis(p, meta, allm, *th); m = metrics(on, y4, meta, cfg, onsets)
            sg = surrogate_segments(on, meta, cfg, onsets, segs, N_SWEEP, fm=fm)
            rows.append({"model": mname, "theta_on": th[0], "theta_off": th[1], **m,
                         "timely_surrogate": sg["timely_surrogate_mean"], "p_timely": sg["timely_p"],
                         "coverage_surrogate": sg["detection_surrogate_mean"], "p_coverage": sg["detection_p"]})
        S = pd.DataFrame(rows)
        feas = (S.false_alarms_per_hour <= FA_MAX) & (S.specificity >= SPEC_MIN)
        S["meets_constraints"] = feas; S["p_timely_holm"] = np.nan; S["p_coverage_holm"] = np.nan
        if feas.any():
            S.loc[feas, "p_timely_holm"] = holm(S.loc[feas, "p_timely"].to_numpy())
            S.loc[feas, "p_coverage_holm"] = holm(S.loc[feas, "p_coverage"].to_numpy())
        S["p_timely_2000"] = np.nan; S["p_coverage_2000"] = np.nan
        for i in S.index[(S.p_timely < 0.05) | (S.p_coverage < 0.05)]:
            on = apply_hysteresis(p, meta, allm, S.at[i, "theta_on"], S.at[i, "theta_off"])
            sg = surrogate_segments(on, meta, cfg, onsets, segs, N_RETEST, fm=fm)
            S.at[i, "p_timely_2000"] = sg["timely_p"]; S.at[i, "p_coverage_2000"] = sg["detection_p"]
        S.round(5).to_csv(f"{OUT}/Q2_sweep_{mname}.csv", index=False)
        F = S[feas]
        r = {"model": mname, "pairs": len(S), "pairs_meeting_constraints": int(feas.sum()),
             "pairs_timely_p<0.05_uncorrected": int((F.p_timely < 0.05).sum()),
             "pairs_timely_Holm<0.05": int((F.p_timely_holm < 0.05).sum()),
             "min_p_timely_constrained": float(F.p_timely.min()) if len(F) else np.nan,
             "pairs_coverage_p<0.05_uncorrected": int((F.p_coverage < 0.05).sum()),
             "pairs_coverage_Holm<0.05": int((F.p_coverage_holm < 0.05).sum())}
        for cap in (25, 50, 100):
            Fc = S[(S.false_alarms_per_hour <= cap) & (S.specificity >= SPEC_MIN)]
            if len(Fc):
                bt_ = Fc.loc[Fc["timely_%"].idxmax()]; bc = Fc.loc[Fc["coverage_%"].idxmax()]
                r[f"best_timely_FA<={cap}"] = f"{bt_['timely_%']:.1f}% [chance {bt_.timely_surrogate:.1f}], p={bt_.p_timely:.3f} at ({bt_.theta_on},{bt_.theta_off})"
                r[f"best_coverage_FA<={cap}"] = f"{bc['coverage_%']:.1f}% [chance {bc.coverage_surrogate:.1f}], p={bc.p_coverage:.3f} at ({bc.theta_on},{bc.theta_off})"
        summ.append(r); print(f"  sweep {mname}: done")
    D = pd.DataFrame(summ)

    A.round(4).to_csv(f"{OUT}/Q1a_pooled.csv", index=False); B.round(4).to_csv(f"{OUT}/Q1b_patient_excess.csv", index=False)
    C.round(5).to_csv(f"{OUT}/Q1c_home_vs_lab.csv", index=False); D.to_csv(f"{OUT}/Q2_summary.csv", index=False)
    print("\nQ1a pooled (2,000 shifts)\n", A.round(3).to_string(index=False))
    print("\nQ1b patient-level excess over chance (percentage points)\n", B.round(3).to_string(index=False))
    print("\nQ1c per patient: home-trained (C-pop) vs frozen laboratory LSTM\n", C.round(4).to_string(index=False))
    print("\nQ2 threshold sweep at home\n", D.T.to_string(header=False))
    print(f"\nDone. Files in {OUT}/ (post hoc, POSTHOC_ADDENDUM.md)")

if __name__ == "__main__":
    main()
