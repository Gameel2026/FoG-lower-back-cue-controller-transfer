import os
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

import bmel_dev as bd
import bmel_posthoc as bp
import bmel_sensitivity as bs
import fog_lstm_controller as flc
import conf_step1_analysis as c1
from fog_pipeline import Config
from fog_study import sequences, CUE_ON
from fog_deep import past_index
from fog_stats import holm

OUT = "results_conference2"
QUICK = False
N_SHIFTS = 2000
RUN = ["defog", "tdcsfog"]
GRID = [(float(a), float(b)) for a, b in bd.GRID]
BUDGETS = [2, 5, 10]               
CONTROLLERS = ["C-pop", "C-personal"] + [f"C-cal-{b}" for b in BUDGETS]
warnings.filterwarnings("ignore")


def load(ds, cfg):
    if ds == "defog":
        X, meta, y4, _ = pickle.load(open(os.path.join(bp.TR, "T_probs_defog.pkl"), "rb"))
    else:
        X, meta, y4, _ = bd.load_data(cfg)
    return X, meta, y4


def fold_probs(ds, k, tr_i, te_i, F, P8, y_on, g, epochs, seed, p_ref):
    ck = os.path.join(OUT, f"{ds}_ckpt_fold{k}.pkl")
    if os.path.exists(ck):
        return pickle.load(open(ck, "rb"))
    old = os.path.join(c1.OUT, f"ckpt_fold{k}.pkl")
    if ds == "defog" and os.path.exists(old):
        oof, pt, _ = pickle.load(open(old, "rb"))
    else:
        oof = np.zeros(len(y_on))
        for ia, ib in GroupKFold(3).split(tr_i, groups=g[tr_i]):
            Z = flc.standardise(F, tr_i[ia], P8); m, _, _ = flc.train(Z, y_on, tr_i[ia], epochs, seed)
            oof[tr_i[ib]] = flc.prob(m, Z[tr_i[ib]])
        if p_ref is not None:
            pt = p_ref[te_i]
        else:
            Z = flc.standardise(F, tr_i, P8); m, _, _ = flc.train(Z, y_on, tr_i, epochs, seed); pt = flc.prob(m, Z[te_i])
    pickle.dump((oof, pt), open(ck, "wb"))
    return oof, pt


def f1_counts(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def run(ds):
    cfg = Config(causal_filter=True, sensors=("tr",)); n_shifts = 20 if QUICK else N_SHIFTS
    epochs = json.load(open("frozen_config.json"))["lstm_final_epochs"]
    X, meta, y4 = load(ds, cfg); N = len(y4)
    y_on = np.isin(y4, CUE_ON).astype(int); t = y_on.astype(bool); g = meta.subject.to_numpy()
    run_key = pd.factorize(meta.subject.astype(str) + "_" + meta.run.astype(str))[0]
    runs = pd.DataFrame({"key": run_key, "subject": g, "run": meta.run.to_numpy()}).drop_duplicates("key").set_index("key")
    seqs_all = [idx for _, idx in sequences(meta)]
    F = np.nan_to_num(X.to_numpy(np.float64)); P8 = past_index(meta, 8)
    p_ref = None
    if ds == "defog" and os.path.exists(os.path.join(bp.OUT, "ref_lstm_defog.npy")):
        p_ref = np.load(os.path.join(bp.OUT, "ref_lstm_defog.npy"))
    print(f"{ds}: {len(np.unique(g))} patients, {len(runs)} recordings, {N} windows; LSTM epochs {epochs}")

    p_test = np.zeros(N); glob_of = np.empty(N, dtype=object); splits = list(GroupKFold(10).split(X, groups=g))
    if QUICK: splits = splits[:2]
    for k, (tr_i, te_i) in enumerate(splits):
        t0 = time.time(); oof, pt = fold_probs(ds, k, tr_i, te_i, F, P8, y_on, g, epochs, cfg.seed, p_ref); p_test[te_i] = pt
        tr = np.zeros(N, bool); tr[tr_i] = True; seqs_tr = [i for i in seqs_all if tr[i[0]]]; best = (GRID[0], -1.0)
        for th in GRID:
            o = c1.hysteresis_var(oof, seqs_tr, np.full(N, th[0]), np.full(N, th[1])); v = c1.f1(o[tr], t[tr])
            if v > best[1]: best = (th, v)
        for i in te_i: glob_of[i] = best[0]
        print(f"  fold {k}: population pair {best[0]}  ({(time.time() - t0) / 60:.1f} min)")
    ev = np.zeros(N, bool); ev[np.concatenate([s[1] for s in splits])] = True

    # every grid pair applied once to the held-out probabilities; per-recording confusion counts
    O = {}; cnt = np.zeros((len(GRID), len(runs), 3)); TP = np.zeros((len(GRID), N), bool); FP = TP.copy(); FN = TP.copy()
    for j, th in enumerate(GRID):
        o = c1.hysteresis_var(p_test, seqs_all, np.full(N, th[0]), np.full(N, th[1])); O[th] = o
        TP[j] = o & t; FP[j] = o & ~t; FN[j] = ~o & t
        cnt[j, :, 0] = np.bincount(run_key, o & t, len(runs)); cnt[j, :, 1] = np.bincount(run_key, o & ~t, len(runs))
        cnt[j, :, 2] = np.bincount(run_key, ~o & t, len(runs))
    pos_run = np.bincount(run_key, t, len(runs)); minutes = np.bincount(run_key, minlength=len(runs)) * cfg.win_sec / 60

    order = meta.sort_values(["seg", "pos"]).index.to_numpy()              
    run_idx = {r: order[run_key[order] == r] for r in runs.index}
    win_per_min = int(round(60 / cfg.win_sec))
    on = {c: np.zeros(N, bool) for c in CONTROLLERS}; rows_th = []
    for r in runs.index:
        w = (run_key == r) & ev
        if not w.any(): continue
        s = runs.loc[r, "subject"]; pop = glob_of[np.flatnonzero(w)[0]]; on["C-pop"][w] = O[pop][w]
        others = [q for q in runs.index[runs.subject == s] if q != r]
        cal_all = np.concatenate([run_idx[q] for q in sorted(others, key=lambda q: runs.loc[q, "run"])]) if others else np.zeros(0, int)
        choices = [("C-personal", cal_all)] + [(f"C-cal-{b}", cal_all[: b * win_per_min]) for b in BUDGETS]
        for c, cal in choices:
            if len(cal) and t[cal].any():
                f = f1_counts(TP[:, cal].sum(1), FP[:, cal].sum(1), FN[:, cal].sum(1)); th = GRID[int(np.argmax(f))]; fb = False
            else:
                th = pop; fb = True
            on[c][w] = O[th][w]
            rows_th.append({"subject": s, "run": runs.loc[r, "run"], "controller": c, "theta_on": th[0], "theta_off": th[1],
                            "population_pair": str(pop), "fallback_to_population": fb,
                            "calibration_minutes": len(cal) * cfg.win_sec / 60})

    meta_e = meta[ev].reset_index(drop=True); y4_e = y4[ev]; t_e = t[ev]; g_e = g[ev]
    onsets = bd.eligible_onsets(meta_e["y2"].to_numpy(), meta_e, cfg); fm = bs.FastMetrics(meta_e, cfg, onsets); segs = bs.recording_segments(meta_e)
    rows, per = [], {}
    for c in CONTROLLERS:
        o = on[c][ev]; sm = bd.summarise(y4_e, o, meta_e, cfg, onsets); sg = bs.surrogate_segments(o, meta_e, cfg, onsets, segs, n_shifts, fm=fm)
        rows.append({"controller": c, "cue_f1_pooled": c1.f1(o, t_e), "cue_specificity": sm["cue_specificity"],
                     "cue_sensitivity": sm["cue_sensitivity"], "false_alarms_per_hour": sm["false_alarms_per_hour"],
                     "coverage_%": sm["detected_%"], "coverage_surrogate": sg["detection_surrogate_mean"], "p_coverage": sg["detection_p"],
                     "timely_%": sm["timely_eligible_%"], "timely_surrogate": sg["timely_surrogate_mean"], "p_timely": sg["timely_p"]})
        pp = c1.per_patient(o, y4_e, meta_e, cfg).set_index("subject")
        pp["cue_f1"] = [c1.f1(o[g_e == s], t_e[g_e == s]) if t_e[g_e == s].any() else np.nan for s in pp.index]
        per[c] = pp
    K1 = pd.DataFrame(rows)
    K2 = pd.concat([p.assign(controller=c).reset_index() for c, p in per.items()])
    tests = []
    for c in CONTROLLERS[1:]:
        a = per[c]["cue_f1"].to_numpy(); b = per["C-pop"]["cue_f1"].to_numpy(); p, n = c1.wtest(a, b)
        tests.append({"comparison": f"{c} vs C-pop", "metric": "cue_f1", "role": "primary" if c == "C-personal" else "secondary",
                      "n_patients": n, "mean_controller": float(np.nanmean(a)), "mean_pop": float(np.nanmean(b)),
                      "median_difference": float(np.nanmedian(a - b)), "patients_improved": int(np.nansum(a > b)),
                      "patients_worse": int(np.nansum(a < b)), "p_wilcoxon": p, "p_holm": np.nan})
        fam = []
        for metric in ["false_alarms_per_hour", "coverage_%"]:
            a = per[c][metric].to_numpy(); b = per["C-pop"][metric].to_numpy(); p, n = c1.wtest(a, b)
            fam.append({"comparison": f"{c} vs C-pop", "metric": metric, "role": "secondary", "n_patients": n,
                        "mean_controller": float(np.nanmean(a)), "mean_pop": float(np.nanmean(b)),
                        "median_difference": float(np.nanmedian(a - b)), "patients_improved": int(np.nansum((a < b) if metric.startswith("false") else (a > b))),
                        "patients_worse": int(np.nansum((a > b) if metric.startswith("false") else (a < b))), "p_wilcoxon": p})
        for f, h in zip(fam, holm(np.array([f["p_wilcoxon"] for f in fam]))): f["p_holm"] = h
        tests += fam
    K3 = pd.DataFrame(tests); K4 = pd.DataFrame(rows_th)
    bud = (K3.metric == "cue_f1") & K3.comparison.str.startswith("C-cal-")               
    K3.loc[bud, "p_holm"] = holm(K3.loc[bud, "p_wilcoxon"].to_numpy())
    pr = K3.iloc[0]; S = K1.set_index("controller")
    fa_ok = S.loc["C-personal", "false_alarms_per_hour"] <= 1.10 * S.loc["C-pop", "false_alarms_per_hour"]
    cov_ok = S.loc["C-pop", "coverage_%"] - S.loc["C-personal", "coverage_%"] <= 5.0
    better = pr.mean_controller > pr.mean_pop and pr.median_difference >= 0
    if ds == "defog":
        verdict = "SUPPORTED" if (better and pr.p_wilcoxon < 0.05) else "NOT supported"
    else:
        verdict = "REPLICATED (direction)" if better else "NOT replicated"
    verdict += f" | clinically favourable: {bool(fa_ok and cov_ok)} (FA within +10%: {bool(fa_ok)}, coverage within -5 pp: {bool(cov_ok)})"
    K3["verdict_pre_specified"] = ""; K3.loc[0, "verdict_pre_specified"] = verdict
    S0 = K1.set_index("controller"); minimum = "none"
    for b in BUDGETS:
        c = f"C-cal-{b}"; rr = K3[(K3.comparison == f"{c} vs C-pop") & (K3.metric == "cue_f1")].iloc[0]
        ok = (rr.mean_controller > rr.mean_pop and rr.median_difference >= 0 and rr.p_holm < 0.05
              and S0.loc[c, "false_alarms_per_hour"] <= 1.10 * S0.loc["C-pop", "false_alarms_per_hour"]
              and S0.loc["C-pop", "coverage_%"] - S0.loc[c, "coverage_%"] <= 5.0)
        if ok: minimum = f"{b} min"; break
    curve = K3[K3.metric == "cue_f1"][["comparison", "mean_controller", "mean_pop", "median_difference", "patients_improved",
                                       "patients_worse", "p_wilcoxon", "p_holm"]]
    d = (per["C-personal"]["cue_f1"] - per["C-pop"]["cue_f1"]).rename("delta_f1")
    cm = K4[K4.controller == "C-personal"].groupby("subject")["calibration_minutes"].mean()
    j = pd.concat([d, cm], axis=1).dropna(); rho = spearmanr(j.delta_f1, j.calibration_minutes) if len(j) > 4 else (np.nan, np.nan)

    pre = os.path.join(OUT, f"{ds}_")
    K1.round(4).to_csv(f"{pre}K1_summary.csv", index=False); K2.round(4).to_csv(f"{pre}K2_per_patient.csv", index=False)
    K3.round(5).to_csv(f"{pre}K3_tests.csv", index=False); K4.to_csv(f"{pre}K4_thresholds.csv", index=False)
    print(f"\nK1 pooled ({n_shifts} shifts)\n", K1.round(3).to_string(index=False))
    print("\nK3 per-patient tests\n", K3.drop(columns="verdict_pre_specified").round(4).to_string(index=False))
    fb = K4.groupby("controller")["fallback_to_population"].mean() * 100
    print("\nrecordings using the population pair (fallback, %):", fb.round(1).to_dict())
    print(f"Spearman rho(delta F1, calibration minutes) = {rho[0]:.3f}, p = {rho[1]:.4g}")
    print("\nCalibration curve (per-patient F1)\n", curve.round(4).to_string(index=False))
    print(f"\n{ds} PRE-SPECIFIED VERDICT: {verdict}")
    print(f"{ds} MINIMUM CALIBRATION (smallest budget better than C-pop, Holm p < 0.05, clinically favourable): {minimum}")


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    for ds in RUN:
        print(f"\n==================== {ds} ====================")
        run(ds)


if __name__ == "__main__":
    main()