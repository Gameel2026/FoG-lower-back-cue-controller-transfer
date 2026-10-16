import os
import sys
import time
import json
import pickle
import platform
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NEED_MOD = ["frozen_config.json", "bmel_transfer.py", "bmel_dev.py", "bmel_sensitivity.py", "bmel_robustness.py",
            "bmel_posthoc.py", "fog_pipeline.py", "fog_study.py", "fog_deep.py", "fog_lstm_controller.py", "fog_improve.py"]
NEED_DATA = {"defog": ("results_bmel_transfer", "T_probs_defog.pkl"),
             "FoG-STAR": ("results_bmel_transfer", "T_probs_FoG-STAR.pkl"),
             "DAPHNET": ("results_bmel_transfer", "T_probs_DAPHNET.pkl"),
             "DEV": ("results_bmel_dev", "cache_tdcsfog_back.pkl")}
OPTIONAL = {"MAP": ("daphnet_format", "kaggle_subject_mapping.csv"),
            "TASKS": ("Competition Dataset", "tasks.csv"),
            "S3": ("results_bmel_sensitivity", "S3_deployment_cost.csv")}
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

import torch
import sklearn
import bmel_dev as bd
import bmel_robustness as br
import bmel_transfer as bt
import bmel_posthoc as bp
import fog_improve as fi
import fog_lstm_controller as flc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from bmel_sensitivity import FastMetrics, surrogate_segments, recording_segments
from fog_pipeline import Config, FOG, NOFOG
from fog_study import apply_hysteresis, sequences, event_metrics, CUE_ON
from fog_deep import past_index, gather

OUT = "results_bmel_posthoc5"
N_SHIFTS = 2000
N_SHIFTS_TASK = 500
N_BOOT = 5000
TESTS = ["defog", "FoG-STAR", "DAPHNET"]
CATS = ["timely_new_activation", "already_on_over_4s", "on_after_onset", "missed"]
warnings.filterwarnings("ignore")


def hyst(p, meta, th):
    return apply_hysteresis(np.asarray(p), meta, np.ones(len(meta), bool), *th)


def evaluate(on, meta, y4, cfg, n_shifts=N_SHIFTS):
    onsets = bd.eligible_onsets(meta["y2"].to_numpy(), meta, cfg)
    s = bd.summarise(y4, on, meta, cfg, onsets)
    sg = surrogate_segments(on, meta, cfg, onsets, recording_segments(meta), n_shifts)
    return {"specificity": s["cue_specificity"], "false_alarms_per_hour": s["false_alarms_per_hour"],
            "cue_on_%": 100 * on.mean(), "timely_%": s["timely_eligible_%"], "timely_chance": sg["timely_surrogate_mean"],
            "p_timely": sg["timely_p"], "coverage_%": s["detected_%"], "coverage_chance": sg["detection_surrogate_mean"],
            "p_coverage": sg["detection_p"]}


def excess(on, meta, cfg):
    P = br.patient_level(on, meta, cfg); out = {}
    for o in ["timely", "coverage"]:
        e = br.summarise_excess(P, f"{o}_obs", f"{o}_null")
        out[f"{o}_excess_median"] = e["median_excess"]; out[f"{o}_excess_CI"] = f"[{e['CI_low']:.1f}, {e['CI_high']:.1f}]"
        out[f"{o}_patients_above"] = f"{e['patients_above_chance']}/{e['n_patients']}"; out[f"{o}_excess_p"] = e["wilcoxon_p_one_sided"]
    return out


def grid_select(p, y_on, meta, onsets, cfg, budget=50.0, floor=0.70):
    y2 = meta["y2"].to_numpy(); t_ = y_on.astype(bool); best = (None, (-1, -1, 0)); allm = np.ones(len(meta), bool)
    for th in bd.GRID:
        o = apply_hysteresis(p, meta, allm, *th); spec = (~o[~t_]).mean()
        e = event_metrics(y2, np.where(o, FOG, NOFOG), meta, cfg)
        if spec >= floor and e["false_alarms_per_hour"] <= budget:
            v = (1, bd.timely_elig(o, onsets, cfg), e["detected_%"])
        else:
            v = (0, -e["false_alarms_per_hour"], 0)
        if v > best[1]:
            best = (th, v)
    return tuple(float(x) for x in best[0])


def per_subject(on, meta, cfg):
    rows = []; y2 = meta["y2"].to_numpy()
    for s in sorted(meta.subject.unique()):
        m = meta.subject.eq(s).to_numpy(); idx = np.flatnonzero(m)
        sub = meta.iloc[idx].reset_index(drop=True)
        if len(bd.contiguous_runs(sub["y2"].to_numpy() == 2)) == 0:
            continue
        ons = bd.eligible_onsets(sub["y2"].to_numpy(), sub, cfg)
        t, c = FastMetrics(sub, cfg, ons)(on[idx])
        fa = event_metrics(y2, np.where(on, FOG, NOFOG), meta, cfg, m)["false_alarms_per_hour"]
        rows.append({"subject": s, "timely_%": t if len(ons) else np.nan, "coverage_%": c, "false_alarms_per_hour": fa})
    return pd.DataFrame(rows)


def b1_overlap(D, th, cfg):
    if "MAP" not in PATHS:
        return pd.DataFrame([{"check": "kaggle_subject_mapping.csv not found"}]), None
    mp = pd.read_csv(PATHS["MAP"])
    t = set(mp.loc[mp.folder == "tdcsfog_back", "Subject"].astype(str)); d = set(mp.loc[mp.folder == "defog_back", "Subject"].astype(str))
    ov = sorted(t & d)
    rows = [{"tdcsfog_subjects": len(t), "defog_subjects": len(d), "shared_subject_codes": len(ov), "codes": ";".join(ov)}]
    res = None
    if ov:
        m_ = mp[(mp.folder == "defog_back") & mp.Subject.astype(str).isin(ov)]
        nums = sorted({int(s[1:3]) for s in m_.S})
        X, meta, y4, probs = D["defog"]; keep = ~meta.subject.isin(nums).to_numpy()
        meta_k = meta[keep].reset_index(drop=True); on = hyst(probs["LSTM"], meta, th)[keep]
        res = pd.DataFrame([{"analysis": "defog without shared subjects", "excluded_subjects": str(nums),
                             "patients": int(meta_k.subject.nunique()), **evaluate(on, meta_k, y4[keep], cfg), **excess(on, meta_k, cfg)}])
    return pd.DataFrame(rows), res


def defog_tasks(meta):
    if "MAP" not in PATHS or "TASKS" not in PATHS:
        return None
    mp = pd.read_csv(PATHS["MAP"]); mp = mp[mp.folder == "defog_back"]
    ids = dict(zip(mp["S"], mp["Id"])); tasks = pd.read_csv(PATHS["TASKS"])
    tt = bp.window_times(bt.TESTS["defog"][0], meta); task = np.array(["(no task)"] * len(meta), dtype=object)
    key = np.array([ids.get(f"S{int(s):02d}R{int(r):02d}") for s, r in zip(meta.subject, meta.run)], dtype=object)
    for Id, grp in tasks.groupby("Id"):
        idx = np.flatnonzero(key == Id)
        for _, r in grp.iterrows():
            sel = idx[(tt[idx] >= r.Begin) & (tt[idx] < r.End)]; task[sel] = r.Task
    return task


def episodes(meta, cfg):
    y2 = meta["y2"].to_numpy(); rows = []
    for idx, s in bd.eligible_onsets(y2, meta, cfg):
        e = s
        while e < len(idx) and y2[idx[e]] == 2:
            e += 1
        rows.append((idx, s, e))
    return rows


def classify(on, eps, W):
    cat = np.empty(len(eps), dtype=object)
    for i, (idx, s, e) in enumerate(eps):
        o = on[idx]
        if o[s - 1]:
            a = s - 1
            while a > 0 and o[a - 1]:
                a -= 1
            cat[i] = "timely_new_activation" if s - a <= W else "already_on_over_4s"
        else:
            cat[i] = "on_after_onset" if o[s:e].any() else "missed"
    return cat


def b2_by_task(on, meta, task, cfg, seed=5):
    eps = episodes(meta, cfg); et = np.array([task[idx[s]] for idx, s, e in eps], dtype=object)
    obs = classify(on, eps, cfg.pre_w); seqs = [idx for _, idx in sequences(meta)]; rng = np.random.default_rng(seed)
    groups = [g for g in pd.unique(et) if (et == g).sum() >= 10]; null = {g: [] for g in groups}
    for _ in range(N_SHIFTS_TASK):
        sh = on.copy()
        for idx in seqs:
            if len(idx) > 1:
                sh[idx] = np.roll(on[idx], rng.integers(1, len(idx)))
        c = classify(sh, eps, cfg.pre_w)
        for g in groups:
            m = et == g; null[g].append([100 * np.mean(c[m] == k) for k in CATS])
    rows = []
    for g in groups:
        m = et == g; nl = np.array(null[g])
        for j, k in enumerate(CATS):
            o = 100 * np.mean(obs[m] == k)
            rows.append({"task_at_onset": g, "eligible_episodes": int(m.sum()), "category": k, "observed_%": o,
                         "chance_%": nl[:, j].mean(), "p_greater": (1 + (nl[:, j] >= o).sum()) / (1 + N_SHIFTS_TASK)})
    return pd.DataFrame(rows)


def lstm_fit(Z, y, idx, epochs, seed):
    m, _, _ = flc.train(Z, y, idx, epochs, seed)
    return m


def zstack(F, mu, sd, P):
    return gather(np.clip((F - mu) / sd, -10, 10).astype(np.float32), P)


def bal_weights(y):
    c = np.bincount(y, minlength=2).astype(float); w = len(y) / (2 * np.maximum(c, 1)); return w[y]


def rf_compact(Xtr, ytr, cfg):
    feats = fi.rank_mi(Xtr, ytr, cfg.seed).index[:fi.K_FEATS].tolist()
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=cfg.seed)
    rf.fit(Xtr[feats].to_numpy(), ytr, sample_weight=bal_weights(ytr)); return rf, feats


def b3_models(Xd, md, y4d, D, th_frozen, epochs, cfg):
    yd = np.isin(y4d, CUE_ON).astype(int); Fd = np.nan_to_num(Xd.to_numpy(np.float64)); g = md.subject.to_numpy()
    ons_d = bd.eligible_onsets(md["y2"].to_numpy(), md, cfg)
    hist = {"LSTM, 4 s history (replication)": 8, "LSTM, current window only": 1}
    Pd = {k: past_index(md, n) for k, n in hist.items()}
    names = list(hist) + ["RF, depth 10 (compact)"]
    oof = {k: np.zeros(len(yd)) for k in names}
    for f, (tr, te) in enumerate(GroupKFold(min(10, len(np.unique(g)))).split(Xd, groups=g)):
        mu = Fd[tr].mean(0); sd = Fd[tr].std(0); sd = np.where(sd > 0, sd, 1.0)
        for k in hist:
            Z = zstack(Fd, mu, sd, Pd[k]); m = lstm_fit(Z, yd, tr, epochs, cfg.seed); oof[k][te] = flc.prob(m, Z[te])
        rf, feats = rf_compact(Xd.iloc[tr], yd[tr], cfg); oof[names[2]][te] = rf.predict_proba(Xd.iloc[te][feats].to_numpy())[:, 1]
        print(f"  fold {f + 1} done")
    rows, sel = [], {}
    for k in names:
        th = grid_select(oof[k], yd, md, ons_d, cfg); sel[k] = th
        on = hyst(oof[k], md, th); s = bd.summarise(y4d, on, md, cfg, ons_d)
        rows.append({"model": k, "dataset": "tdcsfog OOF (selection)", "theta": str(th), "specificity": s["cue_specificity"],
                     "false_alarms_per_hour": s["false_alarms_per_hour"], "cue_on_%": 100 * on.mean(),
                     "timely_%": s["timely_eligible_%"], "coverage_%": s["detected_%"]})
    mu = Fd.mean(0); sd = Fd.std(0); sd = np.where(sd > 0, sd, 1.0); cost = []
    final = {}
    for k, n in hist.items():
        final[k] = lstm_fit(zstack(Fd, mu, sd, Pd[k]), yd, np.arange(len(yd)), epochs, cfg.seed)
    rf, feats = rf_compact(Xd, yd, cfg)
    rf.set_params(n_jobs=1); x1 = Xd.iloc[[0]][feats].to_numpy(); t0 = time.perf_counter()
    for _ in range(200):
        rf.predict_proba(x1)
    cost.append({"model": names[2], "size_MB": len(pickle.dumps(rf)) / 1e6, "ms_per_window_inference": (time.perf_counter() - t0) / 200 * 1000,
                 "nodes": int(sum(e.tree_.node_count for e in rf.estimators_))})
    for k in hist:
        try:
            n_par = int(sum(p.numel() for p in final[k].parameters()))
            cost.append({"model": k, "size_MB": n_par * 4 / 1e6, "parameters": n_par})
        except Exception:
            pass
    pt = []
    for n_ in TESTS:
        X, meta, y4, probs = D[n_]; F = np.nan_to_num(X.to_numpy(np.float64))
        P = {}
        for k, h in hist.items():
            P[k] = flc.prob(final[k], zstack(F, mu, sd, past_index(meta, h)))
        P[names[2]] = rf.predict_proba(X[feats].to_numpy())[:, 1]
        for k in names:
            on = hyst(P[k], meta, sel[k])
            rows.append({"model": k, "dataset": n_, "theta": str(sel[k]), **evaluate(on, meta, y4, cfg), **excess(on, meta, cfg)})
        on_ref = hyst(probs["LSTM"], meta, th_frozen); ref = per_subject(on_ref, meta, cfg)
        for k in names[1:]:
            ps = per_subject(hyst(P[k], meta, sel[k]), meta, cfg)
            for col in ["timely_%", "coverage_%", "false_alarms_per_hour"]:
                d = (ref[col] - ps[col]).dropna().to_numpy()
                p = wilcoxon(d).pvalue if len(d) >= 3 and np.any(d != 0) else np.nan
                pt.append({"dataset": n_, "comparison": f"frozen LSTM minus {k}", "metric": col, "patients": len(d),
                           "median_difference": np.median(d) if len(d) else np.nan, "wilcoxon_p_two_sided": p})
        print(f"  {n_}: done")
    return pd.DataFrame(rows), pd.DataFrame(cost), pd.DataFrame(pt)


def b4_bootstrap(D, th, cfg, seed=17):
    rng = np.random.default_rng(seed); rows = []
    for n_ in TESTS:
        X, meta, y4, probs = D[n_]; ps = per_subject(hyst(probs["LSTM"], meta, th), meta, cfg)
        for col in ["timely_%", "coverage_%", "false_alarms_per_hour"]:
            v = ps[col].dropna().to_numpy(); b = [np.median(rng.choice(v, len(v))) for _ in range(N_BOOT)]
            rows.append({"dataset": n_, "metric": col, "patients": len(v), "median": np.median(v),
                         "CI_low": np.percentile(b, 2.5), "CI_high": np.percentile(b, 97.5)})
    return pd.DataFrame(rows)


def b5_environment():
    cpu = platform.processor()
    try:
        import subprocess
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=20).stdout.split("\n")
        names = [l.strip() for l in out if l.strip() and l.strip().lower() != "name"]
        if names:
            cpu = names[0]
    except Exception:
        pass
    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "scikit-learn": sklearn.__version__, "torch": torch.__version__, "os": platform.platform(), "cpu": cpu,
           "logical_cores": os.cpu_count()}
    return pd.DataFrame([env])


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=100)
    fz = json.load(open("frozen_config.json"))
    th = tuple(fz["operating_points"]["LSTM"][k] for k in ("theta_on", "theta_off")); epochs = fz["lstm_final_epochs"]
    D = {n: pickle.load(open(PATHS[n], "rb")) for n in TESTS}

    print("\nB5 software and hardware")
    E = b5_environment(); E.to_csv(f"{OUT}/B5_environment.csv", index=False); print(E.T.to_string(header=False))
    if "S3" in PATHS:
        print("\nregistered deployment-cost file (for the 8.2 vs 8.3 ms check)\n", pd.read_csv(PATHS["S3"]).to_string(index=False))

    print("\nB1 subject overlap between tdcsfog and defog")
    O, R = b1_overlap(D, th, cfg); O.to_csv(f"{OUT}/B1_subject_overlap.csv", index=False); print(O.to_string(index=False))
    if R is not None:
        R.round(4).to_csv(f"{OUT}/B1_defog_without_shared.csv", index=False); print(R.round(3).T.to_string(header=False))

    print("\nB4 per-patient medians with bootstrap 95% CI (frozen laboratory LSTM)")
    B4 = b4_bootstrap(D, th, cfg); B4.round(3).to_csv(f"{OUT}/B4_patient_bootstrap.csv", index=False); print(B4.round(2).to_string(index=False))

    print("\nB2 timing categories by protocol task at onset (defog, frozen laboratory LSTM)")
    X, meta, y4, probs = D["defog"]
    try:
        task = defog_tasks(meta)
    except Exception as ex:
        task = None; print("  task labels could not be built:", ex)
    if task is None:
        print("  B2 skipped")
    else:
        B2 = b2_by_task(hyst(probs["LSTM"], meta, th), meta, task, cfg)
        B2.round(4).to_csv(f"{OUT}/B2_timing_by_task.csv", index=False); print(B2.round(2).to_string(index=False))

    print("\nB3 temporal-context ablation and compact random forest (same freezing procedure)")
    Xd, md, y4d, _ = pickle.load(open(PATHS["DEV"], "rb")) if "DEV" in PATHS else bd.load_data(cfg)
    B3, C3, T3 = b3_models(Xd, md, y4d, D, th, epochs, cfg)
    B3.round(4).to_csv(f"{OUT}/B3_models.csv", index=False); C3.round(4).to_csv(f"{OUT}/B3_cost.csv", index=False)
    T3.round(4).to_csv(f"{OUT}/B3_frozen_LSTM_vs_models.csv", index=False)
    print(B3.round(3).to_string(index=False)); print(C3.round(3).to_string(index=False)); print(T3.round(4).to_string(index=False))
    print(f"\nDone. Files in {OUT}/ (post hoc, POSTHOC_ADDENDUM_4.md)")


if __name__ == "__main__":
    main()
