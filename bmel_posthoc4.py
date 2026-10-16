import os
import sys
import ast
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NEED_MOD = ["frozen_config.json", "bmel_transfer.py", "bmel_dev.py", "bmel_sensitivity.py", "bmel_robustness.py",
            "fog_pipeline.py", "fog_study.py", "fog_deep.py", "fog_lstm_controller.py"]
NEED_DATA = {"defog": ("results_bmel_transfer", "T_probs_defog.pkl"),
             "FoG-STAR": ("results_bmel_transfer", "T_probs_FoG-STAR.pkl"),
             "DAPHNET": ("results_bmel_transfer", "T_probs_DAPHNET.pkl"),
             "DEV": ("results_bmel_dev", "cache_tdcsfog_back.pkl")}
OPTIONAL = {"REF": ("results_bmel_posthoc", "ref_lstm_defog.npy"),
            "K4": ("results_conference2", "defog_K4_thresholds.csv")}
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
                    print(f"  {f} not found: the home-trained controller is skipped in A1")
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
from bmel_sensitivity import FastMetrics, surrogate_segments, recording_segments
from fog_pipeline import Config, FOG, NOFOG
from fog_study import apply_hysteresis, sequences, event_metrics, CUE_ON
from fog_deep import past_index, gather

OUT = "results_bmel_posthoc4"
N_SHIFTS = 2000
N_SHIFTS_A1 = 500
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TESTS = ["defog", "FoG-STAR", "DAPHNET"]
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


def episode_table(meta, cfg):
    y2 = meta["y2"].to_numpy(); rows = []
    for idx, s in bd.eligible_onsets(y2, meta, cfg):
        e = s
        while e < len(idx) and y2[idx[e]] == 2:
            e += 1
        rows.append((idx, s, e))
    return rows


def classify(on, eps, W):
    cat = np.empty(len(eps), dtype=object); lat = np.full(len(eps), np.nan)
    for i, (idx, s, e) in enumerate(eps):
        o = on[idx]
        if o[s - 1]:
            a = s - 1
            while a > 0 and o[a - 1]:
                a -= 1
            cat[i] = "timely_new_activation" if s - a <= W else "already_on_over_4s"
        else:
            hit = np.flatnonzero(o[s:e])
            if len(hit):
                cat[i] = "on_after_onset"; lat[i] = hit[0]
            else:
                cat[i] = "missed"
    return cat, lat


CATS = ["timely_new_activation", "already_on_over_4s", "on_after_onset", "missed"]


def decompose(name, ctrl, on, meta, cfg, seed=3):
    eps = episode_table(meta, cfg); W = cfg.pre_w
    cat, lat = classify(on, eps, W)
    obs = {c: 100 * np.mean(cat == c) for c in CATS}
    seqs = [idx for _, idx in sequences(meta)]; rng = np.random.default_rng(seed); null = []
    for _ in range(N_SHIFTS_A1):
        sh = on.copy()
        for idx in seqs:
            if len(idx) > 1:
                sh[idx] = np.roll(on[idx], rng.integers(1, len(idx)))
        c, _ = classify(sh, eps, W)
        null.append([100 * np.mean(c == k) for k in CATS])
    null = np.array(null); rows = []
    for j, c in enumerate(CATS):
        rows.append({"dataset": name, "controller": ctrl, "category": c, "eligible_episodes": len(eps),
                     "observed_%": obs[c], "chance_%": null[:, j].mean(),
                     "p_greater": (1 + (null[:, j] >= obs[c]).sum()) / (1 + N_SHIFTS_A1),
                     "p_less": (1 + (null[:, j] <= obs[c]).sum()) / (1 + N_SHIFTS_A1)})
    l = lat[~np.isnan(lat)] * cfg.win_sec
    q = (lambda p: float(np.percentile(l, p)) if len(l) else np.nan)
    L = {"dataset": name, "controller": ctrl, "episodes_on_after_onset": len(l), "latency_median_s": q(50),
         "latency_IQR_s": f"{q(25):.1f}-{q(75):.1f}", "latency_p90_s": q(90),
         "cue_active_at_onset_%": obs["timely_new_activation"] + obs["already_on_over_4s"],
         "cue_active_at_onset_chance_%": null[:, 0].mean() + null[:, 1].mean()}
    return rows, L


def home_trained(meta):
    if "REF" not in PATHS or "K4" not in PATHS:
        return None
    try:
        import conf_step1_analysis as c1
        p_home = np.load(PATHS["REF"])
        K4 = pd.read_csv(PATHS["K4"]); K4 = K4[K4.controller == "C-personal"].drop_duplicates(["subject", "run"])
        pair = {(int(r.subject), int(r.run)): ast.literal_eval(r.population_pair) for r in K4.itertuples()}
        keys = list(zip(meta.subject.astype(int), meta.run.astype(int)))
        a = np.array([pair[k][0] for k in keys]); b = np.array([pair[k][1] for k in keys])
        seqs = [idx for _, idx in sequences(meta)]
        return c1.hysteresis_var(p_home, seqs, a, b)
    except Exception as ex:
        print("  home-trained controller skipped:", ex)
        return None


def load_dev(cfg):
    if "DEV" in PATHS:
        return pickle.load(open(PATHS["DEV"], "rb"))
    return bd.load_data(cfg)


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


def baseline_raw(X, kind):
    if kind == "freeze_index":
        cols = [c for c in X.columns if c.startswith("Freezeindex_")]
        return X[cols].to_numpy(float).mean(1)
    cols = [c for c in X.columns if c.startswith("Sumpower_")]
    return X[cols].to_numpy(float).sum(1)


def per_subject(on, meta, cfg):
    rows = []; y2 = meta["y2"].to_numpy()
    for s in sorted(meta.subject.unique()):
        m = meta.subject.eq(s).to_numpy(); idx = np.flatnonzero(m)
        sub = meta.iloc[idx].reset_index(drop=True)
        n_ep = len(bd.contiguous_runs(sub["y2"].to_numpy() == 2))
        if n_ep == 0:
            continue
        ons = bd.eligible_onsets(sub["y2"].to_numpy(), sub, cfg)
        t, c = FastMetrics(sub, cfg, ons)(on[idx])
        fa = event_metrics(y2, np.where(on, FOG, NOFOG), meta, cfg, m)["false_alarms_per_hour"]
        rows.append({"subject": s, "eligible": len(ons), "timely_%": t if len(ons) else np.nan, "coverage_%": c,
                     "false_alarms_per_hour": fa})
    return pd.DataFrame(rows)


def compare(a, b, col):
    d = (a[col] - b[col]).dropna().to_numpy()
    if len(d) < 3 or not np.any(d != 0):
        return np.nan, np.median(d) if len(d) else np.nan, len(d)
    return wilcoxon(d).pvalue, np.median(d), len(d)


def standardise_all(Fd, P8d, others):
    mu = Fd.mean(0); sd = Fd.std(0); sd = np.where(sd > 0, sd, 1.0)
    z = lambda F, P8: gather(np.clip((F - mu) / sd, -10, 10).astype(np.float32), P8)
    return z(Fd, P8d), {k: z(F, P8) for k, (F, P8) in others.items()}


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",))
    fz = json.load(open("frozen_config.json"))
    th = tuple(fz["operating_points"]["LSTM"][k] for k in ("theta_on", "theta_off")); epochs = fz["lstm_final_epochs"]
    D = {n: pickle.load(open(PATHS[n], "rb")) for n in TESTS}

    print("\nA1 timely-activation decomposition")
    A1, A1L = [], []
    for n in TESTS:
        X, meta, y4, probs = D[n]
        ctrls = {"frozen laboratory LSTM": hyst(probs["LSTM"], meta, th)}
        if n == "defog":
            on_h = home_trained(meta)
            if on_h is not None:
                ctrls["home-trained LSTM (P7)"] = on_h
        for c, on in ctrls.items():
            r, L = decompose(n, c, on, meta, cfg); A1 += r; A1L.append(L)
        print(f"  {n}: done")
    A1 = pd.DataFrame(A1); A1L = pd.DataFrame(A1L)
    A1.round(4).to_csv(f"{OUT}/A1_timely_decomposition.csv", index=False); A1L.round(3).to_csv(f"{OUT}/A1_latency.csv", index=False)
    print(A1.round(3).to_string(index=False)); print(A1L.round(2).to_string(index=False))

    print("\nA3 non-learned baselines (thresholds chosen on tdcsfog with the frozen rule)")
    Xd, md, y4d, _ = load_dev(cfg); yd = np.isin(y4d, CUE_ON).astype(int)
    ons_d = bd.eligible_onsets(md["y2"].to_numpy(), md, cfg)
    A3, A3S, A3E, A3C = [], [], [], []
    for kind in ["freeze_index", "movement_power"]:
        ref = np.sort(baseline_raw(Xd, kind))
        score = lambda v: np.searchsorted(ref, v, side="right") / len(ref)
        p_dev = score(baseline_raw(Xd, kind)); th_b = grid_select(p_dev, yd, md, ons_d, cfg)
        on_dev = hyst(p_dev, md, th_b); s_dev = bd.summarise(y4d, on_dev, md, cfg, ons_d)
        A3.append({"baseline": kind, "dataset": "tdcsfog (selection)", "theta": str(th_b),
                   "specificity": s_dev["cue_specificity"], "false_alarms_per_hour": s_dev["false_alarms_per_hour"],
                   "cue_on_%": 100 * on_dev.mean(), "timely_%": s_dev["timely_eligible_%"], "coverage_%": s_dev["detected_%"]})
        for n in TESTS:
            X, meta, y4, probs = D[n]
            on_b = hyst(score(baseline_raw(X, kind)), meta, th_b); on_l = hyst(probs["LSTM"], meta, th)
            A3.append({"baseline": kind, "dataset": n, "theta": str(th_b), **evaluate(on_b, meta, y4, cfg)})
            pb = per_subject(on_b, meta, cfg); pl = per_subject(on_l, meta, cfg)
            pb.insert(0, "controller", kind); pl.insert(0, "controller", "frozen laboratory LSTM")
            for P_ in (pb, pl):
                P_.insert(0, "dataset", n)
            A3S += [pb, pl]
            for col in ["timely_%", "coverage_%", "false_alarms_per_hour"]:
                p, med, k = compare(pl, pb, col)
                A3C.append({"baseline": kind, "dataset": n, "metric": col, "patients": k,
                            "median_LSTM_minus_baseline": med, "wilcoxon_p_two_sided": p})
            P = br.patient_level(on_b, meta, cfg)
            for o, nn in [("timely", "timely"), ("coverage", "coverage")]:
                ex = br.summarise_excess(P, f"{o}_obs", f"{o}_null")
                A3E.append({"baseline": kind, "dataset": n, "outcome": nn, **ex})
            print(f"  {kind} / {n}: done")
    lstm_rows = []
    for n in TESTS:
        X, meta, y4, probs = D[n]
        lstm_rows.append({"baseline": "frozen laboratory LSTM (reference)", "dataset": n, "theta": str(th),
                          **evaluate(hyst(probs["LSTM"], meta, th), meta, y4, cfg)})
    A3 = pd.concat([pd.DataFrame(A3), pd.DataFrame(lstm_rows)], ignore_index=True)
    A3.round(4).to_csv(f"{OUT}/A3_baselines.csv", index=False)
    pd.concat(A3S).round(3).to_csv(f"{OUT}/A3_per_subject.csv", index=False)
    pd.DataFrame(A3E).round(4).to_csv(f"{OUT}/A3_patient_excess.csv", index=False)
    pd.DataFrame(A3C).round(4).to_csv(f"{OUT}/A3_LSTM_vs_baseline.csv", index=False)
    print(A3.round(3).to_string(index=False)); print(pd.DataFrame(A3E).round(3).to_string(index=False))
    print(pd.DataFrame(A3C).round(4).to_string(index=False))

    print("\nA2 seed variability of the frozen LSTM (retrained on all tdcsfog, frozen thresholds)")
    Fd = np.nan_to_num(Xd.to_numpy(np.float64)); P8d = past_index(md, 8)
    others = {}
    for n in TESTS:
        X, meta, _, _ = D[n]; others[n] = (np.nan_to_num(X.to_numpy(np.float64)), past_index(meta, 8))
    Zd, Zt = standardise_all(Fd, P8d, others); del others, Fd, P8d
    A2 = []
    for seed in SEEDS:
        model, _, _ = flc.train(Zd, yd, np.arange(len(yd)), epochs, seed)
        for n in TESTS:
            X, meta, y4, probs = D[n]; p = flc.prob(model, Zt[n])
            r = {"seed": seed, "dataset": n, "corr_with_frozen_probs": float(np.corrcoef(p, probs["LSTM"])[0, 1]),
                 **evaluate(hyst(p, meta, th), meta, y4, cfg)}
            A2.append(r)
        pd.DataFrame(A2).round(4).to_csv(f"{OUT}/A2_seeds.csv", index=False)
        print(f"  seed {seed}: done")
    A2 = pd.DataFrame(A2); new = A2[A2.seed != 42]; summ = []
    for n in TESTS:
        g = new[new.dataset == n]; row = {"dataset": n, "seeds": len(g)}
        for col, d in [("false_alarms_per_hour", 1), ("specificity", 3), ("timely_%", 1), ("coverage_%", 1)]:
            row[col] = f"{g[col].median():.{d}f} [{g[col].min():.{d}f}-{g[col].max():.{d}f}]"
        row["seeds_timely_p<0.05"] = int((g.p_timely < 0.05).sum()); row["seeds_coverage_p<0.05"] = int((g.p_coverage < 0.05).sum())
        summ.append(row)
    S = pd.DataFrame(summ); S.to_csv(f"{OUT}/A2_seed_summary.csv", index=False)
    print(A2.round(3).to_string(index=False)); print(S.to_string(index=False))
    print(f"\nDone. Files in {OUT}/ (post hoc, POSTHOC_ADDENDUM_3.md)")


if __name__ == "__main__":
    main()