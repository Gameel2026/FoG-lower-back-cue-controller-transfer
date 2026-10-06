import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from fog_pipeline import Config, FOG, NOFOG, contiguous_runs, relabel_four_class
from fog_study import apply_hysteresis, sequences, event_metrics, binary_window_metrics, CUE_ON
from fog_fix3 import Model, event_f1
from fog_stats import holm

SRC, OUT = "results_causal", "results_improve"
QUICK = False
RUN = {"ABLATION": True, "SENSORS": True, "STATS": True}
SENSORS = {"Shank": "Shank", "Thigh": "Thigh", "Trunk": "Trunk", "All three": "All_three"}
K_FEATS = 23
MI_MAX_ROWS = 15000          
N_SHIFTS = 200
SPEC_MIN = 0.70             


def cfg_():
    return Config(causal_filter=True, n_estimators=30 if QUICK else 100)


def load(cache):
    _, X, meta, _ = pickle.load(open(f"{SRC}/cache_{cache}.pkl", "rb"))
    return X, meta


def add_context(X, meta):
    base = X.to_numpy(float); cols = list(X.columns)
    m4, m8, s8, d1 = (np.zeros_like(base) for _ in range(4))
    for _, idx in sequences(meta):
        f = pd.DataFrame(base[idx])
        m4[idx] = f.rolling(4, min_periods=1).mean().to_numpy()
        m8[idx] = f.rolling(8, min_periods=1).mean().to_numpy()
        s8[idx] = f.rolling(8, min_periods=2).std().fillna(0).to_numpy()
        d1[idx] = f.diff().fillna(0).to_numpy()
    parts = [X] + [pd.DataFrame(a, columns=[f"{c}{sfx}" for c in cols], index=X.index)
                   for a, sfx in [(m4, "_m2s"), (m8, "_m4s"), (s8, "_sd4s"), (d1, "_d1")]]
    return pd.concat(parts, axis=1)


def rank_mi(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)) if len(y) <= MI_MAX_ROWS else np.sort(rng.choice(len(y), MI_MAX_ROWS, replace=False))
    mi = mutual_info_classif(X.iloc[idx].to_numpy(), y[idx], random_state=seed)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def timely(y2, on, meta, cfg, mask=None):
    n_ep = n_t = 0
    for _, idx in sequences(meta, mask):
        o = on[idx]; runs = contiguous_runs(o)
        for s, e in contiguous_runs(y2[idx] == 2):
            n_ep += 1
            if s > 0 and o[s - 1]:
                start = next(a for a, b in runs if a < s <= b)
                n_t += (s - start) <= cfg.pre_w
    return 100 * n_t / max(n_ep, 1)


def controller(X, y4, meta, cfg, models=("RF",), objective="window_f1", tag=""):
    y_on = np.isin(y4, CUE_ON).astype(int); y2 = meta["y2"].to_numpy()
    groups = meta["subject"].to_numpy(); on_all = np.zeros(len(y4), bool); choice = {}
    step = 0.10 if objective in ("window_f1", "f1_spec") else 0.05
    grid = [(round(a, 2), round(b, 2)) for a in np.arange(0.30, 0.91, 0.05) for b in np.arange(0.10, a + 1e-9, step)]
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        feats = rank_mi(X[tr], y_on[tr], cfg.seed).index[:K_FEATS].tolist()
        tr_idx = np.flatnonzero(tr)
        best = (None, None, (-1, -1.0)); f1_best = (None, None, -1.0)
        for mname in models:
            oof = np.zeros(len(y4))
            for a, b in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):
                m = Model(mname, cfg, n_trees=min(cfg.n_estimators, 50)).fit(X.iloc[tr_idx[a]][feats].to_numpy(), y_on[tr_idx[a]])
                P, cl = m.predict_proba(X.iloc[tr_idx[b]][feats].to_numpy()); oof[tr_idx[b]] = P[:, list(cl).index(1)]
            t_ = y_on[tr].astype(bool)
            for th in grid:
                on = apply_hysteresis(oof, meta, tr, *th); p_ = on[tr]
                sens, spec = p_[t_].mean(), (~p_[~t_]).mean()
                f1 = f1_score(y_on[tr], p_, zero_division=0)
                if f1 > f1_best[2]: f1_best = (mname, th, f1)
                if objective == "window_f1":
                    v = (1, f1)
                elif objective == "balanced_acc":            
                    v = (1, 0.5 * (sens + spec))
                elif objective == "f1_spec":                
                    v = (1, f1) if spec >= SPEC_MIN else (0, 0.5 * (sens + spec))
                else:
                    v = (1, event_f1(y2, on, meta, tr, cfg))
                if v > best[2]: best = (mname, th, v)
        mname, th, _ = best
        m = Model(mname, cfg).fit(X.iloc[tr_idx][feats].to_numpy(), y_on[tr_idx])
        P, cl = m.predict_proba(X.loc[te, feats].to_numpy()); score = np.zeros(len(y4)); score[te] = P[:, list(cl).index(1)]
        on_all[te] = apply_hysteresis(score, meta, te, *th)[te]
        choice[s] = {"model": mname, "theta_on": th[0], "theta_off": th[1],
                     "constraint_changed_choice": (mname, th) != (f1_best[0], f1_best[1]) if objective == "f1_spec" else False,
                     "constraint_feasible": best[2][0] == 1}
        print(f"  [{tag}] subject {s:02d}: {mname} on={th[0]} off={th[1]}")
    return on_all, pd.DataFrame(choice).T


def summarise(y4, on, meta, cfg):
    pred = np.where(on, FOG, NOFOG); y2 = meta["y2"].to_numpy(); allm = np.ones(len(y4), bool)
    return {**binary_window_metrics(y4, pred), **event_metrics(y2, pred, meta, cfg),
            "timely_pre_onset_%": timely(y2, on, meta, cfg), "event_f1": event_f1(y2, on, meta, allm, cfg)}


def per_subject(name, y4, on, meta, cfg):
    pred = np.where(on, FOG, NOFOG); y2 = meta["y2"].to_numpy(); rows = []
    for s in sorted(meta["subject"].unique()):
        m = meta["subject"].eq(s).to_numpy()
        rows.append({"method": name, "subject": s, **binary_window_metrics(y4[m], pred[m]),
                     **event_metrics(y2, pred, meta, cfg, m), "timely_pre_onset_%": timely(y2, on, meta, cfg, m)})
    return pd.DataFrame(rows)


def surrogate(y4, on, meta, cfg):
    y2 = meta["y2"].to_numpy(); rng = np.random.default_rng(cfg.seed)
    obs = summarise(y4, on, meta, cfg); null = []
    for _ in range(N_SHIFTS):
        sh = on.copy()
        for _, idx in sequences(meta):
            if len(idx) > 1: sh[idx] = np.roll(on[idx], rng.integers(1, len(idx)))
        e = event_metrics(y2, np.where(sh, FOG, NOFOG), meta, cfg)
        null.append({"predicted_before_onset_%": e["predicted_before_onset_%"],
                     "timely_pre_onset_%": timely(y2, sh, meta, cfg), "detected_%": e["detected_%"]})
    null = pd.DataFrame(null)
    return pd.DataFrame({k: {"observed": obs[k], "surrogate_mean": null[k].mean(), "surrogate_95th": null[k].quantile(0.95),
                             "p_one_sided": (1 + (null[k] >= obs[k]).sum()) / (1 + N_SHIFTS)} for k in null}).T


METRICS = {"predicted_before_onset_%": True, "timely_pre_onset_%": True, "detected_%": True,
           "median_lead_time_s": True, "false_alarms_per_hour": False, "balanced_acc": True}


def tests(table, comps):
    rows = []; rng = np.random.default_rng(0)
    for a, b in comps:
        fam = []
        A = table[table.method == a].set_index("subject"); B = table[table.method == b].set_index("subject")
        for met, hb in METRICS.items():
            if met not in A or met not in B or B[met].isna().all():
                continue
            common = A.index.intersection(B.index)
            subs = common if met == "false_alarms_per_hour" else common[A.loc[common, "episodes"] > 0]
            d = (A.loc[subs, met] - B.loc[subs, met]).to_numpy(float)
            nz = d[d != 0]; r = np.argsort(np.argsort(np.abs(nz))) + 1
            rbs = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum() if len(nz) else 0.0
            boot = [np.median(rng.choice(d, len(d))) for _ in range(5000)]
            try: p = wilcoxon(A.loc[subs, met], B.loc[subs, met]).pvalue
            except ValueError: p = np.nan
            fam.append({"comparison": f"{a} vs {b}", "metric": met, "n": len(subs),
                        "median_A": A.loc[subs, met].median(), "median_B": B.loc[subs, met].median(),
                        "median_diff": np.median(d), "CI95_low": np.percentile(boot, 2.5), "CI95_high": np.percentile(boot, 97.5),
                        "rank_biserial": rbs, "A_better": f"{int((d > 0).sum() if hb else (d < 0).sum())}/{len(d)}", "p_raw": p})
        fam = pd.DataFrame(fam); fam["p_holm"] = holm(fam["p_raw"].to_numpy()); rows.append(fam)
    return pd.concat(rows, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True); cfg = cfg_()
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    X0, meta = load("Shank"); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    Xc = add_context(X0, meta); print(f"features: base {X0.shape[1]}, with context {Xc.shape[1]}")
    persub = []

    if RUN["ABLATION"]:
        steps = [("C0 base", X0, ("RF",), "window_f1"), ("C1 +context", Xc, ("RF",), "window_f1"),
                 ("C2 +event tuning", Xc, ("RF",), "event_f1"), ("C3 +model selection", Xc, ("RF", "GBoost"), "event_f1")]
        rows = {}
        for name, X, models, obj in steps:
            print(f"\n{name} ...")
            on, ch = controller(X, y4, meta, cfg, models, obj, name)
            ch.to_csv(f"{OUT}/I_choices_{name.split()[0]}_Shank.csv")
            rows[name] = summarise(y4, on, meta, cfg)
            pd.DataFrame(rows).T.round(3).to_csv(f"{OUT}/I1_ablation_shank.csv")
            persub.append(per_subject(f"{name.split()[0]} Shank", y4, on, meta, cfg))
            pd.concat(persub).round(3).to_csv(f"{OUT}/I3_per_subject.csv", index=False)
            if name.startswith("C3"):
                sur = surrogate(y4, on, meta, cfg).round(4); sur.to_csv(f"{OUT}/I5_surrogate_C3_shank.csv"); print(sur)
        print("\nAblation\n", pd.DataFrame(rows).T.round(3))

    if RUN["SENSORS"]:
        rows = {}
        for name, cache in SENSORS.items():
            if name == "Shank" and RUN["ABLATION"]:
                continue                                          
            print(f"\nC3 {name} ...")
            Xs, ms = load(cache); ys = relabel_four_class(ms, cfg.pre_w, cfg.post_w)
            on, ch = controller(add_context(Xs, ms), ys, ms, cfg, ("RF", "GBoost"), "event_f1", f"C3 {name}")
            ch.to_csv(f"{OUT}/I_choices_C3_{cache}.csv")
            rows[name] = summarise(ys, on, ms, cfg); pd.DataFrame(rows).T.round(3).to_csv(f"{OUT}/I2_sensors_C3.csv")
            persub.append(per_subject(f"C3 {name}", ys, on, ms, cfg))
            pd.concat(persub).round(3).to_csv(f"{OUT}/I3_per_subject.csv", index=False)
        print("\nSensors (C3)\n", pd.DataFrame(rows).T.round(3))

    if RUN["STATS"]:
        t = pd.read_csv(f"{OUT}/I3_per_subject.csv")
        old = pd.read_csv(f"{SRC}/S1_per_subject.csv")
        old = old[old.method.isin(["RF (per-window)", "HMM forward (causal)"])]
        table = pd.concat([t, old], ignore_index=True)
        comps = [("C3 Shank", "C0 Shank"), ("C3 Shank", "RF (per-window)"),
                 ("C3 Shank", "HMM forward (causal)"), ("C3 Trunk", "C3 Shank")]
        comps = [c for c in comps if c[0] in set(table.method) and c[1] in set(table.method)]
        res = tests(table, comps).round(4); res.to_csv(f"{OUT}/I4_wilcoxon.csv", index=False); print("\nTests\n", res)
    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()
