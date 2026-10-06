import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import fog_improve as fi
from fog_pipeline import relabel_four_class
from fog_study import apply_hysteresis, CUE_ON
from fog_fix3 import Model

OUT = "results_floor"
FLOORS = [None, 0.60, 0.65, 0.70, 0.75, 0.80]          
DATASETS = {"DAPHNET": ["results_causal", os.path.join("New Results", "results_causal")],
            "Multimodal FoG": ["results_li_causal", os.path.join("New Results", "results_li_causal")]}
REF_KEY = {"DAPHNET": "DAPHNET C0spec", "Multimodal FoG": "Li2021 C0spec"}
QUICK = False
REPORT_ONLY = False     
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
fl_name = lambda f: "no floor" if f is None else f"floor {f:.2f}"


def find_src(ds):
    for d in DATASETS[ds]:
        if os.path.exists(os.path.join(d, "cache_Shank.pkl")):
            return d
    raise FileNotFoundError(f"cache_Shank.pkl for {ds} not found in {DATASETS[ds]}")


def select(oof, y_on, meta, tr, grid, floor):
    t_ = y_on[tr].astype(bool); best = (None, (-1, -1.0)); best_f1 = (None, -1.0); cache = {}
    for th in grid:
        if th not in cache:
            p_ = apply_hysteresis(oof, meta, tr, *th)[tr]
            cache[th] = (p_[t_].mean(), (~p_[~t_]).mean(), f1_score(y_on[tr], p_, zero_division=0))
        sens, spec, f1 = cache[th]
        if f1 > best_f1[1]: best_f1 = (th, f1)
        v = (1, f1) if (floor is None or spec >= floor) else (0, 0.5 * (sens + spec))
        if v > best[1]: best = (th, v)
    return best[0], best[1][0] == 1, best[0] != best_f1[0]


def run_dataset(ds, cfg):
    fi.SRC = find_src(ds)
    X, meta = fi.load("Shank"); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    y_on = np.isin(y4, CUE_ON).astype(int); groups = meta["subject"].to_numpy()
    grid = [(round(a, 2), round(b, 2)) for a in np.arange(0.30, 0.91, 0.05) for b in np.arange(0.10, a + 1e-9, 0.10)]
    on = {f: np.zeros(len(y4), bool) for f in FLOORS}; choices = []
    subjects = np.unique(groups)[:3] if QUICK else np.unique(groups)
    for s in subjects:
        tr, te = groups != s, groups == s
        feats = fi.rank_mi(X[tr], y_on[tr], cfg.seed).index[:fi.K_FEATS].tolist()
        tr_idx = np.flatnonzero(tr); oof = np.zeros(len(y4))
        for a, b in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):          
            m = Model("RF", cfg, n_trees=min(cfg.n_estimators, 50)).fit(X.iloc[tr_idx[a]][feats].to_numpy(), y_on[tr_idx[a]])
            P, cl = m.predict_proba(X.iloc[tr_idx[b]][feats].to_numpy()); oof[tr_idx[b]] = P[:, list(cl).index(1)]
        m = Model("RF", cfg).fit(X.iloc[tr_idx][feats].to_numpy(), y_on[tr_idx])      
        P, cl = m.predict_proba(X.loc[te, feats].to_numpy()); score = np.zeros(len(y4)); score[te] = P[:, list(cl).index(1)]
        for f in FLOORS:
            th, feasible, changed = select(oof, y_on, meta, tr, grid, f)
            on[f][te] = apply_hysteresis(score, meta, te, *th)[te]
            choices.append({"dataset": ds, "floor": fl_name(f), "subject": s, "theta_on": th[0], "theta_off": th[1],
                            "floor_changed_F1_choice": changed if f is not None else False, "floor_feasible": feasible})
        print(f"  [{ds}] subject {s:02d} done")
    if QUICK:
        keep = meta["subject"].isin(subjects).to_numpy()
        meta, y4 = meta[keep].reset_index(drop=True), y4[keep]
        on = {f: v[keep] for f, v in on.items()}
    return y4, meta, on, pd.DataFrame(choices)


def report(S, sur, T):
    ref = next((p for p in ["results_constrained/B1_summary.csv", os.path.join("New Results", "results_constrained", "B1_summary.csv")] if os.path.exists(p)), None)
    if ref and not QUICK:
        b1 = pd.read_csv(ref, index_col=0)          
        for ds, key in REF_KEY.items():
            r = S[(S.dataset == ds) & (S.floor == "floor 0.70")].iloc[0]
            diffs = {k: abs(r[k] - b1.loc[key, k]) for k in ["timely_pre_onset_%", "detected_%", "false_alarms_per_hour", "cue_specificity"]}
            ok = all(v <= 1e-3 + 1e-9 for v in diffs.values())
            print(f"Check {ds}: floor 0.70 reproduces the main analysis -> {'YES' if ok else 'NO'}  "
                  + ", ".join(f"{k}: {r[k]:.3f} vs {b1.loc[key, k]:.3f}" for k in diffs))
    try:
        figure(S, sur)
    except Exception as e:
        print(f"Figure could not be saved ({e}); the CSV results are complete.")
    cols = ["dataset", "floor", "cue_specificity", "balanced_acc", "timely_pre_onset_%", "detected_%",
            "median_lead_time_s", "false_alarms_per_hour", "median_cue_off_delay_s", "folds_changed_vs_F1", "infeasible_folds"]
    print("\nSummary\n", S[cols].round(3).to_string(index=False))
    print("\nSurrogate (timely activation)\n", sur.query("metric=='timely_pre_onset_%'").round(3).to_string(index=False))
    print("\nPer-patient tests vs floor 0.70 (Holm within each comparison)\n", T.to_string(index=False))


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    if REPORT_ONLY:
        report(pd.read_csv(f"{OUT}/FS1_summary.csv"), pd.read_csv(f"{OUT}/FS2_surrogate.csv"), pd.read_csv(f"{OUT}/FS4_tests.csv"))
        return
    cfg = fi.cfg_(); summ, sur, per, tests, chs = [], [], [], [], []
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        y4, meta, on, ch = run_dataset(ds, cfg); chs.append(ch)
        table = []
        for f in FLOORS:
            name = fl_name(f); c = ch[ch.floor == name]
            summ.append({"dataset": ds, "floor": name, **fi.summarise(y4, on[f], meta, cfg),
                         "folds_changed_vs_F1": int(c.floor_changed_F1_choice.sum()),
                         "infeasible_folds": int((~c.floor_feasible.astype(bool)).sum()), "n_folds": len(c)})
            sg = fi.surrogate(y4, on[f], meta, cfg)
            for k in ["timely_pre_onset_%", "detected_%"]:
                sur.append({"dataset": ds, "floor": name, "metric": k, **sg.loc[k].to_dict()})
            table.append(fi.per_subject(name, y4, on[f], meta, cfg))
        table = pd.concat(table, ignore_index=True); table.insert(0, "dataset", ds); per.append(table)
        comps = [(fl_name(f), "floor 0.70") for f in FLOORS if f != 0.70]
        t = fi.tests(table.drop(columns="dataset"), comps); t.insert(0, "dataset", ds); tests.append(t)
    S = pd.DataFrame(summ); S.round(4).to_csv(f"{OUT}/FS1_summary.csv", index=False)
    pd.DataFrame(sur).round(4).to_csv(f"{OUT}/FS2_surrogate.csv", index=False)
    pd.concat(per).round(4).to_csv(f"{OUT}/FS3_per_subject.csv", index=False)
    T = pd.concat(tests).round(4); T.to_csv(f"{OUT}/FS4_tests.csv", index=False)
    pd.concat(chs).to_csv(f"{OUT}/FS5_choices.csv", index=False)
    report(S, pd.DataFrame(sur), T)
    print(f"\nDone. Files in {OUT}/")


def figure(S, sur):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    xs = {fl_name(f): (0.50 if f is None else f) for f in FLOORS}
    for ax, (lab, ds) in zip(axes, [("a", "DAPHNET"), ("b", "Multimodal FoG")]):
        d = S[S.dataset == ds].copy(); d["x"] = d.floor.map(xs); d = d.sort_values("x")
        s = sur[(sur.dataset == ds) & (sur.metric == "timely_pre_onset_%")].copy(); s["x"] = s.floor.map(xs); s = s.sort_values("x")
        ax.plot(d.x, 100 * d.cue_specificity, "-o", color="#1f4e79", ms=3.5, label="Specificity (%)")
        ax.plot(d.x, d["detected_%"], "-s", color="#555555", ms=3.5, label="Episode detection (%)")
        ax.plot(d.x, d["timely_pre_onset_%"], "-^", color="#c0392b", ms=3.5, label="Timely activation (%)")
        ax.plot(s.x, s.surrogate_mean, ":", color="#c0392b", lw=1, label="Timely activation, surrogate mean")
        ax.axvline(0.70, color="black", lw=0.6, ls="--")
        ax2 = ax.twinx(); ax2.plot(d.x, d.false_alarms_per_hour, "-d", color="#e67e22", ms=3.5, label="False alarms (/h)")
        ax2.set_ylim(0, max(1, S.false_alarms_per_hour.max()) * 1.15)
        if lab == "b": ax2.set_ylabel("False alarms per hour")
        ax.set_xticks([0.50, 0.60, 0.65, 0.70, 0.75, 0.80]); ax.set_xticklabels(["none", "0.60", "0.65", "0.70", "0.75", "0.80"])
        ax.set_xlabel("Specificity floor"); ax.set_ylim(0, 105); ax.set_title(f"({lab}) {ds}", loc="left", fontsize=8)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        if lab == "a":
            ax.set_ylabel("Percentage")
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{OUT}/FS_floor_sensitivity.png", dpi=300, bbox_inches="tight")
    try:
        fig.savefig(f"{OUT}/FS_floor_sensitivity.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    except Exception:
        fig.savefig(f"{OUT}/FS_floor_sensitivity.tif", dpi=600, bbox_inches="tight")   
    plt.close(fig)


if __name__ == "__main__":
    main()
