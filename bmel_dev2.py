import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bmel_dev as bd
import fog_improve as fi
import fog_lstm_controller as flc
from fog_pipeline import Config, FOG, NOFOG
from fog_study import apply_hysteresis, event_metrics, CUE_ON
from fog_deep import past_index, DEVICE

OUT = "results_bmel_dev2"
QUICK = False
MIN_EPOCHS = 5
SOURCES = ["RF", "LSTM", "LSTM5", "FUS5"]
OBJECTIVES = ["F1S", "TS"]
FA_LEVELS = [25, 50, 80, 100, 150]          
warnings.filterwarnings("ignore")


def lstm_two(F, P8, y_on, groups, tr, te, seed, epochs):
    tr_idx = np.flatnonzero(tr)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(tr_idx, groups=groups[tr_idx]))
    Z = flc.standardise(F, tr_idx[a], P8)
    _, n_ep, vb = flc.train(Z, y_on, tr_idx[a], epochs, seed, val_idx=tr_idx[b])
    res = {}
    for name, ne in [("LSTM", n_ep), ("LSTM5", max(n_ep, MIN_EPOCHS))]:
        if name == "LSTM5" and ne == n_ep:
            res[name] = res["LSTM"]; continue
        oof = np.zeros(len(y_on))
        for ia, ib in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):
            Z = flc.standardise(F, tr_idx[ia], P8)
            m, _, _ = flc.train(Z, y_on, tr_idx[ia], ne, seed); oof[tr_idx[ib]] = flc.prob(m, Z[tr_idx[ib]])
        Z = flc.standardise(F, tr_idx, P8)
        m, _, _ = flc.train(Z, y_on, tr_idx, ne, seed)
        sc = np.zeros(len(y_on)); sc[te] = flc.prob(m, Z[te])
        res[name] = (oof, sc, ne)
    return res, n_ep, vb


def run_fold(k, tr, te, X, F, P8, y_on, meta, groups, cfg):
    t0 = time.time(); y2 = meta["y2"].to_numpy(); onsets_tr = bd.eligible_onsets(y2, meta, cfg, tr)
    rf_oof, rf_te, _, _ = bd.rf_scores(X, y_on, groups, tr, te, cfg)
    ls, n_ep, vb = lstm_two(F, P8, y_on, groups, tr, te, cfg.seed, 2 if QUICK else bd.MAX_EPOCHS)
    oof = {"RF": rf_oof, "LSTM": ls["LSTM"][0], "LSTM5": ls["LSTM5"][0]}
    sc = {"RF": rf_te, "LSTM": ls["LSTM"][1], "LSTM5": ls["LSTM5"][1]}
    on, choices = {}, []
    for obj in OBJECTIVES:
        for s in ["RF", "LSTM", "LSTM5"]:
            th, feas = bd.select(oof[s], y_on, meta, tr, obj, onsets_tr, cfg)
            on[f"{s}/{obj}"] = apply_hysteresis(sc[s], meta, te, *th)[te]
            choices.append({"fold": k, "controller": f"{s}/{obj}", "w_rf": np.nan, "theta_on": th[0], "theta_off": th[1], "feasible": feas})
        best = None
        for w in (0.25, 0.5, 0.75):
            p = w * oof["RF"] + (1 - w) * oof["LSTM5"]
            th, feas = bd.select(p, y_on, meta, tr, obj, onsets_tr, cfg)
            o = apply_hysteresis(p, meta, tr, *th); p_ = o[tr]
            v = (int(feas), bd.f1_score(y_on[tr], p_, zero_division=0) if obj == "F1S" else bd.timely_elig(o, onsets_tr, cfg))
            if best is None or v > best[0]: best = (v, w, th, feas)
        _, w, th, feas = best
        on[f"FUS5/{obj}"] = apply_hysteresis(w * sc["RF"] + (1 - w) * sc["LSTM5"], meta, te, *th)[te]
        choices.append({"fold": k, "controller": f"FUS5/{obj}", "w_rf": w, "theta_on": th[0], "theta_off": th[1], "feasible": feas})
    for c in choices:
        c.update({"lstm_epochs": n_ep, "lstm5_epochs": ls["LSTM5"][2], "lstm_val_bacc": vb})
    probs = {s: sc[s][te] for s in sc}
    print(f"  fold {k}: LSTM epochs {n_ep} -> LSTM5 {ls['LSTM5'][2]}  ({(time.time() - t0) / 60:.1f} min)")
    return on, choices, probs


def curves(probs, y4, meta, cfg, onsets):
    y2 = meta["y2"].to_numpy(); allm = np.ones(len(y4), bool); rows = []
    for s, p in probs.items():
        for th in bd.GRID:
            o = apply_hysteresis(p, meta, allm, *th)
            e = event_metrics(y2, np.where(o, FOG, NOFOG), meta, cfg)
            t_ = np.isin(y4, CUE_ON)
            rows.append({"source": s, "theta_on": th[0], "theta_off": th[1], "false_alarms_per_hour": e["false_alarms_per_hour"],
                         "timely_eligible_%": bd.timely_elig(o, onsets, cfg), "detected_%": e["detected_%"],
                         "cue_specificity": (~o[~t_]).mean()})
    return pd.DataFrame(rows)


def matched(C):
    rows = []
    for s, g in C.groupby("source"):
        for fa in FA_LEVELS:
            h = g[g.false_alarms_per_hour <= fa]
            r = h.loc[h["timely_eligible_%"].idxmax()] if len(h) else None
            rows.append({"source": s, "fa_budget_per_hour": fa,
                         "best_timely_%": r["timely_eligible_%"] if r is not None else np.nan,
                         "detected_%": r["detected_%"] if r is not None else np.nan,
                         "specificity": r["cue_specificity"] if r is not None else np.nan,
                         "theta_on": r["theta_on"] if r is not None else np.nan,
                         "theta_off": r["theta_off"] if r is not None else np.nan})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=30 if QUICK else 100)
    fi.QUICK = QUICK; bd.QUICK = QUICK
    print(f"Device: {DEVICE}")
    X, meta, y4, _ = bd.load_data(cfg)                       # reads results_bmel_dev/cache_tdcsfog_back.pkl
    y_on = np.isin(y4, CUE_ON).astype(int); groups = meta["subject"].to_numpy()
    F = np.nan_to_num(X.to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0); P8 = past_index(meta, 8)
    folds = list(GroupKFold(bd.N_OUTER).split(X, groups=groups))[: 2 if QUICK else bd.N_OUTER]
    names = [f"{s}/{o}" for o in OBJECTIVES for s in SOURCES]
    on = {n: np.zeros(len(y4), bool) for n in names}; P = {s: np.zeros(len(y4)) for s in ["RF", "LSTM", "LSTM5"]}
    choices = []; keep = np.zeros(len(y4), bool)
    for k, (tr_i, te_i) in enumerate(folds):
        tr = np.zeros(len(y4), bool); tr[tr_i] = True; te = ~tr; keep |= te
        ck = os.path.join(OUT, f"ckpt2_fold{k}.pkl")
        if os.path.exists(ck):
            o, ch, pr = pickle.load(open(ck, "rb")); print(f"  fold {k}: loaded from checkpoint")
        else:
            o, ch, pr = run_fold(k, tr, te, X, F, P8, y_on, meta, groups, cfg); pickle.dump((o, ch, pr), open(ck, "wb"))
        for n in names: on[n][te] = o[n]
        for s in P: P[s][te] = pr[s]
        choices += ch
    meta_k = meta[keep].reset_index(drop=True); y4_k = y4[keep]
    onsets = bd.eligible_onsets(meta_k["y2"].to_numpy(), meta_k, cfg)
    probs = {s: P[s][keep] for s in P}; probs["FUS5 (w=0.5)"] = 0.5 * probs["RF"] + 0.5 * probs["LSTM5"]
    pickle.dump((probs, meta_k, y4_k), open(f"{OUT}/probs.pkl", "wb"))
    summ, sur = [], []
    for n in names:
        o = on[n][keep]
        summ.append({"controller": n, **bd.summarise(y4_k, o, meta_k, cfg, onsets)})
        sg = bd.surrogate(o, meta_k, cfg, onsets, 20 if QUICK else bd.N_SHIFTS)
        for met in sg.index: sur.append({"controller": n, "metric": met, **sg.loc[met].to_dict()})
    S, G = pd.DataFrame(summ), pd.DataFrame(sur)
    C = curves(probs, y4_k, meta_k, cfg, onsets); M = matched(C)
    S.round(4).to_csv(f"{OUT}/E1_summary.csv", index=False); G.round(4).to_csv(f"{OUT}/E2_surrogate.csv", index=False)
    C.round(4).to_csv(f"{OUT}/E3_tradeoff_curves.csv", index=False); M.round(3).to_csv(f"{OUT}/E4_matched_false_alarms.csv", index=False)
    pd.DataFrame(choices).round(4).to_csv(f"{OUT}/E5_choices.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, g in C.groupby("source"):
        g = g.sort_values("false_alarms_per_hour"); env = g["timely_eligible_%"].cummax()
        ax.plot(g["false_alarms_per_hour"], env, label=s, lw=1.6)
    for n in ["RF/F1S", "LSTM/F1S", "LSTM5/F1S", "FUS5/F1S"]:
        r = S.set_index("controller").loc[n]; ax.scatter(r["false_alarms_per_hour"], r["timely_eligible_%"], s=30, zorder=3)
        ax.annotate(n, (r["false_alarms_per_hour"], r["timely_eligible_%"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("False alarms per hour (log)"); ax.set_ylabel("Timely activation, eligible episodes (%)")
    ax.set_title("Development (tdcsfog): best timely activation at each false-alarm rate"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/E_tradeoff.png", dpi=200)
    cols = ["controller", "cue_specificity", "balanced_acc", "timely_eligible_%", "detected_%", "median_lead_time_s",
            "false_alarms_per_hour", "median_cue_off_delay_s"]
    print("\nSummary\n", S[cols].round(3).to_string(index=False))
    print("\nSurrogate (timely)\n", G[G.metric == "timely_eligible_%"].round(3).to_string(index=False))
    print("\nBest timely activation at fixed false-alarm budgets (development, descriptive)\n",
          M.pivot(index="fa_budget_per_hour", columns="source", values="best_timely_%").round(1).to_string())
    print(f"\nDone. Files in {OUT}/  (plot: E_tradeoff.png)")


if __name__ == "__main__":
    main()