import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import bmel_dev as bd
import fog_improve as fi
import fog_lstm_controller as flc
import kaggle_convert as kc
from fog_pipeline import Config, load_daphnet, segment_windows, extract_features, relabel_four_class, FOG, NOFOG
from fog_study import apply_hysteresis, event_metrics, binary_window_metrics, CUE_ON
from fog_deep import past_index, gather, DEVICE
from paths import DAPHNET_ROOT

OUT = "results_bmel_transfer"
QUICK = False
RUN_REFERENCE = True
TESTS = {"defog": (r"D:\New research fog work\kaggle_fog\daphnet_format\defog_back", False),
         "FoG-STAR": (r"D:\FoG-STAR\daphnet_format_back", True),
         "DAPHNET": (DAPHNET_ROOT, True)}
PRIMARY = "defog"
warnings.filterwarnings("ignore")


def load_test(path, merge, cfg):
    raw = load_daphnet(path)
    if merge:                                                        
        for _, idx in raw.groupby(["subject", "run"]).groups.items():
            raw.loc[idx, "label"] = kc.merge_gaps(raw.loc[idx, "label"].to_numpy().astype(int))
    windows, meta = segment_windows(raw, cfg)
    X = extract_features(windows, cfg)
    return X, meta, relabel_four_class(meta, cfg.pre_w, cfg.post_w)


def evaluate(p, th, X, meta, y4, cfg, n_shifts):
    allm = np.ones(len(y4), bool); on = apply_hysteresis(p, meta, allm, *th)
    onsets = bd.eligible_onsets(meta["y2"].to_numpy(), meta, cfg)
    s = bd.summarise(y4, on, meta, cfg, onsets); g = bd.surrogate(on, meta, cfg, onsets, n_shifts)
    per = []
    y2 = meta["y2"].to_numpy(); pred = np.where(on, FOG, NOFOG)
    for sub in sorted(meta.subject.unique()):
        m = meta.subject.eq(sub).to_numpy(); ons = bd.eligible_onsets(y2, meta, cfg, m)
        per.append({"subject": sub, **binary_window_metrics(y4[m], pred[m]), **event_metrics(y2, pred, meta, cfg, m),
                    "timely_pre_onset_%": bd.timely_elig(on, ons, cfg) if ons else np.nan})
    return s, g, pd.DataFrame(per)


def fit_models(X, F, P8, y_on, idx, epochs, cfg):
    feats = fi.rank_mi(X.iloc[idx], y_on[idx], cfg.seed).index[:fi.K_FEATS].tolist()
    rf = fi.Model("RF", cfg).fit(X.iloc[idx][feats], y_on[idx])
    mu = F[idx].mean(0); sd = F[idx].std(0); sd = np.where(sd > 0, sd, 1.0)
    Z = gather(np.clip((F - mu) / sd, -10, 10).astype(np.float32), P8)
    lstm, _, _ = flc.train(Z, y_on, idx, epochs, cfg.seed)
    return {"rf": rf, "feats": feats, "lstm": lstm, "mu": mu, "sd": sd}


def predict(M, X, F, P8):
    P, cl = M["rf"].predict_proba(X[M["feats"]]); p_rf = P[:, list(cl).index(1)]
    Z = gather(np.clip((F - M["mu"]) / M["sd"], -10, 10).astype(np.float32), P8)
    return {"RF": p_rf, "LSTM": flc.prob(M["lstm"], Z)}


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=30 if QUICK else 100)
    fi.QUICK = QUICK; n_shifts = 20 if QUICK else bd.N_SHIFTS
    fz = json.load(open("frozen_config.json"))
    th = {k: (v["theta_on"], v["theta_off"]) for k, v in fz["operating_points"].items()}
    epochs = 2 if QUICK else fz["lstm_final_epochs"]
    print(f"Device: {DEVICE} | frozen operating points {th} | LSTM epochs {epochs}")
    Xd, md, y4d, _ = bd.load_data(cfg)
    yd = np.isin(y4d, CUE_ON).astype(int); Fd = np.nan_to_num(Xd.to_numpy(np.float64)); P8d = past_index(md, 8)
    ck = os.path.join(OUT, "T_models_lab.pkl")
    if os.path.exists(ck):
        M = pickle.load(open(ck, "rb"))
    else:
        M = fit_models(Xd, Fd, P8d, yd, np.arange(len(yd)), epochs, cfg); pickle.dump(M, open(ck, "wb"))
    summ, sur, per, tests, desc, ref = [], [], [], [], [], []
    for name, (path, merge) in TESTS.items():
        ckd = os.path.join(OUT, f"T_probs_{name}.pkl")
        if os.path.exists(ckd):
            X, meta, y4, probs = pickle.load(open(ckd, "rb"))
        else:
            X, meta, y4 = load_test(path, merge, cfg)
            F = np.nan_to_num(X.to_numpy(np.float64)); P8 = past_index(meta, 8)
            probs = predict(M, X, F, P8); pickle.dump((X, meta, y4, probs), open(ckd, "wb"))
        y2 = meta["y2"].to_numpy(); ons = bd.eligible_onsets(y2, meta, cfg)
        desc.append({"dataset": name, "subjects": meta.subject.nunique(), "windows": len(meta),
                     "hours": len(meta) * cfg.win_sec / 3600, "episodes": len(bd.contiguous_runs(y2 == 2)),
                     "eligible_episodes": len(ons), "cue_needed_%": 100 * np.isin(y4, CUE_ON).mean()})
        pt = {}
        for ctrl in ["LSTM", "RF"]:
            s, g, p = evaluate(probs[ctrl], th[ctrl], X, meta, y4, cfg, n_shifts)
            summ.append({"dataset": name, "controller": ctrl, **s})
            for met in g.index: sur.append({"dataset": name, "controller": ctrl, "metric": met, **g.loc[met].to_dict()})
            p.insert(0, "method", ctrl); p.insert(0, "dataset", name); per.append(p); pt[ctrl] = p
        t = fi.tests(pd.concat([pt["LSTM"], pt["RF"]]), [("LSTM", "RF")]); t.insert(0, "dataset", name); tests.append(t)
        if RUN_REFERENCE:                                           # secondary: retrain inside the dataset
            F = np.nan_to_num(X.to_numpy(np.float64)); P8 = past_index(meta, 8); yo = np.isin(y4, CUE_ON).astype(int)
            g_ = meta.subject.to_numpy(); P_ref = {"RF": np.zeros(len(yo)), "LSTM": np.zeros(len(yo))}
            for tr_i, te_i in GroupKFold(min(10, len(np.unique(g_)))).split(X, groups=g_):
                Mi = fit_models(X, F, P8, yo, tr_i, epochs, cfg)
                pr = predict(Mi, X, F, P8)
                for c in P_ref: P_ref[c][te_i] = pr[c][te_i]
            for ctrl in ["LSTM", "RF"]:
                allm = np.ones(len(y4), bool); on = apply_hysteresis(P_ref[ctrl], meta, allm, *th[ctrl])
                ref.append({"dataset": name, "controller": ctrl, "training": "within-dataset (grouped CV)",
                            **bd.summarise(y4, on, meta, cfg, ons)})
        print(f"  {name}: done")
    S, G, Pt, T, D = pd.DataFrame(summ), pd.DataFrame(sur), pd.concat(per), pd.concat(tests), pd.DataFrame(desc)
    S.round(4).to_csv(f"{OUT}/T1_summary.csv", index=False); G.round(4).to_csv(f"{OUT}/T2_surrogate.csv", index=False)
    Pt.round(4).to_csv(f"{OUT}/T3_per_subject.csv", index=False); T.round(4).to_csv(f"{OUT}/T4_tests.csv", index=False)
    D.round(3).to_csv(f"{OUT}/T5_datasets.csv", index=False)
    cols = ["dataset", "controller", "cue_specificity", "timely_eligible_%", "detected_%", "false_alarms_per_hour",
            "median_cue_off_delay_s", "median_lead_time_s"]
    print("\nDatasets\n", D.round(2).to_string(index=False))
    print("\nTransfer of the frozen laboratory controllers\n", S[cols].round(3).to_string(index=False))
    print("\nSurrogate\n", G.round(3).to_string(index=False))
    if RUN_REFERENCE:
        R = pd.DataFrame(ref); R.round(4).to_csv(f"{OUT}/T6_reference.csv", index=False)
        print("\nReference: retrained within each dataset\n", R[cols].round(3).to_string(index=False))
    gp = G[(G.dataset == PRIMARY) & (G.controller == "LSTM")].set_index("metric")["p_one_sided"]
    fa = S[(S.dataset == PRIMARY) & (S.controller == "LSTM")]["false_alarms_per_hour"].iloc[0]
    print(f"\nPre-registered hypotheses ({PRIMARY}, LSTM controller):")
    p1, p2 = gp["timely_eligible_%"], gp["detected_%"]                    
    lo, hi = sorted([p1, p2]); adj = {lo: min(1, 2 * lo)}; adj[hi] = min(1, max(adj[lo], hi))
    print(f"  H1 timely activation > surrogate : p = {p1:.3f} (Holm {adj[p1]:.3f})")
    print(f"  H2 detection > surrogate         : p = {p2:.3f} (Holm {adj[p2]:.3f})")
    print(f"  H3 (descriptive) false alarms per hour at home: {fa:.1f} (laboratory budget {fz['fa_budget_per_hour']:.0f})")
    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()