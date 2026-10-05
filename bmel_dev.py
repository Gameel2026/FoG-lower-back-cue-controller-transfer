import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import torch

import fog_improve as fi
from fog_pipeline import (Config, load_daphnet, segment_windows, extract_features, relabel_four_class,
                          contiguous_runs, FOG, NOFOG)
from fog_study import apply_hysteresis, sequences, event_metrics, binary_window_metrics, CUE_ON
from fog_deep import FoGLSTM, past_index, gather, set_seed, DEVICE
from fog_lstm_controller import standardise, prob
import fog_lstm_controller as flc

DATA = r"D:\New research fog work\kaggle_fog\daphnet_format\tdcsfog_back"
OUT = "results_bmel_dev"
QUICK = False                       
N_OUTER, FLOOR, N_SHIFTS = 10, 0.70, 200
MAX_EPOCHS = 30
SOURCES = ["RF", "LSTM", "LSTMc", "FUS"]
OBJECTIVES = ["F1S", "TS"]
BASELINE = "RF/F1S"
GRID = [(round(a, 2), round(b, 2)) for a in np.arange(0.30, 0.951, 0.05) for b in np.arange(0.10, a + 1e-9, 0.10)]
warnings.filterwarnings("ignore")


def load_data(cfg):
    path = os.path.join(OUT, "cache_tdcsfog_back.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    windows, meta = segment_windows(load_daphnet(DATA), cfg)
    X = extract_features(windows, cfg)
    y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    pickle.dump((X, meta, y4, windows[:2000]), open(path, "wb"))
    return X, meta, y4, windows[:2000]


def eligible_onsets(y2, meta, cfg, mask=None):
    out = []
    for _, idx in sequences(meta, mask):
        eps = contiguous_runs(y2[idx] == 2); prev_end = 0
        for s, e in eps:
            if s - prev_end >= cfg.pre_w:
                out.append((idx, s))
            prev_end = e
    return out


def timely_elig(on, onsets, cfg):
    hit = 0
    for idx, s in onsets:
        o = on[idx]
        if o[s - 1]:
            start = s - 1
            while start > 0 and o[start - 1]:
                start -= 1
            hit += (s - start) <= cfg.pre_w
    return 100 * hit / max(len(onsets), 1)


def summarise(y4, on, meta, cfg, onsets):
    pred = np.where(on, FOG, NOFOG); y2 = meta["y2"].to_numpy()
    return {**binary_window_metrics(y4, pred), **event_metrics(y2, pred, meta, cfg),
            "timely_eligible_%": timely_elig(on, onsets, cfg), "eligible_episodes": len(onsets)}


def surrogate(on, meta, cfg, onsets, n_shifts):
    y2 = meta["y2"].to_numpy(); rng = np.random.default_rng(cfg.seed)
    obs_t = timely_elig(on, onsets, cfg); obs_d = event_metrics(y2, np.where(on, FOG, NOFOG), meta, cfg)["detected_%"]
    null = []
    for _ in range(n_shifts):
        sh = on.copy()
        for _, idx in sequences(meta):
            if len(idx) > 1:
                sh[idx] = np.roll(on[idx], rng.integers(1, len(idx)))
        null.append((timely_elig(sh, onsets, cfg), event_metrics(y2, np.where(sh, FOG, NOFOG), meta, cfg)["detected_%"]))
    null = np.array(null)
    rows = {}
    for k, (obs, col) in {"timely_eligible_%": (obs_t, 0), "detected_%": (obs_d, 1)}.items():
        rows[k] = {"observed": obs, "surrogate_mean": null[:, col].mean(), "surrogate_95th": np.percentile(null[:, col], 95),
                   "p_one_sided": (1 + (null[:, col] >= obs).sum()) / (1 + n_shifts)}
    return pd.DataFrame(rows).T


def select(p, y_on, meta, tr, objective, onsets_tr, cfg):
    t_ = y_on[tr].astype(bool); best = (None, (-1, -1.0, -1.0))
    for th in GRID:
        o = apply_hysteresis(p, meta, tr, *th); p_ = o[tr]
        sens, spec = p_[t_].mean(), (~p_[~t_]).mean()
        if spec >= FLOOR:
            f1 = f1_score(y_on[tr], p_, zero_division=0)
            v = (1, f1, 0.0) if objective == "F1S" else (1, timely_elig(o, onsets_tr, cfg), f1)
        else:
            v = (0, 0.5 * (sens + spec), 0.0)
        if v > best[1]:
            best = (th, v)
    return best[0], best[1][0] == 1


def platt(p_fit, y_fit, p_apply):
    z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    lr = LogisticRegression(C=1e6).fit(z(p_fit).reshape(-1, 1), y_fit)
    return lr.predict_proba(z(p_apply).reshape(-1, 1))[:, 1], (float(lr.coef_[0, 0]), float(lr.intercept_[0]))


def rf_scores(X, y_on, groups, tr, te, cfg):
    feats = fi.rank_mi(X[tr], y_on[tr], cfg.seed).index[:fi.K_FEATS].tolist()
    tr_idx = np.flatnonzero(tr); oof = np.zeros(len(y_on))
    for a, b in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):
        m = fi.Model("RF", cfg, n_trees=50).fit(X.iloc[tr_idx[a]][feats], y_on[tr_idx[a]])
        P, cl = m.predict_proba(X.iloc[tr_idx[b]][feats]); oof[tr_idx[b]] = P[:, list(cl).index(1)]
    m = fi.Model("RF", cfg).fit(X.iloc[tr_idx][feats], y_on[tr_idx])
    score = np.zeros(len(y_on)); P, cl = m.predict_proba(X.loc[te, feats]); score[te] = P[:, list(cl).index(1)]
    return oof, score, m, feats


def run_fold(k, tr, te, X, F, P8, y4, y_on, meta, groups, cfg):
    t0 = time.time(); y2 = meta["y2"].to_numpy()
    onsets_tr = eligible_onsets(y2, meta, cfg, tr)
    rf_oof, rf_te, rf_model, feats = rf_scores(X, y_on, groups, tr, te, cfg)
    ls_oof, ls_te, n_ep, vb = flc.lstm_scores(F, P8, y_on, groups, tr, te, cfg.seed, 2 if QUICK else MAX_EPOCHS)
    lc_oof, coef = platt(ls_oof[tr], y_on[tr], ls_oof); lc_te, _ = platt(ls_oof[tr], y_on[tr], ls_te)
    src_oof = {"RF": rf_oof, "LSTM": ls_oof, "LSTMc": lc_oof}; src_te = {"RF": rf_te, "LSTM": ls_te, "LSTMc": lc_te}
    on, choices = {}, []
    for obj in OBJECTIVES:
        for s in ["RF", "LSTM", "LSTMc"]:
            th, feas = select(src_oof[s], y_on, meta, tr, obj, onsets_tr, cfg)
            on[f"{s}/{obj}"] = apply_hysteresis(src_te[s], meta, te, *th)[te]
            choices.append({"fold": k, "controller": f"{s}/{obj}", "w_rf": np.nan, "theta_on": th[0], "theta_off": th[1], "feasible": feas})
        best = None                                                     
        for w in (0.25, 0.5, 0.75):
            p_oof = w * rf_oof + (1 - w) * lc_oof
            th, feas = select(p_oof, y_on, meta, tr, obj, onsets_tr, cfg)
            o = apply_hysteresis(p_oof, meta, tr, *th); p_ = o[tr]; t_ = y_on[tr].astype(bool)
            spec = (~p_[~t_]).mean()
            v = (int(feas), f1_score(y_on[tr], p_, zero_division=0) if obj == "F1S" else timely_elig(o, onsets_tr, cfg))
            if best is None or v > best[0]:
                best = (v, w, th, feas)
        _, w, th, feas = best
        on[f"FUS/{obj}"] = apply_hysteresis(w * rf_te + (1 - w) * lc_te, meta, te, *th)[te]
        choices.append({"fold": k, "controller": f"FUS/{obj}", "w_rf": w, "theta_on": th[0], "theta_off": th[1], "feasible": feas})
    for c in choices:
        c.update({"lstm_epochs": n_ep, "lstm_val_bacc": vb, "platt_a": coef[0], "platt_b": coef[1]})
    print(f"  fold {k}: {te.sum()} test windows, LSTM epochs {n_ep}  ({(time.time() - t0) / 60:.1f} min)")
    return on, choices, rf_model, feats


def compute_cost(rf_model, feats, windows, F, P8, cfg):
    rows = []
    t0 = time.perf_counter(); extract_features(windows[:500], cfg); fe = (time.perf_counter() - t0) / 500 * 1000
    rf = rf_model.m if hasattr(rf_model, "m") else rf_model
    nodes = sum(t.tree_.node_count for t in rf.estimators_)
    x1 = pd.DataFrame(np.zeros((1, len(feats))), columns=feats)
    t0 = time.perf_counter()
    for _ in range(200): rf.predict_proba(x1)
    rows.append({"model": "RF (100 trees)", "parameters_or_nodes": nodes, "feature_ms_per_window": fe,
                 "inference_ms_per_window": (time.perf_counter() - t0) / 200 * 1000})
    m = FoGLSTM(F.shape[1], n_cls=2).cpu().eval(); x = torch.zeros(1, P8.shape[1], F.shape[1])
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(200): m(x)
    rows.append({"model": "LSTM (2 x 64)", "parameters_or_nodes": sum(p.numel() for p in m.parameters()),
                 "feature_ms_per_window": fe, "inference_ms_per_window": (time.perf_counter() - t0) / 200 * 1000})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=30 if QUICK else 100)
    fi.QUICK = QUICK; flc.MAX_EPOCHS = 2 if QUICK else MAX_EPOCHS
    print(f"Device: {DEVICE}")
    X, meta, y4, win_sample = load_data(cfg)
    y_on = np.isin(y4, CUE_ON).astype(int); groups = meta["subject"].to_numpy(); y2 = meta["y2"].to_numpy()
    F = np.nan_to_num(X.to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0); P8 = past_index(meta, 8)
    print(f"tdcsfog: {meta.subject.nunique()} subjects, {len(meta)} windows, cue-needed {100 * y_on.mean():.1f}%")
    folds = list(GroupKFold(N_OUTER).split(X, groups=groups))[: 2 if QUICK else N_OUTER]
    names = [f"{s}/{o}" for o in OBJECTIVES for s in SOURCES]
    on = {n: np.zeros(len(y4), bool) for n in names}; choices = []; rf_model = feats = None
    keep = np.zeros(len(y4), bool)
    for k, (tr_i, te_i) in enumerate(folds):
        tr = np.zeros(len(y4), bool); tr[tr_i] = True; te = ~tr; keep |= te
        ck = os.path.join(OUT, f"ckpt_fold{k}.pkl")
        if os.path.exists(ck):
            o, ch = pickle.load(open(ck, "rb")); print(f"  fold {k}: loaded from checkpoint")
        else:
            o, ch, rf_model, feats = run_fold(k, tr, te, X, F, P8, y4, y_on, meta, groups, cfg)
            pickle.dump((o, ch), open(ck, "wb"))
        for n in names: on[n][te] = o[n]
        choices += ch
    if rf_model is None:                                    
        _, _, rf_model, feats = rf_scores(X, y_on, groups, tr, te, cfg)
    meta_k = meta[keep].reset_index(drop=True); y4_k = y4[keep]
    onsets = eligible_onsets(meta_k["y2"].to_numpy(), meta_k, cfg)
    summ, sur, per = [], [], []
    for n in names:
        o = on[n][keep]
        summ.append({"controller": n, **summarise(y4_k, o, meta_k, cfg, onsets)})
        sg = surrogate(o, meta_k, cfg, onsets, 20 if QUICK else N_SHIFTS)
        for met in sg.index: sur.append({"controller": n, "metric": met, **sg.loc[met].to_dict()})
        for s in sorted(meta_k.subject.unique()):
            m = meta_k.subject.eq(s).to_numpy(); pred = np.where(o, FOG, NOFOG)
            ons_s = eligible_onsets(meta_k["y2"].to_numpy(), meta_k, cfg, m)
            per.append({"method": n, "subject": s, **binary_window_metrics(y4_k[m], pred[m]),
                        **event_metrics(meta_k["y2"].to_numpy(), pred, meta_k, cfg, m),
                        "timely_pre_onset_%": timely_elig(o, ons_s, cfg) if ons_s else np.nan})
    S = pd.DataFrame(summ); G = pd.DataFrame(sur); Pt = pd.DataFrame(per)
    tests = fi.tests(Pt, [(n, BASELINE) for n in names if n != BASELINE])
    S.round(4).to_csv(f"{OUT}/D1_summary.csv", index=False); G.round(4).to_csv(f"{OUT}/D2_surrogate.csv", index=False)
    Pt.round(4).to_csv(f"{OUT}/D3_per_subject.csv", index=False); tests.round(4).to_csv(f"{OUT}/D4_tests.csv", index=False)
    pd.DataFrame(choices).round(4).to_csv(f"{OUT}/D5_choices.csv", index=False)
    C = compute_cost(rf_model, feats, win_sample, F, P8, cfg); C.round(4).to_csv(f"{OUT}/D6_compute.csv", index=False)
    cols = ["controller", "cue_specificity", "balanced_acc", "timely_eligible_%", "detected_%", "median_lead_time_s",
            "false_alarms_per_hour", "median_cue_off_delay_s"]
    print("\nSummary\n", S[cols].round(3).to_string(index=False))
    print("\nSurrogate\n", G.round(3).to_string(index=False))
    print("\nCompute\n", C.round(3).to_string(index=False))

    base_t = S.set_index("controller").loc[BASELINE, "timely_eligible_%"]
    pt = G[G.metric == "timely_eligible_%"].set_index("controller")["p_one_sided"]
    ok = S[(S.controller.map(pt) <= 0.05) & (S["timely_eligible_%"] >= base_t - 5)]
    if len(ok):
        print(f"\nSelected by the pre-specified rule: {ok.sort_values('false_alarms_per_hour').iloc[0]['controller']}")
    else:
        print("\nNo controller met the pre-specified rule.")
    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()