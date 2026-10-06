from __future__ import annotations

import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

from fog_pipeline import (CLASS_NAMES, FOG, NOFOG, POST, PRE, Config,
                          contiguous_runs, extract_features, load_daphnet,
                          make_rf, paper_protocol, per_class_metrics,
                          rank_by_mi, relabel_four_class, segment_windows,
                          smote, weighted_scores, HYBRID_23)

N_CLASSES = 4
CUE_ON = (PRE, FOG, POST)         


def sequences(meta: pd.DataFrame, mask: np.ndarray | None = None):
    m = meta if mask is None else meta[mask]
    for key, g in m.groupby(["subject", "run", "seg"], sort=True):
        yield key, g.sort_values("pos").index.to_numpy()


def estimate_hmm(y: np.ndarray, meta: pd.DataFrame, mask: np.ndarray, alpha=1.0):
    A = np.full((N_CLASSES, N_CLASSES), alpha)
    pi = np.full(N_CLASSES, alpha)
    for _, idx in sequences(meta, mask):
        s = y[idx]
        pi[s[0]] += 1
        np.add.at(A, (s[:-1], s[1:]), 1)
    return A / A.sum(1, keepdims=True), pi / pi.sum()


def forward_filter(E: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    T = len(E)
    out = np.empty_like(E)
    a = pi * E[0]
    out[0] = a / a.sum()
    for t in range(1, T):
        a = (out[t - 1] @ A) * E[t]
        out[t] = a / (a.sum() + 1e-300)
    return out


def viterbi(E: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    T, K = E.shape
    lE, lA = np.log(E + 1e-12), np.log(A)
    d = np.log(pi) + lE[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        s = d[:, None] + lA
        back[t] = s.argmax(0)
        d = s.max(0) + lE[t]
    path = np.empty(T, dtype=int)
    path[-1] = d.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def causal_moving_average(P: np.ndarray, k: int = 3) -> np.ndarray:
    c = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), axis=0)
    idx = np.arange(1, len(P) + 1)
    lo = np.maximum(0, idx - k)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


def loso_decoders(X, y4, meta, cfg: Config, k_feats=23, ma_k=3):
    n = len(y4)
    preds = {name: np.full(n, -1) for name in
             ["RF (per-window)", f"Moving avg (k={ma_k})",
              "HMM forward (causal)", "Viterbi (offline)"]}
    groups = meta["subject"].to_numpy()
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        feats = rank_by_mi(X[tr], y4[tr], cfg.seed).index[:k_feats].tolist()
        Xb, yb = smote(X.loc[tr, feats].to_numpy(), y4[tr], cfg.smote_k, cfg.seed)
        model = make_rf(cfg).fit(Xb, yb)
        A, pi = estimate_hmm(y4, meta, tr)

        for _, idx in sequences(meta, te):
            P = np.full((len(idx), N_CLASSES), 1e-6)
            P[:, model.classes_] += model.predict_proba(X.loc[idx, feats].to_numpy())
            P /= P.sum(1, keepdims=True)
            preds["RF (per-window)"][idx] = P.argmax(1)
            preds[f"Moving avg (k={ma_k})"][idx] = causal_moving_average(P, ma_k).argmax(1)
            preds["HMM forward (causal)"][idx] = forward_filter(P, A, pi).argmax(1)
            preds["Viterbi (offline)"][idx] = viterbi(P, A, pi)
        print(f"  fold subject {s:02d} done")
    return preds


def event_metrics(y2, y_pred, meta, cfg: Config, mask=None) -> dict:
    w = cfg.win_sec
    n_ep = n_pred = n_det = n_fa = 0
    leads, delays = [], []
    total_windows = 0
    for _, idx in sequences(meta, mask):
        true_fog = y2[idx] == 2
        on = np.isin(y_pred[idx], CUE_ON)
        total_windows += len(idx)
        episodes = contiguous_runs(true_fog)
        runs = contiguous_runs(on)
        zone = np.zeros(len(idx), bool)
        for s, e in episodes:
            zone[max(0, s - cfg.pre_w):min(len(idx), e + cfg.post_w)] = True
        for a, b in runs:
            if not zone[a:b].any():
                n_fa += 1
        for s, e in episodes:
            n_ep += 1
            if on[s:e].any():
                n_det += 1
            if s > 0 and on[s - 1]:
                n_pred += 1
                start = next(a for a, b in runs if a < s <= b)
                leads.append((s - start) * w)
            if on[e - 1]:
                end = next(b for a, b in runs if a < e <= b)
                delays.append((end - e) * w)
    hours = total_windows * w / 3600
    return {
        "episodes": n_ep,
        "predicted_before_onset_%": 100 * n_pred / max(n_ep, 1),
        "detected_%": 100 * n_det / max(n_ep, 1),
        "median_lead_time_s": float(np.median(leads)) if leads else 0.0,
        "false_alarms_per_hour": n_fa / max(hours, 1e-9),
        "median_cue_off_delay_s": float(np.median(delays)) if delays else 0.0,
    }


def rf_binary(cfg: Config, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, criterion="gini",
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=cfg.seed)


def hysteresis(score: np.ndarray, on_thr: float, off_thr: float) -> np.ndarray:
    out = np.zeros(len(score), dtype=bool)
    state = False
    for t, s in enumerate(score):
        state = (s >= on_thr) if not state else (s >= off_thr)
        out[t] = state
    return out


def apply_hysteresis(score, meta, mask, on_thr, off_thr) -> np.ndarray:
    on = np.zeros(len(score), dtype=bool)
    for _, idx in sequences(meta, mask):
        on[idx] = hysteresis(score[idx], on_thr, off_thr)
    return on


def tune_hysteresis(score, y_on, meta, mask):
    best, best_f1 = (0.5, 0.5), -1.0
    ys = y_on[mask]
    for on_thr in np.arange(0.30, 0.91, 0.05):
        for off_thr in np.arange(0.10, on_thr + 1e-9, 0.10):
            pred = apply_hysteresis(score, meta, mask, on_thr, off_thr)[mask]
            f1 = f1_score(ys, pred, zero_division=0)
            if f1 > best_f1:
                best, best_f1 = (round(on_thr, 2), round(off_thr, 2)), f1
    return best


def binary_cue_loso(X, y4, meta, cfg: Config, k_feats=23, inner_splits=3):
    y_on = np.isin(y4, CUE_ON).astype(int)
    groups = meta["subject"].to_numpy()
    score = np.zeros(len(y4))
    on_all = np.zeros(len(y4), dtype=bool)
    chosen = {}
    n_inner = min(cfg.n_estimators, 50)
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        feats = rank_by_mi(X[tr], y_on[tr], cfg.seed).index[:k_feats].tolist()
        tr_idx = np.flatnonzero(tr)

        oof = np.zeros(len(y4))
        for a, b in GroupKFold(inner_splits).split(tr_idx, groups=groups[tr_idx]):
            m = rf_binary(cfg, n_inner).fit(X.iloc[tr_idx[a]][feats], y_on[tr_idx[a]])
            oof[tr_idx[b]] = m.predict_proba(X.iloc[tr_idx[b]][feats])[:, 1]
        on_thr, off_thr = tune_hysteresis(oof, y_on, meta, tr)

        m = rf_binary(cfg, cfg.n_estimators).fit(X.iloc[tr_idx][feats], y_on[tr_idx])
        score[te] = m.predict_proba(X.loc[te, feats])[:, 1]
        on_all[te] = apply_hysteresis(score, meta, te, on_thr, off_thr)[te]
        chosen[s] = {"on_thr": on_thr, "off_thr": off_thr}
        print(f"  E4 fold subject {s:02d}: on={on_thr} off={off_thr}")

    pred = np.where(on_all, FOG, NOFOG)
    return pred, score, pd.DataFrame(chosen).T


def binary_window_metrics(y4, pred) -> dict:
    t, p = np.isin(y4, CUE_ON), np.isin(pred, CUE_ON)
    tp, tn = float(np.sum(t & p)), float(np.sum(~t & ~p))
    fp, fn = float(np.sum(~t & p)), float(np.sum(t & ~p))
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"cue_sensitivity": sens, "cue_specificity": spec, "cue_precision": prec,
            "cue_f1": 2 * prec * sens / max(prec + sens, 1e-12),
            "balanced_acc": (sens + spec) / 2}


SENSOR_SETS = {
    "Shank": ("ank",),
    "Thigh": ("thi",),
    "Trunk": ("tr",),
    "Shank+Thigh+Trunk": ("ank", "thi", "tr"),
}


def sensor_ablation(root: str, quick: bool = False, sets: dict = SENSOR_SETS):
    os.makedirs("results", exist_ok=True)
    raw = load_daphnet(root)
    rows = {}
    for name, sensors in sets.items():
        cfg = Config(n_estimators=30 if quick else 100, sensors=sensors)
        print(f"\nE5: {name} ...")
        windows, meta = segment_windows(raw, cfg)
        X = extract_features(windows, cfg)
        y2 = meta["y2"].to_numpy()
        y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
        pred, _, _ = binary_cue_loso(X, y4, meta, cfg)
        rows[name] = {"n_features": X.shape[1],
                      **binary_window_metrics(y4, pred),
                      **event_metrics(y2, pred, meta, cfg)}
        pd.DataFrame(rows).T.round(3).to_csv("results/E5_sensor_ablation.csv")  # save as we go
    e5 = pd.DataFrame(rows).T.round(3)
    print("\nE5\n", e5)
    return e5


def main(root: str, quick: bool = False, run_e123: bool = True):
    cfg = Config(n_estimators=30 if quick else 100)
    os.makedirs("results", exist_ok=True)

    print("Loading + features ...")
    windows, meta = segment_windows(load_daphnet(root), cfg)
    X = extract_features(windows, cfg)
    y2 = meta["y2"].to_numpy()
    y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)

    print("E4: binary cue ON/OFF with hysteresis (LOSO) ...")
    pred_bin, _, thr = binary_cue_loso(X, y4, meta, cfg)
    thr.to_csv("results/E4_thresholds_per_fold.csv")
    e4 = pd.DataFrame({"Binary cue + hysteresis": {
        **binary_window_metrics(y4, pred_bin), **event_metrics(y2, pred_bin, meta, cfg)}}).T
    e4.round(3).to_csv("results/E4_binary_cue.csv")
    print("\nE4\n", e4.round(3))
    per_subj = {s: event_metrics(y2, pred_bin, meta, cfg, meta["subject"].eq(s).to_numpy())
                for s in sorted(meta["subject"].unique())}
    pd.DataFrame(per_subj).T.round(3).to_csv("results/E4_per_subject.csv")
    if not run_e123:
        return

    print("E1: paper protocol ...")
    cv, _, _ = paper_protocol(X, y4, HYBRID_23, cfg)
    print("E1/E2: LOSO decoders ...")
    preds = loso_decoders(X, y4, meta, cfg)

    e1 = pd.DataFrame({"Paper protocol (10-fold, SMOTE before split)": cv,
                       "LOSO (leakage-free, RF per-window)":
                           pd.Series(weighted_scores(y4, preds["RF (per-window)"]))}).T
    e1.round(4).to_csv("results/E1_protocol_comparison.csv")
    print("\nE1\n", e1.round(4))

    rows, macro = {}, {}
    for name, p in preds.items():
        rows[name] = weighted_scores(y4, p)
        pc = per_class_metrics(y4, p)
        macro[name] = pc.loc[["Pre-FoG", "Post-FoG"], ["recall", "f1"]].mean()
        pc.to_csv(f"results/E2_per_class_{name.split()[0]}.csv")
    e2 = pd.concat([pd.DataFrame(rows).T,
                    pd.DataFrame(macro).T.add_prefix("transition_")], axis=1)
    e2.round(4).to_csv("results/E2_decoders.csv")
    print("\nE2\n", e2.round(4))

    e3 = pd.DataFrame({name: {**binary_window_metrics(y4, p), **event_metrics(y2, p, meta, cfg)}
                       for name, p in preds.items()}).T
    e3.round(3).to_csv("results/E3_event_level.csv")
    print("\nE3\n", e3.round(3))

    best = "HMM forward (causal)"
    per_subj = {s: event_metrics(y2, preds[best], meta, cfg,
                                 meta["subject"].eq(s).to_numpy())
                for s in sorted(meta["subject"].unique())}
    pd.DataFrame(per_subj).T.round(3).to_csv("results/E3_per_subject_hmm.csv")
    print(f"\nE3 per subject ({best})\n", pd.DataFrame(per_subj).T.round(3))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    from paths import DAPHNET_ROOT
    ROOT = args[0] if args else DAPHNET_ROOT
    sensor_ablation(ROOT, quick=False)            
