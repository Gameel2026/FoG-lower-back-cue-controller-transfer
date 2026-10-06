import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:                                   
    from sklearn.experimental import enable_hist_gradient_boosting  
    from sklearn.ensemble import HistGradientBoostingClassifier

from fog_pipeline import (Config, NOFOG, PRE, FOG, POST, contiguous_runs, rank_by_mi,
                          relabel_four_class, smote, weighted_scores, per_class_metrics)
from fog_study import apply_hysteresis, sequences, event_metrics, binary_window_metrics, CUE_ON

OUT = "results_causal"
QUICK = False
RUN = {"T1": True, "T2": True, "T3": True}


def cfg_for(sensors=("ank",), pre=4.0, post=3.0):
    return Config(causal_filter=True, sensors=sensors, n_estimators=30 if QUICK else 100,
                  pre_fog_sec=pre, post_fog_sec=post)


def load(name):
    _, X, meta, _ = pickle.load(open(f"{OUT}/cache_{name}.pkl", "rb"))
    return X, meta


def balanced_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[v] for v in y])


class Model:
    def __init__(self, name, cfg, n_trees=None):
        self.name, self.cfg = name, cfg
        self.n_trees = n_trees or cfg.n_estimators

    def fit(self, X, y, balance="weights"):
        n, cfg = self.name, self.cfg
        if balance == "smote" or n == "MLP":            
            X, y = smote(X, y, cfg.smote_k, cfg.seed); sw = None
        else:
            sw = balanced_weights(y)
        if n == "RF":
            self.m = RandomForestClassifier(n_estimators=self.n_trees, n_jobs=-1, random_state=cfg.seed)
        elif n == "LogReg":
            self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        elif n == "GBoost":
            self.m = HistGradientBoostingClassifier(max_iter=200, random_state=cfg.seed)
        elif n == "MLP":
            self.m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                                   max_iter=300, random_state=cfg.seed))
        if sw is None:
            self.m.fit(X, y)
        else:
            last = self.m.steps[-1][0] + "__sample_weight" if hasattr(self.m, "steps") else "sample_weight"
            self.m.fit(X, y, **{last: sw})
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X), self.m.classes_


def event_counts(y2, on, meta, mask, cfg):
    det = n_ep = fa = runs_tot = 0
    for _, idx in sequences(meta, mask):
        o, fog = on[idx], y2[idx] == 2
        zone = np.zeros(len(idx), bool)
        for s, e in contiguous_runs(fog):
            n_ep += 1; det += bool(o[s:e].any())
            zone[max(0, s - cfg.pre_w):min(len(idx), e + cfg.post_w)] = True
        for a, b in contiguous_runs(o):
            runs_tot += 1; fa += not zone[a:b].any()
    return det, n_ep, fa, runs_tot


def event_f1(y2, on, meta, mask, cfg):
    det, n_ep, fa, runs = event_counts(y2, on, meta, mask, cfg)
    rec = det / max(n_ep, 1); prec = (runs - fa) / max(runs, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12)


def controller_loso(X, y4, meta, cfg, model="RF", objective="window_f1", k_feats=23):
    y_on = np.isin(y4, CUE_ON).astype(int); y2 = meta["y2"].to_numpy()
    groups = meta["subject"].to_numpy(); on_all = np.zeros(len(y4), bool); chosen = {}
    grid = [(round(a, 2), round(b, 2)) for a in np.arange(0.30, 0.91, 0.05)
            for b in np.arange(0.10, a + 1e-9, 0.05 if objective == "event_f1" else 0.10)]
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        feats = rank_by_mi(X[tr], y_on[tr], cfg.seed).index[:k_feats].tolist()
        tr_idx = np.flatnonzero(tr); oof = np.zeros(len(y4))
        for a, b in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):
            m = Model(model, cfg, n_trees=min(cfg.n_estimators, 50)).fit(X.iloc[tr_idx[a]][feats].to_numpy(), y_on[tr_idx[a]])
            P, cl = m.predict_proba(X.iloc[tr_idx[b]][feats].to_numpy()); oof[tr_idx[b]] = P[:, list(cl).index(1)]
        best, best_v = grid[0], -1
        for on_t, off_t in grid:
            on = apply_hysteresis(oof, meta, tr, on_t, off_t)
            v = f1_score(y_on[tr], on[tr], zero_division=0) if objective == "window_f1" else event_f1(y2, on, meta, tr, cfg)
            if v > best_v: best, best_v = (on_t, off_t), v
        m = Model(model, cfg).fit(X.iloc[tr_idx][feats].to_numpy(), y_on[tr_idx])
        P, cl = m.predict_proba(X.loc[te, feats].to_numpy()); score = np.zeros(len(y4)); score[te] = P[:, list(cl).index(1)]
        on_all[te] = apply_hysteresis(score, meta, te, *best)[te]; chosen[s] = best
        print(f"  [{model}/{objective}] subject {s:02d}: on={best[0]} off={best[1]}")
    pred = np.where(on_all, FOG, NOFOG)
    ev = event_metrics(y2, pred, meta, cfg); det, n_ep, fa, runs = event_counts(y2, on_all, meta, np.ones(len(y4), bool), cfg)
    return {**binary_window_metrics(y4, pred), **ev, "event_f1": event_f1(y2, on_all, meta, np.ones(len(y4), bool), cfg),
            "theta_off_min": min(c[1] for c in chosen.values()), "theta_off_max": max(c[1] for c in chosen.values())}


def fourclass_loso(X, y4, meta, cfg, model="RF", k_feats=23):
    groups = meta["subject"].to_numpy(); pred = np.empty_like(y4)
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        feats = rank_by_mi(X[tr], y4[tr], cfg.seed).index[:k_feats].tolist()
        m = Model(model, cfg).fit(X.loc[tr, feats].to_numpy(), y4[tr], balance="smote")
        P, cl = m.predict_proba(X.loc[te, feats].to_numpy()); pred[te] = np.asarray(cl)[P.argmax(1)]
        print(f"  [4-class {model}] subject {s:02d} done")
    pc = per_class_metrics(y4, pred)
    return {**weighted_scores(y4, pred), "PreFoG_recall": pc.loc["Pre-FoG", "recall"],
            "PostFoG_recall": pc.loc["Post-FoG", "recall"],
            "transition_recall": pc.loc[["Pre-FoG", "Post-FoG"], "recall"].mean()}


def save(rows, name):
    df = pd.DataFrame(rows).T.round(4); df.to_csv(f"{OUT}/{name}.csv"); print(f"\n{name}\n", df); return df


def main():
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    if RUN["T1"]:
        rows = {}
        for sensor, cache, sens in [("Shank", "Shank", ("ank",)), ("Trunk", "Trunk", ("tr",))]:
            X, meta = load(cache); cfg = cfg_for(sens); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
            rows[f"{sensor} event-F1 tuning"] = controller_loso(X, y4, meta, cfg, "RF", "event_f1")
            save(rows, "F3_T1_event_tuning")
    if RUN["T2"]:
        X, meta = load("Shank"); cfg = cfg_for(); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
        four, ctrl = {}, {}
        for model in ["LogReg", "GBoost", "MLP"]:
            four[model] = fourclass_loso(X, y4, meta, cfg, model); save(four, "F3_T2_classifiers_fourclass")
            ctrl[model] = controller_loso(X, y4, meta, cfg, model, "window_f1"); save(ctrl, "F3_T2_classifiers_controller")
    if RUN["T3"]:
        X, meta = load("Shank"); four, ctrl = {}, {}
        for pre, post in [(2.0, 1.5), (6.0, 4.5)]:
            cfg = cfg_for(pre=pre, post=post); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
            key = f"Pre {pre:g} s / Post {post:g} s"
            four[key] = fourclass_loso(X, y4, meta, cfg, "RF"); save(four, "F3_T3_labels_fourclass")
            ctrl[key] = controller_loso(X, y4, meta, cfg, "RF", "window_f1"); save(ctrl, "F3_T3_labels_controller")
    print(f"\nDone. Files F3_* in {OUT}/")


if __name__ == "__main__":
    main()
