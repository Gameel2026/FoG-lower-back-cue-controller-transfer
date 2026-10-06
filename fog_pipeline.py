from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

NOFOG, PRE, FOG, POST = 0, 1, 2, 3
CLASS_NAMES = {NOFOG: "No-FoG", PRE: "Pre-FoG", FOG: "FoG", POST: "Post-FoG"}


@dataclass
class Config:
    fs: int = 64                         
    win_sec: float = 0.5                 
    band: tuple = (0.5, 20.0)            
    filt_order: int = 4                  
    fog_min_sec: float = 0.3             
    loco_band: tuple = (0.5, 3.0)        
    freeze_band: tuple = (3.0, 8.0)      
    nfft: int = 256                      
    mode_resolution: float = 10.0        
    pre_fog_sec: float = 4.0             
    post_fog_sec: float = 3.0            
    regression_feats: tuple = ("Spectralpeak_1", "max_2")  
    ref_subject: int = 4                 
    thr_k: float = 2.0                   
    n_estimators: int = 100              
    n_splits: int = 10
    test_size: float = 0.30
    smote_k: int = 5                     
    seed: int = 42
    sensors: tuple = ("ank",)            
    causal_filter: bool = True           

    @property
    def win(self) -> int:
        return int(round(self.fs * self.win_sec))

    @property
    def pre_w(self) -> int:
        return int(round(self.pre_fog_sec / self.win_sec))    

    @property
    def post_w(self) -> int:
        return int(round(self.post_fog_sec / self.win_sec))   


HYBRID_23 = [
    "Spectralpeak_1", "Spectralpeak_3", "Meanf_1", "medf_1", "Freezeindex_1",
    "spectralcentroid_1", "spectralEntropy_1", "Locomotorypower_2", "Freezeindex_2",
    "Meanf_2", "medf_2", "Sumpower_2", "Meanf_3", "medf_3",
    "max_2", "max_3", "max_1", "min_1", "mode_1", "RMS_2", "kurtosis_1",
    "Variance_2", "StDev_2",
]

EPS = 1e-12

COLS = ["t_ms", "ank_x", "ank_y", "ank_z", "thi_x", "thi_y", "thi_z",
        "tr_x", "tr_y", "tr_z", "label"]


def load_daphnet(root: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(root, "S*R*.txt"))):
        name = os.path.basename(path)
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
        df["subject"], df["run"] = int(name[1:3]), int(name[4:6])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No DAPHNET files found in {root}")
    return pd.concat(frames, ignore_index=True)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def bandpass(x: np.ndarray, cfg: Config) -> np.ndarray:
    if not cfg.causal_filter:
        raise ValueError("Only causal filtering is supported in this release.")
    sos = signal.butter(cfg.filt_order, cfg.band, btype="bandpass", fs=cfg.fs, output="sos")
    zi = signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]   
    y, _ = signal.sosfilt(sos, x, axis=0, zi=zi)
    return y


def segment_windows(df: pd.DataFrame, cfg: Config):
    W, min_fog = cfg.win, int(round(cfg.fog_min_sec * cfg.fs))
    wins, meta = [], []
    cols = [f"{s}_{a}" for s in cfg.sensors for a in "xyz"]
    for (subj, run), g in df.groupby(["subject", "run"], sort=True):
        acc = g[cols].to_numpy(float)
        lab = g["label"].to_numpy()
        for seg, (a, b) in enumerate(contiguous_runs(lab != 0)):   
            if b - a < 3 * W:                                        
                continue
            xf = bandpass(acc[a:b], cfg)                            
            for k in range((b - a) // W):                           
                sl = slice(k * W, (k + 1) * W)
                n_fog = int(np.sum(lab[a:b][sl] == 2))
                y2 = 2 if n_fog >= min_fog else 1                   
                wins.append(xf[sl])
                meta.append((subj, run, seg, k, y2))
    meta = pd.DataFrame(meta, columns=["subject", "run", "seg", "pos", "y2"])
    return np.stack(wins), meta


def statistical_features(x: np.ndarray, cfg: Config) -> dict:
    vals, cnt = np.unique(np.round(x / cfg.mode_resolution), return_counts=True)
    return {
        "mean": x.mean(),
        "median": np.median(x),
        "mode": vals[cnt.argmax()] * cfg.mode_resolution,
        "min": x.min(),
        "max": x.max(),
        "range": np.ptp(x),
        "harmean": stats.hmean(np.abs(x) + EPS),       
        "StDev": x.std(ddof=1),
        "Variance": x.var(ddof=1),
        "Mean_Ab": np.mean(np.abs(x - x.mean())),       
        "Med_Ab": np.mean(np.abs(x - np.median(x))),   
        "kurtosis": stats.kurtosis(x, fisher=False),    
        "skewness": stats.skew(x),
        "RMS": np.sqrt(np.mean(x ** 2)),
    }


def spectral_features(x: np.ndarray, cfg: Config) -> dict:
    f, P = signal.periodogram(x, fs=cfg.fs, nfft=cfg.nfft, window="hann",
                              detrend="constant")
    df_ = f[1] - f[0]
    bp = lambda lo, hi: P[(f >= lo) & (f < hi)].sum() * df_
    loco, frz = bp(*cfg.loco_band), bp(*cfg.freeze_band)

    p = P / (P.sum() + EPS)                      
    mean_f = np.sum(f * p)                       
    mag = np.sqrt(P); mag /= mag.sum() + EPS
    centroid = np.sum(f * mag)                   
    sd = np.sqrt(np.sum((f - mean_f) ** 2 * p)) + EPS
    return {
        "Locomotorypower": loco,
        "Freezepower": frz,
        "Freezeindex": frz / (loco + EPS),        
        "Sumpower": frz + loco,
        "Meanf": mean_f,
        "medf": f[min(np.searchsorted(np.cumsum(P), 0.5 * P.sum()), len(f) - 1)],
        "spectralcentroid": centroid,
        "spectralKurtosis": np.sum((f - mean_f) ** 4 * p) / sd ** 4,
        "spectralEntropy": -np.sum(p * np.log2(p + EPS)) / np.log2(len(p)),
        "Spectralpeak": f[np.argmax(P)],          
    }


def extract_features(windows: np.ndarray, cfg: Config) -> pd.DataFrame:
    rows = []
    for w in windows:
        row = {}
        for ax in range(w.shape[1]):
            x = w[:, ax]
            for k, v in statistical_features(x, cfg).items():
                row[f"{k}_{ax + 1}"] = v
            for k, v in spectral_features(x, cfg).items():
                row[f"{k}_{ax + 1}"] = v
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rank_by_mi(X: pd.DataFrame, y: np.ndarray, seed: int) -> pd.Series:
    mi = mutual_info_classif(X.to_numpy(), y, random_state=seed)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def _ordered_groups(meta: pd.DataFrame):
    for (subj, _, _), g in meta.groupby(["subject", "run", "seg"], sort=True):
        yield subj, g.sort_values("pos").index.to_numpy()


def regression_scores(X: pd.DataFrame, meta: pd.DataFrame, cfg: Config):
    feats = list(cfg.regression_feats)
    reg = LinearRegression().fit(X[feats], meta["y2"])
    r = reg.predict(X[feats])
    ref = meta["subject"].eq(cfg.ref_subject).to_numpy()
    if not ref.any():
        ref = meta["y2"].eq(1).to_numpy()
    thr = r[ref].mean() + cfg.thr_k * r[ref].std()
    return r, thr


def estimate_transition_lengths(meta, r, thr, cfg: Config) -> pd.DataFrame:
    y2 = meta["y2"].to_numpy()
    recs = []
    for subj, idx in _ordered_groups(meta):
        y, rr = y2[idx], r[idx]
        for s, e in contiguous_runs(y == 2):
            pre, i = 0, s - 1
            while i >= 0 and pre < cfg.pre_w and y[i] == 1 and rr[i] > thr:
                pre += 1; i -= 1
            post = 0
            for j in range(e, min(e + cfg.post_w, len(y))):
                if y[j] != 1:
                    break
                if rr[j] > thr:
                    post = j - e + 1
            recs.append({"subject": subj, "pre_w": pre, "post_w": post})
    return pd.DataFrame(recs)


def relabel_four_class(meta: pd.DataFrame, pre_w, post_w) -> np.ndarray:
    y2 = meta["y2"].to_numpy()
    y4 = np.where(y2 == 2, FOG, NOFOG)
    get = lambda v, s: v[s] if isinstance(v, dict) else v
    for subj, idx in _ordered_groups(meta):
        P, Q = get(pre_w, subj), get(post_w, subj)
        eps = contiguous_runs(y2[idx] == 2)
        n = len(idx)
        pre_start = []
        for m, (s, _) in enumerate(eps):
            prev_end = eps[m - 1][1] if m else 0
            pre = min(P, s - prev_end)
            y4[idx[s - pre:s]] = PRE
            pre_start.append(s - pre)
        for m, (_, e) in enumerate(eps):
            limit = pre_start[m + 1] if m + 1 < len(eps) else n
            y4[idx[e:e + min(Q, limit - e)]] = POST
    return y4


def smote(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0):
    try:
        from imblearn.over_sampling import SMOTE
        return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    Xs, ys = [X], [y]
    for c, n in zip(classes, counts):
        need = counts.max() - n
        if need == 0 or n < 2:
            continue
        Xc = X[y == c]
        nn = NearestNeighbors(n_neighbors=min(k, n - 1) + 1).fit(Xc)
        neigh = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        base = rng.integers(0, n, need)
        mate = neigh[base, rng.integers(0, neigh.shape[1], need)]
        gap = rng.random((need, 1))                     
        Xs.append(Xc[base] + gap * (Xc[mate] - Xc[base]))
        ys.append(np.full(need, c))
    return np.vstack(Xs), np.concatenate(ys)


def make_rf(cfg: Config) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=cfg.n_estimators, criterion="gini",
                                  bootstrap=True, n_jobs=-1, random_state=cfg.seed)


def _div(a, b):
    return a / b if b else 0.0


def per_class_metrics(y_true, y_pred, labels=(NOFOG, PRE, FOG, POST)) -> pd.DataFrame:
    rows = {}
    for c in labels:
        t, p = y_true == c, y_pred == c
        tp, tn = float(np.sum(t & p)), float(np.sum(~t & ~p))
        fp, fn = float(np.sum(~t & p)), float(np.sum(t & ~p))
        n = tp + tn + fp + fn
        acc = (tp + tn) / n
        prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
        rows[CLASS_NAMES[c]] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": _div(2 * prec * rec, prec + rec), "error_rate": 1 - acc,
            "kappa": _div(acc - pe, 1 - pe), "jaccard": _div(tp, tp + fp + fn),
            "mcc": _div(tp * tn - fp * fn,
                        np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))),
        }
    df = pd.DataFrame(rows).T
    df.loc["Average"] = df.mean()
    return df.round(4)


def weighted_scores(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                 zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p,
            "recall": r, "f1": f}


def paper_protocol(X: pd.DataFrame, y4: np.ndarray, feats: list, cfg: Config):
    Xb, yb = smote(X[feats].to_numpy(), y4, cfg.smote_k, cfg.seed)
    skf = StratifiedKFold(cfg.n_splits, shuffle=True, random_state=cfg.seed)
    folds = []
    for tr, te in skf.split(Xb, yb):
        m = make_rf(cfg).fit(Xb[tr], yb[tr])
        folds.append(weighted_scores(yb[te], m.predict(Xb[te])))
    cv = pd.DataFrame(folds).mean()

    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=cfg.test_size,
                                          stratify=yb, random_state=cfg.seed)
    pred = make_rf(cfg).fit(Xtr, ytr).predict(Xte)
    return cv, per_class_metrics(yte, pred), confusion_matrix(yte, pred)


def subject_independent(X: pd.DataFrame, y4: np.ndarray, groups: np.ndarray,
                        cfg: Config, k_feats: int = 23):
    y_true, y_pred = [], []
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        if not np.any(y4[te] != NOFOG):          
            continue
        feats = rank_by_mi(X[tr], y4[tr], cfg.seed).index[:k_feats].tolist()
        Xb, yb = smote(X.loc[tr, feats].to_numpy(), y4[tr], cfg.smote_k, cfg.seed)
        m = make_rf(cfg).fit(Xb, yb)
        y_true.append(y4[te]); y_pred.append(m.predict(X.loc[te, feats].to_numpy()))
    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
    return weighted_scores(yt, yp), per_class_metrics(yt, yp), confusion_matrix(yt, yp)


def main(root: str, cfg: Config = Config()):
    raw = load_daphnet(root)
    windows, meta = segment_windows(raw, cfg)
    X = extract_features(windows, cfg)
    print(f"windows: {len(X)}  features: {X.shape[1]}  FoG windows: {(meta.y2 == 2).sum()}")

    print("Top 2-class MI features:\n", rank_by_mi(X, meta.y2.to_numpy(), cfg.seed).head(6))

    r, thr = regression_scores(X, meta, cfg)
    lengths = estimate_transition_lengths(meta, r, thr, cfg)
    print("Estimated per-subject mean lengths (s):\n",
          (lengths.groupby("subject").mean() * cfg.win_sec).round(2))
    y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)   # paper final: 4 s / 3 s
    print("4-class distribution:",
          {CLASS_NAMES[c]: int(n) for c, n in zip(*np.unique(y4, return_counts=True))})

    mi4 = rank_by_mi(X, y4, cfg.seed)
    print("Top-23 by 4-class MI (this run):", mi4.index[:23].tolist())

    cv, per_cls, cm = paper_protocol(X, y4, HYBRID_23, cfg)
    print("\n[Paper protocol] 10-fold CV (weighted):\n", cv.round(4))
    print(per_cls); print(cm)

    cv2, per_cls2, cm2 = subject_independent(X, y4, meta.subject.to_numpy(), cfg)
    print("\n[LOSO, leakage-free]:", {k: round(v, 4) for k, v in cv2.items()})
    print(per_cls2); print(cm2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset")
