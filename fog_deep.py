import os
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn

from fog_pipeline import Config, load_daphnet, segment_windows, relabel_four_class, rank_by_mi, smote, make_rf
from fog_study import sequences
from fog_stats import holm
from paths import DAPHNET_ROOT, LI_CONVERTED

OUT = "results_deep"
QUICK = False
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
MAX_EPOCHS, PATIENCE, BATCH = 40, 6, 256
LABELS = ["No-FoG", "Pre-FoG", "FoG", "Post-FoG"]
MODELS = {"rf": ("A", "RF (features, 0.5 s)"),
          "cnn05": ("A", "CNN (raw, 0.5 s)"),
          "lstm05": ("A", "LSTM (raw, 0.5 s)"),
          "cnn2": ("B", "CNN (raw, 2 s)"),
          "lstm4": ("B", "LSTM (features, 4 s)")}
COMPARISONS = [("cnn05", "rf", "A: model effect"), ("lstm05", "rf", "A: model effect"),
               ("cnn2", "cnn05", "B: context effect (CNN)"), ("lstm4", "rf", "B: context effect (features)")]
DATASETS = {"DAPHNET": (DAPHNET_ROOT, ["results_causal", os.path.join("New Results", "results_causal")]),
            "Multimodal FoG": (LI_CONVERTED + "_shank", ["results_li_causal", os.path.join("New Results", "results_li_causal")])}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def past_index(meta, n):
    P = np.full((len(meta), n), -1, dtype=np.int64)
    for _, idx in sequences(meta):
        for j, i in enumerate(idx):
            for k in range(n):
                src = j - (n - 1) + k
                if src >= 0: P[i, k] = idx[src]
    return P


def gather(A, P):
    out = A[np.maximum(P, 0)]
    out[P < 0] = 0
    return out


def load_dataset(ds, cfg):
    root, cands = DATASETS[ds]
    cache = next((os.path.join(d, "cache_Shank.pkl") for d in cands if os.path.exists(os.path.join(d, "cache_Shank.pkl"))), None)
    if cache is None: raise FileNotFoundError(f"cache_Shank.pkl for {ds} not found in {cands}")
    _, X, meta_c, _ = pickle.load(open(cache, "rb"))
    windows, meta = segment_windows(load_daphnet(root), cfg)
    key = ["subject", "run", "seg", "pos", "y2"]
    if len(meta) != len(meta_c) or not (meta[key].to_numpy() == meta_c[key].to_numpy()).all():
        raise RuntimeError(f"{ds}: raw windows do not match the cached features - check the data path in paths.py")
    y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    w = windows.astype(np.float32)                                            
    raw2 = gather(w, past_index(meta, 4)).reshape(len(meta), 4 * w.shape[1], w.shape[2])   
    F = np.nan_to_num(X.to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)       
    return {"raw05": w, "raw2": raw2}, F, X, meta, y4, past_index(meta, 8)


class FoGCNN(nn.Module):
    channels_first = True

    def __init__(self, n_in=3, n_cls=4):
        super().__init__()
        def block(i, o, k, pool):
            layers = [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]
            if pool: layers.append(nn.MaxPool1d(2))
            return layers
        self.features = nn.Sequential(*block(n_in, 32, 7, True), *block(32, 64, 5, True), *block(64, 64, 3, False))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):                   
        return self.head(self.features(x))


class FoGLSTM(nn.Module):
    channels_first = False

    def __init__(self, n_in, n_cls=4, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, n_cls))

    def forward(self, x):                  
        out, _ = self.lstm(x)
        return self.head(out[:, -1])        


def to_tensor(x, channels_first):
    x = x.transpose(0, 2, 1) if channels_first else x
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def augment(xb, raw_signal, channels_first):
    if raw_signal:                          
        shape = (xb.shape[0], xb.shape[1], 1) if channels_first else (xb.shape[0], 1, xb.shape[2])
        return xb * (1 + 0.1 * torch.randn(shape, device=xb.device)) + 0.02 * torch.randn_like(xb)
    return xb + 0.05 * torch.randn_like(xb) 


@torch.no_grad()
def predict(model, X):
    model.eval(); Xt = to_tensor(X, model.channels_first); out = []
    for k in range(0, len(Xt), 1024):
        out.append(torch.softmax(model(Xt[k:k + 1024].to(DEVICE)), 1).cpu().numpy())
    return np.concatenate(out)


def fit(model, Xtr, ytr, epochs, raw_signal, Xva=None, yva=None, seed=0):
    set_seed(seed)
    counts = np.bincount(ytr, minlength=4).astype(float); w = len(ytr) / (4 * np.maximum(counts, 1))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    cf = model.channels_first; Xt, yt = to_tensor(Xtr, cf), torch.from_numpy(ytr.astype(np.int64))
    best_bacc, best_ep, wait, log = -1.0, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(len(yt)); tot = 0.0
        for k in range(0, len(yt), BATCH):
            b = perm[k:k + BATCH]; xb, yb = Xt[b].to(DEVICE), yt[b].to(DEVICE)
            opt.zero_grad(); loss = lossf(model(augment(xb, raw_signal, cf)), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += loss.item() * len(b)
        rec = {"epoch": ep, "train_loss": tot / len(yt)}
        if Xva is not None:
            bacc = balanced_accuracy_score(yva, predict(model, Xva).argmax(1)); rec["val_balanced_acc"] = bacc
            sched.step(bacc)
            if bacc > best_bacc: best_bacc, best_ep, wait = bacc, ep, 0
            else: wait += 1
        log.append(rec)
        if Xva is not None and wait >= PATIENCE: break
    return best_bacc, best_ep, log


def train_eval(make, standardise, raw_signal, y4, tr, te, groups, seed, epochs):
    tr_idx = np.flatnonzero(tr)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(tr_idx, groups=groups[tr_idx]))
    itr, iva = tr_idx[a], tr_idx[b]
    Xs = standardise(itr)
    set_seed(seed); m = make().to(DEVICE)
    bacc, best_ep, log = fit(m, Xs[itr], y4[itr], epochs, raw_signal, Xs[iva], y4[iva], seed)
    Xs = standardise(tr_idx)
    set_seed(seed); m = make().to(DEVICE)
    fit(m, Xs[tr_idx], y4[tr_idx], max(best_ep, 1), raw_signal, seed=seed)
    return predict(m, Xs[te]).argmax(1), bacc, best_ep, log


def run(ds, cfg):
    raw, F, X, meta, y4, P8 = load_dataset(ds, cfg)
    groups = meta["subject"].to_numpy(); subjects = np.unique(groups)[:2] if QUICK else np.unique(groups)
    epochs = 3 if QUICK else MAX_EPOCHS
    pred = {k: np.full(len(y4), -1) for k in MODELS}; logs = []

    def std_raw(name):                     
        A = raw[name]
        def f(idx):
            mu = A[idx].astype(np.float64).mean(axis=(0, 1), keepdims=True)
            sd = A[idx].astype(np.float64).std(axis=(0, 1), keepdims=True) + 1e-6
            return ((A - mu) / sd).astype(np.float32)
        return f

    def std_seq(idx):                       
        mu = F[idx].mean(0); sd = F[idx].std(0)                         
        sd = np.where(sd > 0, sd, 1.0)
        Z = np.clip((F - mu) / sd, -10, 10)                            
        return gather(Z.astype(np.float32), P8)                        

    nets = {"cnn05": (lambda: FoGCNN(3), std_raw("raw05"), True),
            "lstm05": (lambda: FoGLSTM(3), std_raw("raw05"), True),
            "cnn2": (lambda: FoGCNN(3), std_raw("raw2"), True),
            "lstm4": (lambda: FoGLSTM(F.shape[1]), std_seq, False)}
    for s in subjects:
        t0 = time.time(); tr, te = groups != s, groups == s; info = []
        for k, (make, stdz, is_raw) in nets.items():
            p, bacc, ep, lg = train_eval(make, stdz, is_raw, y4, tr, te, groups, cfg.seed, epochs)
            pred[k][te] = p; info.append(f"{k} ep {ep} (val {bacc:.2f})")
            for r in lg: logs.append({"dataset": ds, "subject": s, "model": k, **r})
        feats = rank_by_mi(X[tr], y4[tr], cfg.seed).index[:23].tolist()              
        Xb, yb = smote(X.loc[tr, feats].to_numpy(), y4[tr], cfg.smote_k, cfg.seed)
        pred["rf"][te] = make_rf(cfg).fit(Xb, yb).predict(X.loc[te, feats].to_numpy())
        print(f"  [{ds}] subject {s:02d}: " + ", ".join(info) + f"  ({time.time() - t0:.0f} s)")
    keep = pred["rf"] >= 0
    return y4[keep], {k: v[keep] for k, v in pred.items()}, meta[keep].reset_index(drop=True), pd.DataFrame(logs)


def scores(y, p):
    rec = recall_score(y, p, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {"accuracy": accuracy_score(y, p), "balanced_acc": balanced_accuracy_score(y, p),
            "macro_F1": f1_score(y, p, labels=[0, 1, 2, 3], average="macro", zero_division=0),
            **{f"recall_{LABELS[k]}": rec[k] for k in range(4)}, "transition_recall": (rec[1] + rec[3]) / 2}


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    cfg = Config(causal_filter=True, sensors=("ank",), n_estimators=30 if QUICK else 100)
    print(f"Device: {DEVICE}")
    summ, per, tests, logs = [], [], [], []
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        y, pr, meta, log = run(ds, cfg); logs.append(log)
        for k, (part, name) in MODELS.items():
            summ.append({"dataset": ds, "part": part, "model": name, **scores(y, pr[k])})
            pd.DataFrame(confusion_matrix(y, pr[k], labels=[0, 1, 2, 3]), index=LABELS, columns=LABELS) \
              .to_csv(f"{OUT}/DL4_confusion_{ds.replace(' ', '_')}_{k}.csv")
        rows = []
        for s in sorted(meta["subject"].unique()):
            m = meta["subject"].eq(s).to_numpy()
            if not np.isin([1, 3], y[m]).all(): continue           
            row = {"dataset": ds, "subject": s}
            for k in MODELS: row.update({f"{met}_{k}": v for met, v in scores(y[m], pr[k][m]).items()})
            rows.append(row)
        t = pd.DataFrame(rows); per.append(t)
        for a, b, question in COMPARISONS:
            fam = []
            for met in ["transition_recall", "balanced_acc", "macro_F1"]:
                d = t[f"{met}_{a}"] - t[f"{met}_{b}"]
                try: p = wilcoxon(t[f"{met}_{a}"], t[f"{met}_{b}"]).pvalue
                except ValueError: p = np.nan
                fam.append({"dataset": ds, "question": question, "comparison": f"{MODELS[a][1]} vs {MODELS[b][1]}",
                            "metric": met, "n": len(t), "median_first": t[f"{met}_{a}"].median(),
                            "median_second": t[f"{met}_{b}"].median(), "first_better": f"{int((d > 0).sum())}/{len(d)}", "p_raw": p})
            fam = pd.DataFrame(fam); fam["p_holm"] = holm(fam["p_raw"].to_numpy()); tests.append(fam)
    S = pd.DataFrame(summ).round(4); S.to_csv(f"{OUT}/DL1_summary.csv", index=False)
    pd.concat(per).round(4).to_csv(f"{OUT}/DL2_per_subject.csv", index=False)
    T = pd.concat(tests).round(4); T.to_csv(f"{OUT}/DL3_tests.csv", index=False)
    pd.concat(logs).round(4).to_csv(f"{OUT}/DL_training_log.csv", index=False)
    try:
        figure(S)
    except Exception as e:
        print(f"Figure could not be saved ({e}); the CSV results are complete.")
    cols = ["dataset", "part", "model", "accuracy", "balanced_acc", "macro_F1", "recall_No-FoG", "recall_Pre-FoG",
            "recall_FoG", "recall_Post-FoG", "transition_recall"]
    print("\nSummary (pooled over held-out patients)\n", S[cols].to_string(index=False))
    print("\nPer-patient tests (Holm within each comparison)\n", T.to_string(index=False))
    print(f"\nDone. Files in {OUT}/")


def figure(S):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    colors = ["#7f7f7f", "#1f4e79", "#5b9bd5", "#c0392b", "#e67e22"]; hatches = ["", "", "", "//", "//"]
    for ax, (lab, ds) in zip(axes, [("a", "DAPHNET"), ("b", "Multimodal FoG")]):
        d = S[S.dataset == ds].set_index("model"); x = np.arange(4); wbar = 0.16
        for i, (k, (part, name)) in enumerate(MODELS.items()):
            vals = [100 * d.loc[name, f"recall_{c}"] for c in LABELS]
            ax.bar(x + (i - 2) * wbar, vals, wbar, color=colors[i], hatch=hatches[i], edgecolor="black", lw=0.4, label=name)
        ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylim(0, 100); ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax.set_title(f"({lab}) {ds}", loc="left", fontsize=8)
        if lab == "a": ax.set_ylabel("Recall (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{OUT}/DL_recall.png", dpi=300, bbox_inches="tight")
    try:
        fig.savefig(f"{OUT}/DL_recall.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    except Exception:
        try: fig.savefig(f"{OUT}/DL_recall.tif", dpi=600, bbox_inches="tight")
        except Exception: print("TIFF not supported here; PNG saved.")
    plt.close(fig)


if __name__ == "__main__":
    main()
