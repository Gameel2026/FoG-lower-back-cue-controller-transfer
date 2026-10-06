import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import torch
import torch.nn as nn

import fog_improve as fi
import fog_floor_sensitivity as fs
from fog_pipeline import relabel_four_class
from fog_study import apply_hysteresis, CUE_ON
from fog_deep import FoGLSTM, past_index, gather, set_seed, DEVICE

OUT = "results_lstm_controller"
QUICK = False
FLOOR = 0.70
MAX_EPOCHS, PATIENCE, BATCH = 40, 6, 256
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")


def standardise(F, idx, P8):
    mu = F[idx].mean(0); sd = F[idx].std(0); sd = np.where(sd > 0, sd, 1.0)
    return gather(np.clip((F - mu) / sd, -10, 10).astype(np.float32), P8)         


def train(Z, y, idx, epochs, seed, val_idx=None):
    set_seed(seed); model = FoGLSTM(Z.shape[2], n_cls=2).to(DEVICE)
    cnt = np.bincount(y[idx], minlength=2).astype(float); w = len(idx) / (2 * np.maximum(cnt, 1))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(Z[idx]); yt = torch.from_numpy(y[idx].astype(np.int64))
    best, best_ep, wait = -1.0, 1, 0
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(len(yt))
        for k in range(0, len(yt), BATCH):
            b = perm[k:k + BATCH]; xb, yb = Xt[b].to(DEVICE), yt[b].to(DEVICE)
            opt.zero_grad(); loss = lossf(model(xb + 0.05 * torch.randn_like(xb)), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if val_idx is not None:
            bacc = balanced_accuracy_score(y[val_idx], (prob(model, Z[val_idx]) >= 0.5).astype(int))
            if bacc > best: best, best_ep, wait = bacc, ep, 0
            else:
                wait += 1
                if wait >= PATIENCE: break
    return model, best_ep, best


@torch.no_grad()
def prob(model, Zs):
    model.eval(); out = []
    for k in range(0, len(Zs), 1024):
        out.append(torch.softmax(model(torch.from_numpy(Zs[k:k + 1024]).to(DEVICE)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def lstm_scores(F, P8, y_on, groups, tr, te, seed, epochs):
    tr_idx = np.flatnonzero(tr)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(tr_idx, groups=groups[tr_idx]))
    Z = standardise(F, tr_idx[a], P8)
    _, n_ep, vb = train(Z, y_on, tr_idx[a], epochs, seed, val_idx=tr_idx[b])          
    oof = np.zeros(len(y_on))
    for ia, ib in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):                  
        Z = standardise(F, tr_idx[ia], P8)
        m, _, _ = train(Z, y_on, tr_idx[ia], n_ep, seed); oof[tr_idx[ib]] = prob(m, Z[tr_idx[ib]])
    Z = standardise(F, tr_idx, P8)
    m, _, _ = train(Z, y_on, tr_idx, n_ep, seed)                                    
    score = np.zeros(len(y_on)); score[te] = prob(m, Z[te])
    return oof, score, n_ep, vb


def run_dataset(ds, cfg):
    fi.SRC = fs.find_src(ds)
    X, meta = fi.load("Shank"); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    y_on = np.isin(y4, CUE_ON).astype(int); groups = meta["subject"].to_numpy()
    F = np.nan_to_num(X.to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0); P8 = past_index(meta, 8)
    grid = [(round(a, 2), round(b, 2)) for a in np.arange(0.30, 0.91, 0.05) for b in np.arange(0.10, a + 1e-9, 0.10)]
    on = np.zeros(len(y4), bool); choices = []
    subjects = np.unique(groups)[:3] if QUICK else np.unique(groups)   
    for s in subjects:
        t0 = time.time(); tr, te = groups != s, groups == s
        oof, score, n_ep, vb = lstm_scores(F, P8, y_on, groups, tr, te, cfg.seed, 3 if QUICK else MAX_EPOCHS)
        th, feasible, changed = fs.select(oof, y_on, meta, tr, grid, FLOOR)
        on[te] = apply_hysteresis(score, meta, te, *th)[te]
        choices.append({"dataset": ds, "subject": s, "epochs": n_ep, "val_balanced_acc": vb, "theta_on": th[0],
                        "theta_off": th[1], "floor_changed_F1_choice": changed, "floor_feasible": feasible})
        print(f"  [{ds}] subject {s:02d}: epochs {n_ep}, thresholds ({th[0]:.2f}, {th[1]:.2f})  ({time.time() - t0:.0f} s)")
    fs.FLOORS = [FLOOR]; fs.QUICK = QUICK
    y4_rf, meta_rf, on_rf, _ = fs.run_dataset(ds, cfg)
    if QUICK:
        keep = meta["subject"].isin(subjects).to_numpy()
        return y4[keep], meta[keep].reset_index(drop=True), on[keep], on_rf[FLOOR], pd.DataFrame(choices)
    return y4, meta, on, on_rf[FLOOR], pd.DataFrame(choices)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    cfg = fi.cfg_(); summ, sur, per, tests, chs = [], [], [], [], []
    print(f"Device: {DEVICE}")
    for ds in fs.DATASETS:
        print(f"\n=== {ds} ===")
        y4, meta, on_lstm, on_rf, ch = run_dataset(ds, cfg); chs.append(ch)
        table = []
        for name, on in [("RF controller (primary)", on_rf), ("LSTM controller (secondary)", on_lstm)]:
            summ.append({"dataset": ds, "controller": name, **fi.summarise(y4, on, meta, cfg)})
            sg = fi.surrogate(y4, on, meta, cfg)
            for k in ["timely_pre_onset_%", "detected_%", "predicted_before_onset_%"]:
                sur.append({"dataset": ds, "controller": name, "metric": k, **sg.loc[k].to_dict()})
            table.append(fi.per_subject(name, y4, on, meta, cfg))
        table = pd.concat(table, ignore_index=True)
        t = fi.tests(table, [("LSTM controller (secondary)", "RF controller (primary)")]); t.insert(0, "dataset", ds)
        tests.append(t); table.insert(0, "dataset", ds); per.append(table)
    S = pd.DataFrame(summ); S.round(4).to_csv(f"{OUT}/LC1_summary.csv", index=False)
    G = pd.DataFrame(sur); G.round(4).to_csv(f"{OUT}/LC2_surrogate.csv", index=False)
    pd.concat(per).round(4).to_csv(f"{OUT}/LC3_per_subject.csv", index=False)
    T = pd.concat(tests).round(4); T.to_csv(f"{OUT}/LC4_tests.csv", index=False)
    pd.concat(chs).round(4).to_csv(f"{OUT}/LC5_choices.csv", index=False)
    cols = ["dataset", "controller", "cue_specificity", "balanced_acc", "timely_pre_onset_%", "predicted_before_onset_%",
            "detected_%", "median_lead_time_s", "false_alarms_per_hour", "median_cue_off_delay_s"]
    print("\nSummary\n", S[cols].round(3).to_string(index=False))
    print("\nSurrogate test\n", G.round(3).to_string(index=False))
    print("\nPer-patient tests, LSTM vs RF controller (Holm within dataset)\n", T.to_string(index=False))
    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()
