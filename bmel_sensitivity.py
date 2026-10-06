import os
import sys
import io
import json
import time
import pickle
import platform
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))          
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch
import bmel_dev as bd
import bmel_transfer as bt
import bmel_posthoc as bp
import kaggle_convert as kc
from fog_pipeline import Config, load_daphnet, segment_windows, extract_features, relabel_four_class, FOG, NOFOG
from fog_study import apply_hysteresis, event_metrics, sequences, CUE_ON
from fog_deep import past_index, gather

OUT = "results_bmel_sensitivity"
N_SHIFTS = 2000
warnings.filterwarnings("ignore")


class FastMetrics:
    def __init__(self, meta, cfg, onsets):
        y2 = meta["y2"].to_numpy(); W = cfg.pre_w; self.n = len(meta)
        ep, starts = [], []
        for _, idx in sequences(meta):
            for a, b in bd.contiguous_runs(y2[idx] == 2):
                starts.append(len(ep)); ep.extend(idx[a:b])
        self.ep = np.asarray(ep, int); self.ep_starts = np.asarray(starts, int)
        win = np.full((len(onsets), W + 1), -1, int)
        for i, (idx, s) in enumerate(onsets):
            for j in range(W + 1):
                p = s - W - 1 + j
                if p >= 0: win[i, j] = idx[p]
        self.win = np.where(win < 0, self.n, win)                       
    def __call__(self, on):
        pad = np.append(on, False)
        det = 100 * np.maximum.reduceat(on[self.ep].astype(np.int8), self.ep_starts).mean() if len(self.ep_starts) else np.nan
        w = pad[self.win]; tim = 100 * (w[:, -1] & ~w.all(1)).mean() if len(self.win) else np.nan
        return tim, det


def surrogate_segments(on, meta, cfg, onsets, segs, n_shifts, seed=42, fm=None):
    fm = fm or FastMetrics(meta, cfg, onsets)
    obs_t, obs_d = fm(on)
    ref_t = bd.timely_elig(on, onsets, cfg); ref_d = event_metrics(meta["y2"].to_numpy(), np.where(on, FOG, NOFOG), meta, cfg)["detected_%"]
    assert abs(obs_t - ref_t) < 1e-6 and abs(obs_d - ref_d) < 1e-6, (obs_t, ref_t, obs_d, ref_d)
    ids = np.unique(segs[segs >= 0]); n = len(on)
    grp = np.full(n, -1); pos = np.zeros(n, int); start = np.zeros(len(ids), int); length = np.zeros(len(ids), int)
    for k, g in enumerate(ids):
        w = np.flatnonzero(segs == g); grp[w] = k; pos[w] = np.arange(len(w)); start[k] = w[0]; length[k] = len(w)
        if not np.all(np.diff(w) == 1):                                  
            raise ValueError("segments must be contiguous blocks of windows")
    rng = np.random.default_rng(seed); m = grp >= 0; null = np.empty((n_shifts, 2))
    for i in range(n_shifts):
        k = np.where(length > 1, rng.integers(1, np.maximum(length, 2)), 0)
        src = np.arange(n); src[m] = start[grp[m]] + (pos[m] - k[grp[m]]) % length[grp[m]]
        null[i] = fm(on[src])
    out = {}
    for name, (obs, c) in {"timely": (obs_t, 0), "detection": (obs_d, 1)}.items():
        out[f"{name}_observed"] = obs; out[f"{name}_surrogate_mean"] = null[:, c].mean()
        out[f"{name}_p"] = (1 + (null[:, c] >= obs).sum()) / (1 + n_shifts)
    return out


def recording_segments(meta):
    seg = np.zeros(len(meta), int)
    for k, (_, idx) in enumerate(sequences(meta)):
        seg[idx] = k
    return seg


def p3_fold(probs_lstm, y_on, meta, y2, tr_i, th_default, cfg):
    tr = np.zeros(len(y_on), bool); tr[tr_i] = True
    th_k = bp.grid_select(probs_lstm, y_on, meta, tr, bd.eligible_onsets(y2, meta, cfg, tr), cfg) or th_default
    return tr_i, th_k


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",), n_estimators=100)
    fz = json.load(open("frozen_config.json"))
    th = {k: (v["theta_on"], v["theta_off"]) for k, v in fz["operating_points"].items()}
    M = pickle.load(open(os.path.join(bt.OUT, "T_models_lab.pkl"), "rb"))
    s1, s2 = [], []
    for name, (path, merge) in bt.TESTS.items():
        X, meta, y4, probs = pickle.load(open(os.path.join(bt.OUT, f"T_probs_{name}.pkl"), "rb"))
        y2 = meta["y2"].to_numpy(); y_on = np.isin(y4, CUE_ON).astype(int); allm = np.ones(len(y4), bool)
        onsets = bd.eligible_onsets(y2, meta, cfg); rec = recording_segments(meta); g = meta.subject.to_numpy()
        runs = {}
        for ctrl in ["LSTM", "RF"]:
            runs[f"Transfer, {ctrl}"] = apply_hysteresis(probs[ctrl], meta, allm, *th[ctrl])
        ref = np.load(os.path.join(bp.OUT, f"ref_lstm_{name}.npy"))
        runs["P1 within-dataset LSTM"] = apply_hysteresis(ref, meta, allm, *th["LSTM"])
        t0 = time.time(); print(f"  {name}: P3 threshold re-selection on {os.cpu_count()} cores ...", flush=True)
        folds = list(GroupKFold(min(10, len(np.unique(g)))).split(X, groups=g))
        try:
            res = Parallel(n_jobs=-1)(delayed(p3_fold)(probs["LSTM"], y_on, meta, y2, tr_i, th["LSTM"], cfg) for tr_i, _ in folds)
        except Exception as e:                                      
            print(f"    parallel run failed ({type(e).__name__}); continuing on one core", flush=True)
            res = [p3_fold(probs["LSTM"], y_on, meta, y2, tr_i, th["LSTM"], cfg) for tr_i, _ in folds]
        on3 = np.zeros(len(y4), bool)
        for tr_i, th_k in res:
            te = np.ones(len(y4), bool); te[tr_i] = False
            on3[te] = apply_hysteresis(probs["LSTM"], meta, te, *th_k)[te]
        runs["P3 recalibrated LSTM"] = on3; print(f"    done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        fm = FastMetrics(meta, cfg, onsets)
        for an, on in runs.items():
            t0 = time.time()
            s1.append({"dataset": name, "analysis": an, "shifts": N_SHIFTS, **surrogate_segments(on, meta, cfg, onsets, rec, N_SHIFTS, fm=fm)})
            print(f"    {an}: {N_SHIFTS} shifts ({time.time() - t0:.0f} s)", flush=True)
        if name == "defog":                                           
            mp = pd.read_csv(bp.MAPPING); mp = mp[mp.folder == "defog_back"]; ids = dict(zip(mp["S"], mp["Id"]))
            tasks = pd.read_csv(bp.TASKS); tt = bp.window_times(path, meta)
            key = np.array([ids.get(f"S{int(s):02d}R{int(r):02d}") for s, r in zip(meta.subject, meta.run)], dtype=object)
            task_id = np.full(len(meta), -1); k = 0
            for Id, grp in tasks.groupby("Id"):
                idx = np.flatnonzero(key == Id)
                for _, r in grp.iterrows():
                    sel = idx[(tt[idx] >= r.Begin) & (tt[idx] < r.End)]
                    if len(sel): task_id[sel] = k; k += 1
            seg = np.where(task_id >= 0, task_id * 100000 + rec, -1)         
            _, seg = np.unique(seg, return_inverse=True); seg = np.where(task_id >= 0, seg, -1)
            for ctrl in ["LSTM", "RF"]:
                s2.append({"dataset": name, "analysis": f"Transfer, {ctrl}", "null": "task-preserving shift",
                           "windows_in_tasks_%": 100 * (task_id >= 0).mean(),
                           **surrogate_segments(runs[f"Transfer, {ctrl}"], meta, cfg, onsets, seg, N_SHIFTS, fm=fm)})
                print(f"    task-preserving surrogate, {ctrl}: done", flush=True)
        print(f"  {name}: done")
    path, _ = bt.TESTS["FoG-STAR"]; raw = load_daphnet(path); raw[["tr_x", "tr_y"]] = raw[["tr_y", "tr_x"]].to_numpy()
    for _, idx in raw.groupby(["subject", "run"]).groups.items():
        raw.loc[idx, "label"] = kc.merge_gaps(raw.loc[idx, "label"].to_numpy().astype(int))
    w, meta = segment_windows(raw, cfg); X = extract_features(w, cfg); F = np.nan_to_num(X.to_numpy(np.float64))
    p5 = bt.predict(M, X, F, past_index(meta, 8))["LSTM"]; on5 = apply_hysteresis(p5, meta, np.ones(len(meta), bool), *th["LSTM"])
    s1.append({"dataset": "FoG-STAR", "analysis": "P5 axis-aligned LSTM", "shifts": N_SHIFTS,
               **surrogate_segments(on5, meta, cfg, bd.eligible_onsets(meta["y2"].to_numpy(), meta, cfg), recording_segments(meta), N_SHIFTS)})
    try:
        rf = M["rf"].m if hasattr(M["rf"], "m") else M["rf"]
        buf = io.BytesIO(); torch.save(M["lstm"].state_dict(), buf)
        win = w[:1]; feats = M["feats"]; mu, sd = M["mu"], M["sd"]
        seq = np.zeros((1, 8, len(mu)), np.float32); lstm = M["lstm"].cpu().eval()
        t_feat, t_lstm, t_rf = [], [], []
        for _ in range(1000):
            t0 = time.perf_counter(); x = extract_features(win, cfg); t1 = time.perf_counter()
            z = np.clip((np.nan_to_num(x.to_numpy(np.float64)) - mu) / sd, -10, 10).astype(np.float32)
            seq[0, :-1] = seq[0, 1:]; seq[0, -1] = z
            with torch.no_grad(): lstm(torch.from_numpy(seq))
            t2 = time.perf_counter(); rf.predict_proba(x[feats]); t3 = time.perf_counter()
            t_feat.append(t1 - t0); t_lstm.append(t2 - t1); t_rf.append(t3 - t2)
        ms = lambda a: (1000 * np.mean(a), 1000 * np.std(a))
        cpu = platform.processor() or platform.machine()
        S3 = pd.DataFrame([
            {"model": "LSTM", "parameters": sum(p.numel() for p in lstm.parameters()), "tree_nodes": np.nan,
             "serialised_size_MB": len(buf.getvalue()) / 1e6, "feature_ms_mean_sd": "%.2f (%.2f)" % ms(t_feat),
             "inference_ms_mean_sd": "%.2f (%.2f)" % ms(t_lstm), "end_to_end_ms": 1000 * (np.mean(t_feat) + np.mean(t_lstm))},
            {"model": "RF", "parameters": np.nan, "tree_nodes": sum(t.tree_.node_count for t in rf.estimators_),
             "serialised_size_MB": len(pickle.dumps(rf)) / 1e6, "feature_ms_mean_sd": "%.2f (%.2f)" % ms(t_feat),
             "inference_ms_mean_sd": "%.2f (%.2f)" % ms(t_rf), "end_to_end_ms": 1000 * (np.mean(t_feat) + np.mean(t_rf))}])
        S3["processor"] = cpu; S3["threads"] = torch.get_num_threads(); S3["repetitions"] = 1000
    except Exception as e:
        print('S3 skipped:', e); S3 = pd.DataFrame()
    A, B = pd.DataFrame(s1), pd.DataFrame(s2)
    A.round(4).to_csv(f"{OUT}/S1_surrogate_2000.csv", index=False); B.round(4).to_csv(f"{OUT}/S2_task_preserving.csv", index=False)
    S3.round(3).to_csv(f"{OUT}/S3_deployment_cost.csv", index=False)
    print("\nS1 - 2,000 circular shifts\n", A.round(4).to_string(index=False))
    print("\nS2 - task-preserving surrogate (defog)\n", B.round(4).to_string(index=False))
    print("\nS3 - deployment cost\n", S3.round(3).to_string(index=False))
    print(f"\nDone. Files in {OUT}/ (post hoc sensitivity analyses)")


if __name__ == "__main__":
    main()