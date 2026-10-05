import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = r"D:\New research fog work\kaggle_fog\Competition Dataset"
OUT = r"D:\New research fog work\kaggle_fog\daphnet_format"     
DRY_RUN = False
INCLUDE_NOTYPE = True          
FS_OUT, PRE_SEC = 64, 4.0
MERGE_GAP_S = 2.0              
SPEC = {"tdcsfog": dict(fs=128, up=1, down=2, scale=1000.0 / 9.80665, meta="tdcsfog_metadata.csv"),
        "defog":   dict(fs=100, up=16, down=25, scale=1000.0, meta="defog_metadata.csv"),
        "notype":  dict(fs=100, up=16, down=25, scale=1000.0, meta="defog_metadata.csv")}


def runs_of(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def convert_file(path, kind):
    sp = SPEC[kind]
    d = pd.read_csv(path)
    if kind == "notype":
        fog = d["Event"].to_numpy() > 0
    else:
        fog = d[["StartHesitation", "Turn", "Walking"]].sum(axis=1).to_numpy() > 0
    valid = np.ones(len(d), bool)
    if kind in ("defog", "notype"):
        valid = d["Valid"].astype(bool).to_numpy() & d["Task"].astype(bool).to_numpy()
    acc = d[["AccAP", "AccV", "AccML"]].to_numpy(float) * sp["scale"]           
    n_in = len(d)
    if n_in < 2 * sp["fs"]:
        return None
    rs = resample_poly(acc, sp["up"], sp["down"], axis=0)
    n = len(rs)
    src = np.minimum(np.round(np.arange(n) * sp["fs"] / FS_OUT).astype(int), n_in - 1)
    lab = np.where(fog[src], 2, 1)
    lab[~valid[src]] = 0
    if MERGE_GAP_S:
        lab = merge_gaps(lab)
    return rs, lab


def merge_gaps(lab, gap_s=MERGE_GAP_S):
    lab = lab.copy()
    for a, b in runs_of(lab != 0):
        seg = lab[a:b]
        eps = runs_of(seg == 2)
        for (s1, e1), (s2, e2) in zip(eps[:-1], eps[1:]):
            if (s2 - e1) / FS_OUT < gap_s:
                seg[e1:s2] = 2
        lab[a:b] = seg
    return lab


def pre_onset(lab):
    out = []
    for a, b in runs_of(lab != 0):
        seg = lab[a:b]
        for s, _ in runs_of(seg == 2):
            prev = np.flatnonzero(seg[:s] == 2)
            start = prev[-1] + 1 if len(prev) else 0
            out.append((s - start) / FS_OUT)
    return out


def main():
    groups = {"tdcsfog_back": ["tdcsfog"], "defog_back": ["defog"] + (["notype"] if INCLUDE_NOTYPE else [])}
    rows, pre, mapping = [], [], []
    for folder, kinds in groups.items():
        out_dir = os.path.join(OUT, folder)
        if not DRY_RUN:
            os.makedirs(out_dir, exist_ok=True)
        meta = pd.read_csv(os.path.join(ROOT, SPEC[kinds[0]]["meta"]))
        files = [(k, f) for k in kinds for f in sorted(glob.glob(os.path.join(ROOT, "train", k, "*.csv")))]
        ids = {os.path.basename(f)[:-4]: (k, f) for k, f in files}
        meta = meta[meta["Id"].isin(ids)].copy()
        subj_no = {s: i + 1 for i, s in enumerate(sorted(meta["Subject"].unique()))}
        if len(subj_no) > 99:
            raise ValueError("more than 99 subjects - file naming S??R?? would overflow")
        run_no = {}
        for _, m in meta.sort_values(["Subject", "Visit", "Id"]).iterrows():
            kind, path = ids[m["Id"]]
            res = convert_file(path, kind)
            if res is None:
                continue
            rs, lab = res
            s = subj_no[m["Subject"]]
            run_no[s] = run_no.get(s, 0) + 1
            r = run_no[s]
            if r > 99:
                raise ValueError("more than 99 runs for one subject")
            eps = len(runs_of(lab == 2))
            p = pre_onset(lab)
            pre += [(folder, s, x) for x in p]
            rows.append(dict(folder=folder, subject=s, run=r, minutes=len(lab) / FS_OUT / 60,
                             excluded_pct=100 * (lab == 0).mean(), fog_pct=100 * (lab == 2).mean(),
                             episodes=eps, medication=m.get("Medication"), kind=kind))
            mapping.append(dict(folder=folder, S=f"S{s:02d}R{r:02d}", Id=m["Id"], Subject=m["Subject"],
                                Visit=m.get("Visit"), Medication=m.get("Medication"), Test=m.get("Test"),
                                kind=kind))
            if not DRY_RUN:
                t = np.arange(len(lab)) * 1000.0 / FS_OUT
                z = np.zeros((len(lab), 3))
                out = np.column_stack([t, z, z, rs, lab])                  
                np.savetxt(os.path.join(out_dir, f"S{s:02d}R{r:02d}.txt"), out, fmt="%.6g")
    R, P = pd.DataFrame(rows), pd.DataFrame(pre, columns=["folder", "subject", "pre_sec"])
    pd.set_option("display.width", 200)
    for folder, g in R.groupby("folder"):
        per = g.groupby("subject").agg(runs=("run", "count"), minutes=("minutes", "sum"), episodes=("episodes", "sum"))
        pp = P[P.folder == folder]["pre_sec"]
        print(f"\n=== {folder} ===")
        print(f"subjects {per.shape[0]} (with FoG {(per.episodes > 0).sum()}), runs {len(g)}, "
              f"hours {g.minutes.sum() / 60:.2f}, episodes {g.episodes.sum()}, "
              f"FoG {np.average(g.fog_pct, weights=g.minutes):.1f}% of samples, "
              f"excluded {np.average(g.excluded_pct, weights=g.minutes):.1f}%")
        print(f"median run length {g.minutes.median() * 60:.0f} s; medication {g.medication.value_counts().to_dict()}")
        if len(pp):
            print(f"episodes with >= {PRE_SEC:.0f} s of FoG-free recording before onset: "
                  f"{100 * (pp >= PRE_SEC).mean():.1f}%  (median {pp.median():.1f} s)")
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "kaggle_runs_summary.csv"), index=False)
    pd.DataFrame(mapping).to_csv(os.path.join(OUT, "kaggle_subject_mapping.csv"), index=False)
    print("\nDRY RUN - no signal files written." if DRY_RUN else f"\nFiles written to {OUT}")
    print("Summary saved: kaggle_runs_summary.csv, kaggle_subject_mapping.csv")


if __name__ == "__main__":
    main()