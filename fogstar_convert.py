import os
import re
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

CSV = r"D:\New research fog work\FoG STAR\sensor_data.csv"          
FOGSTAR_OUT = r"D:\FoG-STAR\daphnet_format"     
DRY_RUN = False
FS_IN, FS_OUT, UP, DOWN = 60, 64, 16, 15
GUARD = 10                                     
COLMAP = {}                                   


def find(cols, *must, exclude=()):
    for c in cols:
        lc = c.lower()
        if all(m in lc for m in must) and not any(e in lc for e in exclude):
            return c
    return None


def mapping(cols):
    m = {"subject": find(cols, "subject"), "session": find(cols, "session"), "task": find(cols, "task"),
         "time": find(cols, "time"), "fog": find(cols, "fog", exclude=("sever",))}
    for loc, keys in {"l_ankle": [("l", "ankle"), ("left", "ankle")], "r_ankle": [("r", "ankle"), ("right", "ankle")],
                      "back": [("back",), ("lumb",)]}.items():
        for ax in "xyz":
            col = None
            for k in keys:
                cand = [c for c in cols if all(x in c.lower() for x in k) and "acc" in c.lower()
                        and re.search(rf"(^|[_\s.-]){ax}($|[_\s.-])", c.lower())]
                if loc == "l_ankle": cand = [c for c in cand if not re.search(r"(^|[_\s.-])r(ight)?[_\s.-]", c.lower())]
                if loc == "r_ankle": cand = [c for c in cand if not re.search(r"(^|[_\s.-])l(eft)?[_\s.-]", c.lower())]
                if cand: col = cand[0]; break
            m[f"{loc}_{ax}"] = col
    m.update(COLMAP)
    return m


def episodes(lab):
    d = np.diff(np.r_[0, (lab == 2).astype(int), 0]); return int((d == 1).sum())


def main():
    df = pd.read_csv(CSV)
    cols = list(df.columns); m = mapping(cols)
    print("Columns in file:\n ", cols)
    print("\nColumn mapping used:"); [print(f"  {k:12s} -> {v}") for k, v in m.items()]
    missing = [k for k, v in m.items() if v is None and k != "time"]
    if missing:
        print(f"\nNOT FOUND: {missing}. Fill COLMAP at the top of the script and run again."); return
    acc = {loc: df[[m[f"{loc}_{a}"] for a in "xyz"]].to_numpy(float) * 1000.0 for loc in ("l_ankle", "r_ankle", "back")}   # g -> mg
    fog = (pd.to_numeric(df[m["fog"]], errors="coerce").fillna(0).to_numpy() > 0)
    subj = df[m["subject"]].to_numpy()
    # ankle side per subject: fewer missing samples (label-independent rule)
    side = {}
    for s in np.unique(subj):
        k = subj == s
        nl, nr = np.isnan(acc["l_ankle"][k]).any(1).mean(), np.isnan(acc["r_ankle"][k]).any(1).mean()
        side[s] = "l_ankle" if nl <= nr else "r_ankle"
    groups = df.groupby([m["subject"], m["session"], m["task"]], sort=True).indices
    summary, files = [], {"ankle": ("ank",), "back": ("tr",), "ankleback": ("ank", "tr")}
    if not DRY_RUN:
        for tag in files: os.makedirs(f"{FOGSTAR_OUT}_{tag}", exist_ok=True)
    run_no = {}
    for (s, sess, task), idx in groups.items():
        idx = np.sort(idx)
        if m["time"]: idx = idx[np.argsort(df[m["time"]].to_numpy()[idx], kind="stable")]
        a, o = side[s], ("r_ankle" if side[s] == "l_ankle" else "l_ankle")
        sig = {"ank": acc[a][idx], "thi": acc[o][idx], "tr": acc["back"][idx]}
        n_in = len(idx)
        if n_in < 2 * FS_IN: continue                                    # shorter than 2 s: skipped
        rs = {k: resample_poly(np.nan_to_num(v), UP, DOWN, axis=0) for k, v in sig.items()}
        n = min(len(v) for v in rs.values())
        src = np.minimum(np.round(np.arange(n) * FS_IN / FS_OUT).astype(int), n_in - 1)
        label = np.where(fog[idx][src], 2, 1)
        run_no[s] = run_no.get(s, 0) + 1; r = run_no[s]
        for tag, need in files.items():
            bad_in = np.zeros(n_in, bool)
            for k in need: bad_in |= np.isnan(sig[k]).any(1)
            bad_in = np.convolve(bad_in.astype(int), np.ones(2 * GUARD + 1, int), mode="same") > 0
            lab = label.copy(); lab[bad_in[src]] = 0
            if tag == "ankle":
                summary.append({"subject": int(s), "session": sess, "task": task, "run": r, "ankle": a,
                                "minutes": n / FS_OUT / 60, "excluded_%": 100 * (lab == 0).mean(),
                                "FoG_%": 100 * (lab == 2).mean(), "FoG_episodes": episodes(lab)})
            if not DRY_RUN:
                t = np.arange(n) * 1000.0 / FS_OUT
                out = np.column_stack([t, rs["ank"][:n], rs["thi"][:n], rs["tr"][:n], lab])
                np.savetxt(os.path.join(f"{FOGSTAR_OUT}_{tag}", f"S{int(s):02d}R{r:02d}.txt"), out, fmt="%.6g")
    S = pd.DataFrame(summary)
    per = S.groupby("subject").agg(runs=("run", "count"), ankle=("ankle", "first"), minutes=("minutes", "sum"),
                                   FoG_episodes=("FoG_episodes", "sum"), excluded_pct=("excluded_%", "mean"))
    pd.set_option("display.width", 200)
    print("\nPer subject:\n", per.round(2).to_string())
    print(f"\nTotal: {len(per)} subjects, {len(S)} runs, {S.minutes.sum():.1f} min, "
          f"{S.FoG_episodes.sum()} FoG episodes (ankle files; episodes split by gaps are counted separately)")
    print("\nDRY RUN - nothing written." if DRY_RUN else f"\nFiles written to {FOGSTAR_OUT}_ankle / _back / _ankleback")


if __name__ == "__main__":
    main()
