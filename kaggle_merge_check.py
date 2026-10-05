import numpy as np, pandas as pd, glob, os
import kaggle_convert as kc
kc.MERGE_GAP_S = 0 

def merge(lab, gap_s):
    lab = lab.copy()
    for a, b in kc.runs_of(lab != 0):
        seg = lab[a:b]
        eps = kc.runs_of(seg == 2)
        for (s1, e1), (s2, e2) in zip(eps[:-1], eps[1:]):
            if (s2 - e1) / 64 < gap_s:
                seg[e1:s2] = 2
        lab[a:b] = seg
    return lab

labs = {"tdcsfog": [], "defog": []}
for folder, kinds in {"tdcsfog": ["tdcsfog"], "defog": ["defog", "notype"]}.items():
    for kind in kinds:
        for f in glob.glob(os.path.join(kc.ROOT, "train", kind, "*.csv")):
            res = kc.convert_file(f, kind)
            if res is not None: labs[folder].append(res[1])

for ds, L in labs.items():
    print(f"\n=== {ds} ===")
    for g in [0, 0.5, 1, 2, 3]:
        n, pre4, dur = 0, [], []
        for lab in L:
            m = merge(lab, g) if g else lab
            pre = kc.pre_onset(m); pre4 += pre
            for a, b in kc.runs_of(m != 0):
                for s, e in kc.runs_of(m[a:b] == 2): dur.append((e - s) / 64)
        pre4 = np.array(pre4); dur = np.array(dur)
        print(f"merge gaps < {g:>3} s: episodes {len(dur):5d}, median duration {np.median(dur):4.1f} s, "
              f"<1 s {100*(dur<1).mean():4.1f}%, with 4 s before onset {100*(pre4>=4).mean():4.1f}%, "
              f"FoG {100*np.mean(np.concatenate([x[x!=0]==2 for x in L if (x!=0).any()])):.1f}%")