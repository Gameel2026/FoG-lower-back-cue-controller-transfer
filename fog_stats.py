from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from fog_pipeline import (Config, extract_features, load_daphnet,
                          relabel_four_class, segment_windows)
from fog_study import (binary_cue_loso, binary_window_metrics, event_metrics,
                       loso_decoders)

COMPARISONS = [
    ("E4 Shank", "RF (per-window)"),
    ("E4 Shank", "HMM forward (causal)"),
    ("E4 Trunk", "E4 Shank"),
]

METRICS = {
    "predicted_before_onset_%": True,
    "detected_%": True,
    "median_lead_time_s": True,
    "false_alarms_per_hour": False,
    "balanced_acc": True,
}
ALL_SUBJECT_METRICS = {"false_alarms_per_hour"}  


def prepare(raw, sensors, quick):
    cfg = Config(n_estimators=30 if quick else 100, sensors=sensors)
    windows, meta = segment_windows(raw, cfg)
    X = extract_features(windows, cfg)
    y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    return cfg, X, meta, y4


def per_subject(name, pred, y4, meta, cfg) -> pd.DataFrame:
    y2 = meta["y2"].to_numpy()
    rows = []
    for s in sorted(meta["subject"].unique()):
        m = meta["subject"].eq(s).to_numpy()
        rows.append({"method": name, "subject": s,
                     **binary_window_metrics(y4[m], pred[m]),
                     **event_metrics(y2, pred, meta, cfg, m)})
    return pd.DataFrame(rows)


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def run_tests(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in COMPARISONS:
        fam = []
        for metric, higher_better in METRICS.items():
            A = table[table.method == a].set_index("subject")
            B = table[table.method == b].set_index("subject")
            common = A.index.intersection(B.index)
            subs = common if metric in ALL_SUBJECT_METRICS else common[A.loc[common, "episodes"] > 0]
            x, y = A.loc[subs, metric].to_numpy(), B.loc[subs, metric].to_numpy()
            d = x - y
            better = int(np.sum(d > 0) if higher_better else np.sum(d < 0))
            try:
                p = wilcoxon(x, y).pvalue          
            except ValueError:                     
                p = np.nan
            fam.append({"comparison": f"{a} vs {b}", "metric": metric,
                        "n_subjects": len(subs),
                        f"median_A": np.median(x), f"median_B": np.median(y),
                        "median_diff_A_minus_B": np.median(d),
                        "subjects_A_better": f"{better}/{len(subs)}",
                        "p_raw": p})
        fam = pd.DataFrame(fam)
        fam["p_holm"] = holm(fam["p_raw"].to_numpy())
        rows.append(fam)
    return pd.concat(rows, ignore_index=True)


def main(root: str, quick: bool = False):
    os.makedirs("results", exist_ok=True)
    raw = load_daphnet(root)
    tables = []

    print("Shank features ...")
    cfg, X, meta, y4 = prepare(raw, ("ank",), quick)
    print("4-class LOSO decoders (RF, HMM) ...")
    preds = loso_decoders(X, y4, meta, cfg)
    for name in ["RF (per-window)", "HMM forward (causal)"]:
        tables.append(per_subject(name, preds[name], y4, meta, cfg))
    print("E4 Shank ...")
    pred, _, _ = binary_cue_loso(X, y4, meta, cfg)
    tables.append(per_subject("E4 Shank", pred, y4, meta, cfg))

    print("Trunk features + E4 Trunk ...")
    cfg, X, meta, y4 = prepare(raw, ("tr",), quick)
    pred, _, _ = binary_cue_loso(X, y4, meta, cfg)
    tables.append(per_subject("E4 Trunk", pred, y4, meta, cfg))

    table = pd.concat(tables, ignore_index=True)
    table.round(3).to_csv("results/S1_per_subject_all_methods.csv", index=False)

    tests = run_tests(table)
    tests.round(4).to_csv("results/S2_wilcoxon_tests.csv", index=False)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    print("\nWilcoxon signed-rank (two-sided, Holm within each comparison)\n",
          tests.round(4))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    from paths import DAPHNET_ROOT
    main(args[0] if args else DAPHNET_ROOT, quick=False)
