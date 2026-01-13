#!/usr/bin/env python3
import argparse
import os
from typing import Tuple, List

import numpy as np
import pandas as pd


def _load_confusion(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Loads labeled confusion CSV (index+header are labels) or raw numeric KxK CSV.
    Returns (cm, labels).
    """
    try:
        df = pd.read_csv(path, index_col=0)
        cm = df.values.astype(np.int64)
        labels = list(df.index.astype(str))
        return cm, labels
    except Exception:
        cm = np.loadtxt(path, delimiter=",").astype(np.int64)
        labels = [str(i) for i in range(cm.shape[0])]
        return cm, labels


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", required=True, help="Confusion CSV")
    ap.add_argument("--out_prefix", required=True, help="Prefix for output files, e.g. outputs/report_vitb32")
    ap.add_argument("--topk", type=int, default=15, help="How many confusions/classes to show")
    args = ap.parse_args()

    cm, labels = _load_confusion(args.confusion)
    K = cm.shape[0]
    n = int(cm.sum())

    # per-class counts + accuracy
    row_sum = cm.sum(axis=1)
    diag = np.diag(cm)
    per_class_acc = np.array([_safe_div(diag[i], row_sum[i]) for i in range(K)])

    overall_acc = _safe_div(diag.sum(), n)

    # top confusions (excluding diagonal)
    conf_list = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                conf_list.append((c, i, j))
    conf_list.sort(reverse=True, key=lambda x: x[0])
    top_conf = conf_list[: args.topk]

    # lowest per-class accuracy (only where we have some samples)
    idxs = [i for i in range(K) if row_sum[i] > 0]
    idxs.sort(key=lambda i: per_class_acc[i])
    low_classes = idxs[: args.topk]

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    out_md = args.out_prefix + ".md"

    with open(out_md, "w") as f:
        f.write(f"# muScale report\n\n")
        f.write(f"- Confusion matrix: `{args.confusion}`\n")
        f.write(f"- K classes: {K}\n")
        f.write(f"- n samples (in confusion): {n}\n")
        f.write(f"- Overall weak-label accuracy (on sampled gold set): {overall_acc:.4f}\n\n")

        f.write("## Top confusions (count; percent within true class)\n")
        if not top_conf:
            f.write("- (none)\n")
        else:
            for c, i, j in top_conf:
                pct = 100.0 * _safe_div(c, row_sum[i])
                f.write(f"- {labels[i]} → {labels[j]}: {c} ({pct:.1f}% of that true class)\n")

        f.write("\n## Lowest per-class accuracy (quick scan)\n")
        for i in low_classes:
            f.write(f"- {labels[i]}: acc={per_class_acc[i]:.3f} (n={int(row_sum[i])})\n")

    print(f"Wrote report -> {out_md}")


if __name__ == "__main__":
    main()

