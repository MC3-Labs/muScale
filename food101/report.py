#!/usr/bin/env python3
import argparse
import os
from typing import Tuple, List

import numpy as np
import pandas as pd


def _load_confusion(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Robust confusion loader.
    Supports:
      A) labeled CSV: first column index labels, header labels -> square
      B) raw numeric CSV (no header) -> square
    If a read produces a non-square matrix, fall back or repair.
    """
    # Attempt labeled
    try:
        df = pd.read_csv(path, index_col=0)
        cm = df.values
        # If this is the "numeric-without-header but index_col=0" trap, it will be non-square
        if cm.shape[0] == cm.shape[1]:
            return cm.astype(np.int64), list(df.index.astype(str))
    except Exception:
        pass

    # Fallback: raw numeric with no header
    df2 = pd.read_csv(path, header=None)
    cm2 = df2.values
    if cm2.shape[0] != cm2.shape[1]:
        # As a last resort, force to square by truncating to min dimension
        m = min(cm2.shape[0], cm2.shape[1])
        cm2 = cm2[:m, :m]
    labels2 = [str(i) for i in range(cm2.shape[0])]
    return cm2.astype(np.int64), labels2


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", required=True, help="Confusion CSV")
    ap.add_argument("--out_prefix", required=True, help="Prefix for output files, e.g. outputs/report_vitb32")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    cm, labels = _load_confusion(args.confusion)
    K = cm.shape[0]
    n = int(cm.sum())

    row_sum = cm.sum(axis=1)
    diag = np.diag(cm)
    per_class_acc = np.array([_safe_div(diag[i], row_sum[i]) for i in range(K)])
    overall_acc = _safe_div(diag.sum(), n)

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

    idxs = [i for i in range(K) if row_sum[i] > 0]
    idxs.sort(key=lambda i: per_class_acc[i])
    low_classes = idxs[: args.topk]

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    out_md = args.out_prefix + ".md"

    with open(out_md, "w") as f:
        f.write("# muScale report\n\n")
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

