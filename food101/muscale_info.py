#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


def _load_confusion(path: str):
    """
    Supports:
    - labeled CSV saved by make_confusion.py (index + header are labels)
    - unlabeled numeric CSV (no header), KxK
    """
    try:
        df = pd.read_csv(path, index_col=0)
        # If first column became index and remaining columns are numeric-like strings, this is labeled.
        # If columns look like '0','1',... and index are also numeric-like, still fine.
        cm = df.values.astype(np.int64)
        labels = list(df.index.astype(str))
        return cm, labels
    except Exception:
        # fallback: raw numeric
        cm = np.loadtxt(path, delimiter=",").astype(np.int64)
        labels = [str(i) for i in range(cm.shape[0])]
        return cm, labels


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", required=True, help="Confusion CSV")
    args = ap.parse_args()

    cm, labels = _load_confusion(args.confusion)
    n = cm.sum()
    if n <= 0:
        raise ValueError("Confusion matrix has zero total count.")

    # joint distribution P(Y, Y~)
    pxy = cm / n
    py = pxy.sum(axis=1, keepdims=False)
    pyp = pxy.sum(axis=0, keepdims=False)

    HY = _entropy_bits(py)
    HYp = _entropy_bits(pyp)

    # mutual information I(Y;Y~) = sum p(x,y) log p(x,y)/(p(x)p(y))
    eps = 1e-12
    denom = (py[:, None] * pyp[None, :]) + eps
    ratio = (pxy + eps) / denom
    I = float((pxy * np.log2(ratio)).sum())

    lam = 0.0 if HY == 0 else I / HY

    print("=== muScale info (from confusion matrix) ===")
    print(f"n_total        : {n}")
    print(f"H(Y) bits      : {HY:.4f}")
    print(f"I(Y;Y~) bits   : {I:.4f}")
    print(f"lambda_pred    : {lam:.4f}")
    print("")
    print("Interpretation:")
    print(f"  1 weak label ~= {lam:.3f} gold labels (random-error regime assumption).")


if __name__ == "__main__":
    main()

