#!/usr/bin/env python3
import argparse
import os
import random
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

try:
    from torchvision.datasets import Food101
except Exception:
    Food101 = None


# ---------- helpers ----------

def _food101_class_names(root: str = "data") -> Optional[List[str]]:
    """
    Returns the canonical Food101 class name list (length 101), or None if torchvision not available.
    """
    if Food101 is None:
        return None
    ds = Food101(root=root, split="train", download=True)
    return list(ds.classes)


def _infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to infer (true_label_col, weak_label_col) from common names.
    """
    true_cands = ["y_true", "true", "label", "target", "gold", "gt", "y"]
    weak_cands = ["y_weak", "weak", "pred", "weak_pred", "teacher", "pseudo", "yhat", "y_pred"]

    true_col = next((c for c in true_cands if c in df.columns), None)
    weak_col = next((c for c in weak_cands if c in df.columns), None)

    if true_col is None or weak_col is None:
        raise ValueError(
            f"Could not infer label columns.\n"
            f"Found columns: {list(df.columns)}\n"
            f"Expected one of true={true_cands} and weak={weak_cands}."
        )
    return true_col, weak_col


def _to_int_labels(s: pd.Series) -> pd.Series:
    """
    Convert labels to int if possible; otherwise leave as-is.
    """
    # already numeric
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return s.astype(int)
    # strings that are digits
    if s.astype(str).str.fullmatch(r"\d+").all():
        return s.astype(int)
    return s


def _maybe_map_to_names(labels: pd.Series, class_names: Optional[List[str]]) -> pd.Series:
    """
    If labels look like integer indices and we have class_names, map to names.
    Otherwise return as-is.
    """
    if class_names is None:
        return labels

    labels_int = _to_int_labels(labels)
    # if we successfully got integers and they look in-range, map
    if pd.api.types.is_integer_dtype(labels_int):
        mx = int(labels_int.max())
        mn = int(labels_int.min())
        if mn >= 0 and mx < len(class_names):
            return labels_int.map(lambda i: class_names[int(i)])
    return labels


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", required=True, help="Path to weak labels parquet/csv produced by run_weak_labels.py")
    ap.add_argument("--n_gold", type=int, default=500, help="Number of gold-labeled samples to subsample for confusion")
    ap.add_argument("--out", required=True, help="Output confusion CSV path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=str, default="data", help="Root used by torchvision Food101 (for class names)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load weak labels file
    if args.weak.endswith(".parquet"):
        df = pd.read_parquet(args.weak)
    else:
        df = pd.read_csv(args.weak)

    true_col, weak_col = _infer_columns(df)

    # Subsample "gold" set (we assume df already contains true labels for test split)
    if args.n_gold > len(df):
        raise ValueError(f"--n_gold {args.n_gold} > available rows {len(df)} in {args.weak}")

    df_sub = df.sample(n=args.n_gold, random_state=args.seed).copy()

    # Canonical Food101 names
    class_names = _food101_class_names(root=args.data_root)

    # Map labels to names if possible
    y_true_raw = df_sub[true_col]
    y_weak_raw = df_sub[weak_col]

    y_true = _maybe_map_to_names(y_true_raw, class_names)
    y_weak = _maybe_map_to_names(y_weak_raw, class_names)

    # Determine label set/order
    # Prefer canonical order if we have it; else use sorted unique from y_true
    if class_names is not None:
        labels = class_names
    else:
        labels = sorted(pd.unique(y_true.astype(str)).tolist())

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    K = len(labels)

    # Build confusion
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true.astype(str), y_weak.astype(str)):
        if t not in label_to_idx or p not in label_to_idx:
            # If anything unexpected appears, skip rather than crash
            continue
        cm[label_to_idx[t], label_to_idx[p]] += 1

    # Save labeled confusion matrix
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(args.out, index=True)

    print(f"Wrote confusion matrix (K={K}, n={cm.sum()}) -> {args.out}")


if __name__ == "__main__":
    main()

