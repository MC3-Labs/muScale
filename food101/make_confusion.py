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


def _food101_class_names(root: str = "data") -> Optional[List[str]]:
    """Return canonical Food101 class name list (length 101), or None if unavailable."""
    if Food101 is None:
        return None
    ds = Food101(root=root, split="train", download=True)
    return list(ds.classes)


def _infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Infer (true_label_col, weak_label_col) from common names.
    Your schema: ['index','split','y_gold','y_weak','model','pretrained'].
    """
    true_cands = ["y_gold", "gold", "y_true", "true", "label", "target", "gt", "y"]
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


def _to_int_if_possible(s: pd.Series) -> pd.Series:
    """Convert labels to int if they look numeric; otherwise return as strings."""
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return s.astype(int)
    st = s.astype(str)
    if st.str.fullmatch(r"\d+").all():
        return st.astype(int)
    return st


def _maybe_map_to_names(labels: pd.Series, class_names: Optional[List[str]]) -> pd.Series:
    """If labels are int indices and class_names exist, map them to names."""
    if class_names is None:
        return labels.astype(str)

    lab = _to_int_if_possible(labels)
    if pd.api.types.is_integer_dtype(lab):
        mn, mx = int(lab.min()), int(lab.max())
        if mn >= 0 and mx < len(class_names):
            return lab.map(lambda i: class_names[int(i)]).astype(str)

    return labels.astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", required=True, help="Weak labels parquet/csv from run_weak_labels.py")
    ap.add_argument("--n_gold", type=int, default=500, help="Number of samples to subsample for confusion")
    ap.add_argument("--out", required=True, help="Output confusion CSV path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=str, default="data", help="Root for torchvision Food101 download")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load weak labels file
    if args.weak.endswith(".parquet"):
        df = pd.read_parquet(args.weak)
    else:
        df = pd.read_csv(args.weak)

    true_col, weak_col = _infer_columns(df)

    if args.n_gold > len(df):
        raise ValueError(f"--n_gold {args.n_gold} > available rows {len(df)} in {args.weak}")

    df_sub = df.sample(n=args.n_gold, random_state=args.seed).copy()

    class_names = _food101_class_names(root=args.data_root)

    y_true = _maybe_map_to_names(df_sub[true_col], class_names)
    y_weak = _maybe_map_to_names(df_sub[weak_col], class_names)

    # Use canonical label order if available; else use union of observed labels
    if class_names is not None:
        labels = list(class_names)
    else:
        labels = sorted(set(y_true.tolist()) | set(y_weak.tolist()))

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    K = len(labels)

    cm = np.zeros((K, K), dtype=np.int64)
    skipped = 0
    for t, p in zip(y_true, y_weak):
        if t not in label_to_idx or p not in label_to_idx:
            skipped += 1
            continue
        cm[label_to_idx[t], label_to_idx[p]] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(args.out, index=True)

    n_used = int(cm.sum())
    print(f"Wrote confusion matrix (K={K}, n={n_used}, skipped={skipped}) -> {args.out}")


if __name__ == "__main__":
    main()

