#!/usr/bin/env python3
"""
Generate a lightweight practitioner-facing report from a confusion matrix.

Inputs:
  - confusion CSV produced by make_confusion.py (rows=true, cols=pred, plus optional metadata)

Outputs:
  - <out_prefix>.json : machine-readable summary
  - <out_prefix>.md   : one-page markdown report
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def _entropy_bits(p: np.ndarray) -> float:
    p = p.astype(np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mutual_information_bits(joint: np.ndarray) -> float:
    joint = joint.astype(np.float64)
    total = joint.sum()
    if total <= 0:
        return float("nan")
    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    # Only sum over nonzero entries to avoid log(0)
    nz = pxy > 0
    return float((pxy[nz] * (np.log2(pxy[nz]) - np.log2(px[nz.any(axis=1), :][:, 0][:, None]) - np.log2(py[:, nz.any(axis=0)][0, :][None, :]))).sum())


def _mi_bits_stable(joint: np.ndarray) -> float:
    # Stable MI computation: sum_{i,j} p(i,j) log p(i,j)/(p(i)p(j))
    joint = joint.astype(np.float64)
    total = joint.sum()
    if total <= 0:
        return float("nan")
    pxy = joint / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    for i in range(pxy.shape[0]):
        if px[i] == 0:
            continue
        for j in range(pxy.shape[1]):
            pij = pxy[i, j]
            if pij <= 0 or py[j] == 0:
                continue
            mi += pij * (np.log2(pij) - np.log2(px[i]) - np.log2(py[j]))
    return float(mi)


def _top_confusions(cm: np.ndarray, labels: List[str], k: int = 10) -> List[Dict]:
    # Off-diagonal largest counts
    K = cm.shape[0]
    pairs: List[Tuple[int, int, int]] = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                pairs.append((i, j, c))
    pairs.sort(key=lambda x: x[2], reverse=True)
    out = []
    for i, j, c in pairs[:k]:
        out.append(
            {
                "true": labels[i],
                "pred": labels[j],
                "count": c,
                "row_frac": _safe_div(c, float(cm[i, :].sum())),
            }
        )
    return out


def _per_class_acc(cm: np.ndarray, labels: List[str]) -> List[Dict]:
    out = []
    for i, name in enumerate(labels):
        row_sum = float(cm[i, :].sum())
        correct = float(cm[i, i])
        out.append(
            {
                "class": name,
                "n": int(row_sum),
                "acc": _safe_div(correct, row_sum),
            }
        )
    # Sort worst-first for quick scanning
    out.sort(key=lambda d: (d["acc"], -d["n"]))
    return out


def _guess_label_list(df: pd.DataFrame) -> List[str]:
    # If df has an explicit 'label' column, use it; else infer from index/columns
    if "label" in df.columns:
        return [str(x) for x in df["label"].tolist()]
    # Often confusion is stored with an unnamed first column = row labels
    if df.columns[0] != df.columns[1] and df.columns[0].lower() in ("", "unnamed: 0", "true", "y"):
        # If first column looks like row labels, use it
        return [str(x) for x in df.iloc[:, 0].tolist()]
    # Otherwise use columns as labels
    return [str(c) for c in df.columns]


def _load_confusion(path: Path) -> Tuple[np.ndarray, List[str], Dict]:
    df = pd.read_csv(path)

    meta: Dict = {}
    # If the file contains metadata rows/cols, keep it minimal; assume the core is a square matrix.
    # Common pattern: first col contains row labels, remaining cols are class names.
    if df.shape[1] >= 2 and df.columns[0].startswith("Unnamed"):
        # first column is row labels
        labels = [str(x) for x in df.iloc[:, 0].tolist()]
        mat = df.iloc[:, 1:].to_numpy()
        # If columns include the same labels, great; otherwise trust row labels.
        cm = mat.astype(np.int64)
        return cm, labels, meta

    # Otherwise assume full dataframe is numeric square with column labels
    labels = [str(c) for c in df.columns]
    cm = df.to_numpy().astype(np.int64)
    return cm, labels, meta


def write_report(confusion_csv: Path, out_prefix: Path, dataset: str | None, weak_labeler: str | None) -> None:
    cm, labels, meta = _load_confusion(confusion_csv)
    total = int(cm.sum())
    K = int(cm.shape[0])

    # empirical p(y)
    py = cm.sum(axis=1).astype(np.float64)
    py = py / py.sum() if py.sum() > 0 else py
    H = _entropy_bits(py)

    I = _mi_bits_stable(cm)
    lam = _safe_div(I, H)

    diag = cm.diagonal().sum()
    acc = _safe_div(float(diag), float(total))

    top_conf = _top_confusions(cm, labels, k=10)
    per_class = _per_class_acc(cm, labels)

    # simple “collapse” / coverage checks
    pred_counts = cm.sum(axis=0).astype(np.float64)
    pred_dist = pred_counts / pred_counts.sum() if pred_counts.sum() > 0 else pred_counts
    pred_entropy = _entropy_bits(pred_dist)

    payload = {
        "dataset": dataset or "Food-101",
        "weak_labeler": weak_labeler or "open_clip",
        "confusion_path": str(confusion_csv),
        "n_total": total,
        "K": K,
        "overall_acc_on_eval": acc,
        "H_Y_bits": H,
        "I_Y_Ytilde_bits": I,
        "lambda_pred": lam,
        "pred_entropy_bits": pred_entropy,
        "top_confusions": top_conf,
        "per_class_accuracy_worst_first": per_class[:25],
    }
    payload.update(meta)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # markdown summary (human readable)
    lines = []
    lines.append(f"# muScale report: {payload['dataset']}")
    lines.append("")
    lines.append(f"**Weak labeler:** {payload['weak_labeler']}")
    lines.append(f"**Eval size:** n={payload['n_total']} (K={payload['K']} classes)")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- H(Y): **{payload['H_Y_bits']:.4f} bits**")
    lines.append(f"- I(Y; Y~): **{payload['I_Y_Ytilde_bits']:.4f} bits**")
    lines.append(f"- λ_pred = I/H: **{payload['lambda_pred']:.4f}**")
    lines.append(f"- Overall accuracy on eval: **{payload['overall_acc_on_eval']:.4f}**")
    lines.append(f"- Prediction entropy (sanity check): **{payload['pred_entropy_bits']:.4f} bits**")
    lines.append("")
    lines.append("## Interpretation")
    lines.append(f"Assuming roughly random / independent errors, **1 weak label ≈ {payload['lambda_pred']:.3f} gold labels** in variance-reduction value.")
    lines.append("")
    lines.append("## Top confusions (true → predicted)")
    for c in payload["top_confusions"]:
        lines.append(f"- {c['true']} → {c['pred']}: {c['count']} ({c['row_frac']:.1%} of that true class)")
    lines.append("")
    lines.append("## Lowest per-class accuracy (quick scan)")
    for row in payload["per_class_accuracy_worst_first"][:15]:
        lines.append(f"- {row['class']}: acc={row['acc']:.3f} (n={row['n']})")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote:\n  {json_path}\n  {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", required=True, type=str, help="Path to confusion CSV")
    ap.add_argument("--out_prefix", required=True, type=str, help="Output prefix (no extension)")
    ap.add_argument("--dataset", default=None, type=str, help="Dataset name override")
    ap.add_argument("--weak_labeler", default=None, type=str, help="Weak labeler name override")
    args = ap.parse_args()

    write_report(
        confusion_csv=Path(args.confusion),
        out_prefix=Path(args.out_prefix),
        dataset=args.dataset,
        weak_labeler=args.weak_labeler,
    )


if __name__ == "__main__":
    main()

