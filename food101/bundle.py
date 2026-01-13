#!/usr/bin/env python3
"""
bundle.py
Creates a compact artifact bundle from a confusion matrix:
  - <out_prefix>.json     (machine-readable summary + metadata)
  - <out_prefix>.md       (human-readable report)
  - <out_prefix>_heatmap.png (confusion heatmap, log-scaled)
  - <out_prefix>_per_class.csv (per-class accuracy/support)

Works with both labeled confusion CSVs (index+header) and raw numeric matrices.
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any

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
        if cm.shape[0] == cm.shape[1]:
            labels = [str(x) for x in df.index.tolist()]
            return cm.astype(np.int64), labels
    except Exception:
        pass

    # Fallback: raw numeric with no header
    df2 = pd.read_csv(path, header=None)
    cm2 = df2.values
    if cm2.shape[0] != cm2.shape[1]:
        m = min(cm2.shape[0], cm2.shape[1])
        cm2 = cm2[:m, :m]
    labels2 = [str(i) for i in range(cm2.shape[0])]
    return cm2.astype(np.int64), labels2


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _entropy_bits(p: np.ndarray) -> float:
    """Entropy in bits for a probability vector p (assumes p sums to 1)."""
    p = p.astype(np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _mutual_info_bits(cm: np.ndarray) -> float:
    """
    Mutual information I(Y; Y~) in bits from a confusion matrix of counts.
    """
    n = cm.sum()
    if n == 0:
        return 0.0
    pxy = cm / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    # Only sum where pxy>0 to avoid log(0).
    mask = pxy > 0
    return float((pxy[mask] * np.log2(pxy[mask] / (px[mask.any(axis=1), :] * py[:, mask.any(axis=0)]))).sum())


def _mutual_info_bits_stable(cm: np.ndarray) -> float:
    """
    More numerically stable mutual information computation.
    """
    n = cm.sum()
    if n == 0:
        return 0.0
    pxy = cm / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] <= 0:
                continue
            mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    return float(mi)


def _top_confusions(cm: np.ndarray, labels: List[str], topk: int) -> List[Dict[str, Any]]:
    K = cm.shape[0]
    row_sum = cm.sum(axis=1)
    items = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            c = int(cm[i, j])
            if c <= 0:
                continue
            pct = 100.0 * _safe_div(c, row_sum[i])
            items.append(
                {
                    "true": labels[i],
                    "pred": labels[j],
                    "count": c,
                    "pct_within_true": pct,
                }
            )
    items.sort(key=lambda d: d["count"], reverse=True)
    return items[:topk]


def _per_class_table(cm: np.ndarray, labels: List[str]) -> pd.DataFrame:
    row_sum = cm.sum(axis=1)
    diag = np.diag(cm)
    acc = np.array([_safe_div(diag[i], row_sum[i]) for i in range(cm.shape[0])], dtype=np.float64)

    return pd.DataFrame(
        {
            "class": labels,
            "support_n": row_sum.astype(int),
            "correct_n": diag.astype(int),
            "accuracy": acc,
        }
    )


def _muscale_metrics(cm: np.ndarray) -> Dict[str, float]:
    """
    Mirrors the spirit of muscale_info.py:
      H(Y) bits
      I(Y;Y~) bits
      lambda_pred = I / H
    """
    n = cm.sum()
    if n == 0:
        return {"n_total": 0, "H_bits": 0.0, "I_bits": 0.0, "lambda_pred": 0.0}

    py = cm.sum(axis=1) / n
    H = _entropy_bits(py)
    I = _mutual_info_bits_stable(cm)
    lam = _safe_div(I, H) if H > 0 else 0.0

    return {"n_total": int(n), "H_bits": float(H), "I_bits": float(I), "lambda_pred": float(lam)}


def _write_heatmap(cm: np.ndarray, labels: List[str], out_png: str, max_labels: int = 30) -> None:
    """
    Writes a log-scaled heatmap PNG.
    If too many labels, suppress tick labels to keep it readable.
    """
    import matplotlib.pyplot as plt

    # log1p so zeros stay 0, big counts compress
    img = np.log1p(cm.astype(np.float64))

    fig = plt.figure(figsize=(10, 9), dpi=200)
    ax = plt.gca()
    im = ax.imshow(img)

    ax.set_title("Confusion matrix (log1p counts)")
    ax.set_xlabel("Predicted / weak label")
    ax.set_ylabel("True / gold label")

    K = cm.shape[0]
    if K <= max_labels:
        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        # Too dense: no tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.01,
            0.01,
            f"K={K} (tick labels suppressed)",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", required=True, help="Confusion CSV (labeled or raw)")
    ap.add_argument("--out_prefix", required=True, help="Output prefix, e.g. food101/outputs/vitb32_bundle")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--dataset", type=str, default="Food101")
    ap.add_argument("--weak_labeler", type=str, default="", help="Short description of weak labeler (e.g., open_clip ViT-B-32 laion2b...)")
    ap.add_argument("--notes", type=str, default="", help="Optional notes string to embed in JSON")
    args = ap.parse_args()

    cm, labels = _load_confusion(args.confusion)
    K = cm.shape[0]
    n = int(cm.sum())
    diag = np.diag(cm)
    overall_acc = _safe_div(diag.sum(), n)

    metrics = _muscale_metrics(cm)
    per_class = _per_class_table(cm, labels)
    top_conf = _top_confusions(cm, labels, args.topk)

    # Outputs
    out_md = args.out_prefix + ".md"
    out_json = args.out_prefix + ".json"
    out_png = args.out_prefix + "_heatmap.png"
    out_per_class = args.out_prefix + "_per_class.csv"

    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    # Write per-class CSV
    per_class.sort_values(["accuracy", "support_n"], ascending=[True, False]).to_csv(out_per_class, index=False)

    # Write heatmap
    _write_heatmap(cm, labels, out_png)

    # Write markdown report
    with open(out_md, "w") as f:
        f.write("# muScale bundle report\n\n")
        f.write(f"- Dataset: {args.dataset}\n")
        if args.weak_labeler:
            f.write(f"- Weak labeler: {args.weak_labeler}\n")
        f.write(f"- Confusion: `{args.confusion}`\n")
        f.write(f"- K classes: {K}\n")
        f.write(f"- n samples (in confusion): {n}\n")
        f.write(f"- Overall weak-label accuracy (on sampled gold set): {overall_acc:.4f}\n\n")

        f.write("## muScale summary\n")
        f.write(f"- H(Y) (bits): {metrics['H_bits']:.4f}\n")
        f.write(f"- I(Y;Y~) (bits): {metrics['I_bits']:.4f}\n")
        f.write(f"- lambda_pred = I/H: {metrics['lambda_pred']:.4f}\n\n")

        f.write("Interpretation (random-error regime assumption):\n\n")
        f.write(f"- 1 weak label ~= {metrics['lambda_pred']:.3f} gold labels\n\n")

        f.write("## Top confusions (count; percent within true class)\n")
        if not top_conf:
            f.write("- (none)\n")
        else:
            for d in top_conf:
                f.write(f"- {d['true']} → {d['pred']}: {d['count']} ({d['pct_within_true']:.1f}% of that true class)\n")

        f.write("\n## Lowest per-class accuracy (quick scan)\n")
        # show the worst 15 classes with nonzero support
        pc_nonzero = per_class[per_class["support_n"] > 0].sort_values("accuracy", ascending=True).head(args.topk)
        for _, r in pc_nonzero.iterrows():
            f.write(f"- {r['class']}: acc={r['accuracy']:.3f} (n={int(r['support_n'])})\n")

        f.write("\n## Artifacts\n")
        f.write(f"- JSON: `{os.path.basename(out_json)}`\n")
        f.write(f"- Per-class CSV: `{os.path.basename(out_per_class)}`\n")
        f.write(f"- Heatmap: `{os.path.basename(out_png)}`\n")

    # Write JSON bundle
    bundle = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "weak_labeler": args.weak_labeler,
        "notes": args.notes,
        "inputs": {"confusion_path": args.confusion},
        "summary": {
            "K": int(K),
            "n_total": int(n),
            "overall_accuracy": float(overall_acc),
            "H_bits": float(metrics["H_bits"]),
            "I_bits": float(metrics["I_bits"]),
            "lambda_pred": float(metrics["lambda_pred"]),
        },
        "top_confusions": top_conf,
        "artifacts": {
            "report_md": out_md,
            "bundle_json": out_json,
            "per_class_csv": out_per_class,
            "heatmap_png": out_png,
        },
    }
    with open(out_json, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"Wrote:\n  {out_md}\n  {out_json}\n  {out_per_class}\n  {out_png}")


if __name__ == "__main__":
    main()

