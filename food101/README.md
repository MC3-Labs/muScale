# Food-101 demo (OpenCLIP weak labels)

This folder is a lightweight end-to-end demonstration:
1) generate weak labels for Food-101 using OpenCLIP (zero-shot)
2) compute a confusion matrix on a small gold subset
3) compute lambda_pred = I(Y;Y~)/H(Y)

It’s meant to be a boring, reproducible example you can run on a single GPU.

## Quick start

From repo root:

### 1) Generate weak labels (test split)
```bash
python food101/run_weak_labels.py \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --out food101/outputs/weak_labels_vitb32.parquet

