# muScale report: Food-101 (test, 500-sample eval)

**Weak labeler:** CLIP ViT-B/32 (laion2b_s34b_b79k)
**Eval size:** n=500 (K=101 classes)

## Summary
- H(Y): **6.4864 bits**
- I(Y; Y~): **5.9723 bits**
- λ_pred = I/H: **0.9207**
- Overall accuracy on eval: **0.8620**
- Prediction entropy (sanity check): **6.4500 bits**

## Interpretation
Assuming roughly random / independent errors, **1 weak label ≈ 0.921 gold labels** in variance-reduction value.

## Top confusions (true → predicted)
- 58 → 45: 3 (75.0% of that true class)
- 93 → 77: 3 (37.5% of that true class)
- 0 → 8: 1 (33.3% of that true class)
- 3 → 54: 1 (25.0% of that true class)
- 3 → 81: 1 (25.0% of that true class)
- 4 → 3: 1 (20.0% of that true class)
- 5 → 16: 1 (12.5% of that true class)
- 5 → 74: 1 (12.5% of that true class)
- 6 → 12: 1 (10.0% of that true class)
- 6 → 42: 1 (10.0% of that true class)

## Lowest per-class accuracy (quick scan)
- 58: acc=0.250 (n=4)
- 10: acc=0.333 (n=3)
- 93: acc=0.500 (n=8)
- 3: acc=0.500 (n=4)
- 31: acc=0.500 (n=4)
- 55: acc=0.500 (n=2)
- 62: acc=0.500 (n=2)
- 71: acc=0.500 (n=2)
- 99: acc=0.500 (n=2)
- 100: acc=0.600 (n=10)
- 35: acc=0.600 (n=5)
- 57: acc=0.600 (n=5)
- 94: acc=0.600 (n=5)
- 0: acc=0.667 (n=3)
- 15: acc=0.667 (n=3)
