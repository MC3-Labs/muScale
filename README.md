# muScale

muScale is a small toolkit for *microscaling laws* in mixed-supervision training: combining a limited set of **gold** labels with large volumes of **weak** labels from a foundation model (e.g., CLIP / BiomedCLIP / GPT-* vision). The working goal is simple: estimate how much weak supervision is “worth” relative to gold supervision, and use that to plan annotation budgets and label allocation without brute-force grid search.

This repo is the implementation companion to our paper:
**μ-scaling Laws: Bias-Variance Decomposition for Foundation Model Augmentation Under Label Noise**.

> Status: early / actively changing. The API will move.

---

## What’s in here (now)

- Compute confusion-matrix-based quantities (entropy, mutual information, etc.)
- A basic scaling-law fitter for the effective-samples model  
  \( R = \alpha (n_g + \lambda n_w)^{-\beta} + \epsilon \)
- A budget allocation helper (given label costs) for simple what-if planning
- CLI entrypoints for quick runs on CSVs

What’s *not* in here yet:
- A polished dataset loader zoo
- A one-click reproduction pipeline for all experiments
- A fully stabilized bias–variance decomposition model (we’re adding it next)

---

## Installation

**Python 3.10+** recommended.

```bash
pip install -e .
