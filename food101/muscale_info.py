import argparse
import numpy as np
import pandas as pd


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information(joint: np.ndarray) -> float:
    # joint is P(Y, Y~)
    py = joint.sum(axis=1, keepdims=True)
    pz = joint.sum(axis=0, keepdims=True)
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            p = joint[i, j]
            if p <= 0:
                continue
            mi += p * np.log2(p / (py[i, 0] * pz[0, j]))
    return float(mi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", required=True, help="CSV confusion matrix: rows=gold, cols=weak")
    args = ap.parse_args()

    C = pd.read_csv(args.confusion).values.astype(np.float64)
    n = C.sum()
    if n <= 0:
        raise ValueError("Confusion matrix is empty.")

    P = C / n  # joint distribution
    py = P.sum(axis=1)  # P(Y)
    H = entropy(py)
    I = mutual_information(P)
    lam = I / H if H > 0 else 0.0

    print("=== muScale info (from confusion matrix) ===")
    print(f"n_total        : {int(n)}")
    print(f"H(Y) bits      : {H:.4f}")
    print(f"I(Y;Y~) bits   : {I:.4f}")
    print(f"lambda_pred    : {lam:.4f}")
    print("")
    print("Interpretation:")
    print(f"  1 weak label ~= {lam:.3f} gold labels (random-error regime assumption).")


if __name__ == "__main__":
    main()

