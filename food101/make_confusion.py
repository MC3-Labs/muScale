import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", required=True, help="parquet with columns y_gold,y_weak")
    ap.add_argument("--n_gold", type=int, default=500, help="size of gold subset to estimate confusion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.weak)
    if args.n_gold < len(df):
        df = df.sample(n=args.n_gold, random_state=args.seed).reset_index(drop=True)

    K = int(max(df["y_gold"].max(), df["y_weak"].max())) + 1
    C = np.zeros((K, K), dtype=np.int64)

    for yg, yw in zip(df["y_gold"].values, df["y_weak"].values):
        C[int(yg), int(yw)] += 1

    pd.DataFrame(C).to_csv(args.out, index=False)
    print(f"Wrote confusion matrix (K={K}, n={len(df)}) -> {args.out}")


if __name__ == "__main__":
    main()

