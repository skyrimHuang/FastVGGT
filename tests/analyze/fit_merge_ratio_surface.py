import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_similarity(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"sequence_length", "block_layer", "token_similarity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


def build_dataset(df: pd.DataFrame):
    # l in [1,24]
    l = df["block_layer"].to_numpy(dtype=np.float64) + 1.0
    S = df["sequence_length"].to_numpy(dtype=np.float64)
    y = df["token_similarity"].to_numpy(dtype=np.float64)
    return l, S, y


def forward(params, l, S):
    a2, a1, a0, mu = params
    lnS = np.log(S + 1.0)
    return lnS * (a2 * l**2 + a1 * l + a0) + mu


def fit_params_linear_lstsq(l, S, y):
    """Solve linear least squares for
    y = ln(S+1) * (a2*l^2 + a1*l + a0) + mu
    """
    lnS = np.log(S + 1.0)
    X = np.column_stack([
        lnS * (l**2),
        lnS * l,
        lnS,
        np.ones_like(l),
    ])
    params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    return params


def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return rmse, r2


def main():
    parser = argparse.ArgumentParser(description="Fit r(l,S) surface and plot loss curve.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/hba/Documents/FastVGGT/tests/tests_result/token_similarity/token_similarity_results.csv",
        help="Path to token similarity CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hba/Documents/FastVGGT/tests/tests_result/token_similarity",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--residual_sort",
        type=str,
        default="S_then_l",
        choices=["S_then_l", "l_then_S", "none"],
        help="How to sort residuals for the error curve",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_similarity(csv_path)
    l, S, y = build_dataset(df)

    params = fit_params_linear_lstsq(l, S, y)
    y_pred = forward(params, l, S)
    rmse, r2 = compute_metrics(y, y_pred)

    # Residual curve (per-sample error)
    residuals = y_pred - y
    if args.residual_sort == "S_then_l":
        order = np.lexsort((l, S))
    elif args.residual_sort == "l_then_S":
        order = np.lexsort((S, l))
    else:
        order = np.arange(len(y))
    residuals_sorted = residuals[order]

    # Save error curve
    plt.figure(figsize=(7, 4))
    plt.plot(residuals_sorted, color="#2E86AB", linewidth=1.5)
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    plt.xlabel("Sample index (sorted)")
    plt.ylabel("Residual (pred - true)")
    plt.title("Fitting Residual Curve")
    loss_path = output_dir / "fit_residual_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # Print fitted parameters
    a2, a1, a0, mu = params
    print("Fitted parameters:")
    print(f"  a2 (ln(S+1)*l^2) = {a2:.6f}")
    print(f"  a1 (ln(S+1)*l)   = {a1:.6f}")
    print(f"  a0 (ln(S+1))     = {a0:.6f}")
    print(f"  mu     = {mu:.6f}")
    print(f"RMSE: {rmse:.6f}, R2: {r2:.6f}")
    print(f"Residual curve saved to: {loss_path}")


if __name__ == "__main__":
    main()
