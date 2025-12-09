import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_monitor_csv(path: str):
    """
    Loads a Stable-Baselines3 monitor CSV or .monitor file.
    Handles '#' comment headers automatically.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")
    df = pd.read_csv(path, comment="#")
    if not {"r", "l"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path}: {df.columns}")
    return df

def moving_average(x, w):
    """Simple moving average with window size w."""
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def main():
    ap = argparse.ArgumentParser(description="Plot SB3 monitor rewards over time")
    ap.add_argument("--csv", required=True, help="Path to monitor_train.csv.monitor or .csv file")
    ap.add_argument("--smooth", type=int, default=20, help="Smoothing window (episodes)")
    args = ap.parse_args()

    df = load_monitor_csv(args.csv)
    df["timestep"] = df["l"].cumsum()

    # Compute smoothed reward
    r_smooth = moving_average(df["r"], args.smooth)
    x_smooth = df["timestep"].iloc[len(df["timestep"]) - len(r_smooth):]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestep"], df["r"], alpha=0.3, label="Episode Reward (raw)")
    plt.plot(x_smooth, r_smooth, color="C1", linewidth=2.0, label=f"Moving Avg ({args.smooth})")

    plt.xlabel("Training Timesteps")
    plt.ylabel("Episode Reward")
    plt.title(os.path.basename(args.csv))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # --- auto save to same folder ---
    out_dir = os.path.dirname(os.path.abspath(args.csv))
    base = os.path.splitext(os.path.basename(args.csv))[0]
    out_path = os.path.join(out_dir, f"{base}_reward_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    main()
