# plot_evaluations.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_eval_npz(path):
    if not os.path.exists(path):
        # try common alt name
        alt = path.replace("evaluation.npz", "evaluations.npz")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Could not find: {path}")
    data = np.load(path, allow_pickle=True)
    # SB3 keys: "timesteps", "results", "ep_lengths" (+ maybe "successes")
    timesteps = data["timesteps"]
    results   = data["results"]          # shape: (n_evals, n_episodes_per_eval)
    ep_lens   = data["ep_lengths"]       # shape: (n_evals, n_episodes_per_eval)
    successes = data["successes"] if "successes" in data.files else None
    return timesteps, results, ep_lens, successes

def main():
    ap = argparse.ArgumentParser("Plot SB3 evaluations.npz")
    ap.add_argument("--npz", required=True, help="Path to evaluations.npz (or evaluation.npz)")
    ap.add_argument("--out", default=None, help="Optional path to save PNG")
    ap.add_argument("--title", default=None, help="Optional plot title")
    ap.add_argument("--smooth", type=int, default=1, help="Moving average over eval means (window)")
    args = ap.parse_args()

    t, results, ep_lens, successes = load_eval_npz(args.npz)

    # Compute per-eval stats
    rew_mean = results.mean(axis=1)
    rew_std  = results.std(axis=1)
    len_mean = ep_lens.mean(axis=1)
    len_std  = ep_lens.std(axis=1)

    # Optional smoothing on reward mean/std via simple moving average
    w = max(1, args.smooth)
    if w > 1:
        def sma(x, k):
            if k <= 1: return x
            c = np.convolve(x, np.ones(k)/k, mode="valid")
            # pad to original length
            pad = np.full_like(x, np.nan, dtype=float)
            pad[len(x)-len(c):] = c
            return pad
        rew_mean_s = sma(rew_mean, w)
        rew_std_s  = sma(rew_std,  w)
    else:
        rew_mean_s, rew_std_s = rew_mean, rew_std

    # Plot
    fig, ax1 = plt.subplots(figsize=(9,5))
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Eval Reward (mean ± 1 std)")
    ax1.plot(t, rew_mean_s, label="Reward mean")
    ax1.fill_between(t, rew_mean_s - rew_std_s, rew_mean_s + rew_std_s, alpha=0.25)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Second axis: episode length
    ax2 = ax1.twinx()
    ax2.set_ylabel("Episode Length (mean)")
    ax2.plot(t, len_mean, linestyle=":", label="Ep len mean")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Title & annotate last point
    title = args.title or f"Evaluations — {os.path.basename(os.path.dirname(args.npz))}"
    plt.title(title)

    def last_non_nan(x):
        idx = np.where(~np.isnan(x))[0]
        return (idx[-1], x[idx[-1]]) if len(idx) else (None, None)

    i_last, r_last = last_non_nan(rew_mean_s)
    if i_last is not None:
        txt = f"last: {r_last:.3f} ± {rew_std[i_last]:.3f} | len={len_mean[i_last]:.1f}"
        ax1.annotate(txt, xy=(t[i_last], rew_mean_s[i_last]), xytext=(10,10),
                     textcoords="offset points", bbox=dict(boxstyle="round", alpha=0.15))

    # Optional successes %
    if successes is not None:
        suc_mean = successes.mean(axis=1) * 100.0
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.08))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        ax3.set_ylabel("Success (%)")
        ax3.plot(t, suc_mean, linestyle="--", label="Success %")
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="best")

    fig.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
