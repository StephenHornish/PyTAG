#!/usr/bin/env python3
import os
import argparse
import numpy as np
import gymnasium as gym

# Registers TAG/* envs
import pytag.gym_wrapper  # noqa: F401

from gymnasium.wrappers import TimeLimit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

import matplotlib
matplotlib.use("Agg")  # headless backend for saving plots
import matplotlib.pyplot as plt

from collections import Counter
from pathlib import Path
from datetime import datetime
import csv


def get_action_mask(env):
    u = env.unwrapped
    if hasattr(u, "get_action_mask"):
        m = u.get_action_mask()
        return np.asarray(m, dtype=bool)
    for attr in ("_last_info", "last_info", "info"):
        info = getattr(u, attr, None)
        if isinstance(info, dict) and "action_mask" in info:
            return np.asarray(info["action_mask"], dtype=bool)
    return np.ones(env.action_space.n, dtype=bool)


def make_env(env_id, max_steps, *, n_players, opponents, obs_type, log_path=None):
    assert len(opponents) == n_players - 1, (
        f"Need {n_players - 1} opponents for {n_players}-player game, got {len(opponents)}"
    )
    agent_ids = ["python"] + opponents
    e = gym.make(env_id, agent_ids=agent_ids, obs_type=obs_type)
    e = TimeLimit(e, max_episode_steps=max_steps)
    e = ActionMasker(e, get_action_mask)
    if log_path:
        e = Monitor(e, filename=log_path)
    return e


def winners_from_java(env, n_players):
    """
    Ask the underlying Java env for player results (robust & N-player safe).
    Returns a list of winner indices (can be multiple in case of ties).
    """
    try:
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_env"):
            jenva = getattr(env.unwrapped._env, "_java_env", None)
            if jenva is None:
                return []
            raw = jenva.getPlayerResults()  # Java enum array, e.g., ["WIN", "LOSE", ...]
            results = [str(r) for r in raw] if raw is not None else []
            # Normalize length defensively
            if len(results) < n_players:
                results += ["UNKNOWN"] * (n_players - len(results))
            elif len(results) > n_players:
                results = results[:n_players]
            winners = [i for i, r in enumerate(results) if "WIN" in r.upper()]
            return winners
    except Exception:
        pass
    return []


def winners_from_info(info, n_players):
    """
    Best-effort fallbacks based on info dict.
    Supports 'winner' (int), 'winners' (list), 'results' (list of strings/enums).
    """
    if not isinstance(info, dict):
        return []
    # single int winner
    if "winner" in info:
        w = info["winner"]
        if isinstance(w, (int, np.integer)) and 0 <= int(w) < n_players:
            return [int(w)]
    # explicit winners list
    if "winners" in info and isinstance(info["winners"], (list, tuple)):
        out = []
        for w in info["winners"]:
            if isinstance(w, (int, np.integer)) and 0 <= int(w) < n_players:
                out.append(int(w))
        if out:
            return out
    # results strings like ["WIN","LOSE",...]
    if "results" in info and isinstance(info["results"], (list, tuple)):
        winners = []
        for i, r in enumerate(info["results"]):
            if isinstance(r, str) and "WIN" in r.upper() and i < n_players:
                winners.append(i)
        if winners:
            return winners
    return []


def detect_winners(env, info, n_players):
    """
    1) Try Java getPlayerResults()
    2) Fall back to info hints
    """
    w = winners_from_java(env, n_players)
    if w:
        return w
    return winners_from_info(info, n_players)


def get_action_label(env, action, info):
    # 1) info -> action_names
    if isinstance(info, dict):
        names = info.get("action_names")
        if isinstance(names, (list, tuple)) and 0 <= action < len(names):
            return str(names[action])
    # 2) gym-style meanings
    try:
        meanings = env.get_action_meanings()
        if 0 <= action < len(meanings):
            return str(meanings[action])
    except Exception:
        pass
    # 3) fallback
    return str(action)


def current_mask(env, info):
    # prefer info if provided by ActionMasker
    if isinstance(info, dict) and "action_mask" in info:
        m = np.asarray(info["action_mask"], dtype=bool)
        return m
    # otherwise ask the env directly
    return get_action_mask(env)


def parse_args():
    p = argparse.ArgumentParser("Evaluate a saved PPO model with win rate & action stats")
    p.add_argument("--model", required=True, help="Path to final_model.zip or checkpoint")
    p.add_argument("--env-id", default="TAG/PowerGrid-v0")
    p.add_argument("--n-players", type=int, default=4)
    p.add_argument(
        "--opponents",
        nargs="+",
        default=None,
        help=(
            "List of opponent types for each non-python player, e.g. "
            "--opponents random mcts\n"
            "If a single value is given, it is replicated for all opponents."
        ),
    )
    p.add_argument("--obs-type", type=str, default="vector", choices=["vector", "json"])
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save plots/CSVs (default: <model_dir>/eval_<timestamp>)",
    )
    p.add_argument(
        "--only-decision-actions",
        action="store_true",
        help="Count actions only when more than one legal move exists",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ----- normalize/validate opponents -----
    allowed_opponents = {"random", "osla", "mcts"}

    if args.opponents is None:
        opponents = ["random"] * (args.n_players - 1)
    else:
        for opp in args.opponents:
            if opp not in allowed_opponents:
                raise ValueError(
                    f"Invalid opponent '{opp}'. Allowed: {sorted(allowed_opponents)}"
                )
        if len(args.opponents) == 1:
            opponents = args.opponents * (args.n_players - 1)
        elif len(args.opponents) == args.n_players - 1:
            opponents = args.opponents
        else:
            raise ValueError(
                f"For n_players={args.n_players}, you must provide either 1 opponent "
                f"type or exactly {args.n_players - 1}. Got {len(args.opponents)}."
            )
        # ----- build agent labels for plotting/CSV -----
    # Seat 0 is always the PPO python agent
    agent_labels = []
    agent_labels.append("PPO_1")  # python at seat 0

    pretty_map = {
        "random": "Random",
        "mcts": "MCTS",
        "osla": "OSLA",
    }

    for seat in range(1, args.n_players):
        opp_type = opponents[seat - 1]
        pretty = pretty_map.get(opp_type, opp_type.upper())
        # seat index is 1-based in the label
        agent_labels.append(f"{pretty}_{seat + 1}")


    # ----- output directory -----
    if args.out_dir is None:
        model_path = Path(args.model).resolve()
        base = model_path.parent
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base / f"eval_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Eval outputs] {out_dir}")
    print(f"[Opponents] python vs {opponents}")

    # ----- env & model -----
    env = make_env(
        args.env_id,
        args.max_episode_steps,
        n_players=args.n_players,
        opponents=opponents,
        obs_type=args.obs_type,
    )

    model = MaskablePPO.load(args.model, device="auto")

    rewards = []
    steps_per_episode = []
    my_wins = 0
    wins_per_player = [0] * args.n_players
    action_counts = Counter()
    action_label_last_seen = {}  # action_id -> label

    # ----- episodes loop -----
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        episode_winners = []
        step_count = 0

        while not (done or truncated):
            # get CURRENT mask
            mask = current_mask(env, info)
            valid_idxs = np.flatnonzero(mask) if mask is not None else []

            # choose action
            action, _ = model.predict(
                obs,
                deterministic=args.deterministic,
                action_masks=mask,
            )
            action = int(action)

            # is this a "decision" state (more than one legal action)?
            is_decision = (mask is None) or (mask.sum() > 1)

            # log to console when it's a nontrivial decision
            if is_decision:
                label = get_action_label(env, action, info)
                print(
                    f"Ep {ep+1} | decision | valid={len(valid_idxs)} "
                    f"first10={valid_idxs[:10].tolist()} chosen={action} ({label})"
                )

            # count actions (either always, or only at decision states)
            if (not args.only_decision_actions) or is_decision:
                action_counts[action] += 1
                label_for_store = get_action_label(env, action, info)
                action_label_last_seen[action] = label_for_store

            # step env
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            step_count += 1

            if args.render:
                env.render()

            # finished episode: detect winners
            if done or truncated:
                episode_winners = detect_winners(env, info, args.n_players)

        rewards.append(ep_reward)
        steps_per_episode.append(step_count)

        # win stats
        for w in episode_winners:
            if 0 <= w < args.n_players:
                wins_per_player[w] += 1
        if 0 in episode_winners:  # python agent is seat 0
            my_wins += 1

        # pretty print
        if episode_winners:
            if len(episode_winners) == 1:
                wstr = f"winner={episode_winners[0]}"
            else:
                wstr = f"winners={episode_winners}"
        else:
            wstr = "winner=?"
        print(f"Ep {ep+1}/{args.episodes} | reward={ep_reward:.2f} | {wstr}")

    env.close()

    # ---------- summary numbers ----------
    mean_r = float(np.mean(rewards)) if rewards else 0.0
    std_r = float(np.std(rewards)) if rewards else 0.0
    mean_steps = float(np.mean(steps_per_episode)) if steps_per_episode else 0.0
    std_steps = float(np.std(steps_per_episode)) if steps_per_episode else 0.0
    win_rate_me = 100.0 * my_wins / max(1, args.episodes)

    # per-player win rates for printing + CSV
    players = list(range(args.n_players))
    win_rates = [wins_per_player[i] / max(1, args.episodes) for i in players]

    # ---------- SUMMARY (print + save to file) ----------
    summary_lines = []
    summary_lines.append("=== Evaluation Summary ===")
    summary_lines.append(f"Episodes:      {args.episodes}")
    summary_lines.append(f"Mean Reward:   {mean_r:.3f} ± {std_r:.3f}")
    summary_lines.append(f"My Win Rate:   {win_rate_me:.1f}% ({my_wins}/{args.episodes})")
    summary_lines.append(f"Deterministic: {args.deterministic}")
    summary_lines.append(f"Avg Steps:     {mean_steps:.1f} ± {std_steps:.1f}")
    summary_lines.append("")
    summary_lines.append("Win Rates by Player:")

    for i, rate in enumerate(win_rates):
        summary_lines.append(
            f"  {agent_labels[i]}: {rate*100:.1f}% ({wins_per_player[i]}/{args.episodes})"
        )

    # Print to console
    print("\n" + "\n".join(summary_lines))

    # Save to text file
    summary_path = out_dir / "Eval_Summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSaved summary to: {summary_path}")

    # ---------- win rate chart + CSV ----------
    with open(out_dir / "win_rates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_index", "agent_label", "wins", "episodes", "win_rate"])
        for i in players:
            w.writerow([i, agent_labels[i], wins_per_player[i], args.episodes, win_rates[i]])

    plt.figure()
    # Use agent_labels so axis shows PPO_1 / Random_2 / MCTS_4, etc.
    plt.bar(agent_labels, win_rates)
    plt.title("Win Rates by Player")
    plt.xlabel("Agent (Type_Seat)")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "win_rates.png", dpi=150)
    plt.close()

    print(f"\nSaved:")
    print(f"  {out_dir / 'win_rates.png'}")
    print(f"  {out_dir / 'win_rates.csv'}")

    # ---------- bucketed action counts (discard/buy/build/run/auction) ----------
    def filter_actions_by_range(lo, hi):
        """Return list of (action_id, label, count) where lo <= id <= hi and id != 0."""
        data = []
        for a in sorted(action_counts.keys()):
            if a == 0:
                continue  # skip Pass Round globally
            if lo <= a <= hi:
                lbl = action_label_last_seen.get(a, str(a))
                cnt = action_counts[a]
                data.append((a, lbl, cnt))
        return data

    # Define semantic buckets by action_id ranges
    buckets = [
        {"name": "Discard Card", "filename_stub": "discard_card", "range": (3, 5)},
        {"name": "Buy Resource", "filename_stub": "buy_resource", "range": (6, 38)},
        {"name": "Build City", "filename_stub": "build_city", "range": (39, 85)},
        {"name": "Run Plant", "filename_stub": "run_plant", "range": (86, 135)},
        {"name": "Auction Plant", "filename_stub": "auction_plant", "range": (136, 177)},
    ]

    saved_bucket_files = []

    for bucket in buckets:
        lo, hi = bucket["range"]
        rows = filter_actions_by_range(lo, hi)
        if not rows:
            continue  # nothing from this bucket used at all

        # --- CSV for this bucket ---
        csv_path = out_dir / f"{bucket['filename_stub']}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["action_id", "label", "count"])
            for (a, lbl, cnt) in rows:
                w.writerow([a, lbl, cnt])
        saved_bucket_files.append(csv_path)

        # --- Plot for this bucket ---
        lbls = [lbl for (_, lbl, _) in rows]
        cnts = [cnt for (_, _, cnt) in rows]

        png_path = out_dir / f"{bucket['filename_stub']}.png"
        plt.figure()
        plt.bar(range(len(lbls)), cnts)
        plt.xticks(range(len(lbls)), lbls, rotation=45, ha="right")
        plt.title(f"{bucket['name']} Action Usage")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        saved_bucket_files.append(png_path)

    if saved_bucket_files:
        print("\nBucketed action files:")
        for pth in saved_bucket_files:
            print(f"  {pth}")


if __name__ == "__main__":
    main()
