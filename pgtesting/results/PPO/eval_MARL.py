#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import csv as _csv
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # add this before importing pyplot
import matplotlib.pyplot as plt

from pytag import MultiAgentPyTAG
from sb3_contrib import MaskablePPO
from pathlib import Path
from datetime import datetime
import os


def parse_args():
    p = argparse.ArgumentParser("Fixed-seat head-to-head: Model A (seat 0) vs Model B (seat 1)")
    # Java/Gym env config
    p.add_argument("--game-id", type=str, default="PowerGrid",
                   help="Java GameType name (e.g., PowerGrid, SushiGo, Diamant)")
    p.add_argument("--obs-type", type=str, default="vector", choices=["vector", "json"])
    p.add_argument("--n-players", type=int, default=2,
                   help="Total seats in the Java game. Seats 0 and 1 are python; others become random.")
    # Evaluation
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Greedy actions instead of sampling")
    # Models
    p.add_argument("--model-a", type=str, required=True, help="Path to Model A (.zip)")
    p.add_argument("--model-b", type=str, required=True, help="Path to Model B (.zip)")
    # Optional CSV
    p.add_argument("--csv", type=str, default="",
                   help="If set, write per-episode results to this CSV path")
    # Outputs
    p.add_argument("--out-dir", type=str, default=None,
                   help="Directory to save per-agent action CSVs/plots (default: alongside model A)")
    p.add_argument("--only-decision-actions", action="store_true",
                   help="Count actions only when more than one legal move exists")
    # Opponents
    p.add_argument(
        "--opponents",
        nargs="+",
        default=None,
        help=("List of opponent types for remaining non-python seats "
              "(random, mcts, osla). Example:\n"
              "--opponents mcts mcts random"),
    )
    # Display names
    p.add_argument("--name-a", type=str, default="Model_A",
                   help="Display name for seat 0 (default: Model_A)")
    p.add_argument("--name-b", type=str, default="Model_B",
                   help="Display name for seat 1 (default: Model_B)")
    return p.parse_args()


# ---------- SB3 -> callable(obs, mask) adapter ----------

def _predict_with_masks_sb3(model: MaskablePPO, obs_vec: np.ndarray,
                            mask_bool: np.ndarray, deterministic: bool) -> int:
    """
    Attempts model.predict(..., action_masks=...) (newer sb3-contrib).
    Falls back to manual masked logits if unavailable.
    """
    obs_batch = np.asarray(obs_vec, dtype=np.float32).reshape(1, -1)
    mask_batch = np.asarray(mask_bool, dtype=bool).reshape(1, -1)

    # Fast path if your sb3-contrib version accepts action_masks
    try:
        action, _ = model.predict(
            obs_batch,
            deterministic=deterministic,
            action_masks=mask_batch
        )
        return int(action[0])
    except TypeError:
        pass

    # Fallback: manual masking on logits
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs_batch)
        latent_pi, latent_vf = model.policy._get_latent(obs_t)  # type: ignore
        dist = model.policy._get_action_dist_from_latent(latent_pi, latent_vf)  # type: ignore

        if hasattr(dist.distribution, "logits"):
            logits = dist.distribution.logits.clone()
        else:
            probs = dist.distribution.probs
            logits = torch.log(probs + 1e-8)

        invalid = ~torch.as_tensor(mask_batch)
        logits[invalid] = -1e9

        if deterministic:
            action_t = torch.argmax(logits, dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            action_t = torch.multinomial(probs, 1).squeeze(1)

        return int(action_t.cpu().numpy()[0])


def make_policy(model_path: str, deterministic: bool):
    model = MaskablePPO.load(model_path, device="auto")

    def _policy(obs: np.ndarray, mask: np.ndarray) -> int:
        return _predict_with_masks_sb3(model, obs, mask, deterministic)

    return _policy


# ---------- Fixed-seat match loop (seat 0 = A, seat 1 = B) ----------

def run_match_fixed(
    policyA, policyB, *,
    game_id: str,
    obs_type: str,
    episodes: int,
    seed: int,
    n_players: int,
    opponents=None,
    csv_path: str = "",
    out_dir: Path = None,
    only_decision_actions: bool = False,
    log_decisions: bool = False,
    name_a="Model_A",
    name_b="Model_B",
):
    """
    Plays seat 0 (A) vs seat 1 (B). Logs per-episode winner CSV (if csv_path set),
    and saves per-agent action usage CSVs + per-bucket comparison plots (two bars per action).
    """

    # ---------- setup ----------
    if opponents is None:
        players = ["python", "python"] + ["random"] * max(0, n_players - 2)
    else:
        if len(opponents) == 1:
            opponents = opponents * (n_players - 2)
        elif len(opponents) != (n_players - 2):
            raise ValueError(
                f"For n_players={n_players}, you must specify 1 or {n_players-2} opponents."
            )
        players = ["python", "python"] + opponents

    env = MultiAgentPyTAG(players, game_id=game_id, seed=seed, obs_type=obs_type)

    writer = None
    f = None
    if csv_path:
        f = open(csv_path, "w", newline="", encoding="utf-8")
        writer = _csv.writer(f)
        writer.writerow(["episode", "result", "reward_A", "reward_B"])

    # per-agent action accounting
    action_counts_A = Counter()
    action_counts_B = Counter()
    action_label_last_seen_A = {}  # id -> label
    action_label_last_seen_B = {}

    A_wins = B_wins = 0
    wins_per_player = [0] * n_players
    other_wins = 0
    unknown_outcomes = 0
    episode_rewards_A = []
    episode_rewards_B = []
    episode_steps = []

    for ep in range(1, episodes + 1):
        step_count = 0

        obs, info = env.reset()
        done = False

        while not done:
            pid = env.getPlayerID()
            mask = env.get_action_mask().astype(bool)
            valid_idxs = np.flatnonzero(mask)

            action_names = info.get("action_names") if isinstance(info, dict) else None

            if pid == 0:
                action = policyA(obs, mask)
            elif pid == 1:
                action = policyB(obs, mask)
            else:
                action = int(np.random.choice(valid_idxs)) if valid_idxs.size else 0

            if not env.is_valid_action(int(action)):
                action = int(np.random.choice(valid_idxs)) if valid_idxs.size else 0

            is_decision = (mask.sum() > 1)
            if (not only_decision_actions) or is_decision:
                lbl = (action_names[action] if action_names and 0 <= action < len(action_names)
                       else str(action))
                if pid == 0:
                    action_counts_A[action] += 1
                    action_label_last_seen_A[action] = lbl
                elif pid == 1:
                    action_counts_B[action] += 1
                    action_label_last_seen_B[action] = lbl

            obs, reward, done, info = env.step(int(action))
            if(pid ==0 ): #dont double count steps for two agents makes game appear twice as long
                step_count += 1

            if done:
                # --- winners across all seats ---
                winners = None
                if hasattr(env, "terminal_rewards"):
                    term_rewards = env.terminal_rewards()
                    winners = [i for i, r in enumerate(term_rewards) if r > 0]
                if not winners:
                    # robust fallback: pick argmax over terminal_reward(i)
                    all_r = [env.terminal_reward(i) for i in range(n_players)]
                    max_r = max(all_r)
                    winners = [i for i, r in enumerate(all_r) if r == max_r]

                # record per-seat wins
                for w in winners:
                    if 0 <= w < n_players:
                        wins_per_player[w] += 1

                if 0 in winners:
                    A_wins += 1
                    outcome = "A"
                elif 1 in winners:
                    B_wins += 1
                    outcome = "B"
                elif winners:
                    other_wins += 1
                    outcome = "O"
                else:
                    unknown_outcomes += 1
                    outcome = "?"

                rA = env.terminal_reward(0)
                rB = env.terminal_reward(1)
                episode_rewards_A.append(rA)
                episode_rewards_B.append(rB)
                episode_steps.append(step_count)

                # --- Pretty Episode Summary ---
                winner_name = (
                    name_a if (0 in winners)
                    else name_b if (1 in winners)
                    else f"Player{winners[0]}" if winners else "?"
                )

                print(
                    f"Ep {ep}/{episodes} finished in {step_count} steps | "
                    f"Winner: {winner_name} | rA={rA:.2f} rB={rB:.2f}"
                )

                if writer:
                    writer.writerow([ep, outcome, rA, rB])


    # ---------- save per-agent action logs (after all episodes) ----------
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        def save_actions_csv(prefix, counts, labels):
            path = out_dir / f"actions_{prefix}.csv"
            with open(path, "w", newline="", encoding="utf-8") as g:
                w = _csv.writer(g)
                w.writerow(["action_id", "label", "count"])
                for a in sorted(counts.keys()):
                    w.writerow([a, labels.get(a, str(a)), counts[a]])
            return path

        pathA = save_actions_csv(name_a, action_counts_A, action_label_last_seen_A)
        pathB = save_actions_csv(name_b, action_counts_B, action_label_last_seen_B)

        # bucket definitions
        buckets = [
            {"name": "Discard Card", "stub": "discard_card", "range": (3, 5)},
            {"name": "Buy Resource", "stub": "buy_resource", "range": (6, 38)},
            {"name": "Build City",   "stub": "build_city",   "range": (39, 85)},
            {"name": "Run Plant",    "stub": "run_plant",    "range": (86, 135)},
            {"name": "Auction Plant","stub": "auction_plant","range": (136, 177)},
        ]

        def rows_in_range(counts, labels, lo, hi):
            rows = []
            for a in sorted(counts.keys()):
                if a == 0:
                    continue  # optional: skip Pass
                if lo <= a <= hi:
                    rows.append((a, labels.get(a, str(a)), counts[a]))
            return rows

        saved_files = []
        for b in buckets:
            lo, hi = b["range"]
            rowsA = rows_in_range(action_counts_A, action_label_last_seen_A, lo, hi)
            rowsB = rows_in_range(action_counts_B, action_label_last_seen_B, lo, hi)

            # Write separate CSVs per agent (unchanged)
            def write_bucket_csv(prefix_name, rows):
                csv_file = out_dir / f"{b['stub']}_{prefix_name.replace(' ', '_')}.csv"
                with open(csv_file, "w", newline="", encoding="utf-8") as g:
                    w = _csv.writer(g)
                    w.writerow(["action_id", "label", "count"])
                    for (aid, lbl, cnt) in rows:
                        w.writerow([aid, lbl, cnt])
                return csv_file

            if rowsA:
                saved_files.append(write_bucket_csv(name_a, rowsA))
            if rowsB:
                saved_files.append(write_bucket_csv(name_b, rowsB))

            # Build union of action_ids in this bucket
            idsA = {aid for (aid, _, _) in rowsA}
            idsB = {aid for (aid, _, _) in rowsB}
            all_ids = sorted(idsA.union(idsB))
            if not all_ids:
                continue  # nothing to plot for this bucket

            # For each action id, get counts from A and B (0 if missing) and a good label
            countsA = []
            countsB = []
            labels = []
            # make quick dicts
            dictA = {aid: (lbl, cnt) for (aid, lbl, cnt) in rowsA}
            dictB = {aid: (lbl, cnt) for (aid, lbl, cnt) in rowsB}
            for aid in all_ids:
                lblA, cntA = dictA.get(aid, (None, 0))
                lblB, cntB = dictB.get(aid, (None, 0))
                # prefer a non-empty label from A, else B, else fallback to id
                lbl = lblA or lblB or str(aid)
                labels.append(lbl)
                countsA.append(cntA)
                countsB.append(cntB)

            # Side-by-side bar chart per action (two bars per action)
            x = np.arange(len(all_ids))
            width = 0.4
            fig, ax = plt.subplots(figsize=(max(8, len(all_ids) * 0.4), 5))
            ax.bar(x - width/2, countsA, width, label=name_a)  # different colors automatically per bar call
            ax.bar(x + width/2, countsB, width, label=name_b)
            ax.set_title(f"{b['name']} — per action")
            ax.set_xlabel("Action")
            ax.set_ylabel("Count")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90)
            ax.legend()
            fig.tight_layout()
            png_path = out_dir / f"{b['stub']}_per_action_{name_a.replace(' ','_')}_vs_{name_b.replace(' ','_')}.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            saved_files.append(png_path)

        print("\nSaved per-agent action logs and comparison plots:")
        print(f"  {pathA}")
        print(f"  {pathB}")
        for p in saved_files:
            print(f"  {p}")

    if f:
        f.close()
    env.close()

    return A_wins, B_wins, wins_per_player, other_wins, unknown_outcomes, episode_rewards_A, episode_rewards_B, episode_steps


def main():
    args = parse_args()

    # Base output dir (default: alongside model A, or user-specified)
    if args.out_dir is None:
        base = Path(args.model_a).resolve().parent
        base_out = base / "MARL_Eval_results"
    else:
        base_out = Path(args.out_dir).resolve()

    # Build subfolder: MARL_<nPlayers>Player_<timestamp>
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_name = f"MARL_{args.n_players}Player_{stamp}"
    out_dir = base_out / subfolder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Head-to-head outputs] {out_dir}")

    # Build policies
    policy_a = make_policy(args.model_a, args.deterministic)
    policy_b = make_policy(args.model_b, args.deterministic)
    csv_path = str((out_dir / args.csv)) if args.csv else ""

    # Run matches
    a_wins, b_wins, wins_per_player, other_wins, unknown_outcomes, rewards_a, rewards_b, steps = run_match_fixed(
        policy_a, policy_b,
        game_id=args.game_id,
        obs_type=args.obs_type,
        episodes=args.episodes,
        seed=args.seed,
        n_players=args.n_players,
        opponents=args.opponents,
        csv_path=csv_path,
        out_dir=out_dir,
        only_decision_actions=args.only_decision_actions,
        name_a=args.name_a,
        name_b=args.name_b,
    )

    total = args.episodes
    print("\n=== Head-to-Head (Fixed Seats) ===")
    print(f"Episodes : {total}")
    print(f"{args.name_a} wins : {a_wins}  ({100.0 * a_wins/total:.1f}%)")
    print(f"{args.name_b} wins : {b_wins}  ({100.0 * b_wins/total:.1f}%)")
    print(f"Other wins : {other_wins}  ({100.0 * other_wins/total:.1f}%)")
    if unknown_outcomes:
        print(f"Unknown outcomes : {unknown_outcomes}")

    # Per-player win rates (like single-agent script)
    players = list(range(args.n_players))
    win_rates = [wins_per_player[i] / max(1, total) for i in players]

    # Save CSV + bar plot to the same out_dir
    with open(out_dir / "win_rates.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["player", "wins", "episodes", "win_rate"])
        for i in players:
            w.writerow([i, wins_per_player[i], total, win_rates[i]])

    # ---------- summary stats ----------
    mean_rA = np.mean(rewards_a) if rewards_a else 0.0
    std_rA  = np.std(rewards_a)  if rewards_a else 0.0
    mean_rB = np.mean(rewards_b) if rewards_b else 0.0
    std_rB  = np.std(rewards_b)  if rewards_b else 0.0
    mean_steps = np.mean(steps) if steps else 0.0
    std_steps  = np.std(steps)  if steps else 0.0

    print(f"\n=== Evaluation Summary ===")
    print(f"Episodes:     {total}")
    print(f"{args.name_a} mean reward: {mean_rA:.2f} ± {std_rA:.2f}")
    print(f"{args.name_b} mean reward: {mean_rB:.2f} ± {std_rB:.2f}")
    print(f"Avg Steps:    {mean_steps:.1f} ± {std_steps:.1f}")

    # Build friendly x-labels
    x_labels = []
    for i in players:
        if i == 0:
            x_labels.append(name_a)
        elif i == 1:
            x_labels.append(name_b)
        else:
            # the corresponding opponent type (from args.opponents)
            opp = args.opponents[i-2] if args.opponents else "random"
            x_labels.append(f"Opponent_{i} ({opp})")

    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, win_rates)
    plt.title("Win Rates by Player")
    plt.xlabel("Agent")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "win_rates.png", dpi=150)
    plt.close()


    print(f"\nSaved:")
    print(f"  {out_dir / 'win_rates.png'}")
    print(f"  {out_dir / 'win_rates.csv'}")


if __name__ == "__main__":
    main()
