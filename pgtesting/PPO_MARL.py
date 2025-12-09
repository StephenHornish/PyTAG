import argparse
import numpy as np
import torch

from pytag_multiagent_wrapper import MultiAgentPyTAG
from sb3_contrib import MaskablePPO

#Runs two trained RL agents against one another in an environment 
def parse_args():
    p = argparse.ArgumentParser("Fixed-seat head-to-head: Model A (seat 0) vs Model B (seat 1)")
    # Java/Gym env config
    p.add_argument("--game-id", type=str, default="PowerGrid")
    p.add_argument("--obs-type", type=str, default="vector", choices=["vector", "json"])
    p.add_argument("--n-players", type=int, default=2)
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
    return p.parse_args()



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
        # Let policy handle preprocessing
        obs_t, _ = model.policy.obs_to_tensor(obs_batch)
        latent_pi, latent_vf = model.policy._get_latent(obs_t)  # type: ignore
        dist = model.policy._get_action_dist_from_latent(latent_pi, latent_vf)  # type: ignore

        # Try to get logits (Categorical). If probs only, take log.
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


#Fixed-seat match loop (seat 0 = A, seat 1 = B)
def run_match_fixed(policyA, policyB, *, game_id: str, obs_type: str,
                    episodes: int, seed: int, n_players: int, csv_path: str = ""):
    """
    Returns (A_wins, B_wins, draws).
    """
    import csv as _csv

    players = ["python", "python"] + ["random"] * max(0, n_players - 2)
    env = MultiAgentPyTAG(players, game_id=game_id, seed=seed, obs_type=obs_type)

    writer = None
    f = None
    if csv_path:
        f = open(csv_path, "w", newline="", encoding="utf-8")
        writer = _csv.writer(f)
        writer.writerow(["episode", "result", "reward_A", "reward_B"])

    A_wins = B_wins = draws = 0

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False

        while not done:
            pid = env.getPlayerID()
            mask = env.get_action_mask().astype(bool)

            if pid == 0:
                action = policyA(obs, mask)
            elif pid == 1:
                action = policyB(obs, mask)
            else:
                valid = np.flatnonzero(mask)
                action = int(np.random.choice(valid)) if valid.size else 0

            if not env.is_valid_action(int(action)):
                valid = np.flatnonzero(mask)
                action = int(np.random.choice(valid)) if valid.size else 0

            obs, reward, done, info = env.step(int(action))

            if done:
                rA = env.terminal_reward(0)
                rB = env.terminal_reward(1)
                if rA > rB:
                    A_wins += 1
                    outcome = "A"
                elif rB > rA:
                    B_wins += 1
                    outcome = "B"
                else:
                    draws += 1
                    outcome = "D"

                if writer:
                    writer.writerow([ep, outcome, rA, rB])

    env.close()
    if f:
        f.close()
    return A_wins, B_wins, draws


def main():
    args = parse_args()

    # Build policies
    policy_a = make_policy(args.model_a, args.deterministic)
    policy_b = make_policy(args.model_b, args.deterministic)

    # Run matches
    a_wins, b_wins, draws = run_match_fixed(
        policy_a, policy_b,
        game_id=args.game_id,
        obs_type=args.obs_type,
        episodes=args.episodes,
        seed=args.seed,
        n_players=args.n_players,
        csv_path=args.csv
    )

    total = args.episodes
    print("\n=== Head-to-Head (Fixed Seats) ===")
    print(f"Episodes : {total}")
    print(f"A wins   : {a_wins}  ({100.0 * a_wins/total:.1f}%)")
    print(f"B wins   : {b_wins}  ({100.0 * b_wins/total:.1f}%)")
    print(f"Draws    : {draws}   ({100.0 * draws/total:.1f}%)")


if __name__ == "__main__":
    main()
