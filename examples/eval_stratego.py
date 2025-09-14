# eval_stratego.py
import os, math, random, argparse
import numpy as np
import torch, torch.nn as nn
import gymnasium as gym

# IMPORTANT: this registers TAG/* envs and starts the JVM
import pytag.gym_wrapper  # do not remove

ENV_ID = "TAG/Stratego-v0"

# ---------- utils ----------
def sanitize_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if not np.all(np.isfinite(obs)):
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs

def get_mask(info):
    if not isinstance(info, dict):
        return None
    if info.get("action_mask") is not None:
        return np.asarray(info["action_mask"], dtype=bool)
    if info.get("mask") is not None:
        return np.asarray(info["mask"], dtype=bool)
    if info.get("legal_actions") is not None:
        aidx = np.asarray(info["legal_actions"], dtype=np.int64)
        m = np.zeros(int(aidx.max()) + 1, dtype=bool) if aidx.size else np.zeros(1, dtype=bool)
        if aidx.size: m[aidx] = True
        return m
    return None

def masked_categorical(logits: torch.Tensor, mask_np: np.ndarray):
    if mask_np is None:
        m = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
    else:
        m = torch.as_tensor(mask_np, dtype=torch.bool, device=logits.device)
        if m.numel() != logits.numel() or m.sum() == 0:
            m = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
    neg_inf = torch.tensor(-1e9, device=logits.device)
    masked_logits = torch.where(m, logits, neg_inf)
    masked_logits = torch.nan_to_num(masked_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    return torch.distributions.Categorical(logits=masked_logits)

# ---------- policy (must match training) ----------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        in_dim = max(1, int(obs_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        if x.shape[1] == 0:
            x = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
        return self.net(x)  # logits

# ---------- main ----------
def main(args):
    # single env = more stable with JPype
    env = gym.make(ENV_ID, obs_type="vector", disable_env_checker=True)
    obs, info = env.reset(seed=args.seed)

    # infer dims from env
    obs_dim = int(getattr(env.observation_space, "shape", [0])[0] or 0)
    act_dim = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # build & load policy
    policy = PolicyNet(obs_dim, act_dim).to(device)
    state_dict = torch.load(args.model, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()

    rng = np.random.default_rng(args.seed)
    ep_returns, ep_steps, ep_wins = [], [], []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(1<<31)))
        obs = sanitize_obs(obs)
        prev_score = 0.0
        done = False
        steps = 0
        ret = 0.0
        max_steps = 2000  # runaway safety

        while not done and steps < max_steps:
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            logits = policy(x)[0]
            mask = get_mask(info)
            dist = masked_categorical(logits, mask)
            action = int(dist.sample().item())

            obs, score, terminated, truncated, info = env.step(action)
            obs = sanitize_obs(obs)

            # PyTAG reward is often absolute score; use delta for return
            delta = float(score) - prev_score
            prev_score = float(score)
            ret += delta

            steps += 1
            done = bool(terminated or truncated)

        # win flag if exposed by env (may differ by game)
        has_won = False
        if isinstance(info, dict):
            if "has_won" in info:
                has_won = bool(info["has_won"])
            elif "final_info" in info and info["final_info"]:
                # some wrappers store final info here
                fi = info["final_info"]
                if isinstance(fi, (list, tuple)) and fi:
                    has_won = bool(fi[0].get("has_won", False))
                elif isinstance(fi, dict):
                    has_won = bool(fi.get("has_won", False))

        ep_returns.append(ret)
        ep_steps.append(steps)
        ep_wins.append(1.0 if has_won else 0.0)
        print(f"[Episode {ep+1}/{args.episodes}] return={ret:.3f} steps={steps} win={int(has_won)}")

    env.close()

    print("\n==== Evaluation Summary ====")
    print(f"Episodes:   {args.episodes}")
    print(f"Avg Return: {np.mean(ep_returns):.3f}")
    print(f"Avg Steps:  {np.mean(ep_steps):.1f}")
    if any(isinstance(w, (int,float)) for w in ep_wins):
        print(f"Win Rate:   {np.mean(ep_wins)*100:.1f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to stratego_policy.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    main(args)
