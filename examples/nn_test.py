# stratego_reinforce_with_plot.py
import math, random
import numpy as np
import torch, torch.nn as nn
import gymnasium as gym
import pytag.gym_wrapper  # registers TAG/*
import matplotlib
matplotlib.use("Agg")  # render without needing a GUI
import matplotlib.pyplot as plt
import os

ENV_ID  = "TAG/Stratego-v0"
GAMMA   = 0.99
LR      = 3e-4
EPISODES = 5000
SMA_K    = 100  # moving-average window

# ----------------- utils -----------------
def sanitize_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if not np.all(np.isfinite(obs)):
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs

def masked_categorical(logits: torch.Tensor, mask_np: np.ndarray) -> torch.distributions.Categorical:
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

def compute_returns(rewards, gamma=GAMMA):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))

def moving_average(x, k=SMA_K):
    if len(x) == 0:
        return np.array([])
    k = max(1, min(k, len(x)))
    w = np.ones(k) / k
    return np.convolve(np.asarray(x, dtype=float), w, mode="valid")

# ----------------- policy -----------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        in_dim = max(1, int(obs_dim))  # handle 0-dim feature stubs
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

# ----------------- main -----------------
if __name__ == "__main__":
    env = gym.make(ENV_ID, obs_type="vector", disable_env_checker=True)
    obs, info = env.reset()

    obs_dim = int(getattr(env.observation_space, "shape", [0])[0] or 0)
    act_dim = int(env.action_space.n)

    policy = PolicyNet(obs_dim, act_dim)
    optim  = torch.optim.Adam(policy.parameters(), lr=LR)

    ep_returns = []
    ep_lengths = []

    for ep in range(EPISODES):
        obs, info = env.reset()
        obs = sanitize_obs(obs)

        logps, rewards = [], []
        steps, done = 0, False

        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32)
            logits = policy(x)[0]

            mask = np.asarray(info.get("action_mask"), dtype=bool) if "action_mask" in info else None
            dist = masked_categorical(logits, mask)

            action = int(dist.sample().item())
            logps.append(dist.log_prob(torch.tensor(action)))

            obs, reward, terminated, truncated, info = env.step(action)
            obs = sanitize_obs(obs)
            rewards.append(float(reward))
            done = bool(terminated or truncated)
            steps += 1

            if steps > 2000:  # runaway safety
                break

        # REINFORCE update
        if logps:
            returns = compute_returns(rewards, GAMMA)
            returns_t = torch.as_tensor(returns, dtype=torch.float32)
            std = returns_t.std(unbiased=False)
            if std > 1e-8:
                returns_t = (returns_t - returns_t.mean()) / (std + 1e-8)

            loss = -(torch.stack(logps) * returns_t).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            if torch.isfinite(loss):
                optim.step()

        ep_ret = float(sum(rewards))
        ep_returns.append(ep_ret)
        ep_lengths.append(steps)
        print(f"Episode {ep+1}/{EPISODES}  steps={steps}  return={ep_ret:.2f}")

    env.close()
    torch.save(policy.state_dict(), "stratego_policy.pt")
    print(f"Saved weights to: {os.path.abspath('stratego_policy.pt')}")

    # -------- Plot & save learning curve --------
    plt.figure(figsize=(8,4.5), dpi=140)
    ma = moving_average(ep_returns, SMA_K)
    if len(ma) > 0:
        plt.plot(range(SMA_K-1, SMA_K-1+len(ma)), ma, label=f"{SMA_K}-ep moving avg", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Stratego — REINFORCE (masked)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png = "stratego_returns.png"
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved plot to: {os.path.abspath(out_png)}")
