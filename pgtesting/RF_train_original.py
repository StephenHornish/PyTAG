import os, math, random
import numpy as np
import torch, torch.nn as nn
import gymnasium as gym
import pytag.gym_wrapper  # registers TAG/* and starts the JVM
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
import json
import matplotlib.pyplot as plt

# ---------- config ----------
ENV_ID   = "TAG/PowerGrid-v0"
GAMMA    = 0.995
LR       = 3e-5
EPISODES = 100000
SMA_K    = 100
MAX_STEPS_PER_EP = 3000

# ---------- utils ----------
def get_action_label(env, action: int, info):
    # 1) If info dict gives us explicit names
    if isinstance(info, dict):
        names = info.get("action_names")
        if isinstance(names, (list, tuple)) and 0 <= action < len(names):
            return str(names[action])

    try:
        if hasattr(env.unwrapped, "get_action_meanings"):
            meanings = env.unwrapped.get_action_meanings()
            if 0 <= action < len(meanings):
                return str(meanings[action])
    except Exception:
        pass

    try:
        label_fn = getattr(env.unwrapped, "action_label", None)
        if callable(label_fn):
            return str(label_fn(action))
    except Exception:
        pass

    try:
        je = getattr(env.unwrapped, "_env", None)
        j  = getattr(je, "_java_env", None) if je is not None else None
        if j is not None and hasattr(j, "getActionName"):
            return str(j.getActionName(action))
    except Exception:
        pass

    return f"action_{action}"


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

# ---------- policy ----------
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
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] == 0:
            x = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
        expected = self.net[0].in_features
        if x.shape[1] != expected:
            raise ValueError(f"Observation dim mismatch: got {x.shape[1]}, expected {expected}")
        return self.net(x)

# ---------- main ----------
if __name__ == "__main__":
    # Roster
    AGENT_IDS = ["python", "random", "random"]
    NUM_PLAYERS = len(AGENT_IDS)

    # --- Output directories ---
    SAVE_DIR = os.path.join("Results", "reinforce", f"Players{NUM_PLAYERS}_EPS{EPISODES}_K{SMA_K}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_path     = os.path.join(SAVE_DIR, "powergrid_actions_log.jsonl")
    weight_path  = os.path.join(SAVE_DIR, "powergrid_policy.pt")
    returns_plot = os.path.join(SAVE_DIR, "powergrid_returns.png")
    wins_plot    = os.path.join(SAVE_DIR, "powergrid_total_wins.png")

    # --- Environment ---
    env = gym.make(
        ENV_ID,
        obs_type="vector",
        agent_ids=AGENT_IDS,
        disable_env_checker=True
    )
    obs, info = env.reset()
    obs = sanitize_obs(obs)

    obs_dim = int(obs.shape[0])
    act_dim = int(env.action_space.n)
    policy = PolicyNet(obs_dim, act_dim)
    optim  = torch.optim.Adam(policy.parameters(), lr=LR)

    ep_returns, ep_lengths = [], []
    win_counts = np.zeros(NUM_PLAYERS, dtype=int)  # <-- match roster

    # Nice labels for win bar chart
    labels = ["PythonAgent"] + [f"Random{i}" for i in range(1, NUM_PLAYERS)]

    for ep in range(EPISODES):
        obs, info = env.reset()
        obs = sanitize_obs(obs)
        logps, rewards, actions_this_ep = [], [], []
        steps, done = 0, False

        while not done and steps < MAX_STEPS_PER_EP:
            x = torch.as_tensor(obs, dtype=torch.float32)
            logits = policy(x)[0]

            mask = np.asarray(info.get("action_mask"), dtype=bool) if isinstance(info, dict) and "action_mask" in info else None
            dist = masked_categorical(logits, mask)
            action = int(dist.sample().item())
            logps.append(dist.log_prob(torch.tensor(action)))
            actions_this_ep.append(get_action_label(env, action, info))

            obs, reward, terminated, truncated, info = env.step(action)
            obs = sanitize_obs(obs)
            rewards.append(float(reward))
            done = bool(terminated or truncated)
            steps += 1

        # --- Try to detect winner directly from Java (N-player safe) ---
        wins = None
        if hasattr(env.unwrapped, "_env") and hasattr(env.unwrapped._env, "_java_env"):
            jenv = env.unwrapped._env._java_env
            try:
                raw = jenv.getPlayerResults()  # Java enum array
                results = [str(r) for r in raw] if raw is not None else []
                # Conform to NUM_PLAYERS for safe indexing
                if len(results) < NUM_PLAYERS:
                    results += ["UNKNOWN"] * (NUM_PLAYERS - len(results))
                elif len(results) > NUM_PLAYERS:
                    results = results[:NUM_PLAYERS]
                for pid, res in enumerate(results):
                    if "WIN" in res.upper():
                        win_counts[pid] += 1
                wins = results
            except Exception:
                wins = None

        # --- Policy update (REINFORCE) ---
        if logps:
            returns = compute_returns(rewards, GAMMA)
            R = torch.as_tensor(returns, dtype=torch.float32)
            std = R.std(unbiased=False)
            if std > 1e-8:
                R = (R - R.mean()) / (std + 1e-8)
            loss = -(torch.stack(logps) * R).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            if torch.isfinite(loss):
                optim.step()

        ep_ret = float(sum(rewards))
        ep_returns.append(ep_ret)
        ep_lengths.append(steps)

        # --- Logging ---
        log_row = {
            "episode": ep + 1,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "steps": steps,
            "return": ep_ret,
            "actions": actions_this_ep,
            "wins": wins if wins is not None else []
        }
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(log_row, ensure_ascii=False) + "\n")

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{EPISODES} | steps={steps} | return={ep_ret:.2f} | wins={wins}")

    # === End of training loop ===
    env.close()
    torch.save(policy.state_dict(), weight_path)
    print(f"Saved weights to: {os.path.abspath(weight_path)}")

    # ---- Plot: Returns only ----
    plt.figure(figsize=(8,4.5), dpi=140)
    ma = moving_average(ep_returns, SMA_K)
    if len(ma) > 0:
        plt.plot(range(SMA_K-1, SMA_K-1+len(ma)), ma, label=f"{SMA_K}-ep moving avg (Return)", linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Return (Total Reward)")
    plt.title("PowerGrid — REINFORCE (Training Returns)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(returns_plot)
    print(f"Saved return plot to: {os.path.abspath(returns_plot)}")

    # ---- Plot: Total Wins Bar Chart ----
    plt.figure(figsize=(max(6, 1.5*NUM_PLAYERS), 4), dpi=140)
    plt.bar(labels, win_counts)
    for i, v in enumerate(win_counts):
        plt.text(i, v + 0.5, str(v), ha="center", va="bottom", fontweight="bold")
    plt.title("Total Wins after Training"); plt.ylabel("Number of Wins")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(wins_plot)
    print(f"Saved win summary to: {os.path.abspath(wins_plot)}")
