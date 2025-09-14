# train_vs_fixed_opponent.py
import os, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import pytag.gym_wrapper  # registers TAG/* and starts JVM
from pytag.gym_wrapper.envs import TAGMultiplayerGym

# --------------------- utils ---------------------
def to_numpy(x): return x.detach().cpu().numpy()
def sanitize_obs(obs):
    obs = np.asarray(obs, dtype=np.float32)
    if not np.all(np.isfinite(obs)):
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs

def masked_categorical(logits: torch.Tensor, mask_np):
    if mask_np is None:
        m = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
    else:
        m = torch.as_tensor(mask_np, dtype=torch.bool, device=logits.device)
        if m.numel() != logits.numel() or m.sum() == 0:
            m = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
    neg_inf = torch.tensor(-1e9, device=logits.device)
    logits = torch.where(m, logits, neg_inf)
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    return torch.distributions.Categorical(logits=logits)

# ----------------- nets -----------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(max(1, obs_dim), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v  = nn.Linear(hidden, 1)

    def forward(self, x):
        if x.ndim == 1: x = x.unsqueeze(0)
        if x.shape[1] == 0:
            x = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)

class FixedTorchOpponent:
    """Frozen opponent loaded from stratego_policy.pt"""
    def __init__(self, model_path, obs_dim, act_dim, device="cpu"):
        self.device = torch.device(device)
        # opponent can be same architecture as learner's policy head
        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        sd = torch.load(model_path, map_location="cpu")
        # allow loading a pure policy .pt (if it only has pi weights)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        self.net.eval()

    def act(self, obs_np, mask_np):
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        logits, _ = self.net(x)
        dist = masked_categorical(logits[0], mask_np)
        return int(dist.sample().item())

# --------------- PPO buffer (learner-only turns) ---------------
class RolloutBuffer:
    def __init__(self, size, obs_dim, device):
        self.size = size
        self.obs   = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.mask  = torch.zeros((size,), dtype=torch.bool, device=device)  # flattened against act_dim at step time
        self.actions = torch.zeros((size,), dtype=torch.long, device=device)
        self.logp   = torch.zeros((size,), dtype=torch.float32, device=device)
        self.val    = torch.zeros((size,), dtype=torch.float32, device=device)
        self.rew    = torch.zeros((size,), dtype=torch.float32, device=device)
        self.done   = torch.zeros((size,), dtype=torch.bool, device=device)
        self.ptr = 0

    def add(self, obs, mask, action, logp, val, rew, done):
        i = self.ptr
        self.obs[i]    = obs
        self.mask[i]   = True if mask is not None else False
        self.actions[i]= action
        self.logp[i]   = logp
        self.val[i]    = val
        self.rew[i]    = rew
        self.done[i]   = done
        self.ptr += 1

    def full(self): return self.ptr >= self.size
    def reset(self): self.ptr = 0

# --------------- GAE advantage ---------------
def compute_gae(rew, val, done, last_val, gamma=0.99, lam=0.95):
    T = len(rew)
    adv = torch.zeros_like(rew)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - done[t].float()
        nextvalue = last_val if t == T-1 else val[t+1]
        delta = rew[t] + gamma * nextvalue * nextnonterminal - val[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + val
    return adv, ret

# --------------- training ---------------
def train_ppo_vs_fixed(
    model_path="stratego_policy.pt",
    episodes=200,
    rollout_len=1024,             # learner decisions per update
    epochs=4,
    minibatch=256,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    lr=3e-4,
    device="cpu",
    save_every=50,
    save_path="learner_ppo.pt",
):

    # --- make multi-agent env
    env = TAGMultiplayerGym(
        game_id="Stratego",
        agent_ids=["python", "python"],  # PID0 learner, PID1 opponent
        seed=42,
        obs_type="vector",
    )

    # infer shapes
    obs_dict, info_dict = env.reset()
    pid0, pid1 = 0, 1
    obs_dim = int(len(obs_dict[pid0]))
    act_dim = int(env.action_space.n)

    # nets
    ac = ActorCritic(obs_dim, act_dim).to(device)
    opt = optim.Adam(ac.parameters(), lr=lr)

    opponent = FixedTorchOpponent(model_path, obs_dim, act_dim, device=device)

    # rollout storage (only learner turns)
    buf = RolloutBuffer(rollout_len, obs_dim, device)

    # tracking
    ep = 0
    learner_wins = 0
    opp_wins = 0
    draws = 0
    avg_return = []
    avg_steps  = []

    while ep < episodes:
        # play episodes until buffer is full with learner decisions
        buf.reset()
        while not buf.full():
            obs_dict, info_dict = env.reset()
            done = False
            steps = 0
            prev_scores = {pid0: 0.0, pid1: 0.0}
            ret = {pid0: 0.0, pid1: 0.0}

            while not done:
                current = env._current_pid()

                # current player's obs/mask
                obs_np = sanitize_obs(obs_dict[current])
                mask_np = info_dict[current].get("action_mask", None)

                if current == pid0:
                    # learner acts + store transition
                    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                    logits, val = ac(x)
                    dist = masked_categorical(logits[0], mask_np)
                    action = int(dist.sample().item())
                    logp = dist.log_prob(torch.tensor(action, device=device))
                    obs_dict, rew_abs, terminated, truncated, info_dict = env.step({current: action})

                    # delta reward for learner
                    r_abs = float(rew_abs.get(pid0, 0.0))
                    delta = r_abs - prev_scores[pid0]
                    prev_scores[pid0] = r_abs
                    ret[pid0] += delta

                    # store transition for learner
                    buf.add(
                        obs=x,
                        mask=mask_np,
                        action=torch.tensor(action, device=device),
                        logp=logp.detach(),
                        val=val.squeeze(0).detach(),
                        rew=torch.tensor(delta, dtype=torch.float32, device=device),
                        done=torch.tensor(bool(terminated or truncated), device=device),
                    )
                else:
                    # opponent acts; learner does not store a transition
                    action = opponent.act(obs_np, mask_np)
                    obs_dict, rew_abs, terminated, truncated, info_dict = env.step({current: action})
                    # track opponent return (delta)
                    r_abs = float(rew_abs.get(pid1, 0.0))
                    ret[pid1] += (r_abs - prev_scores[pid1])
                    prev_scores[pid1] = r_abs

                steps += 1
                done = bool(terminated or truncated)

                if buf.full():  # enough learner steps to update
                    break

            # determine winner (best-effort)
            wins = {}
            for pid, idict in info_dict.items():
                if isinstance(idict, dict) and 'has_won' in idict:
                    wins[int(pid)] = bool(idict['has_won'])

            learner_won = wins.get(pid0, None)
            opp_won     = wins.get(pid1, None)
            if learner_won is None or opp_won is None:
                if ret[pid0] > ret[pid1]:
                    learner_won, opp_won = True, False
                elif ret[pid1] > ret[pid0]:
                    learner_won, opp_won = False, True
                else:
                    learner_won, opp_won = False, False

            if learner_won and not opp_won: learner_wins += 1
            elif opp_won and not learner_won: opp_wins += 1
            else: draws += 1

            ep += 1
            avg_return.append(ret[pid0])
            avg_steps.append(steps)

            if ep % 5 == 0:
                print(f"[Ep {ep}] R(learner)={np.mean(avg_return[-5:]):.3f} "
                      f"win%={100.0*learner_wins/max(1,ep):.1f} "
                      f"steps(avg)={np.mean(avg_steps[-5:]):.1f}  buf={buf.ptr}/{buf.size}")

            if ep % save_every == 0:
                torch.save(ac.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

            if buf.full():  # leave to update
                break

        # ---------- PPO update ----------
        with torch.no_grad():
            last_val = torch.tensor(0.0, device=device)

        adv, ret = compute_gae(buf.rew[:buf.ptr], buf.val[:buf.ptr], buf.done[:buf.ptr], last_val, gamma, gae_lambda)
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # prepare minibatches
        idxs = np.arange(buf.ptr)
        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, buf.ptr, minibatch):
                mb = idxs[start:start+minibatch]
                obs_mb   = buf.obs[mb]
                act_mb   = buf.actions[mb]
                old_logp = buf.logp[mb]
                adv_mb   = adv[mb]
                ret_mb   = ret[mb]

                logits, v = ac(obs_mb)
                # build dists with masks (we don't reapply mask here to avoid train-time bias;
                # if you want to enforce legality even in training, you can store masks per step)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5*(ret_mb - v.squeeze(-1)).pow(2).mean()
                loss = policy_loss + vf_coef*value_loss - ent_coef*entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()

    env.close()
    torch.save(ac.state_dict(), save_path)
    print(f"\nTraining complete. Saved learner to {save_path}")
    print(f"Final win% (learner): {100.0*learner_wins/max(1,ep):.1f}  "
          f"episodes={ep}  avg_return={np.mean(avg_return):.3f}  avg_steps={np.mean(avg_steps):.1f}")

# ----------------- CLI -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", type=str, default="stratego_policy.pt", help="path to fixed opponent .pt")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")  # "cuda" if you want
    ap.add_argument("--save", type=str, default="learner_ppo.pt")
    args = ap.parse_args()

    train_ppo_vs_fixed(
        model_path=args.opponent,
        episodes=args.episodes,
        device=args.device,
        save_path=args.save,
    )
