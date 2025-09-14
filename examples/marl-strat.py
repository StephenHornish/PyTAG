# marl_selfplay_stratego.py
import argparse, math, numpy as np, torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym
import pytag.gym_wrapper  # registers TAG/*
from pytag.gym_wrapper.envs import TAGMultiplayerGym

# ---------------- utils ----------------
def sanitize_obs(obs):
    obs = np.asarray(obs, dtype=np.float32)
    if not np.all(np.isfinite(obs)):
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs

def masked_categorical(logits: torch.Tensor, mask_np):
    # logits: (B, A) or (A,)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if mask_np is None:
        m = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
    else:
        m = torch.as_tensor(mask_np, dtype=torch.bool, device=logits.device)
        if m.dim() == 1:
            m = m.unsqueeze(0).expand_as(logits)
        if m.shape != logits.shape or m.sum() == 0:
            m = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
    neg_inf = torch.tensor(-1e9, device=logits.device)
    masked = torch.where(m, logits, neg_inf)
    masked = torch.nan_to_num(masked, nan=-1e9, posinf=1e9, neginf=-1e9)
    return torch.distributions.Categorical(logits=masked)

# ---------------- model ----------------
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

# ------------- storage -------------
class RolloutBuf:
    def __init__(self, size, obs_dim, act_dim, device):
        self.size=size; self.device=device
        self.obs   = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.mask  = torch.zeros((size, act_dim), dtype=torch.bool,   device=device)
        self.act   = torch.zeros((size,),         dtype=torch.long,    device=device)
        self.logp  = torch.zeros((size,),         dtype=torch.float32, device=device)
        self.val   = torch.zeros((size,),         dtype=torch.float32, device=device)
        self.rew   = torch.zeros((size,),         dtype=torch.float32, device=device)
        self.done  = torch.zeros((size,),         dtype=torch.bool,    device=device)
        self.ptr=0
    def add(self, obs, mask_vec, act, logp, val, rew, done):
        i=self.ptr
        self.obs[i]=obs; self.mask[i]=mask_vec; self.act[i]=act
        self.logp[i]=logp; self.val[i]=val; self.rew[i]=rew; self.done[i]=done
        self.ptr+=1
    def full(self): return self.ptr>=self.size
    def reset(self): self.ptr=0

def compute_gae(rew, val, done, last_val, gamma=0.99, lam=0.95):
    T=len(rew); adv=torch.zeros_like(rew)
    lastgaelam=0.0
    for t in reversed(range(T)):
        nextnonterm = 1.0 - done[t].float()
        nextv = last_val if t==T-1 else val[t+1]
        delta = rew[t] + gamma*nextv*nextnonterm - val[t]
        lastgaelam = delta + gamma*lam*nextnonterm*lastgaelam
        adv[t]=lastgaelam
    ret = adv + val
    return adv, ret

# ------------- training loop -------------
def train_selfplay(
    episodes=200,
    rollout_len=1024,
    shared=True,                    # parameter sharing toggle
    gamma=0.99, lam=0.95,
    lr=3e-4, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
    epochs=4, minibatch=256,
    device="cpu", save_every=100, save_prefix="selfplay"
):
    env = TAGMultiplayerGym(game_id="Stratego", agent_ids=["python","python"], seed=42, obs_type="vector")
    obs_dict, info_dict = env.reset()
    pid0, pid1 = 0, 1
    obs_dim = int(len(obs_dict[pid0]))
    act_dim = int(env.action_space.n)

    # policies
    if shared:
        net = ActorCritic(obs_dim, act_dim).to(device)
        opt = optim.Adam(net.parameters(), lr=lr)
        net1 = net; opt1 = opt
    else:
        net  = ActorCritic(obs_dim, act_dim).to(device)
        net1 = ActorCritic(obs_dim, act_dim).to(device)
        opt  = optim.Adam(net.parameters(),  lr=lr)
        opt1 = optim.Adam(net1.parameters(), lr=lr)

    # buffers (per seat)
    buf0 = RolloutBuf(rollout_len, obs_dim, act_dim, device)
    buf1 = RolloutBuf(rollout_len, obs_dim, act_dim, device)

    ep = 0
    wins0=wins1=draws=0
    avgR0=[]; avgR1=[]; avgSteps=[]

    def act_with(net, obs_np, mask_np):
        x = torch.as_tensor(sanitize_obs(obs_np), dtype=torch.float32, device=device)
        logits, v = net(x)
        dist = masked_categorical(logits[0], mask_np)
        a = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(a, device=device)).detach()
        mvec = torch.as_tensor(mask_np if mask_np is not None else np.ones(act_dim,bool),
                               dtype=torch.bool, device=device)
        return x, mvec, a, logp, v.squeeze(0).detach()

    def ppo_update(buf, net, opt):
        with torch.no_grad(): last_v = torch.tensor(0.0, device=device)
        adv, ret = compute_gae(buf.rew[:buf.ptr], buf.val[:buf.ptr], buf.done[:buf.ptr], last_v, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False)+1e-8)

        idx = np.arange(buf.ptr)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, buf.ptr, minibatch):
                mb = idx[s:s+minibatch]
                obs_mb   = buf.obs[mb]
                mask_mb  = buf.mask[mb]
                act_mb   = buf.act[mb]
                old_logp = buf.logp[mb]
                adv_mb   = adv[mb]
                ret_mb   = ret[mb]

                logits, v = net(obs_mb)
                dist = masked_categorical(logits, mask_mb)   # reapply mask!
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
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

    while ep < episodes:
        # play until at least one buffer fills
        buf0.reset(); buf1.reset()
        while not (buf0.full() or buf1.full()):
            obs_dict, info_dict = env.reset()
            done=False; steps=0
            prev_scores={pid0:0.0,pid1:0.0}
            R={pid0:0.0,pid1:0.0}

            while not done:
                current = env._current_pid()
                obs = obs_dict[current]; mask = info_dict[current].get("action_mask", None)

                if current == pid0:
                    x, mvec, a, logp, v = act_with(net, obs, mask)
                    obs_dict, rew_abs, term, trunc, info_dict = env.step({current:a})
                    # delta reward for seat 0
                    r_abs = float(rew_abs.get(pid0, 0.0)); delta = r_abs - prev_scores[pid0]
                    prev_scores[pid0] = r_abs; R[pid0]+=delta
                    buf0.add(x, mvec, torch.tensor(a,device=device), logp, v,
                             torch.tensor(delta, dtype=torch.float32, device=device),
                             torch.tensor(bool(term or trunc), device=device))
                else:
                    x, mvec, a, logp, v = act_with(net1, obs, mask)
                    obs_dict, rew_abs, term, trunc, info_dict = env.step({current:a})
                    r_abs = float(rew_abs.get(pid1, 0.0)); delta = r_abs - prev_scores[pid1]
                    prev_scores[pid1] = r_abs; R[pid1]+=delta
                    buf1.add(x, mvec, torch.tensor(a,device=device), logp, v,
                             torch.tensor(delta, dtype=torch.float32, device=device),
                             torch.tensor(bool(term or trunc), device=device))

                steps+=1; done=bool(term or trunc)
                if buf0.full() or buf1.full(): break

            # determine outcome & terminal shaping on last entries if episode ended
            # try 'has_won' first
            w0=w1=None
            for pid, idict in info_dict.items():
                if isinstance(idict, dict) and 'has_won' in idict:
                    if int(pid)==pid0: w0=bool(idict['has_won'])
                    if int(pid)==pid1: w1=bool(idict['has_won'])
            if w0 is None or w1 is None:
                if R[pid0]>R[pid1]: w0,w1=True,False
                elif R[pid1]>R[pid0]: w0,w1=False,True
                else: w0,w1=False,False

            if done:
                bonus0 = 1.0 if w0 and not w1 else (-1.0 if w1 and not w0 else 0.0)
                bonus1 = -bonus0
                if buf0.ptr>0: buf0.rew[buf0.ptr-1] += bonus0
                if buf1.ptr>0: buf1.rew[buf1.ptr-1] += bonus1

                ep += 1
                wins0 += 1 if (w0 and not w1) else 0
                wins1 += 1 if (w1 and not w0) else 0
                draws += 1 if (not w0 and not w1) else 0
                avgR0.append(R[pid0]); avgR1.append(R[pid1]); avgSteps.append(steps)

                if ep % 5 == 0:
                    print(f"[Ep {ep}] win0={100*wins0/max(1,ep):.1f}% "
                          f"R0={np.mean(avgR0[-5:]):.3f} R1={np.mean(avgR1[-5:]):.3f} "
                          f"steps={np.mean(avgSteps[-5:]):.1f} "
                          f"buf0={buf0.ptr}/{buf0.size} buf1={buf1.ptr}/{buf1.size}")

                if ep % save_every == 0:
                    torch.save(net.state_dict(),  f"{save_prefix}_p0.pt")
                    torch.save(net1.state_dict(), f"{save_prefix}_p1.pt")
                    print(f"Saved: {save_prefix}_p0.pt / {save_prefix}_p1.pt")

        # ---- PPO updates ----
        if buf0.ptr>0: ppo_update(buf0, net, opt)
        if buf1.ptr>0 and (not shared): ppo_update(buf1, net1, opt1)
        elif buf1.ptr>0 and shared:
            # shared policy: update with both buffers (concatenate for a single pass)
            # simple version: run another update on net with buf1 to mix gradients
            ppo_update(buf1, net1, opt1)

    env.close()
    torch.save(net.state_dict(),  f"{save_prefix}_p0_final.pt")
    torch.save(net1.state_dict(), f"{save_prefix}_p1_final.pt")
    print("\n=== Done ===")
    print(f"Episodes={ep}  P0 win%={100*wins0/max(1,ep):.1f}  P1 win%={100*wins1/max(1,ep):.1f}  Draws={draws}")
    print(f"Avg R0={np.mean(avgR0):.3f}  Avg R1={np.mean(avgR1):.3f}  Avg steps={np.mean(avgSteps):.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--rollout-len", type=int, default=1024)
    ap.add_argument("--shared", type=int, default=1)  # 1=shared policy, 0=separate
    ap.add_argument("--device", type=str, default="cpu")  # "cuda"
    ap.add_argument("--save-prefix", type=str, default="selfplay_ppo")
    args = ap.parse_args()

    train_selfplay(
        episodes=args.episodes,
        rollout_len=args.rollout_len,
        shared=bool(args.shared),
        device=args.device,
        save_prefix=args.save_prefix,
    )
