# ppo.py
import os
import argparse
import numpy as np
import gymnasium as gym

import pytag.gym_wrapper  # registers TAG/*

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from torch import nn



def parse_args():
    p = argparse.ArgumentParser("PPO (SB3 MaskablePPO) for TAG/PowerGrid-v0")
    # Game settings
    p.add_argument("--env-id", type=str, default="TAG/PowerGrid-v0")
    p.add_argument("--n-players", type=int, default=4)
    p.add_argument("--opponent", type=str, default="random", choices=["random", "osla", "mcts"])
    p.add_argument("--obs-type", type=str, default="vector", choices=["vector", "json"])
    # Training runtime
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    # Infra
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--logdir", type=str, default="./ppo_pg_logs")
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--use-subproc", action="store_true")
    p.add_argument("--policy", type=str, default="auto")
    return p.parse_args()


def get_action_mask(env) -> np.ndarray:
    u = env.unwrapped
    if hasattr(u, "get_action_mask"):
        m = u.get_action_mask()
        return np.asarray(m, dtype=bool)
    for attr in ("_last_info", "last_info", "info"):
        info = getattr(u, attr, None)
        if isinstance(info, dict) and "action_mask" in info:
            return np.asarray(info["action_mask"], dtype=bool)
    return np.ones(env.action_space.n, dtype=bool)


def make_single_env(env_id: str, max_steps: int, *, n_players: int, opponent: str, obs_type: str):
    assert n_players >= 2, "Need at least 2 players (1 learner + 1 bot)."
    agent_ids = ["python"] + [opponent] * (n_players - 1)

    def thunk():
        e = gym.make(env_id, agent_ids=agent_ids, obs_type=obs_type)
        e = TimeLimit(e, max_episode_steps=max_steps)
        e = ActionMasker(e, get_action_mask)
        return e
    return thunk


def choose_policy(sample_env: gym.Env, requested: str) -> str:
    if requested != "auto":
        return requested
    return "MultiInputPolicy" if isinstance(sample_env.observation_space, gym.spaces.Dict) else "MlpPolicy"


def next_run_name(base_dir: str, base_name: str) -> str:
    """
    Returns base_name if unused; otherwise base_name-1, base_name-2, ...
    """
    candidate = base_name
    idx = 1
    while os.path.exists(os.path.join(base_dir, candidate)):
        idx += 1
        candidate = f"{base_name}-{idx}"
    return candidate


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    # ---- Construct auto run name: PPO_np{players}_ns{n_steps}-k ----
    base_name = f"PPO_np{args.n_players}_ns{args.n_steps}"
    run_name = next_run_name(args.logdir, base_name)
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[Run] {run_name}")
    print(f"[Players] {args.n_players} | [Opponent] {args.opponent} | [n_envs] {args.n_envs} | [n_steps] {args.n_steps}")
    print(f"[Logs] {run_dir}")

    # Build vectorized training envs
    env_fns = [
        make_single_env(
            args.env_id, args.max_episode_steps,
            n_players=args.n_players, opponent=args.opponent, obs_type=args.obs_type
        )
        for _ in range(args.n_envs)
    ]
    Vec = SubprocVecEnv if args.use_subproc else DummyVecEnv
    vec_env = Vec(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(run_dir, "train"))

    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=5.0
    )

    eval_env = DummyVecEnv([
    make_single_env(
        args.env_id, args.max_episode_steps,
        n_players=args.n_players, opponent=args.opponent, obs_type=args.obs_type
        )
    ])
    eval_env = VecMonitor(eval_env, filename=os.path.join(run_dir, "eval"))

    # 🧩 Keep eval env fixed (no normalization updates)
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=False,
        norm_reward=False
    )
    # Auto-select policy
    tmp = make_single_env(
        args.env_id, args.max_episode_steps,
        n_players=args.n_players, opponent=args.opponent, obs_type=args.obs_type
    )()
    policy_type = choose_policy(tmp, args.policy)
    tmp.close()

    # Build model
    model = MaskablePPO(
        policy_type,
        vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.lr,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.logdir,  # parent dir for TB
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.ReLU,
            ortho_init=False,
        ),
    )

    # Callbacks with run-specific folders
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best"),
        log_path=run_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
        save_path=os.path.join(run_dir, "ckpts"),
        name_prefix=run_name,  # includes np/ns
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Train (set tb_log_name so TB subdir matches run_name)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_cb, ckpt_cb],
        tb_log_name=run_name
    )

    # Save final model into run dir
    model.save(os.path.join(run_dir, "final_model"))
    vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))


    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
