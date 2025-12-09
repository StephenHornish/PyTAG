# masked_eval_callback.py

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback as SB3EvalCallback

class MaskedEvalCallback(SB3EvalCallback):
    """
    Drop-in replacement for stable_baselines3.common.callbacks.EvalCallback
    that supports action masking when calling model.predict().

    Assumes:
    - eval_env is an env (or VecEnv) wrapped with ActionMasker
      so that either:
        info["action_mask"] exists, or
        eval_env.envs[0].get_action_mask() works.
    - model is something like MaskablePPO that accepts action_masks=...
    """

    def _evaluate_policy(self, model, eval_env, n_eval_episodes, deterministic=True):
        episode_rewards = []
        episode_lengths = []

        for _ in range(n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            truncated = False
            ep_reward = 0.0
            ep_len = 0

            while not (done or truncated):
                # 1) get current legal action mask
                mask = None
                if isinstance(info, dict) and "action_mask" in info:
                    mask = np.asarray(info["action_mask"], dtype=bool)
                else:
                    # VecEnv fallback: try the first underlying env
                    try:
                        raw_mask = eval_env.envs[0].get_action_mask()
                        mask = np.asarray(raw_mask, dtype=bool)
                    except Exception:
                        mask = None

                # 2) predict using the mask
                action, _ = model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=mask,
                )

                # 3) step env forward
                obs, reward, done, truncated, info = eval_env.step(action)
                ep_reward += float(reward)
                ep_len += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        return mean_reward, std_reward
