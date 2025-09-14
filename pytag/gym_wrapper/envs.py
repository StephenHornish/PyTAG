# Gym wrappers for PyTAG (Gymnasium-compatible)
# - TagSingleplayerGym : flat Discrete + action mask, exposes action-tree shape
# - TAGMultiplayerGym  : turn-based multi-agent wrapper around MultiAgentPyTAG
#
# Requires: gymnasium>=0.26, PyTAG installed (pip install -e .), and TAG jars set up.

from typing import List, Dict, Any, Union
import numpy as np
import gymnasium as gym

from pytag import PyTAG, MultiAgentPyTAG


# -------------------------------
# Single-player wrapper
# -------------------------------
class TagSingleplayerGym(gym.Env):
    """
    Single-player Gymnasium wrapper for a PyTAG game.
    - reset(self, *, seed=None, options=None) -> (obs, info)
    - step(self, action) -> (obs, reward, terminated, truncated, info)
    Exposes action-tree meta via get_action_tree_shape().
    """
    metadata = {"render_modes": []}

    def __init__(self, game_id: str, agent_ids: List[str], seed: int = 0, obs_type: str = "vector", render_mode=None):
        super().__init__()
        self._obs_type = obs_type
        self._render_mode = render_mode
        assert agent_ids.count("python") == 1, \
            "Only one python agent is allowed in TagSingleplayerGym. Use TAGMultiplayerGym for multi-player."

        # Initialize underlying PyTAG env (single-agent façade)
        self._env = PyTAG(agent_ids=agent_ids, game_id=game_id, seed=seed, obs_type=obs_type)
        self._playerID = agent_ids.index("python")

        # One reset to size spaces and cache info/mask/tree
        _obs, _info = self._env.reset()

        # Action space (flat discrete over tree-encoded actions)
        self.action_space = gym.spaces.Discrete(int(self._env.action_space))

        # Observation space (vector length provided by PyTAG)
        obs_size = int(self._env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Cache last info/mask and the ACTION TREE SHAPE (RESTORED)
        self._last_info = _info
        self._last_action_mask = self._extract_mask(_info)
        self._action_tree_shape = self._env.get_action_tree_shape()  # <— restored

    # ---- helpers ----
    @staticmethod
    def _extract_mask(info: Dict[str, Any]) -> Union[np.ndarray, None]:
        if not isinstance(info, dict):
            return None
        if "action_mask" in info and info["action_mask"] is not None:
            return np.asarray(info["action_mask"], dtype=np.float32)
        if "mask" in info and info["mask"] is not None:
            return np.asarray(info["mask"], dtype=np.float32)
        if "legal_actions" in info and info["legal_actions"] is not None:
            aidx = np.asarray(info["legal_actions"], dtype=np.int64)
            m = np.zeros(int(aidx.max()) + 1, dtype=np.float32)
            m[aidx] = 1.0
            return m
        return None

    # ---- Gymnasium API ----
    def reset(self, *, seed=None, options=None):
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(int(seed))
        obs, info = self._env.reset()  # PyTAG returns (obs, info)
        obs = np.asarray(obs, dtype=np.float32)
        self._last_info = info
        self._last_action_mask = self._extract_mask(info)
        return obs, info

    def step(self, action: int):
        # PyTAG returns (obs, reward, done, info)
        obs, reward, done, info = self._env.step(int(action))
        obs = np.asarray(obs, dtype=np.float32)
        reward = float(reward)
        terminated = bool(done)
        truncated = False  # TAG doesn't separately expose truncation
        self._last_info = info
        self._last_action_mask = self._extract_mask(info)
        return obs, reward, terminated, truncated, info

    # ---- optional helpers ----
    def sample_rnd_action(self) -> int:
        return self._env.sample_rnd_action()

    def is_valid_action(self, action: int) -> bool:
        if self._last_action_mask is None:
            return True
        a = int(action)
        return 0 <= a < len(self._last_action_mask) and bool(self._last_action_mask[a] > 0)

    def get_action_tree_shape(self):
        """Expose TAG's conditional action-tree branching structure."""
        return self._action_tree_shape


# -------------------------------
# Multi-player wrapper (turn-based)
# -------------------------------
class TAGMultiplayerGym(gym.Env):
    """
    Turn-based multi-agent Gymnasium wrapper for MultiAgentPyTAG.

    Observations/infos/rewards are returned as dicts keyed by player id (0..N-1).
    On each step, only the *current player* acts.
    The `step()` method accepts either:
      - an `int` (action for the current player), or
      - a dict `{pid: action}`, from which the current player's action is extracted.

    API:
      reset(self, *, seed=None, options=None) -> (obs_dict, info_dict)
      step(self, action_or_dict) -> (obs_dict, reward_dict, terminated, truncated, info_dict)

    Notes:
      - `terminated` is True when the game ends. TAG doesn't distinguish truncation, so `truncated=False`.
      - Action legality should be read from the current player's info dict (e.g., info[pid]["action_mask"]).
    """
    metadata = {"render_modes": []}

    def __init__(self, game_id: str, agent_ids: List[str], seed: int = 0, obs_type: str = "vector", render_mode=None):
        super().__init__()
        self._obs_type = obs_type
        self._render_mode = render_mode
        self._players = agent_ids
        self._n_players = len(agent_ids)

        # Initialize underlying turn-based multi-agent env
        self._env = MultiAgentPyTAG(agent_ids, game_id=game_id, seed=seed, obs_type=obs_type)

        # We expose Dict spaces (per-player) to reflect the structure returned by MultiAgentPyTAG
        # Size them via one reset
        _obs, _info = self._env.reset()

        # Build per-player observation spaces (assume same vector size for all players in a given game/obs_type)
        # Fall back to PyTAG vector size if available through a helper attribute
        # If your MultiAgentPyTAG exposes per-player obs sizes directly, adapt this.
        if isinstance(_obs, dict):
            any_pid = next(iter(_obs))
            obs_dim = int(len(np.asarray(_obs[any_pid], dtype=np.float32)))
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict({pid: obs_space for pid in _obs.keys()})
        elif isinstance(_obs, (list, tuple)):
            obs_dim = int(len(np.asarray(_obs[0], dtype=np.float32)))
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict({pid: obs_space for pid in range(len(_obs))})
        else:
            raise ValueError("Unexpected observation structure from MultiAgentPyTAG.reset()")

        # Flat discrete action enumeration shared across players (TAG convention)
        # If action counts differ per player (rare), you can switch to a Dict of Discrete.
        # We infer action_dim from a mask in the first info dict that contains it.
        action_dim = None
        def _find_mask(d):
            if not isinstance(d, dict):
                return None
            for k in ("action_mask", "mask", "legal_actions"):
                if k in d and d[k] is not None:
                    return d[k]
            return None

        if isinstance(_info, dict):
            for pid, idict in _info.items():
                m = _find_mask(idict)
                if m is not None:
                    if isinstance(m, (list, tuple, np.ndarray)):
                        action_dim = int(len(m))
                    else:
                        # legal_actions as index list
                        aidx = np.asarray(m, dtype=np.int64)
                        action_dim = int(aidx.max() + 1)
                    break
        elif isinstance(_info, (list, tuple)):
            for idict in _info:
                m = _find_mask(idict)
                if m is not None:
                    if isinstance(m, (list, tuple, np.ndarray)):
                        action_dim = int(len(m))
                    else:
                        aidx = np.asarray(m, dtype=np.int64)
                        action_dim = int(aidx.max() + 1)
                    break

        if action_dim is None:
            # Fallback if mask not present at reset; you may expose a helper in MultiAgentPyTAG to query this.
            raise RuntimeError("Could not infer action_dim from info; ensure masks/legals are provided at reset.")

        self.action_space = gym.spaces.Discrete(action_dim)

        # track current player
        self._last_obs = _obs
        self._last_info = _info

    # ---- helpers ----
    def _current_pid(self) -> int:
        # MultiAgentPyTAG exposes the active player via getPlayerID()
        return int(self._env.getPlayerID())

    @staticmethod
    def _to_obs_dict(obs) -> Dict[int, np.ndarray]:
        if isinstance(obs, dict):
            return {int(k): np.asarray(v, dtype=np.float32) for k, v in obs.items()}
        elif isinstance(obs, (list, tuple)):
            return {i: np.asarray(v, dtype=np.float32) for i, v in enumerate(obs)}
        else:
            raise ValueError("Unexpected observation type from MultiAgentPyTAG")

    @staticmethod
    def _to_info_dict(info) -> Dict[int, Dict[str, Any]]:
        if isinstance(info, dict):
            return {int(k): (v if isinstance(v, dict) else {}) for k, v in info.items()}
        elif isinstance(info, (list, tuple)):
            return {i: (v if isinstance(v, dict) else {}) for i, v in enumerate(info)}
        else:
            # fall back to single dict for all (not typical)
            return {0: info if isinstance(info, dict) else {}}

    # ---- Gymnasium API ----
    def reset(self, *, seed=None, options=None):
        # MultiAgentPyTAG generally seeds via constructor; forward if available
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(int(seed))
        obs, info = self._env.reset()
        obs_d = self._to_obs_dict(obs)
        info_d = self._to_info_dict(info)
        self._last_obs, self._last_info = obs_d, info_d
        return obs_d, info_d

    def step(self, actions: Union[int, Dict[int, int], Dict[str, int]]):
        """
        Accepts either an int (action for current player), or a dict {pid: action}.
        Only the action for the current player is used this turn.
        """
        pid = self._current_pid()
        if isinstance(actions, int):
            act = int(actions)
        elif isinstance(actions, dict):
            if pid in actions:
                act = int(actions[pid])
            elif str(pid) in actions:
                act = int(actions[str(pid)])
            else:
                raise KeyError(f"No action provided for current player pid={pid}")
        else:
            raise TypeError("actions must be int or dict {pid: action}")

        # Java step
        obs, reward, done, info = self._env.step(act)

        # Normalize obs/info to dicts
        obs_d = self._to_obs_dict(obs)
        info_d = self._to_info_dict(info)

        # ---- Rewards: support dict / list / scalar ----
        if isinstance(reward, dict):
            # per-player mapping already
            rew_d = {int(k): float(v) for k, v in reward.items()}
        elif isinstance(reward, (list, tuple, np.ndarray)):
            rew_d = {i: float(r) for i, r in enumerate(reward)}
        else:
            # scalar -> assign to current player
            rew_d = {pid: float(reward)}

        # ---- Done: support dict / scalar ----
        if isinstance(done, dict):
            terminated = any(bool(v) for v in done.values())
        else:
            terminated = bool(done)

        truncated = False  # TAG doesn't distinguish truncation

        self._last_obs, self._last_info = obs_d, info_d
        return obs_d, rew_d, terminated, truncated, info_d


    def close(self):
        # Underlying Java env doesn't require explicit close, but keep for symmetry
        if hasattr(self._env, "close"):
            self._env.close()
