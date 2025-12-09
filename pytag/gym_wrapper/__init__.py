# register the pyTAG environments as gym environments
import gymnasium as gym
from pytag.gym_wrapper.envs import TAGMultiplayerGym


gym.envs.register(
     id='TAG/Diamant-v0',
     entry_point='pytag.gym_wrapper.envs:TagSingleplayerGym',
     kwargs={"game_id": "Diamant", "agent_ids": ["python", "random"]}
)

gym.envs.register(
     id='TAG/SushiGo-v0',
     entry_point='pytag.gym_wrapper.envs:TagSingleplayerGym',
     kwargs={"game_id": "SushiGo", "agent_ids": ["python", "random"], "obs_type": "vector"}
)

gym.envs.register(
     id='TAG/ExplodingKittens-v0',
     entry_point='pytag.gym_wrapper.envs:TagSingleplayerGym',
     kwargs={"game_id": "ExplodingKittens", "agent_ids": ["python", "random"]}
)

gym.envs.register(
     id='TAG/TicTacToe-v0',
     entry_point='pytag.gym_wrapper.envs:TagSingleplayerGym',
     kwargs={"game_id": "TicTacToe", "agent_ids": ["python", "random"]}
)

gym.envs.register(
     id='TAG/Stratego-v0',
     entry_point='pytag.gym_wrapper.envs:TagSingleplayerGym',
     kwargs={"game_id": "Stratego", "agent_ids": ["python", "random"]}
)

gym.envs.register(
     id='TAG/LoveLetter-v0',
     entry_point='pytag.gym_wrapper.envs:TagSingleplayerGym',
     kwargs={"game_id": "LoveLetter", "agent_ids": ["python", "random"]}
)

gym.envs.register(
     id='TAG/PowerGrid-v0',
     entry_point='pytag.gym_wrapper.envs:TagSingleplayerGym',
     kwargs={
         "game_id": "PowerGrid",          # must match your GameType enum name in TAG
         "agent_ids": ["python", "random"], # you can also use ["python", "random"]
         "obs_type": "vector"             # or "json" if you prefer JSON observations
     }
)