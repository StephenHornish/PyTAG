import gymnasium as gym
import pytag.gym_wrapper  # make sure this is imported so TAG/* registers

env_id = "TAG/PowerGrid-v0"
agent_ids = ["python", "mcts", "random"]  
obs_type = "vector"

envs = []
for i in range(100):
    e = gym.make(env_id, agent_ids=agent_ids, obs_type=obs_type)
    obs, info = e.reset()
    print(f"Env {i}: len(obs) = {len(obs)}")
    envs.append(e)
