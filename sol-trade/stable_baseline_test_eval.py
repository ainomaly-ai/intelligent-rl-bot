# test.py
import gymnasium as gym
from sol_env import TradeEnv
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO, TRPO
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
import pickle
from utils import dashboard



# dashboard.start_dashboard()

# Create the CartPole environment
gym.register(id='Sol-v0', entry_point='sol_env.sol_env:TradeEnv',max_episode_steps=5000)

# Load your dataset
pckl_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/asc/eval/data_eval2.pkl"
with open(pckl_file_path, 'rb') as f:
                df = pickle.load(f)

num_rows = len(df)

sol_token_parent = "H1pqkHGyaHube3HRBJhZ4zWw8SRTXa5nDZ22bRYz2QuJ"
    
    # # Create an instance of the custom environment
# env = gym.make('Sol-v0',)
env = make_vec_env('Sol-v0', n_envs=1, env_kwargs={"segment_index": 0,
                                                    "num_segments": 1,
                                                      "segment_indices": [(0, num_rows-10)], 
                                                    "df_file" : df ,
                                                      "eval": True,
                                                    "max_steps": 5000,
                                                    "parent_id" : sol_token_parent})






# Load the trained PPO model
model_path = "stable_bs/saved_agents/ppo_trade"
model = TRPO.load(model_path, env=env)

# Evaluate the model over a number of episodes
num_episodes = 2
episode_rewards = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        print(info)
    
    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward}")


# Calculate and print the average reward over all episodes
average_reward = sum(episode_rewards) / num_episodes
print(f"Average Reward over {num_episodes} episodes: {average_reward}")





# env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])