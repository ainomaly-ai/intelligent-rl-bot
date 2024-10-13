# test.py
import gymnasium as gym
from sol_env import TradeEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env


# Create the CartPole environment
gym.register(id='Sol-v0', entry_point='sol_env.sol_env:TradeEnv',max_episode_steps=50)
    
    # # Create an instance of the custom environment
env = gym.make('Sol-v0',)
# #env = make_vec_env('CartPole-v1', n_envs=1)

# # Define the PPO model
model = PPO('MlpPolicy', env, verbose=2, batch_size=200, device="cuda")
# model = A2C('MlpPolicy', env, verbose=2, device="cuda")

# # Train the model
model.learn(total_timesteps=2)

# # # Save the model
model.save("ppo_trade")



# env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])




