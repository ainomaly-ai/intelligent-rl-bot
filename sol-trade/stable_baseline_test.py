# test.py
import gymnasium as gym
from sol_env import TradeEnv
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import RecurrentPPO, TRPO

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

import pickle
from utils import dashboard
import numpy as np

# dashboard.start_dashboard()

# Create the Trade environment
gym.register(id='Sol-v0', entry_point='sol_env.sol_env:TradeEnv',max_episode_steps=5000)

# Load your dataset
# pckl_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/asc/data.pkl"
pckl_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/asc/combined_df.pkl"

with open(pckl_file_path, 'rb') as f:
                df = pickle.load(f)

Segment = True
model = TRPO

num_rows = len(df)
num_envs = 30

# Calculate segment sizes for each worker
segment_size = num_rows // num_envs
remaining_rows = num_rows % num_envs

# Create a list of segment indices for each worker
segment_indices = []
start_index = 0
for i in range(num_envs):
    end_index = start_index + segment_size + (1 if i < remaining_rows else 0)
    if Segment == True :
        print(f"Segment {i} start index: {start_index}, end index: {end_index}")
        segment_indices.append((start_index, end_index))
    else:
        start_index = 0
        end_index = num_rows
        segment_indices.append((start_index, end_index))
    start_index = end_index

    # # Create an instance of the custom environment
# env = gym.make('Sol-v0',)
env = make_vec_env('Sol-v0', n_envs=num_envs, env_kwargs={"segment_index": 0,
                                                    "num_segments": 1,
                                                    "segment_indices": segment_indices ,
                                                    "df_file" : df ,
                                                    "eval" : False,
                                                    "max_steps": 5000})
env = VecMonitor(env)

class CustomTensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomTensorBoardCallback, self).__init__(verbose)
        self.episode_finished = [False] * num_envs

    def _on_step(self) -> bool:
        # Log custom metrics to TensorBoard
        # if self.n_calls % 200 == 0:  # Log every 2000 steps for more frequent updates
        # Initialize lists to accumulate values across the episode if they don't exist yet
        if not hasattr(self, 'episode_sol_values'):
            sol_values = []
            avg_usd = []
            profit= []

            for env_idx, info in enumerate(self.locals['infos']):
                if 'sol' and 'avg_usd' and 'profit' in info:
                    sol_values.append(info['sol'])
                    avg_usd.append(info['avg_usd'])
                    profit.append(info['profit'])

             # Check for episode end to log averages
        for env_idx, done in enumerate(self.locals['dones']):
            if done:  # Episode has ended for this environment
                try:
                    # Calculate averages only if we have values accumulated
                    if sol_values and avg_usd and profit:
                        avg_sol = sum(sol_values) / len(sol_values)
                        avg_episode_usd = sum(avg_usd) / len(avg_usd)
                        avg_profit = sum(profit) / len(profit)

                        log_tag_sol = f"episode/avg_sol"
                        log_tag_usd = f"episode/avg_usd"
                        log_tag_profit = f"episode/avg_profit"  # Corrected the variable name here
                        self.logger.record(log_tag_sol, avg_sol)
                        self.logger.record(log_tag_usd, avg_episode_usd)
                        self.logger.record(log_tag_profit, avg_profit)  # Updated to use the correct variable

                    # Reset episode-specific lists for the next episode
                    sol_values = []
                    avg_usd = []
                    profit = []


                # if sol_values and avg_usd and profit:  # Check if we have any valid 'sol' values to average
                #     avg_sol = sum(sol_values) / len(sol_values)
                #     avg_usd = sum(avg_usd) / len(avg_usd)
                #     avg_profit = sum(profit) / len(profit)
                #     log_tag_sol = f"rollout/avg_sol"
                #     log_tag_usd = f"rollout/avg_usd"
                #     log_tag_usd = f"rollout/avg_profit"
                #     self.logger.record(log_tag_sol, avg_sol)
                #     self.logger.record(log_tag_usd, avg_usd)
                #     self.logger.record(log_tag_usd, avg_profit)
                
                except Exception as e:
                    print(f"Error logging average reward at step {self.n_calls}: {e}")
            
            return True


net_arch = [128, 256, 512, 256, 128]  # Two hidden layers with 64 units each



if model == RecurrentPPO:
    #Define the PPO model with adjusted hyperparameters
    model = RecurrentPPO(
        'MlpLstmPolicy',
        env,
        policy_kwargs={"net_arch": net_arch},
        n_steps=2048,  # Increased number of steps per update
        batch_size=128,
        gamma=0.995,
        n_epochs=7,
        learning_rate=0.0003,  # Reduced learning rate
        clip_range=0.1,
        clip_range_vf=None,
        ent_coef=0.01,  # Increased entropy coefficient to encourage exploration
        max_grad_norm=0.5,
        target_kl=None,
        tensorboard_log='./stable_bs/ppo_tensorboard/',
        verbose=1,
        # device="cpu"
    )
elif model == PPO:
    #  #Define the PPO model with adjusted hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs={"net_arch": net_arch},
        n_steps=  2048,  # Increased number of steps per update
        batch_size= 4096,
        gamma=0.99,
        n_epochs=15,
        learning_rate= 0.0005,  # Reduced learning rate
        clip_range=0.10,
        clip_range_vf=None,
        ent_coef=  0.01,  # Increased entropy coefficient to encourage exploration
        max_grad_norm=0.5,
        target_kl=0.1,
        tensorboard_log='./stable_bs/ppo_tensorboard/',
        verbose=1,
        # device="cpu"
    )
elif model == TRPO:
    #  #Define the PPO model with adjusted hyperparameters
    model = TRPO(
    'MlpPolicy',
    env,
    learning_rate=0.003,
    n_steps=2048,
    batch_size=128,
    gamma=0.99,
    cg_max_steps=15,
    cg_damping=0.1,
    tensorboard_log='./stable_bs/ppo_tensorboard/',
    verbose=1,
    # device="cpu"
    )

    

# Train the model
callback = CustomTensorBoardCallback()
model.learn(total_timesteps=5500000, callback=callback)  # Increased training duration


# Save the model
model.save("stable_bs/saved_agents/ppo_trade")




# env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])