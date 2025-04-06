from ray.rllib.algorithms.ppo import PPOConfig
import streamlit
from ray import tune
from ray.tune.registry import register_env
from sol_env.sol_env import TradeEnv
from pprint import pprint
import pandas as pd
import pickle 
import ray
from ray.tune.stopper import Stopper
from utils import dashboard
import gym
from stable_baselines3.common.env_util import make_vec_env


from ray.tune.search.hyperopt import HyperOptSearch

# ray.init(log_to_driver=False) 


hyperopt_search = HyperOptSearch(metric="episode_reward_mean", mode="max")

Segment =True

 
# gym.register(id='Sol-v0', entry_point='sol_env.sol_env:TradeEnv',max_episode_steps=2000)


# context = ray.init()
# print(context.dashboard_url)

# Start the dashboard thread
# dashboard.start_dashboard()


# Load your dataset
pckl_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/asc/combined_df.pkl"
with open(pckl_file_path, 'rb') as f:
                df = pickle.load(f)

num_rows = len(df)

num_envs = 14

df_id = ray.put(df)  # Place DataFrame in the object store

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

# Pre-calculate max timesteps for each segment
max_timesteps_list = []

def max_timesteps(start_index, end_index):
    # Calculate the maximum timesteps for this segment
    return (end_index - start_index) - 10

# env = make_vec_env('Sol-v0', n_envs=15, env_kwargs={"segment_index": 0, 
#                                                     "num_segments": 1, 
#                                                     "segment_indices": [(0, num_rows-10)] ,
#                                                     "df_file" :  ray.get(df_id) , 
#                                                     "eval" : False})

def env_creator(env_config):
    # Retrieve the worker index from the config
    worker_index = env_config.get("worker_index", 0)
    segment_index = worker_index % num_envs  # Ensure segment_index is within bounds
    max_timesteps_list.append(max_timesteps(start_index, end_index))
    print(f"Creating environment with segment_index: {segment_index}")
    max_episode_steps = 2000
    # Retrieve the DataFrame from the object store using its ID
    df_retrieved = ray.get(df_id)
    # return TradeEnv(segment_index, num_envs, segment_indices, {**env_config, "max_env_timesteps": max_env_timesteps})
    return TradeEnv(segment_index, num_envs, segment_indices, df_retrieved, False, max_episode_steps, env_config)

# Register each environment with a unique name
for i in range(num_envs):
    register_env(f"Trade_Env", env_creator)

# Register each environment with a unique name
# for i in range(num_envs):
#     register_env(f"Trade_Env_{i}", env_creator)



config = (
    PPOConfig()
    .environment(env="Trade_Env", env_config={"num_segments": num_envs, "segment_indices": segment_indices})  # Pass the number of segments to the environment)
    # .environment(env="Sol-v0" , #env="Trade_Env",
        # env_config={"num_segments": num_envs, "segment_indices": segment_indices})  # Pass the number of segments to the environment)
    .framework("torch")  # Use "torch" for PyTorch; "tf" if you prefer TensorFlow
    .rollouts(
        num_rollout_workers=num_envs,  # Utilize multiple CPU cores
        rollout_fragment_length="auto"  # Let Ray determine the best fragment length
    )
    .training(
        lr=tune.grid_search([0.0001, 0.0003, 0.0005]),
        # gamma=tune.uniform(0.90, 0.995),
        train_batch_size=tune.grid_search([1024,2048]),  # Adjusted to be compatible with the rollout setup
        sgd_minibatch_size=tune.grid_search([32,64]),  # Optimize for your GPU's memory size
        # lr = 0.0003,
        gamma = 0.995,
        # train_batch_size=2046,  # Adjusted to be compatible with the rollout setup
        # sgd_minibatch_size=64,  # Optimize for your GPU's memory size
        num_sgd_iter=40,  # Adjust for convergence
    )
    .resources(
        num_gpus=1,  # Use your RTX 3090 GPU
        num_cpus_per_worker=1,  # Allocate one CPU per worker
    )
)

# # Specify the log directory and logger config outside of reporting
# config.log_dir = "./results"
# config.logger_config = {
#     "loggers": ["tensorboard"]
# }

# # Set the log level directly on the configuration object
# config.log_level = "INFO"



class CustomStopper(Stopper):
    def __init__(self, num_envs, max_timestep_list):
        self.num_envs = num_envs
        self.max_timesteps_list = max_timestep_list  # Ensure this is passed during initialization
        self.stopped_trials = set()

    def __call__(self, trial_id, result):
        """
        This is the main method that Ray Tune will invoke to check if a trial should stop.
        """
        print(f"result: {result}")
        print(f"trial_id: {trial_id}")
        print(f"max_timesteps_dict keys: {list(self.max_timesteps_dict.keys())}")
        return self.should_stop(trial_id, result)

    def should_stop(self, trial_id, result):
        if not result.get("timesteps_total"):
            return False
        
        # Get the maximum timesteps from the environment
        max_env_timesteps = self.max_timesteps_list[trial_id]
        
        if result["timesteps_total"] >= max_env_timesteps:
            self.stopped_trials.add(trial_id)
        return len(self.stopped_trials) == self.num_envs

    def stop_all(self):
        return len(self.stopped_trials) == self.num_envs


# Define stopping criteria
stop_criteria = {
    # "timesteps_total": num_rows-10,  # Stop after 100,000 timesteps
    "timesteps_total": 7000000,  # Stop after 100,000 timesteps
}

print(max_timesteps_list)

# Run training
tune.run(
    "PPO",                       # Replace with your algorithm
    config=config.to_dict(), # Convert PPOConfig to dictionary
    metric="episode_reward_mean",
    mode="max",     
    stop=stop_criteria,
    log_to_file=True,
    storage_path="/home/abishek/sol-proj/ray/sol-trade/ray_results",  # Specify the log directory for TensorBoard
    verbose=1,                   # Set verbosity level (0: silent, 1: minimal, 2: detailed)
)
    # stop=CustomStopper(num_envs, max_timesteps_list)  # Use the custom stopper

# # Build the PPO algorithm
# algo = config.build()

# # Train the algorithm
# for i in range(100):
#     result = algo.train()
#     result.pop("config")
#     pprint(result)