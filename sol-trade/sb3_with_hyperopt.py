# test.py
import gymnasium as gym
from sol_env import TradeEnv
from stable_baselines3 import PPO, A2C

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import pickle
from utils import dashboard
import numpy as np




# Create the Trade environment
gym.register(id='Sol-v0', entry_point='sol_env.sol_env:TradeEnv',max_episode_steps=2000)

# Load your dataset
# pckl_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/asc/data.pkl"
pckl_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/asc/combined_df.pkl"
with open(pckl_file_path, 'rb') as f:
                df = pickle.load(f)


# Load your dataset
pckl_eval_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/asc/eval/data_eval.pkl"
with open(pckl_eval_file_path, 'rb') as f:
                df_eval = pickle.load(f)

num_rows = len(df)
num_rows_eval=len(df_eval)
    
    # # Create an instance of the custom environment
# env = gym.make('Sol-v0',)
env = make_vec_env('Sol-v0', n_envs=5, env_kwargs={"segment_index": 0, 
                                                    "num_segments": 1, 
                                                    "segment_indices": [(0, num_rows-10)] ,
                                                    "df_file" : df , 
                                                    "eval" : False,
                                                    "max_steps": 2000})
env = VecMonitor(env)


eval_env = make_vec_env('Sol-v0', n_envs=1, env_kwargs={"segment_index": 0,
                                                    "num_segments": 1,
                                                      "segment_indices": [(0, num_rows_eval-10)], 
                                                    "df_file" : df_eval,
                                                      "eval": True,
                                                      "max_steps": 2000})


net_arch = [64, 128, 256, 128, 64]  # Two hidden layers with 64 units each

# Define the objective function
def objective(params):
    """
    Objective function to optimize with Hyperopt.
    Trains and evaluates an RL model with given hyperparameters.
    """
    model = PPO(
        policy="MlpPolicy",
        policy_kwargs={"net_arch": net_arch},
        env=env,
        learning_rate=params['learning_rate'],
        n_steps=int(params['n_steps']),
        batch_size=int(params['batch_size']),
        gamma=params['gamma'],
        ent_coef=params['ent_coef'],
        verbose=0,
    )
    
    # Train the model for a few thousand steps
    model.learn(total_timesteps=200000)
    
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    # episode_lengths = [info['episode_length'] for info in eval_env.episode_history]
    # success_rates = [info['success_rate'] for info in eval_env.episode_history]
    # print(f"Mean Episode Length: {np.mean(episode_lengths)}, Success Rate: {np.mean(success_rates)}")
    
    # Clean up
    env.close()
    eval_env.close()
    del model
    
    # Return negative reward to minimize
    return {'loss': -mean_reward, 'status': STATUS_OK}

    
# Define the hyperparameter space
space = {
    'batch_size': hp.qloguniform('batch_size', 7, 10, 1), 
    'n_steps': hp.qloguniform('n_steps', 7, 10, 1),  # Explore higher n_steps values (128 to 1024)
    'ent_coef': hp.loguniform('ent_coef', -8, -2),  # Slightly adjusted range
    'gamma': hp.uniform('gamma', 0.9, 0.999),  # Narrowed range around more typical values
    'learning_rate': hp.loguniform('learning_rate', -5, -1),  # Wider range for exploration
}

# Run the optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=25,
    trials=trials,
)

print("Best hyperparameters:", best)

with open('best_params.txt', 'w') as f:
    f.write(str(best))



#  #Define the PPO model with adjusted hyperparameters
# model = PPO(
#     'MlpPolicy',
#     env,
#     policy_kwargs={"net_arch": net_arch},
#     n_steps=4096,  # Increased number of steps per update
#     batch_size=5120,
#     gamma=0.99,
#     n_epochs=10,
#     learning_rate=0.01,  # Reduced learning rate
#     clip_range=0.2,
#     clip_range_vf=None,
#     ent_coef=0.01,  # Increased entropy coefficient to encourage exploration
#     max_grad_norm=0.5,
#     target_kl=None,
#     tensorboard_log='./stable_bs/ppo_tensorboard/',
#     verbose=1,
#     # device="cpu"
# )



# # Train the model
# callback = CustomTensorBoardCallback()
# model.learn(total_timesteps=3100000, callback=callback)  # Increased training duration


# # Save the model
# model.save("stable_bs/saved_agents/ppo_trade")




# # env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])