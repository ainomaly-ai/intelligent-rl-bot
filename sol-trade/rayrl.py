import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
# from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from sol_env.sol_env import TradeEnv


def env_creator(env_config):
    return TradeEnv(env_config)


# Register the environment
register_env("TradeEnv", env_creator)

# Initialize Ray
# ray.init()

# Configuration for the PPO algorithm
config = {
    "env": "TradeEnv",
    "env_config": {},  # Config for your custom environment
    "num_workers": 1,  # Number of parallel environments
    "framework": "torch",  # or "tf" if you prefer TensorFlow
    "log_level": "INFO",
    "train_batch_size": 200,
    "rollout_fragment_length": 50,
    "sgd_minibatch_size": 64,
    "num_sgd_iter": 10,
}

# Train the PPO model using Tune for hyperparameter tuning
analysis = tune.run(
    PPO,
    config=config,
    stop={"training_iteration": 10},  # Train for 10 iterations (adjust as needed)
    checkpoint_at_end=True,
)



# Output the best trial's config
print("Best hyperparameters found were: ", analysis.best_config)

# Shut down Ray
ray.shutdown()