import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from sol_env.sol_env import TradeEnv
from ray.tune import CLIReporter

# Environment creator function
def env_creator(env_config):
    return TradeEnv(env_config)

# Register the environment
register_env("TradeEnv", env_creator)

# Define a function for Ray Tune
def tune_ppo(config):
    ppo_config = (
        PPOConfig()
        .environment(env="TradeEnv")
        .framework("torch")
        .rollouts(
            num_rollout_workers=config["num_rollout_workers"],
            rollout_fragment_length="auto"
        )
        .training(
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            num_sgd_iter=config["num_sgd_iter"]
        )
        .resources(
            num_gpus=0.5,  # Use 0.5 GPU per trial if possible; adjust based on your setup
            num_cpus_per_worker=1
        )
    )

    algo = ppo_config.build()

    for i in range(10):
        result = algo.train()
        tune.track(
            mean_reward=result["episode_reward_mean"],
            timesteps_total=result["timesteps_total"]
        )

# Ray Tune experiment setup
analysis = tune.run(
    tune_ppo,
    config={
        "num_rollout_workers": tune.grid_search([2, 4, 6]),  # Reduce the number of workers
        "train_batch_size": tune.grid_search([2048, 4096]),  # Adjusted options for batch size
        "sgd_minibatch_size": tune.grid_search([64, 128]),  # Reduce options for minibatch size
        "num_sgd_iter": tune.choice([10, 20])  # Simplify the choice
    },
    resources_per_trial=tune.PlacementGroupFactory(
        [{'CPU': 4, 'GPU': 0.5}] + [{'CPU': 1}] * 8
    ),  # Adjusted resources for each trial
    local_dir="tune_results",  # Directory to save tuning results
    metric="mean_reward",
    mode="max",
    verbose=1,
    progress_reporter=CLIReporter(
        parameter_columns=["num_rollout_workers", "train_batch_size", "sgd_minibatch_size", "num_sgd_iter"],
        metric_columns=["mean_reward", "timesteps_total"]
    )
)
