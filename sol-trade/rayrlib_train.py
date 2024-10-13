from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from sol_env.sol_env import TradeEnv


def env_creator(env_config):
    return TradeEnv(env_config)


# Register the environment
register_env("TradeEnv", env_creator)

config = {
    "env": "TradeEnv",
    "env_config": {},  # Config for your custom environment
    "num_workers": 24,  # Number of parallel environments
    "framework": "torch",  # or "tf" if you prefer TensorFlow
    "log_level": "INFO",
    "train_batch_size": 500,
    "rollout_fragment_length": 50,
    "sgd_minibatch_size": 64,
    "num_sgd_iter": 10,
}

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("TradeEnv")
    # .env_runners(num_env_runners=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    #.evaluation(evaluation_num_env_runners=1)
)

algo = config.build()  # 2. build the algorithm,

for _ in range(5):
    print(algo.train())  # 3. train it,

algo.evaluate()  # 4. and evaluate it.