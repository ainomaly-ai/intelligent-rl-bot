"""Example of defining a custom gymnasium Env to be learned by an RLlib Algorithm.

This example:
    - demonstrates how to write your own (single-agent) gymnasium Env class, define its
    physics and mechanics, the reward function used, the allowed actions (action space),
    and the type of observations (observation space), etc..
    - shows how to configure and setup this environment class within an RLlib
    Algorithm config.
    - runs the experiment with the configured algo, trying to solve the environment.

To see more details on which env we are building for this example, take a look at the
`SimpleCorridor` class defined below.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack`

Use the `--corridor-length` option to set a custom length for the corridor. Note that
for extremely long corridors, the algorithm should take longer to learn.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
You should see results similar to the following in your console output:

+--------------------------------+------------+-----------------+--------+
| Trial name                     | status     | loc             |   iter |
|--------------------------------+------------+-----------------+--------+
| PPO_SimpleCorridor_78714_00000 | TERMINATED | 127.0.0.1:85794 |      7 |
+--------------------------------+------------+-----------------+--------+

+------------------+-------+----------+--------------------+
|   total time (s) |    ts |   reward |   episode_len_mean |
|------------------+-------+----------+--------------------|
|          18.3034 | 28000 | 0.908918 |            12.9676 |
+------------------+-------+----------+--------------------+
"""
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import pandas as pd
import random
from utils import Portfolio
from utils import DataLoader


from typing import Optional

# from ray.rllib.utils.test_utils import (
#     add_rllib_example_script_args,
#     run_rllib_example_script_experiment,
# )

# parser = add_rllib_example_script_args(
#     default_reward=0.9, default_iters=50, default_timesteps=100000
# )
# parser.add_argument(
#     "--sol",
#     type=int,
#     default=1,
#     help="No: of sol to start with",
# )


class TradeEnv(gym.Env):

    """Example of a custom env in which the agent has to trade sol to different tokens as increase portfolio.
    The trading is simulated inside Portfolio class currently faking priority and quantity. The trade quantity is low 
    mimicking minimum to null impact on price. Initial stratergy is to take multiple small trades to generate more returns.
    Allowed actions are left (0) and right (1).
    The reward function is -0.01 per step taken and a uniform random value between
    0.5 and 1.5 when reaching the goal state.

    You can configure the length of the corridor via the env's config. Thus, in your
    AlgorithmConfig, you can do:
    `config.environment(env_config={"corridor_length": ..})`.
    """

    def __init__(self,data, config: Optional[dict] = None):
        config = config or {}
        # self.end_sol = 3
        # self.cur_sol = config.get("sol", 7)
        # self.token = 0
        # self.action_space = Box(0.05, 0.3, shape=(1,), dtype=np.float32)
        self.action_space = MultiDiscrete([2, 5 ,3])
        self.data = data
        # low_values = [0, 0, 0, 0, 0]
        # high_values = [3.0, 5.0, ,10,10]
        self.prev_action = None


        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.portfolio = Portfolio()
        csv_file_path = "/home/abishek/sol-proj/ray/sol-trade/output.csv"
        data_loader = DataLoader(csv_file_path)
        print(data_loader.take(1))


    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.portfolio = Portfolio()

        # self.cur_sol = self.config.get("sol", 7)
        # self.token = 0
        # Return obs and (empty) info dict.
        self.observation_space = (self.portfolio.sol, self.prev_action,  self.timestamp, self.maker, self.token, self.current_price )

        return self.observation_space, (np.float32), {"env_state": "reset"}

    def step(self, action):
        assert action[0] in [0, 1, 2], "Action must be in [0, 1,2]"
        assert action[1] in [0,1,2,3,4], "Action must be in [0, 1, 2, 3, 4]"
        assert action[2] in [0, 1, 2], "Action must be in [0,1,2]"

        p, sol_amount, slip = action

        trade_dict = {"b_s" : {0 : "buy", 1 : "sell", 2 : "hold"},
                      "sol_amt" : {0 : 0.2, 1 : 0.4, 2 : 0.6, 3 : 0.8, 4 : 1},
                      "slip" : {0 : "high", 1 : "medium", 2 : "low"}}
        
        self.portfolio.trade(trade_dict["b_s"][p], trade_dict["sol_amt"][sol_amount], trade_dict["slip"][slip])

        self.prev_sol = self.portfolio.sol

        self.prev_action = action

        # The environment only ever terminates when we reach the goal state.
        terminated = self.portfolio.sol >= 2
        truncated = False
        # Produce a random reward from [0.5, 1.5] when we reach the goal.
        reward = self.reward()
        infos = {}

        self.observation_space = (self.portfolio.sol, self.prev_action,  self.data.timestamp, self.data.maker, self.data.token, self.data.current_price )

        return (
            np.array([self.cur_sol], np.float32),
            reward,
            terminated,
            truncated,
            infos,
        )
    
    def reward(self):

        # reward for having more sol than portfolio
        if self.portfolio.sol >= 1:
            port_reward = self.portfolio.sol 

        # reward for having more sol than before and negative reward for having less sol than before
        rec_reward =  0.5 * (self.portfolio.sol - self.prev_sol) #  

        total_reward = port_reward + rec_reward

        return total_reward

    @staticmethod
    def token(names, volumes, time, prices, liquditys):
        return {
            "name": names,
            "time": time,
            "liquidity": volumes,
            "price": prices,
            "liqudity": liquditys,
        }



# if __name__ == "__main__":
#     args = parser.parse_args()

#     # Can also register the env creator function explicitly with:
#     # register_env("corridor-env", lambda config: SimpleCorridor())

#     # Or you can hard code certain settings into the Env's constructor (`config`).
#     # register_env(
#     #    "corridor-env-w-len-100",
#     #    lambda config: SimpleCorridor({**config, **{"corridor_length": 100}}),
#     # )

#     # Or allow the RLlib user to set more c'tor options via their algo config:
#     # config.environment(env_config={[c'tor arg name]: [value]})
#     # register_env("corridor-env", lambda config: SimpleCorridor(config))

#     base_config = (
#         get_trainable_cls(args.algo)
#         .get_default_config()
#         .environment(
#             TradeEnv,  # or provide the registered string: "corridor-env"
#             env_config={"sol": args.sol},
#         )
#     )

#     run_rllib_example_script_experiment(base_config, args)