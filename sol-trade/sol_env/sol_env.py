""" Environment for RL trading with solana

The format for the current input csv data is "token, timestamp, buy/sell, tokenprice, Usdamount, amountToken, maker"


"""
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random
from utils import Portfolio
from utils import DataLoaderMevx
from utils import DataLoader
import requests
from utils.dashboard import update_dashboard
import time
from collections import deque, Counter
import talib
import asyncio
import aiohttp
import json
import sys

from typing import Optional
Debug =   False
EVAL_STREAM = True

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

    """

    def __init__(self,segment_index: int, num_segments: int, 
                 segment_indices: list, 
                 df_file: pd.DataFrame,  
                 eval: bool, 
                 max_steps:int,
                 parent_id: str,
                 config: Optional[dict] = None):
        
        config = config or {}
        # self.action_space = MultiDiscrete([2, 5 ,3])
        self.action_space = MultiDiscrete([2, 5])
        # self.max_episode_steps = max_episode_steps                           # Initialize max_episode_steps
        self.window_size = 50

        #Defining previous params
        self.prev_action = None
        self.save_env_state = None
        self.prev_token= None
        self.prev_token_price= None
        self.gain_percent = None
        self.prev_gain_percent = None
        self.profit_wallet = 0
        self.usd_value_avg_queue = deque(maxlen=30)  # Queue with a maximum length of 30
        self.last_agent_trade_action = None 
        self.max_steps = max_steps
        self.parent_id = parent_id

        self.scaler = MinMaxScaler()

        if EVAL_STREAM:
            self.portfolio = Portfolio(self.parent_id)
        else:
            self.portfolio = Portfolio()


        csv_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/output.csv"
        pckl_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/asc/data.pkl"
        pckl_file_path_eval = "/home/abishek/sol-proj/ray/sol-trade/data/asc/eval/data_eval.pkl"

        # data_loader = DataLoader(csv_file_path)
        print(len(df_file))
        if eval and not isinstance(df_file, pd.DataFrame):
            self.data_loader = DataLoaderMevx(None,pckl_file_path_eval,True)
            
        else:    
            # self.data_loader = DataLoaderMevx(None,pckl_file,True)
            self.data_loader = DataLoaderMevx(None,None,df_file,True)
        print("Data loader created.....")
        self.data = self.data_loader.ray_load(eval)

         # Split the data into segments
        self.total_data = self.data
        print(f"num segments.............. .{segment_index, num_segments}")
        self.segment_indices = segment_indices
        self.segment_index = segment_index
        self.segment_start, self.segment_end = segment_indices[segment_index]
        # segment_length = len(self.data) // num_segments
        # self.segment_start = segment_index * segment_length
        # self.segment_end = self.segment_start + segment_length
        #   final data for the specific environment after splitting
        self.data = self.total_data.iloc[self.segment_start:self.segment_end].reset_index(drop=True)

        # print(self.data.take(1))
        
        self.random_trade_prob = 0.1  # Probability of making a trade (initially set to low value to encourage trading)
        

        # n_features = self.data.shape[1]

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(55,6), dtype=np.float64)

################################# reset ##################################################

    def reset(self, *, seed=None, options=None):
        # Run your async reset function synchronously
        seed = seed
        options = options
        result = asyncio.get_event_loop().run_until_complete(self.reset_async(seed =seed, options=options))
        return result


    async def reset_async(self, *, seed=None, options=None):
        
        random.seed(seed)

        self.counter = Counter(next(iter(d.keys())) for d in self.portfolio.trade_log)
        print(f"Total buys: {self.counter['buy']}")
        print(f"Total sells: {self.counter['sell']}")
        self.prev_counter = None

        if EVAL_STREAM:
            self.portfolio = Portfolio(self.parent_id)
        else:
            self.portfolio = Portfolio()

        self.usd_value_avg_queue.clear()  # Clear the queue for a fresh start
        self.gain_percent = None
        self.prev_gain_percent = None
        self.profit_wallet = 0


        """Reset the environment to the initial state."""
        if self.save_env_state is None:
            self.current_step = self.window_size
        else:
            self.current_step = self.save_env_state[0]
        # Return the initial window of past data points (for all features)

        # If test change to realtime stream data
        if EVAL_STREAM:
            print(self.parent_id)
            windowed_data = await self.start_server_request(self.parent_id,50)
            windowed_data = np.array(windowed_data)
        else:    
            windowed_data = np.array((self.data[self.current_step - self.window_size:self.current_step]))


        # print(windowed_data)
        if EVAL_STREAM:
            last_trade = np.array(windowed_data[-1])
            last_trade = np.reshape(last_trade, (1, -1))
        else:
            last_trade = np.array(self.data[self.current_step-1:self.current_step])
   
        last_trade2 = np.array(self.data[self.current_step-1:self.current_step])
        print(last_trade, last_trade2)
        # print(len(self.data))
        # print(len(windowed_data))
        # print(f"arrrrr2_np : {last_trade[0,0]}")

        self.token = last_trade[0,0]
        self.token_price = last_trade[0,3], 
        self.last_trade_type = last_trade[0,2], 
        self.last_usd_amount = last_trade[0,4], 
        self.last_trade_token_sol = last_trade[0,5]

        print(self.token_price, last_trade2[0,3])

        sol, tokens, token_value, usd_value, self.profit_wallet = self.portfolio.get_state()


        trade_info ={"token_price" : self.token_price,"buy_sell" : self.last_trade_type,"usd_amount" : self.last_usd_amount,"token_sol_amount" : self.last_trade_token_sol}
        trade_info2 = {"sol" : self.portfolio.sol, "tokens": tokens, "token_value" : token_value, "usd_value" :usd_value, "avg_usd": usd_value, "profit" : self.profit_wallet}
        # arr3 = np.array([self.window_size])

        sol_left = np.array([self.portfolio.sol])

        avl_tokens = np.array([self.portfolio.token_value])

   
        # arr4 = self.prev_action

        if self.prev_action is None:
            prev_action = np.array([0] * len(self.action_space)) 
        else:
            prev_action = np.array(self.prev_action)

        if self.prev_token is None:
            self.prev_token = self.token

        if self.prev_token_price is None:
            self.prev_token_price = self.token_price

        # Calculating the RSI

        prices = windowed_data[:,3].astype(float)
        
        
        # Filtering for the last 14 prices
        if Debug:
            print(f"Prev token : {self.prev_token},current token :  {self.token}")
            print(f"window {windowed_data[-15 , 3]} , {windowed_data[-1 , 3]}")
        
        rsi = np.array(self.talib_fun(self.token, self.prev_token,windowed_data, prices))

        if Debug:
            print(f"RSI {rsi}")


        

        # arr2_pad = np.pad(arr2, (0,6), mode='constant',constant_values=(0,0))
        sol_left_pad = np.pad(sol_left, (0,5), mode='constant',constant_values=(0,0))
        prev_action_pad = np.pad(prev_action, (0,4), mode='constant',constant_values=(0,0))
        
        # Pad the array with zeros to make its length a multiple of 6
        pad_length = (6 - len(rsi) % 6) % 6
        padded_rsi = np.concatenate([rsi, np.zeros(pad_length)])

        # Reshape into (x, 6)
        reshaped_rsi = padded_rsi.reshape(-1, 6)

        # rsi_reshaped = rsi.reshape(2,7)
        # avl_tokens_pad = np.pad(avl_tokens,(0,(7-len(avl_tokens))), mode='constant',constant_values=(0,0))

        # Normalize windowed_data
        normalized_windowed_data = self.scaler.fit_transform(windowed_data)

        if Debug:
            print(reshaped_rsi.shape, normalized_windowed_data.shape, sol_left_pad.shape)

        if Debug:
            # Debugging statements
            print(f"Shape of normalized_windowed_data: {normalized_windowed_data.shape}")
            print(f"Shape of sol_left_pad: {sol_left_pad.shape}, { sol_left_pad.reshape(1, 6).shape}")
            print(f"Shape of prev_action_pad: {prev_action_pad.shape},{ prev_action_pad.reshape(1, 6).shape}")


        combined = np.concatenate([normalized_windowed_data, sol_left_pad.reshape(1, 6), prev_action_pad.reshape(1, 6),reshaped_rsi], axis=0)

        if Debug:
            # Debugging statements
            print(f"Shape of combined observation_reset: {combined.shape}")


        

        # Ensure the combined observation is of type float64
        combined = combined.astype(np.float64)

#         reset_obs = []

# # # Return obs and (empty) info dict.

        print(f"done reset!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        return combined, {"env_state": "reset", "trade_info" : trade_info2}


######################################Step###################################################



    def step(self, action):
        # Run your async setp function synchronously
        result = asyncio.get_event_loop().run_until_complete(self.step_async(action=action))
        return result

    async def step_async (self, action,):

        self.counter = Counter(next(iter(d.keys())) for d in self.portfolio.trade_log)
        print(f"Total buys: {self.counter['buy']}")
        print(f"Total sells: {self.counter['sell']}")

        self.prev_counter = self.counter
        
        assert action[0] in [0, 1, 2], "Action must be in [0, 1,2]"
        assert action[1] in [0,1,2,3,4], "Action must be in [0, 1, 2, 3, 4]"
        # assert action[2] in [0, 1, 2], "Action must be in [0,1,2]" # will not be used, remove later

        p, sol_amount = action #, slip
        h = 0

      

        """ If the token changes, set action to sell all tokens"""
        if self.prev_token is not  None and self.token != self.prev_token:
            h = 2
        elif next(iter(self.portfolio.token_value.values()), {}).get('token_sol_amount', 0) > 0 and next(val['token_sol_amount'] for val in self.portfolio.token_value.values()) > 150:
            h= 3
        else:
            h= 0

        trade_dict = {"b_s" : {0 : "buy", 1 : "sell", 2 : "hold"},
                      "sol_amt" : {0 : 0.2, 1 : 0.4, 2 : 0.6, 3 : 0.8, 4 : 1},
                      "slip" : {0 : "high", 1 : "medium", 2 : "low"}}
        
        # print(self.data[self.current_step-1:self.current_step])
        
        #slippage simulate after changing from tuple 
        slippage_percentage = ((self.token_price[0] - self.prev_token_price[0]) / self.prev_token_price[0]) * 100

        if h == 0 and (self.prev_token_price[0] and self.token_price[0])!= 0:
            if slippage_percentage <= 0.15:
                slip = 2
            elif slippage_percentage <= 0.50:
                slip = 1
            elif slippage_percentage >=0.50 :
                slip = 0
        else: 
            slip = 2

        if eval:
            self.random_trade_prob = 0.9  # Probability of making a trade (set to near 0 for eval)
        else:
            # Decrease random trade probability over time
            self.random_trade_prob = max(0.01, 0.1 * (0.99 ** (self.current_step / 100000))) # Adjust the divisor and initial value as needed

        print(f"HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE:{self.prev_token, self.token, h}")
        print(self.counter)
        if next(iter(self.portfolio.token_value.values()), {}).get('token_sol_amount', 0) > 0.01 and h == 2 : #sell all as token changes
                # self.portfolio.update_values(self.token_price[0], self.token, next(val['total_tokens_left'] for val in self.portfolio.tokens.values())) # updating values before executing trade
                print("Selling all the tokens ................")
                self.last_agent_trade_action = self.portfolio.trade(self.prev_token, trade_dict["b_s"][1], next(val['token_sol_amount'] for val in self.portfolio.token_value.values()),self.prev_token_price[0], trade_dict["slip"][0])
                self.portfolio.token_value = {}
                self.portfolio.tokens ={}
                print(self.portfolio.token_value)
                print(self.last_agent_trade_action)
                sys.exit
        

        
        if random.random() < self.random_trade_prob:  # Randomly decide whether to trade or not based on the probability
            if next(iter(self.portfolio.token_value.values()), {}).get('token_sol_amount', 0) > 0.01 and h ==3 : # Automatically sell if the agent is in 50 dollar profit 
                self.portfolio.update_values(self.token_price[0], self.token, next(val['total_tokens_left'] for val in self.portfolio.tokens.values())) # updating values before executing trade
                self.last_agent_trade_action = self.portfolio.trade(self.token, trade_dict["b_s"][1], trade_dict["sol_amt"][sol_amount],self.token_price[0], trade_dict["slip"][0]) 
            elif next(iter(self.portfolio.token_value.values()), {}).get('token_sol_amount', 0) > 0.01:
                print(self.portfolio.token_value)
                self.portfolio.update_values(self.token_price[0], self.token, next(val['total_tokens_left'] for val in self.portfolio.tokens.values())) # updating values before executing trade
                self.last_agent_trade_action = self.portfolio.trade(self.token, trade_dict["b_s"][p], trade_dict["sol_amt"][sol_amount],self.token_price[0], trade_dict["slip"][slip])
            else:
                # self.portfolio.update_values(self.token_price[0], self.token, self.portfolio.token_value["token_sol_amount"]) # updating values before executing trade
                self.last_agent_trade_action = self.portfolio.trade(self.token, trade_dict["b_s"][p], trade_dict["sol_amt"][sol_amount],self.token_price[0], trade_dict["slip"][slip])
        else:
            print("Random function: no trade")
            pass  # Do nothing (i.e., hold the position)

        # self.portfolio.trade(trade_dict["b_s"][p], trade_dict["sol_amt"][sol_amount],self.token_price, trade_dict["slip"][slip])


        # Add in function to auto sell if the portfolio reaches certain amount 

        if self.portfolio.get_usd_value() > 350 and self.last_agent_trade_action != "sold":  # Check if USD value exceeds $350 and last action wasn't selling
            trade_dict = {
                "b_s": ["sell"],  # Action: Sell
                "sol_amt": [0.2],  # Amount in SOL to sell
                "slip": [slip]     # Slippage (assuming 'slip' is defined elsewhere for slippage value)
            }
            
            self.last_agent_trade_action = self.portfolio.trade(
                self.token, 
                trade_dict["b_s"][0], 
                trade_dict["sol_amt"][0], 
                self.token_price[0], 
                trade_dict["slip"][0]
            )
            print("Sold 0.2 SOL of token as portfolio USD value exceeded $350.")
        else:
            # Existing logic for other conditions or actions (buy, hold, etc.)
            pass

        

        self.prev_sol = self.portfolio.sol

        self.prev_action = action

        self.prev_token = self.token

        self.prev_token_price = self.token_price


        # The environment only ever terminates when we reach the goal state.
        terminated1 = self.current_step >= len(self.data) - 1  #Terminate for reaching end of dataset
        terminated2 = self.profit_wallet >= 2 # Terminate for success
        terminated3 = self.portfolio.sol < 0  # Terminate for 0 balace
        terminated4 = True if self.current_step % self.max_steps == 0 else False # Terminate for end of episode timestep
        terminated = any([terminated1, terminated2, terminated3, terminated4])
        truncated = False
        if terminated:
            if terminated1:
                print("terminated : end of dataset")
            elif terminated2:
                print("terminated : success...")
            elif terminated3:
                print("terminated : sol less than zero")    
        done = terminated or truncated
      
        # Produce a random reward from [0.5, 1.5] when we reach the goal.
        reward = self.reward()
        

        # self.observation_space = (self.data[self.current_step - self.window_size:self.current_step], self.portfolio.sol, self.prev_action)

        obs = []
        # print(self.current_step)

        # Starting observation here ...
        if EVAL_STREAM:
            windowed_data = await self.start_server_request(self.parent_id,50)
            windowed_data = np.array(windowed_data)
            # print(windowed_data)
        else:
            windowed_data = np.array((self.data[self.current_step - self.window_size:self.current_step]))


        if EVAL_STREAM:
            last_trade = np.array(windowed_data[-1])
            last_trade = np.reshape(last_trade, (1, -1)) # reshaping to dimenstion of csv data
            # print(f"last tradeeee {last_trade}")
        else:
            last_trade = np.array(self.data[self.current_step-1:self.current_step])
        # print(len(self.data))
        # print(len(windowed_data))
        # print(f"arrrrr2_np : {last_trade[0,0]}")
                
        self.token = last_trade[0,0]
        self.token_price = last_trade[0,3], 
        self.last_trade_type = last_trade[0,2], 
        self.last_usd_amount = last_trade[0,4], 
        self.last_trade_token_sol = last_trade[0,5]

        
        print(self.portfolio.token_value)
        if next(iter(self.portfolio.token_value.values()), {}).get('token_sol_amount', 0) > 0.01:
                        # print("Updating token values for the step #################")
                        # print(self.token_price[0], self.token,next(val['total_tokens_left'] for val in self.portfolio.tokens.values()))
                        if self.prev_token != self.token:
                            None
                        else:    
                            self.portfolio.update_values(self.token_price[0], self.token, next(val['total_tokens_left'] for val in self.portfolio.tokens.values())) # updating values 

        

        sol, tokens, token_value, usd_value, self.profit_wallet = self.portfolio.get_state()
        # update_dashboard(sol, tokens, token_value, usd_value)

        # Append new value to the queue (right side, following FIFO principle)
        self.usd_value_avg_queue.append(usd_value)

        if self.usd_value_avg_queue:
                avg_usd = sum(self.usd_value_avg_queue)/len(self.usd_value_avg_queue)
            

        if self.current_step % 10 == 0:  # Print every 10 steps
            print(f"Step {self.current_step}:")
            print(f"SOL: {self.portfolio.sol:.2f}")
            print(f"Tokens: {self.portfolio.tokens}")
            print(f"USD Value: {self.portfolio.usd_value:.2f}")
            print("--------------------")
            # time.sleep(0.1)  # Optional: brief delay for readability

        
                      
        trade_info ={"token_price" : self.token_price,"buy_sell" : self.last_trade_type,"usd_amount" : self.last_usd_amount,"token_sol_amount" : self.last_trade_token_sol}
        trade_info2 = {"sol" : self.portfolio.sol, "tokens": tokens, "token_value" : token_value, "usd_value" :usd_value, "avg_usd": avg_usd, "profit" : self.profit_wallet}

        infos = trade_info2

        sol_left = np.array([self.portfolio.sol])

        avl_tokens = np.array([self.portfolio.token_value])



        if self.prev_action is None:
            prev_action = np.array([0] * len(self.action_space)) 
        else:
            prev_action = np.array(self.prev_action)


        sol_left_pad = np.pad(sol_left, (0,5), mode='constant',constant_values=(0,0))
        prev_action_pad = np.pad(prev_action, (0,4), mode='constant',constant_values=(0,0))

        # Calculating the RSI

        prices = windowed_data[:,3].astype(float)

        
        # Filtering for the last 14 prices
        # print(self.prev_token, self.token)
        # print(windowed_data[-15 , 0] , windowed_data[-1 , 0])
        rsi = np.array(self.talib_fun(self.token, self.prev_token,windowed_data, prices))
        
             # Pad the array with zeros to make its length a multiple of 6
        pad_length = (6 - len(rsi) % 6) % 6
        padded_rsi = np.concatenate([rsi, np.zeros(pad_length)])

        # Reshape into (x, 6)
        reshaped_rsi = padded_rsi.reshape(-1, 6)
    

        # avl_tokens_pad = np.pad(avl_tokens,(0,(7-len(avl_tokens))), mode='constant',constant_values=(0,0))


        # obs = [windowed_data, sol_left_pad, prev_action_pad]

        #  print(obs)

        # Normalize windowed_data
        normalized_windowed_data = self.scaler.transform(windowed_data)

        obs_combined = np.concatenate([normalized_windowed_data, sol_left_pad.reshape(1, 6), prev_action_pad.reshape(1, 6), reshaped_rsi], axis=0)

        if Debug:
            # Debugging statements
            print(f"Shape of normalized_windowed_data: {normalized_windowed_data.shape}")
            print(f"Shape of sol_left_pad: {sol_left_pad.shape}")
            print(f"Shape of prev_action_pad: {prev_action_pad.shape}")
            print(f"Shape of combined observation_step: {obs_combined.shape}")

        # Ensure the combined observation is of type float64
        obs_combined = obs_combined.astype(np.float64)

        # print("Observation space:", (obs_combined))
        

        self.current_step += 1


        self.save_env_state = [self.current_step, self.portfolio.sol]


        if terminated or truncated:
            print(f"Total Reward for the episode is {reward}")
            print(self.portfolio.sol)


        return (
            obs_combined,
            reward,
            # done,
            truncated,
            terminated,
            infos,
        )
    
    def reward(self):

        # TO do :  slippage consideration, profitable trades reward 

        sol_priority_weight=0.8,  # Adjust to prioritize SOL increase (range: 0 to 1)
        threshhold_usd=150,   # Desired minimum USD value
        usd_stagnation_threshold=30  # timesteps without significant USD value increase
        

# **SOL Increase Reward**
        def sol_increase_reward(current_sol, previous_sol):
            if current_sol > previous_sol: 
                return 1.0 
            elif current_sol < previous_sol: 
                return -0.5 
            else:
                return 0.0


# **USD Value Reward with Stagnation Penalty**
        def usd_value_reward_with_stagnation_penalty(current_usd, avg_usd_queue):
            if len(avg_usd_queue)>10:
                avg_usd = sum(avg_usd_queue)/len(avg_usd_queue)

                self.prev_gain_percent = self.gain_percent

                self.gain_percent = ((current_usd - avg_usd) / avg_usd) * 100


                if current_usd > avg_usd:
                    return 0.8
                else:
                    return -0.5
            else:
                return 0

        def threshold_Reward(current_usd):
            if current_usd < threshhold_usd[0]:
                return -2.0  # Harsh penalty for low USD value
            else:
                return 0
            
        def short_trade_reward(current_sol, prev_sol):
            # Calculate the percentage increase in SOL
            percent_increase = ((current_sol - prev_sol) / prev_sol) * 100

            
            # Only provide a reward if there is a 20% or greater increase in SOL
            if self.prev_gain_percent and percent_increase is not None:
                if self.prev_gain_percent >= 40 and  percent_increase >= 20:
                    return 2.0
                else:
                    return 0.0
            else:
                return 0.0
            
        def sol_milestone_reward(profit_sol):
            reward = 0.0
            milestones = [i * 0.1 for i in range(21)]
            
            for milestone in milestones:
                if profit_sol > milestone:
                    reward += milestone*3.0
            
            return reward
        
         # **New Reward Component: Sell Incentive for > $350 USD**
        def sell_incentive_reward(current_usd, sold_tokens_sol):
            threshold_usd = 350.0
            target_sold_sol = 0.2
            
            if current_usd > threshold_usd:
                if sold_tokens_sol > (target_sold_sol * 0.9) and sold_tokens_sol < (target_sold_sol * 1.1):  # ±10% tolerance
                    return 5.0  # Substantial reward for meeting the sell target near $300+ USD
                elif sold_tokens_sol > 0:
                    return 1.0  # Some reward for selling, but not at the target amount
            return 0.0
        
        def trade_penality():
            if self.counter["buy"] > self.prev_counter["buy"]:
                return -0.1
            

            # **Compute Rewards**
        sol_reward = sol_increase_reward(self.portfolio.sol, self.prev_sol)

        short_trade_reward_sol = short_trade_reward(self.portfolio.sol, self.prev_sol)

        usd_reward = usd_value_reward_with_stagnation_penalty(
            self.portfolio.get_usd_value(), 
            self.usd_value_avg_queue)
        
        threshold_port_reward = threshold_Reward(self.portfolio.get_usd_value()) 

        sol_profit_reward = sol_milestone_reward(self.profit_wallet)

        success_reward = 30 if self.profit_wallet > 1 else 0.0

        trade_pen = trade_penality()

        usd_total_reward = usd_reward + threshold_port_reward + success_reward + trade_pen

        
        # Calculate sold tokens in SOL (assuming `self.last_trade_token_sol` is the amount sold in the last trade)
        # sold_tokens_sol = 0.2 if self.last_agent_trade_action == "sold" else 0
        sold_tokens_sol = 0.2 if self.prev_counter["sell"] < self.counter["sell"] else 0
        
        sell_incentive = sell_incentive_reward(self.portfolio.get_usd_value(), sold_tokens_sol)

        # **Combine Rewards with Priority Weighting**
        total_reward = (sol_priority_weight[0] * sol_reward) + ((1 - sol_priority_weight[0]) * usd_total_reward)  + sol_profit_reward + short_trade_reward_sol + sell_incentive


        # max_reward = self.portfolio.sol + 0.5 * self.portfolio.sol +1  # Maximum combined reward

        # normalized_total_reward = total_reward / max_reward  

        return total_reward
    
    def max_timesteps(self):
        # Calculate the maximum timesteps for this segment
        start_index, end_index = self.segment_indices[self.segment_index]
        return (end_index - start_index) - self.window_size
    
    def talib_fun(self, token, prev_token,windowed_data, prices):
        if token != prev_token:
            rsi = [0]*14
        elif windowed_data[-15 , 0] != windowed_data[-1 , 0]: # check if prev token in last 14th place == present token 
            rsi = [0]*14
        else:
             rsi = talib.RSI(prices, timeperiod=14)
             rsi = rsi[-15:-1]

        return rsi
    
    async def get_trades_from_server(self,parent_id, limit=50):
        payload = {
            'parent': parent_id,
            'limit': limit,
            'order_by': "timestamp desc"
        }

        async with aiohttp.ClientSession() as session:
            url = 'http://localhost:8000/get_trades'
            # response = await session.post(url, json=payload)
            async with session.post(url, json=payload) as response:
                print("Request sent; awaiting response...")

        # response = requests.post('http://localhost:8000/get_trades', json=payload)
                if response.status == 200:
                    chunk_list = []

                    async for chunk in response.content.iter_any():  # Read chunks as they arrive
                            if chunk:
                                # chunk_list.append(chunk.decode("utf-8"))  # Decode and collect JSON chunks
                                print(f"Received chunk: {chunk.decode('utf-8')[:100]}...")  # Debug first 100 chars
                                break
                            
                            if len(chunk) == 0:  # Check if chunk is empty (indicates stream end)
                                break
                    

                    # full_data = "".join(chunk_list)  # Combine all chunks into a full JSON string
                    # print(len(full_data))
                    try:
                        data = json.loads(chunk)  # Parse JSON
                        df = pd.DataFrame(data)
                        df = df.astype(float)
                        print("Received JSON data:")
                        # print(f"head<< {df.head()}")
                        # print(f"tail<<<<< {df.tail()}")
                        return df
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        return None
                    
                else:   
                    print(f"Error: {response.status}")
                    return None
                
    async def start_server_request(self, parent_id, limit):
        try:
            result = await asyncio.wait_for(
                self.get_trades_from_server(parent_id),
                timeout=180
            )
        except asyncio.TimeoutError:
            print("Timed out waiting for server response.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        else:
            # print(f"Reult : {result}")
            return result

            