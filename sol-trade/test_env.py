# test.py
import gymnasium as gym
from sol_env import TradeEnv
import pickle 


# Load your dataset
pckl_file_path = "/home/abishek/sol-proj/ray/sol-trade/data/asc/data.pkl"
with open(pckl_file_path, 'rb') as f:
                df = pickle.load(f)

def main():
    # Register the custom environment
    gym.register(id='Sol-v0', entry_point='sol_env.sol_env:TradeEnv')

    sol_token_parent = "EtcwBb9fT7YYAgpkVz6AG1QZwPJxS79Tg2r7TszYuP72"

    
    # Create an instance of the custom environment
    env = gym.make('Sol-v0',segment_index= 0, 
                   num_segments=1, 
                   segment_indices = [(0,200000)],
                   df_file = df, 
                   eval = False, 
                   max_steps=100,
                   max_episode_steps = 50,
                   parent_id = sol_token_parent)

    
    # Reset the environment to its initial state
    observation = env.reset()
    print("Initial Observation:", observation)
    
    for _ in range(20000):
        # Render the environment (optional)
        # env.render()
        
        # Take a random action
        action = env.action_space.sample()
        
        # Step through the environment with the chosen action
        next_observation, reward, terminated, truncated, info = env.step(action)
        # print("Next Observation:", next_observation)
        # print("Reward:", reward)
        # print("Done:", truncated or terminated)
        # print("Info:", info)
        print(f"step: {_}")
        
        if truncated or terminated:
            env.reset()
            # break
            # print("done")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()