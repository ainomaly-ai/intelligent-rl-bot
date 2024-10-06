# test.py
import gymnasium as gym
from sol_env import TradeEnv


def main():
    # Register the custom environment
    gym.register(id='Sol-v0', entry_point='sol_env.sol_env:TradeEnv')
    
    # Create an instance of the custom environment
    env = gym.make('Sol-v0')
    
    # Reset the environment to its initial state
    observation = env.reset()
    print("Initial Observation:", observation)
    
    for _ in range(100):
        # Render the environment (optional)
        # env.render()
        
        # Take a random action
        action = env.action_space.sample()
        
        # Step through the environment with the chosen action
        next_observation, reward, done, info = env.step(action)
        print("Next Observation:", next_observation)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        
        if done:
            break
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()