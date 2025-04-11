from gymnasium.envs.registration import register

register(
     id="sol-v0",
     entry_point="sol_env.sol_env:TradeEnv",
     max_episode_steps=300,
)