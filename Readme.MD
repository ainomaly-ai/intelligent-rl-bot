Sol Trade Bot Project
Overview
The Sol Trade Bot is an automated trading bot designed to simulate token trading operations in a decentralized environment. The project leverages machine learning, specifically reinforcement learning (RL), to optimize trading strategies based on historical data and dynamic market conditions.

Key Components
Portfolio Management:

Tokensimulation Module (
tokensim.py
): Manages the portfolio's SOL tokens, token inventory, and USD value tracking.
Update Mechanisms: Adjusts token holdings and values dynamically based on buy/sell actions, slippage, and market prices.
Trading Environment (
sol_env.py
):

A custom Gymnasium environment tailored for trading simulations. It encapsulates the state space (SOL balance, token inventories) and action space (buy, sell, hold decisions).
Uses historical price data to simulate market conditions and reward agents based on profit maximization.
Reinforcement Learning Integration (
stable_baseline_test.py
):

Trains RL models (TRPO, PPO) using Stable Baselines 3 library to interact with the trading environment.
Logs training metrics to TensorBoard for visualization and analysis.
File Structure
tokensim.py: Contains the 
Portfolio
 class responsible for managing tokens, executing trades, updating values, and logging transactions.
sol_env.py: Defines the custom Gymnasium environment (
TradeEnv
) that simulates the trading process using historical data.
stable_baseline_test.py: Scripts to load training data, configure RL models, train them, and save trained agents.
Key Features
Slippage Handling: Accounts for market impact when executing large trades by adjusting trade sizes based on priority levels (high, med, low).
Dynamic Value Updates: Continuously updates the portfolio's USD value to reflect token prices and SOL balance.
Profit Transfer Logic: Automatically transfers a portion of profits back to the main wallet when certain thresholds are met.
TensorBoard Integration: Logs episode metrics (average SOL, USD value, profit) for training performance tracking.
Setup & Usage
Environment Configuration:

Install dependencies: gymnasium, stable-baselines3, pandas, etc.
Ensure historical data is available at specified paths (e.g., /home/abishek/sol-proj/ray/sol-trade/data/asc/combined_df.pkl).
Training RL Models:

Run 
stable_baseline_test.py
 to train models using the configured environment and hyperparameters.
Adjust parameters like learning rate, batch size, and episode steps as needed.
Monitoring Training:

Use TensorBoard to visualize training progress: tensorboard --logdir=./stable_bs/ppo_tensorboard/.
Contributions & Support
Pull Requests: Welcome for bug fixes, feature enhancements, or improved models.
Documentation: Contributions to improve documentation are appreciated.
For further details on configuration and customization of the environment and RL agents, refer to the respective source files and comments within them.

Note: The project is designed for educational and experimental purposes. Use in real-world scenarios requires thorough testing and adaptation to comply with regulatory requirements
