# RL Trade Bot 

## Overview
The Sol Trade Bot is an automated trading bot designed to simulate token trading operations in a decentralized environment using reinforcement learning (RL). The project leverages historical data and dynamic market conditions to optimize trading strategies through machine learning models.

---

## Key Components

### 1. Portfolio Management (`tokensim.py`)
- **Portfolio Class**: Manages SOL tokens, token inventory, USD value tracking, and transaction logging.
- **Methods**:
  - `transfer_sol(amount)`: Transfers SOL from the portfolio to another wallet.
  - `update_values()`: Dynamically adjusts token holdings and values based on market prices and trades.

### 2. Trading Environment (`sol_env.py`)
- **Custom Gymnasium Environment**: 
  - **TradeEnv Class**: Simulates trading actions (buy, sell, hold) using historical price data.
  - **State Space**: Includes SOL balance, token inventories, and USD value.
  - **Reward Function**: Maximizes profit based on trade outcomes.

### 3. Reinforcement Learning Integration (`stable_baseline_test.py`)
- **Training Script**:
  - Uses Stable Baselines 3 (TRPO, PPO) to train RL agents.
  - Logs training metrics to TensorBoard for visualization.
- **Hyperparameters Tuning**: Adjustable learning rates, batch sizes, and episode steps.

---

## File Structure
- **tokensim.py**: Manages the portfolio's state and transactions.
- **sol_env.py**: Defines the custom trading environment.
- **stable_baseline_test.py**: Loads data, configures RL models, trains them, and saves agents.

---

## Key Features
1. **Slippage Handling**: Adjusts trade sizes based on priority levels (high, medium, low) to simulate market impact.
2. **Dynamic Value Updates**: Continuously tracks USD value changes due to token prices and SOL balance.
3. **Profit Transfer Logic**: Automatically transfers profits back to the main wallet when thresholds are met.
4. **TensorBoard Integration**: Logs episode metrics (average SOL, USD value, profit) for performance tracking.

---

## Setup & Usage

### Prerequisites
- Install dependencies:
  ```bash
  pip install gymnasium stable-baselines3 sb3-contrib pandas matplotlib tensorboard
  ```
- Ensure historical data is available at the specified path (e.g., `combined_df.pkl`).

### Running the Code
1. **Train an RL Model**:
   - Execute `stable_baseline_test.py` to train and save agents.
     ```bash
     python stable_baseline_test.py
     ```

2. **Monitor Training Progress**:
   - Use TensorBoard to visualize metrics:
     ```bash
     tensorboard --logdir=./stable_bs/ppo_tensorboard/
     ```
   
3. **Example Usage** (in `tokensim.py`):
   ```python
   portfolio = Portfolio()
   print(portfolio.get_state())
   for _ in range(10):
       portfolio.update_values()
       print(portfolio.get_state())
   ```

---

## Contributions & Support

### How to Contribute
- **Bug Fixes**: Address any issues with code or functionality.
- **Feature Enhancements**: Propose new features like more complex trading strategies.
- **Documentation**: Improve README and inline comments.

### Contact
- For questions, contact the project maintainers via GitHub Issues.

---

## License & Disclaimer

**License**: MIT  
**Disclaimer**: This project is for educational purposes. Use in real-world scenarios requires thorough testing and compliance with regulations.

--- 

**Note**: The bot simulates trades based on historical data and may not account for all real-world complexities. Always test thoroughly before deployment.
