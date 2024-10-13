import random
import time

class Portfolio:
    def __init__(self):
        self.sol = 1
        self.avl_token = 0 # available token in portfolio
        #self.bal_usd = 100.0  # Initial USD value
        self.sol_value = 158.14  # Initial SOL token amount
        self.token_value = 0 # Initial token value in USD
        self.total_usd = self.sol_value
        

    def trade(self, p, sol_amount, token_price, priority):
        slippage = {"high": 0.5, "med": 0.1, "low": 0.05}
        if p == "buy":
            if priority in ["high", "med", "low"]:
                time.sleep(1)  # Simulate delay for high priority trades
                self.slippage = slippage[priority] / 100
                trade_sol = sol_amount *  self.slippage
                self.sol -= (sol_amount + trade_sol)
                self.avl_token += (trade_sol * self.sol_value) / token_price  
        elif p == "sell":
            if priority in ["high", "med", "low"]:
                time.sleep(1)  # Simulate delay for high priority trades
                self.slippage = slippage[priority] / 100
                trade_sol = sol_amount * self.slippage
                self.sol += (sol_amount - trade_sol)
                self.avl_token -= (trade_sol * self.sol_value) / token_price
        else:
            print("Hold for now")


    def update_values(self, token_value):

        self.token_value = self.avl_token * token_value
        # Update USD value 
        self.usd_value = self.get_usd_value()


        
        # # Update SOL token amount randomly by -5% to +5%
        # self.sol_token_amount *= (1 + random.uniform(-0.05, 0.05))
        
        # # Update other tokens values randomly by -5% to +5%
        # for token in self.other_tokens:
        #     self.other_tokens[token] *= (1 + random.uniform(-0.05, 0.05))

    # def get_observation(self):
    #     # Return the current values as a dictionary
    #     return {
    #         "usd": self.usd_value,
    #         "sol": self.sol_token_amount,
    #         **{k: v for k, v in self.other_tokens.items()}
    #     }
    
    def get_usd_value(self):
        return self.sol_value + self.token_value 


# # Example usage:
# portfolio = Portfolio()
# print("Initial Portfolio Values:", portfolio.get_observation())
# for _ in range(10):
    # portfolio.update_values()
    # print("Updated Portfolio Values:", portfolio.get_observation())


