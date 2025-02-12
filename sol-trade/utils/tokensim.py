import random
import time





Debug = False
class Portfolio:
    def __init__(self):
        self.sol = 1
        self.profilt_sol_wallet = 0
        self.avl_token = 0 # available token in portfolio
        #self.bal_usd = 100.0  # Initial USD value
        self.sol_value = 204.14  # Initial SOL token amount
        self.token_sol_amount = {} # Initial token value in SOl
        self.total_usd = 0
        self.tokens = {}
        self.token_value = {} # Initial token value in USD
        self.usd_value = self.sol_value
        self.trade_log = []

 




    def trade(self,token, p, sol_amount, token_price, priority):
        slippage = {"high": 0.5, "med": 0.1, "low": 0.05}

        # Check if token is already in the tokens dictionary
        if token not in self.tokens:
            # If not, initialize it with an empty list and set total_tokens_left to 0
            self.tokens[token] = {"total_tokens_left": 0}
            self.token_value[token] = {"token_value": 0, "token_sol_amount" : 0}


        if p == "buy":
            if priority in ["high", "med", "low"]:
                # time.sleep(1)  # Simulate delay for high priority trades
                self.slippage = slippage[priority] / 100 # Calculate slippage based on priority(percent)
                trade_sol = sol_amount *  self.slippage
                total_trade_sol = sol_amount + trade_sol
                if self.sol -total_trade_sol < 0.2:
                    print("Not enough SOL to complete trade")
                    return "not bought"
                
                else:   
                    self.sol -= total_trade_sol
                    temp_token_buy_amount = (total_trade_sol * self.sol_value) / token_price
                    self.tokens[token]["total_tokens_left"] += temp_token_buy_amount  
                    self.update_values(token_price, token, self.tokens[token]["total_tokens_left"])     
                    self.trade_log.append({"buy":[token, total_trade_sol,temp_token_buy_amount]})
                 # Update tokens dictionary for the specific token
                # self.tokens[token]["total_tokens_left"] = self.avl_token  
                    if Debug:
                        print(f"Token: {token}, Bought {int((trade_sol * self.sol_value) / token_price)} tokens")
                    return "bought"

            
        elif p == "sell":
            if priority in ["high", "med", "low"] and self.tokens[token]["total_tokens_left"] > 0:
                # time.sleep(1)  # Simulate delay for high priority trades
                self.slippage = slippage[priority] / 100  # Calculate slippage based on priority(percent)
                trade_sol = sol_amount * self.slippage
                total_trade_sol = sol_amount - trade_sol
                temp_token_sell_amount = (total_trade_sol * self.sol_value) / token_price
                if temp_token_sell_amount > self.tokens[token]["total_tokens_left"]:
                    print("Not enough token left to sell that amount")
                    return  "not sold"
                
                print("sell executed")
                self.sol += total_trade_sol
                self.tokens[token]["total_tokens_left"] -= temp_token_sell_amount
                self.update_values(token_price, token, self.tokens[token]["total_tokens_left"])     
                self.trade_log.append({"sell":[token, total_trade_sol,temp_token_sell_amount]})

                # Check if SOL has reached 1.2 and transfer 0.2 SOL
                if self.sol >= 1.2:
                    print("sending profit ............................")
                    self.transfer_sol(0.2)

                if self.sol >= 0.8 and self.usd_value > 350:
                    print("sending profit ............................")
                    self.transfer_sol(0.2)

                 # Update tokens dictionary for the specific token
                # self.tokens[token]["total_tokens_left"] = self.avl_token  
                if Debug:
                    print(f"Token: {token}, Sold {int((trade_sol * self.sol_value) / token_price)} tokens")
                return "sold"
            
            else:
                if Debug:
                    print("no token to sell!")
                return  "not sold"

        else:
            if Debug:
                print("Hold for now")
            return  "hold"



    def update_values(self, token_price, token, total_tokens):

        self.token_value[token]["token_value"] = total_tokens * token_price
        self.token_value[token]["token_sol_amount"] = self.token_value[token]["token_value"] / self.sol_value
        # Update USD value 
        self.usd_value = self.get_usd_value()

    ## To do : self.token_value here to add the values of all of them     
    def get_usd_value(self):
        return self.sol_value + sum(token['token_value'] for token in self.token_value.values())
    
    def get_state(self):
        return self.sol, self.tokens, self.token_value, self.usd_value, self.profilt_sol_wallet
    
    def transfer_sol(self, amount):
        # Implement the logic to transfer SOL to another function or process
        print(f"Transferring {amount} SOL to another wallet")
        self.sol -= amount  # Deduct the transferred amount from the current portfolio
        self.profilt_sol_wallet += amount


# # Example usage:
# portfolio = Portfolio()
# print("Initial Portfolio Values:", portfolio.get_observation())
# for _ in range(10):
    # portfolio.update_values()
    # print("Updated Portfolio Values:", portfolio.get_observation())


