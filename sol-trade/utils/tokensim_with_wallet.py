import random
import time
import subprocess
import re
import json
from utils.jupiter.jupiter_connect import extract_major_values
from decimal import Decimal


def solana_cli(command):
    """Helper function to call Solana CLI commands."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e.stderr}")
        raise

def request_airdrop(pubkey, amount):
    """Request an airdrop using Solana CLI."""
    print(f"Requesting {amount} SOL airdrop for {pubkey}...")
    command = ['solana', 'airdrop', str(amount), '--url', 'http://127.0.0.1:8899', pubkey]
    result = solana_cli(command)
    print(result)


    
def wait_for_finalization(tx_signature, max_attempts=70, delay=0.3):
    """Waits until the transaction reaches 'Finalized' status."""
    for attempt in range(max_attempts):
        command = ['solana', 'confirm', tx_signature, '--url', 'http://127.0.0.1:8899']
        result = solana_cli(command)

        if "Finalized" in result:
            print(f"✅ Transaction {tx_signature} is FINALIZED!")
            return True
        elif "Confirmed" in result:
            print(f"🕒 Transaction {tx_signature} is CONFIRMED but not finalized.")
            return True
        else:
            print(f"⚠️ Attempt {attempt+1}: Status Unknown")

        time.sleep(delay)

    print(f"❌ Transaction {tx_signature} was not finalized after {max_attempts} attempts.")
    return False

def send_transaction(sender_key, recipient_key, amount,
                     compute_unit_price="1000"):

    balance_sender = get_balance(sender_key)
    print(f"before :{balance_sender}")
    balance_receipient = get_balance(recipient_key)
    print(f"before :{balance_receipient}")
    
    """Send a transaction using Solana CLI."""
    print(f"Sending {amount} SOL from {sender_key} to {recipient_key}...")
    command = [
        'solana', 'transfer', recipient_key, str(amount),
        '--from', sender_key, '--url', 'http://127.0.0.1:8899', 
        '--allow-unfunded-recipient',  
        '--with-compute-unit-price', 
        compute_unit_price,
       '--commitment', 'processed' 
    ]
    result = solana_cli(command)
    print(result)

      # Extract transaction signature
    match = re.search(r"Signature:\s+(\S+)", result)
    if not match:
        print("Error: Could not extract transaction signature.")
        print(result)
        return
    
    tx_signature = match.group(1)  # Extract the actual signature
    print(f"Transaction Signature: {tx_signature}")

    
    confirm = wait_for_finalization(tx_signature)

    print(confirm)

    balance_sender = get_balance(sender_key)
    print(f"after :{balance_sender}")
    balance_receipient = get_balance(recipient_key)
    print(f"after :{balance_receipient}")
    

def get_balance(pubkey):
    """Get the balance of a public key using Solana CLI."""
    print(f"Getting balance for {pubkey}...")
    command = ['solana', 'balance', pubkey, '--url', 'http://127.0.0.1:8899']
    result = solana_cli(command)
    print(result)
    return float(result.split()[0])


def burn_all_sol(wallet_key, burner_key):
    """Burn all SOL by sending the balance to a burn address."""
    print(f"Burning all SOL from {wallet_key}...")
    
    # Get the balance of the wallet to burn
    balance_output = get_balance(wallet_key)
    print(balance_output)
    balance = balance_output
    
    # Deduct a small fee (approximately 0.000005 SOL)
    fee = 0.005
    burn_amount = balance - fee
    
    # If the balance is greater than the fee, we can proceed with burning
    if burn_amount > 0:
        # Use a known "burn address" (a randomly generated address or non-existent address)
        burn_address = '11111111111111111111111111111111'  # Common Solana burn address (a special address)

        # Send the amount minus the fee to the burn address
        command = [
            'solana', 'transfer',burner_key, str(burn_amount),
            '--from', wallet_key, '--url', 'http://127.0.0.1:8899', '--allow-unfunded-recipient'
        ]
        result = solana_cli(command)
        print(result)
    else:
        print("Balance is too low to burn.")



# Generate a new keypair for sender and recipient (or use existing ones)
sender_key = '/home/abishek/.config/solana/sender.json'  # Replace with actual sender keypair file path
recipient_key = '/home/abishek/.config/solana/receipient.json'  # Replace with actual recipient public key
burner_key = '/home/abishek/.config/solana/burner.json'  # Replace with actual burner public key
trade_key = '/home/abishek/.config/solana/trade.json'  # Replace with actual trade public key

# params = {
#     "inputMint": input_mint,
#     "outputMint": output_mint,
#     "amount": amount,
#     "slippageBps": slippage_bps
# }


Debug = False
class Portfolio:
    def __init__(self, parent_id=None):
        self.sol = get_balance(sender_key)
        self.profilt_sol_wallet = get_balance(recipient_key)
        self.avl_token = 0 # available token in portfolio
        #self.bal_usd = 100.0  # Initial USD value
        self.sol_value = 185  # Initial SOL token amount
        self.avl_sol_value =185
        self.sol_addr = "So11111111111111111111111111111111111111112"
        self.token_sol_amount = {} # Initial token value in SOl
        self.total_usd = 0
        self.tokens = {}
        self.token_value = {} # Initial token value in USD
        self.usd_value = self.sol_value
        self.trade_log = []
        self.parent_id = parent_id
        

        burn_all_sol(sender_key, burner_key)
        burn_all_sol(recipient_key, burner_key)
        burn_all_sol(trade_key, burner_key)

        # Request airdrop for sender and trader
        request_airdrop(trade_key, 100)
        request_airdrop(sender_key, 1)
        print("Portfolio Initialized")
    
        



    def trade(self,token, p, sol_amount, token_price, priority):
        slippage = {"high": 0.5, "med": 0.1, "low": 0.05}
        self.update_wallets()

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
                    # self.sol -= total_trade_sol
                    if self.parent_id:
                        sol, token_amount = extract_major_values(self.sol_addr, self.parent_id, int(total_trade_sol*1000000000)) #Get quote from jupiter 
                        send_transaction(sender_key, trade_key, sol)
                    else:
                        send_transaction(sender_key, trade_key, total_trade_sol)
                    
                    self.update_wallets()
                    
                    if self.parent_id: 
                        temp_token_buy_amount = token_amount
                    else:
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
                # self.sol += total_trade_sol
                temp_token_sell_amount = (total_trade_sol * self.sol_value) / token_price
                if temp_token_sell_amount > self.tokens[token]["total_tokens_left"]:
                    print("Not enough token left to sell that amount")
                    return  "not sold"
                
                print("sell executed")
                if self.parent_id:
                    temp_token_sell_amount = round(temp_token_sell_amount)
                    print(temp_token_sell_amount)
                    sol, token_amount = extract_major_values( self.parent_id,self.sol_addr, temp_token_sell_amount*1000000) #Get quote from jupiter for sell 
                    print(f"token amountttttt , solllllllllllllllll: {token_amount, sol}")
                    send_transaction(trade_key, sender_key, sol)
                else:
                    send_transaction(trade_key, sender_key, total_trade_sol)

                self.update_wallets()

                self.tokens[token]["total_tokens_left"] -= temp_token_sell_amount
                self.update_values(token_price, token, self.tokens[token]["total_tokens_left"])   
                if self.parent_id:  
                    self.trade_log.append({"sell":[token, sol,temp_token_sell_amount]})
                else:
                    self.trade_log.append({"sell":[token, total_trade_sol,temp_token_sell_amount]})
                

                # Check if SOL has reached 1. and transfer 0.2 SOL
                if self.sol >= 1.2:
                    print("sending profit ............................")
                    send_transaction(sender_key, recipient_key, 0.2)
                    # self.transfer_sol(0.3)
                if self.sol >= 0.8 and self.usd_value > 350:
                    print("sending profit ............................")
                    send_transaction(sender_key, recipient_key, 0.2)

                self.update_wallets()

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
        print(f"total price of tokens { total_tokens , token_price , total_tokens * token_price}")
        self.token_value[token]["token_value"] = total_tokens * token_price
        self.token_value[token]["token_sol_amount"] = self.token_value[token]["token_value"] / self.sol_value
        # Update USD value 
        self.usd_value = self.get_usd_value()

    ## To do : self.token_value here to add the values of all of them     
    def get_usd_value(self):
        self.update_wallets()
        return self.avl_sol_value + sum(token['token_value'] for token in self.token_value.values())
    
    def get_state(self):
        self.update_wallets()
        return self.sol, self.tokens, self.token_value, self.usd_value, self.profilt_sol_wallet
    
    def transfer_sol(self, amount):
        # Implement the logic to transfer SOL to another function or process
        print(f"Transferring {amount} SOL to another wallet")
        self.sol -= amount  # Deduct the transferred amount from the current portfolio
        self.profilt_sol_wallet += amount

    def update_wallets(self):    
        self.sol = get_balance(sender_key)
        self.profilt_sol_wallet = get_balance(recipient_key)
        self.avl_sol_value = self.sol_value * self.sol




# # Example usage:
# portfolio = Portfolio()
# print("Initial Portfolio Values:", portfolio.get_observation())
# for _ in range(10):
    # portfolio.update_values()
    # print("Updated Portfolio Values:", portfolio.get_observation())


