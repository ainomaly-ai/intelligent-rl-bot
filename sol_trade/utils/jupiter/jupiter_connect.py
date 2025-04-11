import os
import base64
import requests
from solana.rpc.api import Client
# from solana.keypair import Keypair
from solana.transaction import Transaction
import time 

# --- CONFIGURATION ---
# Use your preferred Solana RPC endpoint (a dedicated endpoint is recommended for production)
RPC_URL = "https://api.mainnet-beta.solana.com"  # or your custom endpoint
client = Client(RPC_URL)

# Jupiter API endpoints (new hostnames as per Jupiter docs)
QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
SWAP_URL = "https://api.jup.ag/swap/v1/swap"

# Set your tokens (example: swapping SOL to USDC)
# For SOL, the mint is typically "So11111111111111111111111111111111111111112"
# For USDC on Solana, the mint is typically "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
input_mint = "So11111111111111111111111111111111111111112"   # SOL
output_mint = "7oBYdEhV4GkXC19ZfgAvXpJWp2Rn9pm1Bx2cVNxFpump"   # USDC

# Amount to swap (in smallest units, e.g. lamports for SOL)
# amount = "100000000"  # Example: 0.1 SOL (if SOL has 9 decimals)
amount = "151359735"  # Example: 0.1 SOL (if SOL has 9 decimals)
slippage_bps = "50"   # 0.5% slippage



# Function to extract and display major values
def extract_major_values(tokenA, tokenB, amount, slippage= slippage_bps):

    params = {
    "inputMint": tokenA,
    "outputMint": tokenB,
    "amount": amount,
    "slippageBps": slippage
    }

    print(params)
    print(input_mint , tokenA)
    print(input_mint == tokenA)

    print("Requesting quote from Jupiter...")
    quote_response = requests.get(QUOTE_URL, params=params)
    quote_data = quote_response.json()
    if "error" in quote_data:
        raise Exception("Error in quote response: " + str(quote_data["error"]))

    # Convert amounts from lamports to SOL (1 SOL = 100,000,000 lamports)

    # time.sleep(2)
    print(quote_data)
    out_received = int(quote_data["outAmount"]) 
    input_amount = int(quote_data["inAmount"]) 
    fee_amount = int(quote_data["routePlan"][0]["swapInfo"]["feeAmount"]) / 100000000

    # Extract other values
    estimated_usd_value = float(quote_data["swapUsdValue"])
    slippage_bps = quote_data["slippageBps"]
    price_impact_pct = quote_data["priceImpactPct"]
    swap_mode = quote_data["swapMode"]
    platform_fee = quote_data["platformFee"]
    time_taken = quote_data["timeTaken"]


    # Display extracted values
    # print("Extracted Major Values:")
    # print(f"SOL Received: {sol_received:.6f} SOL")
    # print(f"Input Amount: {input_amount:.6f} Tokens")
    # print(f"Fee Amount: {fee_amount:.6f} Tokens")
    # print(f"Estimated USD Value: ${estimated_usd_value:.2f}")
    # print(f"Slippage Tolerance: {slippage_bps / 100}%")
    # print(f"Price Impact: {price_impact_pct}%")
    # print(f"Swap Mode: {swap_mode}")
    # print(f"Platform Fee: {platform_fee}")
    # print(f"Time Taken: {time_taken:.6f} seconds")

    if tokenA == input_mint:
        print(input_amount/ 1000000000 , out_received/1000000)
        return   input_amount/ 1000000000 , out_received/1000000
    else:
        return   out_received / 1000000000, input_amount/1000000
    
