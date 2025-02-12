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
    
# Your wallet details – set these securely!
# Here, PRIVATE_KEY should be set in your environment or loaded securely.
# The key must be in the correct format (for example, a base58 or bytes object)
# PRIVATE_KEY = os.getenv("PRIVATE_KEY")
# if not PRIVATE_KEY:
#     raise Exception("Please set your PRIVATE_KEY environment variable.")

# For example, if your private key is stored as a comma-separated list of ints in JSON,
# you might load it as follows:
# import json
# secret_key = json.loads(PRIVATE_KEY)  # adjust parsing based on your storage format
# wallet = Keypair.from_secret_key(bytes(secret_key))
# user_public_key = str(wallet.public_key)

# --- STEP 1: GET A QUOTE ---


# print(extract_major_values(quote_data))

# --- STEP 2: REQUEST THE SWAP TRANSACTION ---
# payload = {
#     "quoteResponse": quote_data,
#     "userPublicKey": user_public_key,
#     # Setting wrapAndUnwrapSol to True lets Jupiter handle SOL wrapping for you
#     "wrapAndUnwrapSol": True
# }

# print("Requesting swap transaction from Jupiter...")
# swap_resp = requests.post(SWAP_URL, json=payload)
# swap_data = swap_resp.json()
# if "error" in swap_data:
#     raise Exception("Error in swap response: " + str(swap_data["error"]))

# # The returned transaction is base64 encoded
# swap_transaction_base64 = swap_data["swapTransaction"]

# # --- STEP 3: DESERIALIZE, SIGN, AND SEND THE TRANSACTION ---
# print("Deserializing transaction...")
# tx_bytes = base64.b64decode(swap_transaction_base64)

# # Note: Depending on the transaction type (legacy or versioned), the deserialization method may differ.
# # Here we assume a legacy transaction for simplicity.
# transaction = Transaction.deserialize(tx_bytes)

# print("Signing transaction...")
# transaction.sign(wallet)

# print("Sending transaction to Solana network...")
# raw_tx = transaction.serialize()
# send_resp = client.send_raw_transaction(raw_tx)
# tx_sig = send_resp["result"]

# print("Transaction submitted. Signature:", tx_sig)

# # Optionally, wait for confirmation:
# print("Confirming transaction...")
# confirmation = client.confirm_transaction(tx_sig)
# if confirmation["result"]:
#     print("Transaction confirmed! View it at https://solscan.io/tx/" + tx_sig)
# else:
#     print("Transaction not confirmed yet. Please check manually.")
