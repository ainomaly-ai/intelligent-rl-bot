import subprocess
import time


def solana_cli(command):
    """Helper function to call Solana CLI commands."""
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout

def create_wallet(name):
    """Create a new wallet and return the file path and public key."""
    wallet_path = f'/home/abishek/.config/solana/temp/{name}.json'
    print("Creating a new wallet...")
    command = ['solana-keygen', 'new', '--no-passphrase', '-o', wallet_path]
    solana_cli(command)
    pubkey_command = ['solana', 'address', '--keypair', wallet_path]
    pubkey = solana_cli(pubkey_command)
    print(f"New wallet created. Public key: {pubkey}")
    return wallet_path, pubkey

def request_airdrop(pubkey, amount):
    """Request an airdrop using Solana CLI."""
    print(f"Requesting {amount} SOL airdrop for {pubkey}...")
    command = ['solana', 'airdrop', str(amount), '--url', 'http://127.0.0.1:8899', pubkey]
    result = solana_cli(command)
    print(result)

def send_transaction(sender_key, recipient_key, amount):
    """Send a transaction using Solana CLI."""
    print(f"Sending {amount} SOL from {sender_key} to {recipient_key}...")
    command = [
        'solana', 'transfer', recipient_key, str(amount),
        '--from', sender_key, '--url', 'http://127.0.0.1:8899', '--allow-unfunded-recipient'
    ]
    result = solana_cli(command)
    print(result)

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
            'solana', 'transfer', burner_key, str(burn_amount),
            '--from', wallet_key, '--url', 'http://127.0.0.1:8899', '--allow-unfunded-recipient'
        ]
        result = solana_cli(command)
        print(result)
    else:
        print("Balance is too low to burn.")

# Example Usage:
# Generate a new keypair for sender and recipient (or use existing ones)
# sender_key = '/home/abishek/.config/solana/sender.json'  # Replace with actual sender keypair file path
# recipient_key = '/home/abishek/.config/solana/receipient.json'  # Replace with actual recipient public key
# burner_key = '/home/abishek/.config/solana/burner.json'  # Replace with actual burner public key

# Create new wallets for sender, recipient, and burner
sender_key, sender_pubkey = create_wallet("sender")
recipient_key, recipient_pubkey = create_wallet("recipient")
burner_key, burner_pubkey = create_wallet("burner")


# Request airdrop for sender
request_airdrop(sender_key, 2)

# Check balances before the transaction
get_balance(sender_key)
get_balance(recipient_key)
get_balance(burner_key)

# Send a transaction (1 SOL from sender to recipient)
send_transaction(sender_key, recipient_key, 1)

# Check balances after the transaction
get_balance(sender_key)
get_balance(recipient_key)


# Burn all SOL from sender and recipient wallets
burn_all_sol(sender_key, burner_key)
burn_all_sol(recipient_key, burner_key)

# Check balances after the transaction
get_balance(sender_key)
get_balance(recipient_key)
get_balance(burner_key)


