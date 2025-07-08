#!/usr/bin/env python3
"""
Fixed Unichain Configuration for EulerSwap Vault Agent
Using your deployed contract addresses with correct ABI paths
"""

import os
import json
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# ============ UNICHAIN CONFIGURATION ============
UNICHAIN_RPC_URL = "https://unichain-rpc.publicnode.com"  # Unichain MAINNET
UNICHAIN_CHAIN_ID = 130  # Unichain MAINNET
AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============ YOUR DEPLOYED CONTRACT ADDRESSES ============
VAULT_FACTORY_ADDRESS = "0xEc0036beC79dBCf0279dAd3bC59F6231b9F461d9"
STRATEGY_FACTORY_ADDRESS = "0xa5752653c78D9254EB3082d8559b76a38C9E8563"
MOCK_USDC_ADDRESS = "0xC0933C5440c656464D1Eb1F886422bE3466B1459"
MOCK_WETH_ADDRESS = "0xf0f994B4A8dB86A46a1eD4F12263c795b26703Ca"
USDC_VAULT_ADDRESS = "0xFAf4Af2Ed51cDb2967B0d204074cc2e4302F9188"
WETH_VAULT_ADDRESS = "0x112b33c07deE1697d9b4A68b2BA0F66c2417635C"
EULERSWAP_STRATEGY_ADDRESS = "0x807463769044222F3b7B5F98da8d3E25e0aC44B0"

# ============ UNICHAIN PROTOCOL ADDRESSES ============
EULER_SWAP_FACTORY = "0x45b146BC07c9985589B52df651310e75C6BE066A"
USDC_EVAULT = "0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba"
WETH_EVAULT = "0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC"
EVC = "0x2A1176964F5D7caE5406B627Bf6166664FE83c60"

# ============ WEB3 SETUP ============
w3 = Web3(Web3.HTTPProvider(UNICHAIN_RPC_URL))
agent_account = w3.eth.account.from_key(AGENT_PRIVATE_KEY)

print(f"🚀 Unichain Agent: {agent_account.address}")
print(f"🌐 Chain ID: {UNICHAIN_CHAIN_ID}")
print(f"🏭 Your Vault Factory: {VAULT_FACTORY_ADDRESS}")

# ============ LOAD ABIS WITH CORRECT PATHS ============
def load_abi(filename):
    """Load ABI from your contracts directory structure."""
    # Try different possible paths based on your structure
    possible_paths = [
        f"../contracts/abi/core/vault.sol/{filename}",
        f"../contracts/abi/core/vaultfactoryslim.sol/{filename}",
        f"../contracts/abi/core/strategyfactory.sol/{filename}",
        f"../contracts/abi/mocks/mockUSDC.sol/{filename}",
        f"../contracts/abi/mocks/mockWETH.sol/{filename}",
        f"../contracts/abi/strategies/EulerSwapStrategy.sol/{filename}",
        f"contracts/abi/core/vault.sol/{filename}",
        f"contracts/abi/core/vaultfactoryslim.sol/{filename}",
        f"contracts/abi/core/strategyfactory.sol/{filename}",
        f"contracts/abi/mocks/mockUSDC.sol/{filename}",
        f"contracts/abi/mocks/mockWETH.sol/{filename}",
        f"contracts/abi/strategies/EulerSwapStrategy.sol/{filename}",
        filename
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    # Handle Hardhat artifact format
                    if isinstance(data, dict) and "abi" in data:
                        print(f"✅ Loaded ABI from: {path}")
                        return data["abi"]
                    elif isinstance(data, list):
                        print(f"✅ Loaded ABI from: {path}")
                        return data
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
                continue
    
    print(f"❌ Could not find ABI file: {filename}")
    print(f"   Searched in: {possible_paths[:3]}...")
    return []

# ============ STANDARD ERC4626 VAULT ABI ============
# Since you don't have the Vault ABI, here's the standard ERC4626 interface
VAULT_ABI = [
    {"name": "totalAssets", "type": "function", "inputs": [], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "totalSupply", "type": "function", "inputs": [], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "balanceOf", "type": "function", "inputs": [{"type": "address"}], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "deposit", "type": "function", "inputs": [{"type": "uint256"}, {"type": "address"}], "outputs": [{"type": "uint256"}], "stateMutability": "nonpayable"},
    {"name": "withdraw", "type": "function", "inputs": [{"type": "uint256"}, {"type": "address"}, {"type": "address"}], "outputs": [{"type": "uint256"}], "stateMutability": "nonpayable"},
    {"name": "asset", "type": "function", "inputs": [], "outputs": [{"type": "address"}], "stateMutability": "view"},
    {"name": "convertToShares", "type": "function", "inputs": [{"type": "uint256"}], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "convertToAssets", "type": "function", "inputs": [{"type": "uint256"}], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "depositToStrategy", "type": "function", "inputs": [{"type": "address"}, {"type": "uint256"}, {"type": "bytes"}], "outputs": [], "stateMutability": "nonpayable"},
    {"name": "harvestStrategy", "type": "function", "inputs": [{"type": "address"}, {"type": "bytes"}], "outputs": [], "stateMutability": "nonpayable"},
    {"name": "addStrategy", "type": "function", "inputs": [{"type": "address"}], "outputs": [], "stateMutability": "nonpayable"}
]

# Load ABIs from your files
try:
    usdc_abi = load_abi("mockusdc.json")
    weth_abi = load_abi("mockweth.json") 
    strategy_abi = load_abi("eulerswapstrategy.json")
    
    # Use the standard vault ABI since you don't have the specific one
    vault_abi = VAULT_ABI
    
    print("✅ All ABIs loaded")
except Exception as e:
    print(f"❌ ABI loading error: {e}")
    # Fallback to minimal ABIs
    usdc_abi = [
        {"name": "balanceOf", "type": "function", "inputs": [{"type": "address"}], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
        {"name": "approve", "type": "function", "inputs": [{"type": "address"}, {"type": "uint256"}], "outputs": [{"type": "bool"}], "stateMutability": "nonpayable"},
        {"name": "faucet", "type": "function", "inputs": [{"type": "uint256"}], "outputs": [], "stateMutability": "nonpayable"}
    ]
    weth_abi = usdc_abi
    strategy_abi = [{"name": "getBalance", "type": "function", "inputs": [], "outputs": [{"type": "uint256"}], "stateMutability": "view"}]
    vault_abi = VAULT_ABI

# ============ CONTRACT INSTANCES ============
try:
    usdc_vault_contract = w3.eth.contract(address=USDC_VAULT_ADDRESS, abi=vault_abi)
    weth_vault_contract = w3.eth.contract(address=WETH_VAULT_ADDRESS, abi=vault_abi)
    usdc_contract = w3.eth.contract(address=MOCK_USDC_ADDRESS, abi=usdc_abi)
    weth_contract = w3.eth.contract(address=MOCK_WETH_ADDRESS, abi=weth_abi)
    strategy_contract = w3.eth.contract(address=EULERSWAP_STRATEGY_ADDRESS, abi=strategy_abi)
    
    print("✅ All contract instances created")
except Exception as e:
    print(f"❌ Contract creation error: {e}")

print("✅ Unichain configuration loaded with your deployed contracts")