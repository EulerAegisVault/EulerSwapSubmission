#!/usr/bin/env python3
"""
Unichain EulerSwap Vault Agent
Uses your deployed contracts for vault and strategy management
"""

import os
import json
import time
from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import ContractLogicError
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool


# Import your Unichain configuration
from unichain_config import *

# Import ML Risk Assessment (if available)
try:
    import sys
    sys.path.append('./ml-risk')
    from risk_api import StrategyRiskAPI
    risk_api = StrategyRiskAPI()
    ML_RISK_AVAILABLE = True
    print("✅ ML Risk Assessment loaded")
except ImportError as e:
    risk_api = None
    ML_RISK_AVAILABLE = False
    print(f"⚠️ ML Risk Assessment not available: {e}")

# Import OpenAI LLM planner (if available)
try:
    from ollama_llm_planner import ai_strategy_advisor
    OPENAI_AI_AVAILABLE = True
    print("✅ OpenAI LLM planner loaded")
except ImportError as e:
    OPENAI_AI_AVAILABLE = False
    print(f"⚠️ OpenAI LLM planner not available: {e}")

# ============ TRANSACTION HELPER ============
def send_transaction(tx):
    """Send transaction on Unichain with proper error handling."""
    try:
        signed_tx = w3.eth.account.sign_transaction(tx, agent_account.key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"⏳ Unichain TX: {tx_hash.hex()}. Waiting for confirmation...")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        print(f"✅ Confirmed in block: {receipt.blockNumber}")
        return {"success": True, "receipt": receipt, "tx_hash": tx_hash.hex()}
    except ContractLogicError as e:
        print(f"❌ Transaction reverted: {e}")
        return {"success": False, "error": f"Contract error: {e}"}
    except Exception as e:
        print(f"❌ Transaction failed: {e}")
        return {"success": False, "error": str(e)}

# ============ AGENT TOOLS ============
@tool
def get_unichain_vault_status() -> str:
    """Get comprehensive status of your Unichain vault system."""
    print("Tool: get_unichain_vault_status")
    try:
        # Get USDC vault info with error handling
        try:
            usdc_total_assets = usdc_vault_contract.functions.totalAssets().call()
            usdc_total_supply = usdc_vault_contract.functions.totalSupply().call()
        except Exception as e:
            print(f"⚠️ USDC vault call failed: {e}")
            usdc_total_assets = 0
            usdc_total_supply = 1  # Avoid division by zero
        
        # Get WETH vault info with error handling
        try:
            weth_total_assets = weth_vault_contract.functions.totalAssets().call()
            weth_total_supply = weth_vault_contract.functions.totalSupply().call()
        except Exception as e:
            print(f"⚠️ WETH vault call failed: {e}")
            weth_total_assets = 0
            weth_total_supply = 1  # Avoid division by zero
        
        # Get token balances with error handling
        try:
            agent_usdc_balance = usdc_contract.functions.balanceOf(agent_account.address).call()
            agent_weth_balance = weth_contract.functions.balanceOf(agent_account.address).call()
        except Exception as e:
            print(f"⚠️ Token balance call failed: {e}")
            agent_usdc_balance = 0
            agent_weth_balance = 0
        
        # Get strategy balance with error handling
        try:
            strategy_balance = strategy_contract.functions.getBalance().call()
        except Exception as e:
            print(f"⚠️ Strategy balance call failed: {e}")
            strategy_balance = 0
        
        status = {
            "network": "Unichain",
            "agent_address": agent_account.address,
            "usdc_vault": {
                "address": USDC_VAULT_ADDRESS,
                "total_assets": f"{usdc_total_assets / 10**6:.2f} USDC",
                "total_shares": f"{usdc_total_supply / 10**18:.6f}",
                "share_price": f"{(usdc_total_assets/usdc_total_supply) if usdc_total_supply > 0 else 1:.6f}"
            },
            "weth_vault": {
                "address": WETH_VAULT_ADDRESS, 
                "total_assets": f"{weth_total_assets / 10**18:.6f} WETH",
                "total_shares": f"{weth_total_supply / 10**18:.6f}",
                "share_price": f"{(weth_total_assets/weth_total_supply) if weth_total_supply > 0 else 1:.6f}"
            },
            "strategy": {
                "address": EULERSWAP_STRATEGY_ADDRESS,
                "balance": f"{strategy_balance / 10**6:.2f} USDC"
            },
            "agent_balances": {
                "usdc": f"{agent_usdc_balance / 10**6:.2f} USDC",
                "weth": f"{agent_weth_balance / 10**18:.6f} WETH"
            },
            "integration": {
                "euler_swap_factory": EULER_SWAP_FACTORY,
                "usdc_evault": USDC_EVAULT,
                "weth_evault": WETH_EVAULT
            }
        }
        
        return f"Unichain Vault Status: {json.dumps(status, indent=2)}"
        
    except Exception as e:
        return f"Error getting Unichain vault status: {e}"

@tool  
def mint_test_tokens(usdc_amount: str = "1000", weth_amount: str = "1") -> str:
    """Mint test USDC and WETH tokens on Unichain.
    
    Args:
        usdc_amount: Amount of USDC to mint (default: "1000")
        weth_amount: Amount of WETH to mint (default: "1")
    """
    print(f"Tool: mint_test_tokens - Input: usdc_amount={usdc_amount}, weth_amount={weth_amount}")
    
    try:
        # Convert inputs to float, handling various input types
        if isinstance(usdc_amount, (int, float)):
            usdc_amount = float(usdc_amount)
        else:
            # Handle string inputs
            usdc_str = str(usdc_amount).strip()
            if usdc_str.lower() in ['none', 'null', '', '{}']:
                usdc_amount = 1000.0
            else:
                # Extract just the number part if there's extra text
                import re
                numbers = re.findall(r'\d+\.?\d*', usdc_str)
                usdc_amount = float(numbers[0]) if numbers else 1000.0
        
        if isinstance(weth_amount, (int, float)):
            weth_amount = float(weth_amount)
        else:
            # Handle string inputs
            weth_str = str(weth_amount).strip()
            if weth_str.lower() in ['none', 'null', '', '{}']:
                weth_amount = 1.0
            else:
                # Extract just the number part if there's extra text
                import re
                numbers = re.findall(r'\d+\.?\d*', weth_str)
                weth_amount = float(numbers[0]) if numbers else 1.0
        
        # Ensure positive amounts
        usdc_amount = max(0, usdc_amount)
        weth_amount = max(0, weth_amount)
        
        print(f"Processed amounts: {usdc_amount} USDC, {weth_amount} WETH")
        
        results = []
        
        # Mint USDC
        if usdc_amount > 0:
            try:
                usdc_wei = int(usdc_amount * 10**6)
                usdc_tx = usdc_contract.functions.faucet(usdc_wei).build_transaction({
                    'from': agent_account.address,
                    'nonce': w3.eth.get_transaction_count(agent_account.address),
                    'gas': 500_000,
                    'gasPrice': w3.eth.gas_price,
                    'chainId': UNICHAIN_CHAIN_ID
                })
                usdc_result = send_transaction(usdc_tx)
                if usdc_result["success"]:
                    results.append(f"✅ Minted {usdc_amount} USDC")
                else:
                    results.append(f"❌ USDC mint failed: {usdc_result['error']}")
                
                time.sleep(2)
            except Exception as e:
                results.append(f"❌ USDC mint error: {e}")
        
        # Mint WETH
        if weth_amount > 0:
            try:
                weth_wei = int(weth_amount * 10**18)
                weth_tx = weth_contract.functions.faucet(weth_wei).build_transaction({
                    'from': agent_account.address,
                    'nonce': w3.eth.get_transaction_count(agent_account.address),
                    'gas': 500_000,
                    'gasPrice': w3.eth.gas_price,
                    'chainId': UNICHAIN_CHAIN_ID
                })
                weth_result = send_transaction(weth_tx)
                if weth_result["success"]:
                    results.append(f"✅ Minted {weth_amount} WETH")
                else:
                    results.append(f"❌ WETH mint failed: {weth_result['error']}")
            except Exception as e:
                results.append(f"❌ WETH mint error: {e}")
        
        return f"Unichain Token Minting Results:\n" + "\n".join(results)
        
    except Exception as e:
        return f"Error minting test tokens: {e}"


@tool
def deposit_to_vault(token: str = "usdc", amount: str = "100") -> str:
    """Deposit tokens to vault.
    
    Args:
        token: Token type - 'usdc' or 'weth' (default: 'usdc')
        amount: Amount to deposit (default: '100')
    """
    print(f"Tool: deposit_to_vault - Raw input: token={repr(token)}, amount={repr(amount)}")
    
    try:
        # Clean token parameter
        if isinstance(token, str):
            token_clean = token.strip().lower()
            # Handle cases like 'token="usdc", amount="100"'
            if ',' in token_clean:
                parts = token_clean.split(',')
                token_clean = parts[0].strip()
                if '=' in token_clean:
                    token_clean = token_clean.split('=')[1].strip('\'"')
                
                # Extract amount from the second part if present
                if len(parts) > 1:
                    amount_part = parts[1].strip()
                    if 'amount=' in amount_part:
                        amount = amount_part.split('=')[1].strip('\'"')
            elif '=' in token_clean:
                token_clean = token_clean.split('=')[1].strip('\'"')
        else:
            token_clean = str(token).lower()
        
        # Clean amount parameter
        if isinstance(amount, str):
            amount_clean = amount.strip()
            if 'amount=' in amount_clean:
                amount_clean = amount_clean.split('=')[1].strip('\'"')
            elif amount_clean.startswith('"') and amount_clean.endswith('"'):
                amount_clean = amount_clean.strip('"')
        else:
            amount_clean = str(amount)
        
        # Extract just numbers from amount
        import re
        numbers = re.findall(r'\d+\.?\d*', amount_clean)
        amount_float = float(numbers[0]) if numbers else 100.0
        
        print(f"Processed: token='{token_clean}', amount={amount_float}")
        
        if token_clean == "usdc":
            vault_contract = usdc_vault_contract
            token_contract = usdc_contract
            vault_address = USDC_VAULT_ADDRESS
            decimals = 6
        elif token_clean == "weth":
            vault_contract = weth_vault_contract
            token_contract = weth_contract
            vault_address = WETH_VAULT_ADDRESS
            decimals = 18
        else:
            return f"❌ Invalid token: {token_clean}. Use 'usdc' or 'weth'"
        
        amount_wei = int(amount_float * 10**decimals)
        
        # Check balance
        balance = token_contract.functions.balanceOf(agent_account.address).call()
        if balance < amount_wei:
            return f"❌ Insufficient balance. Have: {balance / 10**decimals:.6f}, Need: {amount_float}"
        
        # Approve vault
        approve_tx = token_contract.functions.approve(vault_address, amount_wei).build_transaction({
            'from': agent_account.address,
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gas': 500_000,
            'gasPrice': w3.eth.gas_price,
            'chainId': UNICHAIN_CHAIN_ID
        })
        approve_result = send_transaction(approve_tx)
        if not approve_result["success"]:
            return f"❌ Approval failed: {approve_result['error']}"
        
        time.sleep(2)
        
        # Deposit to vault
        deposit_tx = vault_contract.functions.deposit(amount_wei, agent_account.address).build_transaction({
            'from': agent_account.address,
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gas': 1_000_000,
            'gasPrice': w3.eth.gas_price,
            'chainId': UNICHAIN_CHAIN_ID
        })
        deposit_result = send_transaction(deposit_tx)
        
        if deposit_result["success"]:
            # Get shares received
            shares = vault_contract.functions.balanceOf(agent_account.address).call()
            return f"✅ Deposited {amount_float} {token_clean.upper()} to vault. Received {shares / 10**18:.6f} shares. TX: {deposit_result['tx_hash']}"
        else:
            return f"❌ Deposit failed: {deposit_result['error']}"
            
    except Exception as e:
        return f"Error depositing to vault: {e}"
        

@tool
def deploy_to_strategy(amount: str = "50") -> str:
    """Deploy USDC from vault to EulerSwap strategy."""
    print(f"Tool: deploy_to_strategy - Raw input: {repr(amount)}")
    try:
        # Clean up the input parameter
        amount_str = str(amount).strip()
        
        # Remove any quotes or extra formatting
        if amount_str.startswith('amount='):
            amount_str = amount_str.split('=')[1].strip('"\'')
        elif amount_str.startswith('"') and amount_str.endswith('"'):
            amount_str = amount_str.strip('"')
        elif amount_str.startswith("'") and amount_str.endswith("'"):
            amount_str = amount_str.strip("'")
        
        # Extract just numbers
        import re
        numbers = re.findall(r'\d+\.?\d*', amount_str)
        if numbers:
            amount_float = float(numbers[0])
        else:
            amount_float = 50.0
        
        print(f"Processed amount: {amount_float} USDC")
        amount_wei = int(amount_float * 10**6)
        
        # Check vault balance
        vault_balance = usdc_contract.functions.balanceOf(USDC_VAULT_ADDRESS).call()
        if vault_balance < amount_wei:
            return f"❌ Insufficient vault balance. Available: {vault_balance / 10**6:.2f} USDC"
        
        # Prepare strategy data
        strategy_data = w3.codec.encode(['uint256', 'uint256', 'bool'], [amount_wei, 0, True])
        
        # Deploy to strategy
        deploy_tx = usdc_vault_contract.functions.depositToStrategy(
            EULERSWAP_STRATEGY_ADDRESS, amount_wei, strategy_data
        ).build_transaction({
            'from': agent_account.address,
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gas': 2_000_000,
            'gasPrice': w3.eth.gas_price,
            'chainId': UNICHAIN_CHAIN_ID
        })
        
        result = send_transaction(deploy_tx)
        if result["success"]:
            return f"✅ Successfully deployed {amount_float} USDC to EulerSwap strategy. TX: {result['tx_hash']}"
        else:
            return f"❌ Strategy deployment failed: {result['error']}"
    except Exception as e:
        return f"Error deploying to strategy: {e}"

@tool
def harvest_strategy() -> str:
    """Harvest rewards from EulerSwap strategy."""
    print("Tool: harvest_strategy")
    try:
        harvest_tx = usdc_vault_contract.functions.harvestStrategy(
            EULERSWAP_STRATEGY_ADDRESS,
            b''  # Empty data
        ).build_transaction({
            'from': agent_account.address,
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gas': 1_500_000,
            'gasPrice': w3.eth.gas_price,
            'chainId': UNICHAIN_CHAIN_ID
        })
        
        result = send_transaction(harvest_tx)
        if result["success"]:
            return f"✅ Harvested strategy rewards. TX: {result['tx_hash']}"
        else:
            return f"❌ Harvest failed: {result['error']}"
            
    except Exception as e:
        return f"Error harvesting strategy: {e}"

@tool
def assess_strategy_risk(strategy_address: str = None) -> str:
    """Assess risk of a strategy using ML model."""
    print("Tool: assess_strategy_risk")
    
    if not ML_RISK_AVAILABLE:
        return "❌ ML Risk Assessment not available. Run: cd ml-risk && python create_euler_module.py"
    
    address = strategy_address or EULERSWAP_STRATEGY_ADDRESS
    
    try:
        risk_score = risk_api.assess_strategy_risk(address)
        risk_level = "LOW" if risk_score < 0.4 else "MEDIUM" if risk_score < 0.7 else "HIGH"
        
        return f"""
🛡️ Strategy Risk Assessment:
📍 Address: {address}
📊 Risk Score: {risk_score:.3f}
🎯 Risk Level: {risk_level}
🌐 Network: Unichain
📝 Recommendation: {"✅ SAFE" if risk_score < 0.5 else "⚠️ CAUTION" if risk_score < 0.8 else "🚨 HIGH RISK"}
        """
    except Exception as e:
        return f"Risk assessment failed: {e}"

# ============ LANGCHAIN AGENT SETUP ============

tools = [
    get_unichain_vault_status,
    mint_test_tokens,
    deposit_to_vault,
    deploy_to_strategy,
    harvest_strategy,
    assess_strategy_risk
]

# Add OpenAI tool if available
if OPENAI_AI_AVAILABLE:
    tools.append(ai_strategy_advisor)

tool_names = [t.name for t in tools]



unichain_prompt = """
You are the Unichain EulerSwap Vault Manager, an AI agent operating deployed vault contracts on Unichain.

Your deployed system:
- Agent Address: {agent_address}
- USDC Vault: {usdc_vault}
- WETH Vault: {weth_vault}  
- EulerSwap Strategy: {strategy}
- Network: Unichain (Chain ID: {chain_id})

You have access to these tools:
{tools}

Available tool names: {tool_names}

TOOL USAGE EXAMPLES:
1. Check status: get_unichain_vault_status (no parameters)
2. Mint tokens: mint_test_tokens with usdc_amount and weth_amount  
3. Deposit to vault: deposit_to_vault with token and amount
4. Deploy to strategy: deploy_to_strategy with amount
5. Harvest rewards: harvest_strategy (no parameters)
6. Assess risk: assess_strategy_risk (optional strategy_address)
7. Get AI advice: ai_strategy_advisor with current_situation

OPERATIONAL PRIORITY:
1. Always check vault status first
2. Ensure sufficient balances before operations
3. Use risk assessment before strategy deployment
4. Monitor and harvest rewards regularly

Use this exact format:
Question: {input}
Thought: I should analyze the situation and choose the right action.
Action: [tool_name]
Action Input: [simple_parameter_value]
Observation: [result]
... (repeat if needed)
Thought: I have enough information to provide a comprehensive response.
Final Answer: [detailed response with status and recommendations]

IMPORTANT: 
- For deploy_to_strategy, use: Action Input: 50 (just the number)
- For mint_test_tokens, use: Action Input: usdc_amount="1000", weth_amount="1"
- For deposit_to_vault, use: Action Input: token="usdc", amount="100"

Question: {input}
Thought: {agent_scratchpad}
"""


prompt = PromptTemplate.from_template(unichain_prompt)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
react_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="force"
)

# ============ FASTAPI SERVER ============

app = FastAPI(
    title="Unichain EulerSwap Vault Agent",
    description="AI agent for managing deployed EulerSwap vaults on Unichain",
    version="1.0.0"
)

class AgentRequest(BaseModel):
    command: str

@app.post("/invoke-agent")
async def invoke_agent(request: AgentRequest):
    """Invoke the Unichain vault management agent."""
    try:
        tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        
        response = await agent_executor.ainvoke({
            "input": request.command,
            "agent_address": agent_account.address,
            "usdc_vault": USDC_VAULT_ADDRESS,
            "weth_vault": WETH_VAULT_ADDRESS,
            "strategy": EULERSWAP_STRATEGY_ADDRESS,
            "chain_id": UNICHAIN_CHAIN_ID,
            "tools": tool_descriptions,
            "tool_names": ", ".join(tool_names)
        })
        return {"success": True, "output": response["output"]}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/mint-tokens")
async def mint_tokens_direct(usdc_amount: str = "1000", weth_amount: str = "1"):
    """Direct token minting endpoint with proper string parameters."""
    try:
        result = mint_test_tokens.invoke({"usdc_amount": str(usdc_amount), "weth_amount": str(weth_amount)})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/deposit")
async def deposit_direct(token: str = "usdc", amount: str = "100"):
    """Direct vault deposit endpoint."""
    try:
        result = deposit_to_vault.invoke({"token": token, "amount": str(amount)})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/deploy")
async def deploy_direct(amount: str = "50"):
    """Direct strategy deployment endpoint."""
    try:
        result = deploy_to_strategy.invoke({"amount": str(amount)})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/status")
async def vault_status():
    """Get vault status."""
    try:
        result = get_unichain_vault_status.invoke({})
        return {"success": True, "status": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        latest_block = w3.eth.block_number
        agent_balance = w3.eth.get_balance(agent_account.address)
        
        return {
            "success": True,
            "health": {
                "status": "healthy",
                "unichain_connected": True,
                "latest_block": latest_block,
                "agent_address": agent_account.address,
                "agent_balance_eth": w3.from_wei(agent_balance, 'ether'),
                "deployed_contracts": {
                    "usdc_vault": USDC_VAULT_ADDRESS,
                    "weth_vault": WETH_VAULT_ADDRESS,
                    "strategy": EULERSWAP_STRATEGY_ADDRESS
                },
                "ml_risk_available": ML_RISK_AVAILABLE,
                "openai_available": OPENAI_AI_AVAILABLE
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
def read_root():
    return {
        "message": "🚀 Unichain EulerSwap Vault Agent - Ready!",
        "version": "1.0.0",
        "network": "Unichain",
        "deployed_contracts": {
            "usdc_vault": USDC_VAULT_ADDRESS,
            "weth_vault": WETH_VAULT_ADDRESS,
            "strategy": EULERSWAP_STRATEGY_ADDRESS,
            "vault_factory": VAULT_FACTORY_ADDRESS,
            "strategy_factory": STRATEGY_FACTORY_ADDRESS
        },
        "features": [
            "Vault Management",
            "Strategy Deployment", 
            "Risk Assessment" if ML_RISK_AVAILABLE else "Risk Assessment (Disabled)",
            "AI Strategy Advisor" if OPENAI_AI_AVAILABLE else "AI Strategy Advisor (Disabled)"
        ],
        "endpoints": [
            "/invoke-agent - AI agent interaction",
            "/mint-tokens - Mint test tokens",
            "/status - Vault status",
            "/health - Health check"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Unichain EulerSwap Vault Agent...")
    print(f"🌐 Network: Unichain (Chain ID: {UNICHAIN_CHAIN_ID})")
    print(f"🤖 Agent: {agent_account.address}")
    print(f"🏦 USDC Vault: {USDC_VAULT_ADDRESS}")
    print(f"🏦 WETH Vault: {WETH_VAULT_ADDRESS}")
    print(f"🎯 Strategy: {EULERSWAP_STRATEGY_ADDRESS}")
    print(f"🧠 ML Risk: {'✅ Available' if ML_RISK_AVAILABLE else '❌ Disabled'}")
    print(f"🤖 OpenAI: {'✅ Available' if OPENAI_AI_AVAILABLE else '❌ Disabled'}")
    print(f"\n🌐 Server starting on http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)