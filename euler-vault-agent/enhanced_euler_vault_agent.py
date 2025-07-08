import os
import json
import time
from dotenv import load_dotenv
from web3 import Web3
# Fixed import for Web3.py v6+
import web3.middleware
from web3.exceptions import ContractLogicError
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

# Import risk assessment
import sys
sys.path.append('./ml-risk')
try:
    from risk_api import RiskAssessmentAPI
    RISK_MODEL_AVAILABLE = True
    print("✅ Risk model imported successfully")
except ImportError as e:
    print(f"⚠️ Risk model not available: {e}")
    print("Run: cd ml-risk && python anomaly_risk_model.py")
    RISK_MODEL_AVAILABLE = False

# Import OpenAI LLM planner (reuse your existing one)
try:
    from ollama_llm_planner import ai_strategy_advisor  # Uses OpenAI
    OPENAI_AI_AVAILABLE = True
    print("✅ OpenAI LLM planner imported successfully")
except ImportError as e:
    print(f"⚠️ OpenAI LLM planner not available: {e}")
    OPENAI_AI_AVAILABLE = False

# ==============================================================================
# 1. ENHANCED euler CONFIGURATION AND SETUP
# ==============================================================================

load_dotenv()

# --- euler Configuration ---
RPC_URL = os.getenv("NEAR_TESTNET_RPC_URL")  # euler testnet
CHAIN_ID = int(os.getenv("NEAR_TESTNET_CHAIN_ID"))  # euler chain ID
AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Contract Addresses ---
VAULT_ADDRESS = os.getenv("VAULT_ADDRESS")
VRF_STRATEGY_ADDRESS = os.getenv("VRF_STRATEGY_ADDRESS")
USDC_TOKEN_ADDRESS = os.getenv("USDC_TOKEN_ADDRESS")

# --- euler Strategy Configuration ---
euler_STRATEGIES = {
    "ref_finance": os.getenv("REF_FINANCE_STRATEGY_ADDRESS", ""),
    "trisolaris": os.getenv("TRISOLARIS_STRATEGY_ADDRESS", ""),
    "bastion": os.getenv("BASTION_STRATEGY_ADDRESS", "")
}

# --- Web3 Setup for euler ---
w3 = Web3(Web3.HTTPProvider(RPC_URL))
# Fixed middleware injection for Web3.py v6+
try:
    from web3.middleware import geth_poa_middleware
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    print("✅ PoA middleware injected")
except (ImportError, AttributeError):
    print("⚠️ PoA middleware not available, continuing without it")
    print("   This is often fine for euler testnet")

# --- Agent Account Setup ---
agent_account = w3.eth.account.from_key(AGENT_PRIVATE_KEY)
print(f"🤖 euler Agent Wallet Address: {agent_account.address}")

# --- Risk Model Setup ---
if RISK_MODEL_AVAILABLE:
    try:
        risk_api = RiskAssessmentAPI("ml-risk/models/anomaly_risk_model.joblib")
        print("✅ Risk assessment model loaded")
    except Exception as e:
        risk_api = None
        print(f"⚠️ Risk model loading failed: {e}")
else:
    risk_api = None

# --- Load ABIs ---
def load_abi(filename):
    """Loads a contract ABI from the abi directory with robust path handling."""
    possible_paths = [
        os.path.join("abi", filename),
        os.path.join("..", "abi", filename),
        os.path.join("../abi", filename),
        filename
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    abi_data = json.load(f)
                    if isinstance(abi_data, dict) and "abi" in abi_data:
                        return abi_data["abi"]
                    elif isinstance(abi_data, list):
                        return abi_data
                    else:
                        raise ValueError(f"Invalid ABI format in {filename}")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            continue
    
    raise FileNotFoundError(f"Could not find {filename} in any of these locations: {possible_paths}")

try:
    vault_abi = load_abi("Vault.json")
    vrf_strategy_abi = load_abi("NearVrfYieldStrategy.json")
    usdc_abi = load_abi("MockUSDC.json")
    print("✅ All ABI files loaded successfully")
except FileNotFoundError as e:
    print(f"❌ ABI loading failed: {e}")
    print("Please ensure ABI files are in the correct directory")
    exit(1)

# --- Create Contract Objects ---
vault_contract = w3.eth.contract(address=VAULT_ADDRESS, abi=vault_abi)
vrf_strategy_contract = w3.eth.contract(address=VRF_STRATEGY_ADDRESS, abi=vrf_strategy_abi)
usdc_contract = w3.eth.contract(address=USDC_TOKEN_ADDRESS, abi=usdc_abi)

print("✅ Enhanced euler configuration loaded with risk management")

# ==============================================================================
# 2. ENHANCED AGENT TOOLS WITH euler INTEGRATION
# ==============================================================================

def send_transaction(tx):
    """Signs and sends a transaction with Web3.py v6+ compatibility."""
    try:
        # Sign transaction with proper Web3.py v6+ method
        signed_tx = w3.eth.account.sign_transaction(tx, agent_account.key)
        
        # Handle different Web3.py versions
        if hasattr(signed_tx, 'rawTransaction'):
            raw_tx = signed_tx.rawTransaction
        elif hasattr(signed_tx, 'raw_transaction'):
            raw_tx = signed_tx.raw_transaction
        else:
            # For newer versions, the signed transaction might be the raw transaction itself
            raw_tx = signed_tx
        
        tx_hash = w3.eth.send_raw_transaction(raw_tx)
        print(f"⏳ euler transaction sent: {tx_hash.hex()}. Waiting for confirmation...")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        print(f"✅ euler transaction confirmed in block: {receipt.blockNumber}")
        return {"success": True, "receipt": receipt, "tx_hash": tx_hash.hex()}
        
    except ContractLogicError as e:
        print(f"❌ euler transaction reverted: {e}")
        return {"success": False, "error": f"Contract logic error: {e}"}
    except Exception as e:
        print(f"❌ euler transaction error: {e}")
        # More detailed error for debugging
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@tool
def get_enhanced_euler_protocol_status() -> str:
    """
    Gets comprehensive euler protocol status including risk metrics and yield opportunities.
    """
    print("Tool: get_enhanced_euler_protocol_status")
    try:
        # Basic protocol status
        liquid_usdc_wei = usdc_contract.functions.balanceOf(VAULT_ADDRESS).call()
        prize_pool_wei = vrf_strategy_contract.functions.getBalance().call()
        last_winner = vrf_strategy_contract.functions.lastWinner().call()
        
        liquid_usdc = liquid_usdc_wei / (10**6)
        prize_pool = prize_pool_wei / (10**6)
        
        # Risk assessment for euler VRF strategy
        risk_level = "UNKNOWN"
        if risk_api:
            try:
                vrf_risk = risk_api.assess_strategy_risk(VRF_STRATEGY_ADDRESS)
                risk_level = "LOW" if vrf_risk < 0.3 else "MEDIUM" if vrf_risk < 0.7 else "HIGH"
            except Exception as e:
                risk_level = f"UNAVAILABLE ({str(e)[:50]}...)"
        
        # euler-specific yield opportunity analysis
        yield_opportunities = analyze_euler_yield_opportunities()
        
        status_report = {
            "vault_liquid_usdc": f"{liquid_usdc:.2f} USDC",
            "current_prize_pool": f"{prize_pool:.2f} USDC", 
            "last_lottery_winner": last_winner,
            "euler_vrf_strategy_risk_level": risk_level,
            "best_yield_opportunity": yield_opportunities.get("best", "euler VRF Lottery"),
            "total_deployed": f"{prize_pool:.2f} USDC",
            "agent_address": agent_account.address,
            "vault_address": VAULT_ADDRESS,
            "euler_vrf_strategy_address": VRF_STRATEGY_ADDRESS,
            "euler_chain_id": CHAIN_ID,
            "risk_model_available": RISK_MODEL_AVAILABLE,
            "openai_ai_available": OPENAI_AI_AVAILABLE
        }
        
        return f"Enhanced euler Protocol Status: {json.dumps(status_report, indent=2)}"
    except Exception as e:
        return f"Error getting enhanced euler protocol status: {e}"

def analyze_euler_yield_opportunities():
    """Analyze available yield opportunities across euler ecosystem."""
    opportunities = {
        "euler_vrf": {"apy": 0.0, "risk": 0.2, "type": "prize"},
        "ref_finance": {"apy": 15.2, "risk": 0.4, "type": "dex"},
        "trisolaris": {"apy": 12.8, "risk": 0.45, "type": "dex"},
        "bastion": {"apy": 9.1, "risk": 0.35, "type": "lending"},
        "near_stake": {"apy": 10.5, "risk": 0.3, "type": "staking"}
    }
    
    # Calculate risk-adjusted returns
    for name, opp in opportunities.items():
        opp["risk_adjusted_apy"] = opp["apy"] * (1 - opp["risk"])
    
    best = max(opportunities.items(), key=lambda x: x[1]["risk_adjusted_apy"])
    return {"best": best[0], "opportunities": opportunities}

@tool
def assess_euler_strategy_risk(strategy_address: str) -> str:
    """
    Assess the risk level of an euler DeFi strategy before deployment.
    """
    print(f"Tool: assess_euler_strategy_risk for {strategy_address}")
    
    if not risk_api:
        return "Risk assessment unavailable - model not loaded. Run: cd ml-risk && python anomaly_risk_model.py"
    
    try:
        risk_score = risk_api.assess_strategy_risk(strategy_address)
        detailed_assessment = risk_api.get_detailed_assessment(strategy_address)
        
        risk_level = "LOW" if risk_score < 0.3 else "MEDIUM" if risk_score < 0.7 else "HIGH"
        recommendation = "APPROVE" if risk_score < 0.5 else "CAUTION" if risk_score < 0.8 else "REJECT"
        
        return f"""
euler Strategy Risk Assessment for {strategy_address}:
📊 Risk Score: {risk_score:.3f}
🎯 Risk Level: {risk_level}
💡 Recommendation: {recommendation}
🔍 Details: {detailed_assessment.get('risk_level', 'N/A')}
📋 Error (if any): {detailed_assessment.get('error', 'None')}
🌐 Network: euler (NEAR EVM)
        """
    except Exception as e:
        return f"euler risk assessment failed: {e}"

@tool
def deploy_to_euler_strategy_with_risk_check(strategy_name: str, amount: float = 0.0) -> str:
    """
    Deploy funds to an euler strategy after comprehensive risk assessment.
    
    Args:
        strategy_name: Name of the strategy ("euler_vrf", "ref_finance", etc.)
        amount: Amount of USDC to deploy (0 = use reasonable default)
    """
    print(f"Tool: deploy_to_euler_strategy_with_risk_check - {strategy_name}, {amount} USDC")
    
    try:
        # Handle strategy name mapping
        strategy_address = None
        if strategy_name in euler_STRATEGIES and euler_STRATEGIES[strategy_name]:
            strategy_address = euler_STRATEGIES[strategy_name]
        elif strategy_name == "euler_vrf" or strategy_name == "vrf":
            strategy_address = VRF_STRATEGY_ADDRESS
        elif strategy_name == "ref_finance":
            strategy_address = euler_STRATEGIES.get("ref_finance", VRF_STRATEGY_ADDRESS)
            if not euler_STRATEGIES.get("ref_finance"):
                strategy_name = "euler_vrf"
                print("⚠️ Ref Finance strategy not deployed, redirecting to euler VRF lottery")
        
        if not strategy_address:
            available_strategies = ["euler_vrf"] + [k for k, v in euler_STRATEGIES.items() if v]
            return f"❌ Unknown euler strategy: {strategy_name}. Available: {available_strategies}"
        
        # Risk assessment
        if risk_api and strategy_address != VRF_STRATEGY_ADDRESS:
            try:
                risk_score = risk_api.assess_strategy_risk(strategy_address)
                if risk_score > 0.7:
                    return f"❌ DEPLOYMENT BLOCKED - High risk score: {risk_score:.3f}"
                elif risk_score > 0.5:
                    print(f"⚠️ CAUTION - Medium risk score: {risk_score:.3f}, proceeding...")
            except Exception as e:
                print(f"⚠️ Risk assessment failed: {e}, proceeding without risk check...")
        
        # Check available balance
        liquid_usdc_wei = usdc_contract.functions.balanceOf(VAULT_ADDRESS).call()
        liquid_usdc = liquid_usdc_wei / (10**6)
        
        # Set reasonable amount if not specified
        if amount == 0:
            if strategy_name == "euler_vrf" or strategy_name == "vrf":
                amount = min(150.0, liquid_usdc * 0.5)
            else:
                amount = liquid_usdc * 0.8
        
        if amount > liquid_usdc:
            return f"❌ Insufficient funds: {liquid_usdc:.2f} USDC available, {amount:.2f} USDC requested"
        
        if amount <= 0:
            return "❌ No funds available to deploy"
        
        # For euler VRF strategy, use simulate yield harvest and deposit
        if strategy_name == "euler_vrf" or strategy_address == VRF_STRATEGY_ADDRESS:
            return simulate_euler_yield_harvest_and_deposit.invoke({"amount_usdc": amount})
        
        # Execute deployment for other euler strategies
        amount_wei = int(amount * (10**6))
        
        tx = vault_contract.functions.depositToStrategy(
            strategy_address,
            amount_wei,
            b''
        ).build_transaction({
            'from': agent_account.address,
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gas': 2_000_000,
            'gasPrice': w3.eth.gas_price,
            'chainId': CHAIN_ID
        })
        
        result = send_transaction(tx)
        
        if result["success"]:
            return f"✅ Successfully deployed {amount:.2f} USDC to {strategy_name} strategy on euler. TX: {result['tx_hash']}"
        else:
            return f"❌ euler deployment failed: {result['error']}"
            
    except Exception as e:
        return f"❌ Error in euler risk-checked deployment: {e}"

@tool
def simulate_euler_yield_harvest_and_deposit(amount_usdc: float) -> str:
    """Simulates yield harvest with euler-specific enhanced logging and risk awareness."""
    print(f"Tool: simulate_euler_yield_harvest_and_deposit (Amount: {amount_usdc})")
    
    # Risk check: Don't simulate excessive amounts
    if amount_usdc > 1000:
        return "❌ Risk check failed: Simulated yield amount too high (>1000 USDC)"
    
    if amount_usdc <= 0:
        return "❌ Invalid amount: Must be greater than 0"
    
    try:
        amount_wei = int(amount_usdc * (10**6))

        # 1. Mint "yield" to the agent's wallet
        print(f"Minting {amount_usdc} USDC to euler agent...")
        mint_tx = usdc_contract.functions.mint(
            agent_account.address,
            amount_wei
        ).build_transaction({
            'from': agent_account.address,
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gas': 500_000,
            'gasPrice': w3.eth.gas_price,
            'chainId': CHAIN_ID
        })
        mint_result = send_transaction(mint_tx)
        if not mint_result["success"]:
            return f"Failed to mint mock yield on euler: {mint_result['error']}"
        
        time.sleep(2)

        # 2. Approve the euler VRF Strategy
        print(f"Approving euler VRF strategy to spend {amount_usdc} USDC...")
        approve_tx = usdc_contract.functions.approve(
            VRF_STRATEGY_ADDRESS,
            amount_wei
        ).build_transaction({
            'from': agent_account.address,
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gas': 500_000,
            'gasPrice': w3.eth.gas_price,
            'chainId': CHAIN_ID
        })
        approve_result = send_transaction(approve_tx)
        if not approve_result["success"]:
            return f"Failed to approve yield deposit on euler: {approve_result['error']}"
            
        time.sleep(2)

        # 3. Deposit the "yield" into the euler VRF strategy
        print(f"Depositing {amount_usdc} USDC as euler prize pool...")
        deposit_tx = vrf_strategy_contract.functions.depositYield(
            amount_wei
        ).build_transaction({
            'from': agent_account.address,
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gas': 1_000_000,
            'gasPrice': w3.eth.gas_price,
            'chainId': CHAIN_ID
        })
        deposit_result = send_transaction(deposit_tx)
        
        if deposit_result["success"]:
            return f"✅ Successfully simulated and deposited {amount_usdc} USDC as euler prize pool. TX: {deposit_result['tx_hash']}"
        else:
            return f"Failed to deposit yield on euler: {deposit_result['error']}"

    except Exception as e:
        return f"Error simulating euler yield harvest: {e}"

@tool
def trigger_euler_lottery_draw() -> str:
    """Triggers euler lottery draw with enhanced winner tracking and risk checks."""
    print("Tool: trigger_euler_lottery_draw")
    try:
        prize_pool_wei = vrf_strategy_contract.functions.getBalance().call()
        if prize_pool_wei == 0:
            return "Cannot trigger draw: The euler prize pool is zero. Use simulate_euler_yield_harvest_and_deposit() first."

        prize_amount = prize_pool_wei / 10**6
        
        # Safety check: Don't trigger draws for extremely large amounts without confirmation
        if prize_amount > 10000:
            return f"⚠️ Safety check: euler prize amount is very large ({prize_amount:.2f} USDC). Please confirm this is intended."
        
        print(f"Triggering euler lottery draw for a prize of {prize_amount:.2f} USDC...")
        
        tx = vault_contract.functions.harvestStrategy(
            VRF_STRATEGY_ADDRESS,
            b''
        ).build_transaction({
            'from': agent_account.address,
            'nonce': w3.eth.get_transaction_count(agent_account.address),
            'gas': 2_000_000,
            'gasPrice': w3.eth.gas_price,
            'chainId': CHAIN_ID
        })

        result = send_transaction(tx)
        if result["success"]:
            time.sleep(2)
            new_winner = vrf_strategy_contract.functions.lastWinner().call()
            return f"🎉 euler lottery draw successful! Winner: {new_winner}, Prize: {prize_amount:.2f} USDC, TX: {result['tx_hash']}"
        else:
            return f"Failed to trigger euler lottery draw: {result['error']}"

    except Exception as e:
        return f"Error triggering euler lottery draw: {e}"

@tool
def emergency_euler_risk_assessment() -> str:
    """
    Perform emergency risk assessment of all deployed euler funds.
    """
    print("Tool: emergency_euler_risk_assessment")
    
    try:
        risk_summary = {
            "total_at_risk": 0.0,
            "high_risk_strategies": [],
            "medium_risk_strategies": [],
            "low_risk_strategies": [],
            "recommendations": []
        }
        
        # Check euler VRF strategy
        prize_pool_wei = vrf_strategy_contract.functions.getBalance().call()
        prize_pool = prize_pool_wei / (10**6)
        
        if prize_pool > 0:
            risk_summary["low_risk_strategies"].append({
                "name": "euler VRF Lottery",
                "address": VRF_STRATEGY_ADDRESS,
                "balance": prize_pool,
                "risk_score": 0.2,  # VRF is considered low risk
                "notes": "euler VRF-based lottery system on NEAR EVM"
            })
            risk_summary["total_at_risk"] += prize_pool
        
        # Check other euler strategies
        for strategy_name, address in euler_STRATEGIES.items():
            if address and risk_api:
                try:
                    risk_score = risk_api.assess_strategy_risk(address)
                    balance = 0.0  # Would need strategy contract ABI to get actual balance
                    
                    strategy_info = {
                        "name": strategy_name,
                        "address": address,
                        "balance": balance,
                        "risk_score": risk_score,
                        "network": "euler"
                    }
                    
                    if risk_score > 0.7:
                        risk_summary["high_risk_strategies"].append(strategy_info)
                        risk_summary["recommendations"].append(f"URGENT: Exit euler {strategy_name}")
                    elif risk_score > 0.5:
                        risk_summary["medium_risk_strategies"].append(strategy_info)
                        risk_summary["recommendations"].append(f"MONITOR: Watch euler {strategy_name}")
                    else:
                        risk_summary["low_risk_strategies"].append(strategy_info)
                        
                except Exception as e:
                    print(f"Risk check failed for euler {strategy_name}: {e}")
        
        total_strategies = len(risk_summary["high_risk_strategies"]) + \
                          len(risk_summary["medium_risk_strategies"]) + \
                          len(risk_summary["low_risk_strategies"])
        
        return f"""
🚨 Emergency euler Risk Assessment:
📊 Total Strategies: {total_strategies}
💰 Total Funds at Risk: {risk_summary["total_at_risk"]:.2f} USDC
🔴 High Risk Strategies: {len(risk_summary["high_risk_strategies"])}
🟡 Medium Risk Strategies: {len(risk_summary["medium_risk_strategies"])}
🟢 Low Risk Strategies: {len(risk_summary["low_risk_strategies"])}
🌐 Network: euler (NEAR EVM)

📋 Strategy Details:
{json.dumps(risk_summary, indent=2)}

💡 Recommendations: {risk_summary["recommendations"] if risk_summary["recommendations"] else ["All euler strategies appear safe"]}
        """
        
    except Exception as e:
        return f"Emergency euler risk assessment failed: {e}"



@tool
def analyze_euler_ecosystem() -> str:
    """
    Analyze the entire euler ecosystem for yield opportunities and risks.
    """
    print("Tool: analyze_euler_ecosystem")
    
    try:
        opportunities = analyze_euler_yield_opportunities()
        
        # Group by category
        dex_strategies = {k: v for k, v in opportunities["opportunities"].items() if v["type"] == "dex"}
        lending_strategies = {k: v for k, v in opportunities["opportunities"].items() if v["type"] == "lending"}
        yield_strategies = {k: v for k, v in opportunities["opportunities"].items() if v["type"] in ["yield_farming", "advanced_defi"]}
        staking_strategies = {k: v for k, v in opportunities["opportunities"].items() if v["type"] in ["liquid_staking", "prize"]}
        
        ecosystem_analysis = {
            "best_overall_strategy": opportunities["best"],
            "best_strategy_details": opportunities["best_info"],
            "ecosystem_summary": {
                "total_strategies": len(opportunities["opportunities"]),
                "euler_native_strategies": opportunities["euler_ecosystem_count"],
                "dex_options": len(dex_strategies),
                "lending_options": len(lending_strategies),
                "yield_farming_options": len(yield_strategies),
                "staking_options": len(staking_strategies)
            },
            "top_recommendations": {
                "lowest_risk": min(opportunities["opportunities"].items(), key=lambda x: x[1]["risk"]),
                "highest_apy": max(opportunities["opportunities"].items(), key=lambda x: x[1]["apy"]),
                "best_risk_adjusted": opportunities["best"]
            },
            "euler_advantages": [
                "Lower gas costs than Ethereum",
                "EVM compatibility with NEAR security",
                "Fast transaction finality (2-3 seconds)",
                "Growing DeFi ecosystem",
                "Bridge access to NEAR ecosystem"
            ]
        }
        
        return f"""
🌐 euler Ecosystem Analysis:

Best Overall Strategy: {opportunities["best"]} 
└─ APY: {opportunities["best_info"]["apy"]}%
└─ Risk: {opportunities["best_info"]["risk"]}
└─ Risk-Adjusted APY: {opportunities["best_info"]["risk_adjusted_apy"]:.2f}%
└─ Description: {opportunities["best_info"]["description"]}

📊 Ecosystem Summary:
├─ Total Strategies: {ecosystem_analysis["ecosystem_summary"]["total_strategies"]}
├─ euler Native: {ecosystem_analysis["ecosystem_summary"]["euler_native_strategies"]}
├─ DEX Options: {ecosystem_analysis["ecosystem_summary"]["dex_options"]} (Ref Finance, Trisolaris)
├─ Lending Options: {ecosystem_analysis["ecosystem_summary"]["lending_options"]} (Bastion, Burrow)
├─ Yield Farming: {ecosystem_analysis["ecosystem_summary"]["yield_farming_options"]} (Beefy, Pulsar)
└─ Staking Options: {ecosystem_analysis["ecosystem_summary"]["staking_options"]} (Meta Pool, VRF)

🏆 Top Recommendations:
├─ Lowest Risk: {ecosystem_analysis["top_recommendations"]["lowest_risk"][0]} ({ecosystem_analysis["top_recommendations"]["lowest_risk"][1]["risk"]} risk)
├─ Highest APY: {ecosystem_analysis["top_recommendations"]["highest_apy"][0]} ({ecosystem_analysis["top_recommendations"]["highest_apy"][1]["apy"]}% APY)
└─ Best Risk-Adjusted: {ecosystem_analysis["top_recommendations"]["best_risk_adjusted"][0]}

🌟 euler Advantages:
{chr(10).join([f"├─ {advantage}" for advantage in ecosystem_analysis["euler_advantages"]])}

💡 Strategy Recommendation: Focus on {opportunities["best"]} for optimal risk-adjusted returns while maintaining lottery operations with euler VRF.
        """
        
    except Exception as e:
        return f"Error analyzing euler ecosystem: {e}"

@tool  
def deploy_to_euler_ecosystem_strategy(strategy_name: str, amount: float = 150.0) -> str:
    """
    Deploy to specific euler ecosystem strategies with enhanced integration.
    
    Args:
        strategy_name: euler strategy name (ref_finance, beefy_finance, etc.)
        amount: Amount of USDC to deploy
    """
    print(f"Tool: deploy_to_euler_ecosystem_strategy - {strategy_name}, {amount} USDC")
    
    try:
        # Enhanced strategy mapping with euler ecosystem
        strategy_info = {
            # DEX Strategies
            "ref_finance": {
                "address": euler_STRATEGIES.get("ref_finance", ""),
                "description": "euler's leading DEX",
                "risk_score": 0.4,
                "expected_apy": 15.2
            },
            "trisolaris": {
                "address": euler_STRATEGIES.get("trisolaris", ""),
                "description": "Popular euler AMM",
                "risk_score": 0.45,
                "expected_apy": 12.8
            },
            
            # Yield Farming
            "beefy_finance": {
                "address": euler_STRATEGIES.get("beefy_finance", ""),
                "description": "Auto-compounding vaults",
                "risk_score": 0.5,
                "expected_apy": 18.5
            },
            
            # Lending
            "bastion": {
                "address": euler_STRATEGIES.get("bastion", ""),
                "description": "euler native lending",
                "risk_score": 0.35,
                "expected_apy": 9.1
            },
            
            # Default to VRF if unknown
            "euler_vrf": {
                "address": VRF_STRATEGY_ADDRESS,
                "description": "euler VRF lottery system",
                "risk_score": 0.2,
                "expected_apy": 0.0
            }
        }
        
        # Get strategy details
        strategy = strategy_info.get(strategy_name)
        if not strategy:
            available_strategies = list(strategy_info.keys())
            return f"❌ Unknown euler strategy: {strategy_name}. Available: {available_strategies}"
        
        # Check if strategy is deployed
        if not strategy["address"]:
            return f"""
⚠️ {strategy_name} strategy not yet deployed to euler.
📝 Description: {strategy["description"]}
📊 Expected APY: {strategy["expected_apy"]}%
🎯 Risk Score: {strategy["risk_score"]}

Redirecting to euler VRF lottery system for now.
💡 To deploy {strategy_name}: Set {strategy_name.upper()}_STRATEGY_ADDRESS in .env
            """
        
        # Risk assessment
        if risk_api:
            try:
                risk_score = risk_api.assess_strategy_risk(strategy["address"])
                if risk_score > 0.7:
                    return f"❌ DEPLOYMENT BLOCKED - High risk score: {risk_score:.3f} for {strategy_name}"
                elif risk_score > 0.5:
                    print(f"⚠️ CAUTION - Medium risk score: {risk_score:.3f} for {strategy_name}, proceeding...")
            except Exception as e:
                print(f"⚠️ Risk assessment failed for {strategy_name}: {e}")
        
        # For VRF, use yield simulation
        if strategy_name == "euler_vrf" or strategy["address"] == VRF_STRATEGY_ADDRESS:
            return simulate_euler_yield_harvest_and_deposit.invoke({"amount_usdc": amount})
        
        # For other strategies, would need actual deployment logic
        # For now, simulate the deployment
        return f"""
✅ euler Ecosystem Deployment Simulated:
├─ Strategy: {strategy_name}
├─ Description: {strategy["description"]}  
├─ Amount: {amount:.2f} USDC
├─ Expected APY: {strategy["expected_apy"]}%
├─ Risk Score: {strategy["risk_score"]}
└─ Address: {strategy["address"]}

🌐 euler Benefits: Lower gas costs, EVM compatibility, NEAR ecosystem access
💡 Note: Actual deployment requires strategy contract integration
        """
        
    except Exception as e:
        return f"Error deploying to euler ecosystem strategy: {e}"


# ==============================================================================
# 3. ENHANCED LANGCHAIN AGENT FOR euler
# ==============================================================================

# Build tools list dynamically based on availability
tools = [
    get_enhanced_euler_protocol_status,
    assess_euler_strategy_risk,
    deploy_to_euler_strategy_with_risk_check,
    emergency_euler_risk_assessment,
    simulate_euler_yield_harvest_and_deposit,
    trigger_euler_lottery_draw,
    # New euler ecosystem tools
    analyze_euler_ecosystem,
    deploy_to_euler_ecosystem_strategy
]

# Add OpenAI AI tool if available
if OPENAI_AI_AVAILABLE:
    tools.append(ai_strategy_advisor)
    print("✅ AI strategy advisor tool added")

tool_names = [t.name for t in tools]

enhanced_euler_prompt_template = """
You are the "Enhanced euler Vault Manager," an AI agent with advanced risk management capabilities for operating a no-loss prize savings game on euler (NEAR's EVM layer).

Your address: {agent_address}
Your vault: {vault_address}
Your euler VRF strategy: {euler_vrf_strategy_address}
euler Network: NEAR EVM Layer (Chain ID: {euler_chain_id})

ENHANCED euler CAPABILITIES:
🎯 Risk Assessment: Evaluate euler strategies before deployment
🔍 Multi-Strategy Analysis: Compare yield opportunities across euler protocols
🚨 Emergency Monitoring: Detect and respond to risk events
📊 Comprehensive Reporting: Detailed status and metrics for euler
🤖 AI Strategy Advisor: OpenAI for intelligent recommendations (if available)
🌐 euler Integration: Web3 support for NEAR's EVM layer

You have access to these enhanced euler tools:
{tools}

OPERATIONAL PROCEDURE (Enhanced for euler):
1. **Enhanced Assessment**: Use get_enhanced_euler_protocol_status() for comprehensive overview
2. **Risk Evaluation**: Use assess_euler_strategy_risk() for strategy safety analysis
3. **AI Strategy Planning**: Use ai_strategy_advisor() for intelligent recommendations (if available)
4. **Strategic Deployment**: Use deploy_to_euler_strategy_with_risk_check() for safe fund allocation
5. **Emergency Protocols**: Run emergency_euler_risk_assessment() if you detect anomalies
6. **Yield Optimization**: Balance prize rewards with euler ecosystem opportunities
7. **Lottery Execution**: Only trigger draws after confirming adequate prize pools and safety

Use the following format:
Question: the user's request or task
Thought: Consider current euler state, risk factors, and optimal strategy
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat as needed)
Thought: I now have enough information to provide the final answer.
Final Answer: comprehensive response with euler risk assessment and recommendations

euler-SPECIFIC RISK MANAGEMENT RULES:
- Never deploy to strategies with risk score > 0.7
- Always assess risk before new euler deployments
- euler VRF strategy is considered LOW RISK (NEAR EVM-based lottery system)
- Monitor for unusual patterns or high-risk activities in euler ecosystem
- Prioritize user fund safety over yield maximization
- Run emergency assessment if risk indicators spike
- Use AI advisor for complex strategic decisions when available
- Consider euler gas costs and transaction fees (ETH-style)
- Leverage euler's unique DeFi ecosystem (Ref Finance, Trisolaris, Bastion)

euler ECOSYSTEM NOTES:
- You operate on euler testnet (NEAR's EVM layer)
- euler uses Ethereum-style transactions and gas fees
- Risk model should work well since euler = EVM
- euler VRF strategy leverages NEAR's native randomness
- Focus on lottery operations while monitoring euler yield opportunities
- euler transactions use ETH for gas, not NEAR tokens
- Account model uses Ethereum addresses (0x...)

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(enhanced_euler_prompt_template)

# Initialize enhanced LLM and Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
react_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=react_agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=10,
    early_stopping_method="force"
)

# ==============================================================================
# 4. ENHANCED FASTAPI SERVER FOR euler
# ==============================================================================

app = FastAPI(
    title="Enhanced euler Vault Manager Agent",
    description="AI agent with risk management for euler (NEAR EVM) prize savings protocol",
    version="3.0.0"
)

class AgentRequest(BaseModel):
    command: str

class RiskAssessmentRequest(BaseModel):
    strategy_address: str

class YieldRequest(BaseModel):
    amount_usdc: float

@app.post("/invoke-agent")
async def invoke_agent(request: AgentRequest):
    """Enhanced euler agent endpoint with risk management and AI strategy advisor."""
    try:
        tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        
        response = await agent_executor.ainvoke({
            "input": request.command,
            "agent_address": agent_account.address,
            "vault_address": VAULT_ADDRESS,
            "euler_vrf_strategy_address": VRF_STRATEGY_ADDRESS,
            "euler_chain_id": CHAIN_ID,
            "tools": tool_descriptions,
            "tool_names": ", ".join(tool_names)
        })
        return {"success": True, "output": response["output"]}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/assess-risk")
async def assess_risk(request: RiskAssessmentRequest):
    """Dedicated euler risk assessment endpoint."""
    try:
        result = assess_euler_strategy_risk.invoke({"strategy_address": request.strategy_address})
        return {"success": True, "assessment": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/emergency-status")
async def emergency_status():
    """Emergency euler risk monitoring endpoint."""
    try:
        result = emergency_euler_risk_assessment.invoke({})
        return {"success": True, "emergency_assessment": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/enhanced-status")
async def enhanced_status():
    """Enhanced euler protocol status endpoint."""
    try:
        result = get_enhanced_euler_protocol_status.invoke({})
        return {"success": True, "status": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/ai-strategy")
async def ai_strategy_endpoint(request: AgentRequest):
    """AI strategy advisor endpoint for euler (if OpenAI is available)."""
    try:
        if not OPENAI_AI_AVAILABLE:
            return {
                "success": False, 
                "error": "OpenAI AI not available. Check OPENAI_API_KEY in .env"
            }
        
        result = ai_strategy_advisor.invoke({"current_situation": request.command})
        return {"success": True, "ai_strategy": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/generate-yield")
async def generate_yield_direct(request: YieldRequest):
    """Direct euler yield generation endpoint bypassing agent parameter parsing."""
    try:
        result = simulate_euler_yield_harvest_and_deposit.invoke({"amount_usdc": request.amount_usdc})
        return {"success": True, "yield_result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/trigger-lottery")
async def trigger_lottery_direct():
    """Direct euler lottery trigger endpoint."""
    try:
        result = trigger_euler_lottery_draw.invoke({})
        return {"success": True, "lottery_result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
def read_root():
    return {
        "message": "Enhanced euler Vault Manager Agent is running",
        "version": "3.0.0",
        "vault_address": VAULT_ADDRESS,
        "euler_vrf_strategy_address": VRF_STRATEGY_ADDRESS,
        "usdc_token_address": USDC_TOKEN_ADDRESS,
        "agent_address": agent_account.address,
        "euler_chain_id": CHAIN_ID,
        "euler_rpc": RPC_URL,
        "risk_model_available": RISK_MODEL_AVAILABLE,
        "openai_ai_available": OPENAI_AI_AVAILABLE,
        "features": [
            "euler Risk Assessment",
            "euler VRF Lottery Management", 
            "Emergency Monitoring",
            "Enhanced Status Reporting",
            "AI Strategy Advisor (OpenAI)" if OPENAI_AI_AVAILABLE else "AI Strategy Advisor (Not Available)",
            "euler EVM Integration",
            "euler Ecosystem Support (Ref Finance, Trisolaris, Bastion)"
        ],
        "endpoints": [
            "/invoke-agent",
            "/assess-risk", 
            "/emergency-status",
            "/enhanced-status",
            "/ai-strategy" if OPENAI_AI_AVAILABLE else "/ai-strategy (disabled)",
            "/generate-yield",
            "/trigger-lottery"
        ]
    }

@app.get("/health")
async def health_check():
    """Comprehensive euler health check endpoint."""
    try:
        # Test euler connection
        latest_block = w3.eth.block_number
        
        # Test contract connectivity
        vault_balance = usdc_contract.functions.balanceOf(VAULT_ADDRESS).call()
        prize_pool = vrf_strategy_contract.functions.getBalance().call()
        
        # Test agent wallet balance (ETH for gas on euler)
        agent_balance = w3.eth.get_balance(agent_account.address)
        
        health_status = {
            "status": "healthy",
            "timestamp": int(time.time()),
            "euler_connected": True,
            "euler_chain_id": CHAIN_ID,
            "euler_rpc": RPC_URL,
            "latest_block": latest_block,
            "agent_address": agent_account.address,
            "agent_balance_eth": w3.from_wei(agent_balance, 'ether'),
            "vault_usdc_balance": vault_balance / 10**6,
            "prize_pool_usdc": prize_pool / 10**6,
            "risk_model_loaded": risk_api is not None,
            "openai_ai_available": OPENAI_AI_AVAILABLE,
            "contracts_accessible": True
        }
        
        return {"success": True, "health": health_status}
        
    except Exception as e:
        return {
            "success": False, 
            "health": {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time()),
                "euler_chain_id": CHAIN_ID
            }
        }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced euler Vault Manager Agent...")
    print(f"🔧 Risk Model Available: {RISK_MODEL_AVAILABLE}")
    print(f"🤖 OpenAI AI Available: {OPENAI_AI_AVAILABLE}")
    print(f"🌐 euler Network: NEAR EVM Layer")
    print(f"🆔 euler Chain ID: {CHAIN_ID}")
    print(f"💰 Agent Address: {agent_account.address}")
    print(f"🏦 Vault Address: {VAULT_ADDRESS}")
    print(f"🎲 euler VRF Strategy: {VRF_STRATEGY_ADDRESS}")
    print(f"💵 USDC Token: {USDC_TOKEN_ADDRESS}")
    
    # Feature summary
    features = []
    if RISK_MODEL_AVAILABLE:
        features.append("✅ ML Risk Assessment")
    else:
        features.append("❌ ML Risk Assessment (train model)")
        
    if OPENAI_AI_AVAILABLE:
        features.append("✅ OpenAI Strategy Advisor")
    else:
        features.append("❌ OpenAI Strategy Advisor (setup API key)")
    
    features.append("✅ euler VRF Lottery Management")
    features.append("✅ Emergency Monitoring")
    features.append("✅ Enhanced Status Reporting")
    features.append("✅ euler EVM Integration")
    features.append("✅ euler Ecosystem Support")
    
    print("\n📋 Available Features:")
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n🌐 Starting server on http://localhost:8000")
    print(f"📚 API docs: http://localhost:8000/docs")
    print(f"🔍 Health check: http://localhost:8000/health")
    print(f"\n🎯 Ready for euler DeFi vault management!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)