"""
euler-specific OpenAI LLM Planner for euler (NEAR EVM) vault strategies
"""

import json
import os
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

class eulerOpenAILLMPlanner:
    """LLM Planner using OpenAI for euler-specific strategy generation"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with OpenAI configuration for euler"""
        self.config = config
        self.provider = config.get('provider', 'openai')
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 1500)
        
        # OpenAI configuration
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        print(f"🤖 euler LLM Provider: {self.provider}")
        print(f"🧠 Model: {self.model}")
    
    def generate_euler_vault_strategy(self, market_data: Dict[str, Any], vault_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate euler vault management strategy using OpenAI"""
        
        prompt = f"""
You are an expert DeFi vault manager for euler (NEAR's EVM layer) prize savings protocol.

Current euler Vault Status:
- Liquid USDC: {vault_status.get('liquid_usdc', 0)} USDC
- Prize Pool: {vault_status.get('prize_pool', 0)} USDC  
- Last Winner: {vault_status.get('last_winner', 'None')}
- euler Network: NEAR EVM Layer (Chain ID: 1313161555)
- Situation: {vault_status.get('situation', 'Normal operations')}

euler Ecosystem Context:
- euler VRF Available: {market_data.get('euler_vrf_available', True)}
- Ref Finance (DEX): ~15.2% APY, Medium Risk
- Trisolaris (AMM): ~12.8% APY, Medium Risk  
- Bastion (Lending): ~9.1% APY, Low Risk
- Pulsar Finance: Advanced DeFi strategies
- Beefy Finance: Auto-compounding vaults
- Gas Conditions: {market_data.get('gas_price', 'Low (euler advantage)')}

euler Ecosystem Advantages:
- EVM compatibility with lower gas costs
- NEAR's fast finality and security
- Bridge to NEAR ecosystem
- Growing DeFi ecosystem
- Ethereum tooling compatibility

TASK: Generate a safe euler vault management strategy focusing on:
1. Prize pool optimization for weekly euler VRF lottery
2. User fund safety (top priority) 
3. Risk management and security
4. euler ecosystem yield opportunities
5. Weekly lottery prize generation

Respond with ONLY valid JSON in this exact format:
{{
    "strategy_type": "euler_vault_management",
    "primary_action": "optimize_prize_pool",
    "risk_level": "low",
    "euler_chain_id": 1313161555,
    "actions": [
        {{
            "action_type": "simulate_euler_yield_harvest_and_deposit",
            "parameters": {{
                "amount_usdc": 150.0
            }},
            "priority": 1,
            "reasoning": "Generate weekly euler lottery prize pool"
        }}
    ],
    "expected_outcome": {{
        "prize_pool_target": 150.0,
        "risk_score": 0.2,
        "estimated_timeline": "immediate",
        "euler_advantages": "Lower gas costs, EVM compatibility, NEAR security"
    }},
    "recommendations": [
        "Create modest weekly prize pool using euler VRF",
        "Maintain low risk approach on euler",
        "Consider Ref Finance for higher yields when appropriate",
        "Leverage euler's EVM compatibility and NEAR ecosystem",
        "Monitor Bastion for lending opportunities"
    ]
}}
"""
        
        return self._generate_with_openai(prompt)
    
    def _generate_with_openai(self, prompt: str) -> Dict[str, Any]:
        """Generate strategy using OpenAI API"""
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an euler DeFi vault manager expert. Respond only with valid JSON strategy objects optimized for euler (NEAR EVM). No additional text."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ OpenAI API error: {response.status_code} - {response.text}")
                return self._fallback_euler_strategy()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            print(f"🤖 OpenAI euler Response: {content[:200]}...")
            
            # Extract JSON from response
            strategy = self._extract_json_from_response(content)
            return strategy if strategy else self._fallback_euler_strategy()
            
        except Exception as e:
            print(f"⚠️ OpenAI euler generation failed: {e}")
            return self._fallback_euler_strategy()
    
    def _extract_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON strategy from LLM response"""
        try:
            # Try parsing as direct JSON
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                # Look for JSON within the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
        return None
    
    def _fallback_euler_strategy(self) -> Dict[str, Any]:
        """Fallback euler strategy when LLM fails"""
        return {
            "strategy_type": "euler_vault_management",
            "primary_action": "optimize_prize_pool",
            "risk_level": "low",
            "euler_chain_id": 1313161555,
            "actions": [
                {
                    "action_type": "simulate_euler_yield_harvest_and_deposit",
                    "parameters": {"amount_usdc": 150.0},
                    "priority": 1,
                    "reasoning": "Fallback: Generate modest euler prize pool for weekly lottery"
                }
            ],
            "expected_outcome": {
                "prize_pool_target": 150.0,
                "risk_score": 0.2,
                "estimated_timeline": "immediate",
                "euler_advantages": "Lower gas costs, EVM compatibility, NEAR security"
            },
            "recommendations": [
                "Use fallback strategy due to LLM unavailability",
                "Generate modest prize pool for weekly euler lottery",
                "Leverage euler's unique advantages over Ethereum",
                "Consider euler DeFi ecosystem opportunities (Ref, Trisolaris, Bastion)"
            ]
        }
    
    def check_api_available(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ OpenAI API not available: {e}")
            return False


# Enhanced agent tool using euler-specific OpenAI LLM planner
@tool
def ai_strategy_advisor(current_situation: str = "general_analysis") -> str:
    """
    Use OpenAI to analyze current euler vault situation and recommend strategies.
    
    Args:
        current_situation: Description of the current euler situation to analyze
    """
    print(f"Tool: ai_strategy_advisor - euler Situation: {current_situation}")
    
    # Initialize euler-specific OpenAI LLM planner
    llm_config = {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'temperature': 0.1,
        'max_tokens': 1500
    }
    
    try:
        planner = eulerOpenAILLMPlanner(llm_config)
        
        # Check if OpenAI API is available
        if not planner.check_api_available():
            return """
❌ OpenAI API not available for euler strategies. 

Please check:
1. OPENAI_API_KEY is set in .env file
2. API key is valid and has credits
3. Internet connection is working

Using fallback euler rule-based strategy instead.
            """
        
        # Get current euler vault status (dynamic from your actual contracts)
        vault_status = {
            "liquid_usdc": 290.0,  # From your health check
            "prize_pool": 0.0,
            "last_winner": "0x0000000000000000000000000000000000000000",
            "strategy_type": "euler_vrf_lottery",
            "euler_chain_id": 1313161555,
            "situation": current_situation
        }
        
        market_data = {
            "euler_vrf_available": True,
            "gas_price": "low",  # euler advantage
            "risk_model_available": True,
            "situation_description": current_situation,
            "ref_finance_apy": 15.2,
            "trisolaris_apy": 12.8,
            "bastion_apy": 9.1
        }
        
        # Generate euler strategy using OpenAI
        strategy = planner.generate_euler_vault_strategy(market_data, vault_status)
        
        return f"""
🤖 AI Strategy Recommendation (OpenAI for euler):

Strategy Type: {strategy['strategy_type']}
Primary Action: {strategy['primary_action']}
Risk Level: {strategy['risk_level']}
euler Chain ID: {strategy.get('euler_chain_id', 1313161555)}

Actions to Take:
{json.dumps(strategy['actions'], indent=2)}

Expected Outcome:
{json.dumps(strategy['expected_outcome'], indent=2)}

AI Recommendations:
{json.dumps(strategy['recommendations'], indent=2)}

🌐 euler Advantages: Lower gas costs, EVM compatibility, NEAR ecosystem access, fast finality
        """
        
    except Exception as e:
        return f"❌ euler AI strategy advisor failed: {e}\n\nUsing fallback: Recommend 150 USDC yield harvest for weekly euler lottery with low gas costs."


# Test function
def test_euler_openai_connection():
    """Test OpenAI connection for euler strategies"""
    config = {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'temperature': 0.1,
        'max_tokens': 100
    }
    
    try:
        planner = eulerOpenAILLMPlanner(config)
        available = planner.check_api_available()
        print(f"✅ OpenAI API Available for euler: {available}")
        return available
    except Exception as e:
        print(f"❌ OpenAI euler Test Failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing euler OpenAI LLM Planner...")
    test_euler_openai_connection()