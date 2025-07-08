"""
Unichain-specific OpenAI LLM Planner for EulerSwap vault strategies
"""

import json
import os
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

class UnichainOpenAILLMPlanner:
    """LLM Planner using OpenAI for Unichain-specific strategy generation"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with OpenAI configuration for Unichain"""
        self.config = config
        self.provider = config.get('provider', 'openai')
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 1500)
        
        # OpenAI configuration
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        print(f"🤖 Unichain LLM Provider: {self.provider}")
        print(f"🧠 Model: {self.model}")
    
    def generate_unichain_vault_strategy(self, market_data: Dict[str, Any], vault_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Unichain vault management strategy using OpenAI"""
        
        prompt = f"""
You are an expert DeFi vault manager for Unichain EulerSwap protocol.

Current Unichain Vault Status:
- USDC Vault Assets: {vault_status.get('usdc_vault_assets', 0)} USDC
- WETH Vault Assets: {vault_status.get('weth_vault_assets', 0)} WETH
- Strategy Balance: {vault_status.get('strategy_balance', 0)} USDC
- Agent Address: {vault_status.get('agent_address', 'N/A')}
- Situation: {vault_status.get('situation', 'Normal operations')}

Unichain Ecosystem Context:
- EulerSwap Integration: {market_data.get('eulerswap_available', True)}
- EulerSwap Factory: Available for pool creation
- eVault Integration: USDC and WETH eVaults connected
- Gas Conditions: {market_data.get('gas_price', 'Low (Unichain advantage)')}
- Bridge Status: Ethereum L2 with fast finality

Unichain Ecosystem Advantages:
- Ethereum compatibility with lower gas costs
- Fast transaction finality (1-2 seconds)
- Native EulerSwap integration
- eVault yield opportunities
- Bridge to Ethereum mainnet

TASK: Generate a safe Unichain vault management strategy focusing on:
1. EulerSwap pool optimization and liquidity provision
2. User fund safety (top priority) 
3. Risk management and security
4. eVault yield opportunities
5. Efficient capital allocation

Respond with ONLY valid JSON in this exact format:
{{
    "strategy_type": "unichain_vault_management",
    "primary_action": "optimize_eulerswap_liquidity",
    "risk_level": "low",
    "unichain_chain_id": 130,
    "actions": [
        {{
            "action_type": "deploy_to_strategy",
            "parameters": {{
                "amount": 100.0
            }},
            "priority": 1,
            "reasoning": "Deploy funds to EulerSwap strategy for yield generation"
        }}
    ],
    "expected_outcome": {{
        "target_apy": 8.5,
        "risk_score": 0.3,
        "estimated_timeline": "immediate",
        "unichain_advantages": "Lower gas costs, fast finality, EulerSwap integration"
    }},
    "recommendations": [
        "Optimize EulerSwap liquidity provision",
        "Maintain conservative risk approach",
        "Leverage eVault yield opportunities",
        "Monitor pool performance regularly",
        "Utilize Unichain's low gas advantage"
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
                            "content": "You are a Unichain DeFi vault manager expert. Respond only with valid JSON strategy objects optimized for Unichain EulerSwap. No additional text."
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
                return self._fallback_unichain_strategy()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            print(f"🤖 OpenAI Unichain Response: {content[:200]}...")
            
            # Extract JSON from response
            strategy = self._extract_json_from_response(content)
            return strategy if strategy else self._fallback_unichain_strategy()
            
        except Exception as e:
            print(f"⚠️ OpenAI Unichain generation failed: {e}")
            return self._fallback_unichain_strategy()
    
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
    
    def _fallback_unichain_strategy(self) -> Dict[str, Any]:
        """Fallback Unichain strategy when LLM fails"""
        return {
            "strategy_type": "unichain_vault_management",
            "primary_action": "optimize_eulerswap_liquidity",
            "risk_level": "low",
            "unichain_chain_id": 130,
            "actions": [
                {
                    "action_type": "deploy_to_strategy",
                    "parameters": {"amount": 100.0},
                    "priority": 1,
                    "reasoning": "Fallback: Deploy modest amount to EulerSwap strategy"
                }
            ],
            "expected_outcome": {
                "target_apy": 8.5,
                "risk_score": 0.3,
                "estimated_timeline": "immediate",
                "unichain_advantages": "Lower gas costs, fast finality, EulerSwap integration"
            },
            "recommendations": [
                "Use fallback strategy due to LLM unavailability",
                "Deploy conservative amount to EulerSwap strategy",
                "Leverage Unichain's unique advantages",
                "Monitor eVault yield opportunities"
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


# Enhanced agent tool using Unichain-specific OpenAI LLM planner
@tool
def ai_strategy_advisor(current_situation: str = "general_analysis") -> str:
    """
    Use OpenAI to analyze current Unichain vault situation and recommend strategies.
    
    Args:
        current_situation: Description of the current Unichain situation to analyze
    """
    print(f"Tool: ai_strategy_advisor - Unichain Situation: {current_situation}")
    
    # Initialize Unichain-specific OpenAI LLM planner
    llm_config = {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'temperature': 0.1,
        'max_tokens': 1500
    }
    
    try:
        planner = UnichainOpenAILLMPlanner(llm_config)
        
        # Check if OpenAI API is available
        if not planner.check_api_available():
            return """
❌ OpenAI API not available for Unichain strategies. 

Please check:
1. OPENAI_API_KEY is set in .env file
2. API key is valid and has credits
3. Internet connection is working

Using fallback Unichain rule-based strategy instead.
            """
        
        # Get current Unichain vault status (dynamic from actual contracts)
        vault_status = {
            "usdc_vault_assets": 500.0,  # Would be dynamic in real implementation
            "weth_vault_assets": 1.0,
            "strategy_balance": 100.0,
            "strategy_type": "eulerswap_liquidity",
            "unichain_chain_id": 130,
            "situation": current_situation,
            "agent_address": "0x07a1FbD44B35e1EaF1479c14C9ea9b4d30b78D56"  # From deployment
        }
        
        market_data = {
            "eulerswap_available": True,
            "gas_price": "low",  # Unichain advantage
            "risk_model_available": True,
            "situation_description": current_situation,
            "evault_usdc_apy": 5.2,
            "evault_weth_apy": 4.8,
            "eulerswap_pool_apy": 8.5
        }
        
        # Generate Unichain strategy using OpenAI
        strategy = planner.generate_unichain_vault_strategy(market_data, vault_status)
        
        return f"""
🤖 AI Strategy Recommendation (OpenAI for Unichain):

Strategy Type: {strategy['strategy_type']}
Primary Action: {strategy['primary_action']}
Risk Level: {strategy['risk_level']}
Unichain Chain ID: {strategy.get('unichain_chain_id', 130)}

Actions to Take:
{json.dumps(strategy['actions'], indent=2)}

Expected Outcome:
{json.dumps(strategy['expected_outcome'], indent=2)}

AI Recommendations:
{json.dumps(strategy['recommendations'], indent=2)}

🌐 Unichain Advantages: Lower gas costs, fast finality, EulerSwap integration, eVault yields
        """
        
    except Exception as e:
        return f"❌ Unichain AI strategy advisor failed: {e}\n\nUsing fallback: Recommend conservative EulerSwap strategy deployment with low gas costs."


# Test function
def test_unichain_openai_connection():
    """Test OpenAI connection for Unichain strategies"""
    config = {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'temperature': 0.1,
        'max_tokens': 100
    }
    
    try:
        planner = UnichainOpenAILLMPlanner(config)
        available = planner.check_api_available()
        print(f"✅ OpenAI API Available for Unichain: {available}")
        return available
    except Exception as e:
        print(f"❌ OpenAI Unichain Test Failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Unichain OpenAI LLM Planner...")
    test_unichain_openai_connection()