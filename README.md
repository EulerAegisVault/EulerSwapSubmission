# Aegis Vault: AI-Powered ML-Strategy Vault on Unichain Powered by Euler

**Submission for the EulerSwap Builder Competition**

This project introduces a sophisticated, autonomous asset management system designed to integrate seamlessly with the EulerSwap ecosystem. We have built an AI-powered agent that acts as an on-chain portfolio manager, leveraging machine learning for real-time risk assessment. Our system automates the complex process of liquidity provision and yield farming, creating a "set-and-forget" solution for maximizing returns from EulerSwap's unique architecture.

---

## Table of Contents

1.  [**Project Vision & Goal**](#project-vision--goal)
2.  [**System Architecture**](#system-architecture)
    -   [On-Chain Layer: The Foundation](#on-chain-layer-the-foundation)
    -   [AI Agent Layer: The Brain](#ai-agent-layer-the-brain)
    -   [Machine Learning Layer: The Shield](#machine-learning-layer-the-shield)
3.  [**How It Works: A Complete Workflow**](#how-it-works-a-complete-workflow)
4.  [**Technical Deep Dive**](#technical-deep-dive)
5.  [**Getting Started & How to Run**](#getting-started--how-to-run)
6.  [**Future Roadmap**](#future-roadmap)

---

## Project Vision & Goal

The core challenge in DeFi today is complexity. While protocols like EulerSwap offer powerful new ways to generate yield, effectively managing liquidity, hedging impermanent loss, and harvesting rewards requires constant monitoring and deep expertise.

Our goal is to abstract this complexity away from the end-user. We've built an autonomous system that uses AI to make intelligent, data-driven decisions on behalf of its users. The **EulerSwap AI Vault** is designed to be a fully automated fund manager that not only seeks the best returns within EulerSwap but also actively protects capital using a novel ML-based risk framework.

---

## System Architecture

Our system is built on three distinct but interconnected pillars, creating a powerful synergy between on-chain contracts, off-chain intelligence, and proactive security.

```
+--------------------------+      +------------------------+      +-----------------------+
| ML Risk Assessment Model |<---->|   AI Agent (Python)    |----->| On-Chain Vault        |
| (Scikit-learn)           |      | (LangChain & FastAPI)   |      | (Solidity / ERC-4626) |
+--------------------------+      +------------------------+      +-----------------------+
           ^                                                            | (deploys to)
           | (scores risk)                                              |
           |                                                            v
           +------------------------------------------------------+-----------------------+
                                                                  | EulerSwapStrategy     |
                                                                  +-----------------------+
                                                                            |
                                                                            v
                                                                  +-----------------------+
                                                                  | EulerSwap LP Position |
                                                                  +-----------------------+
                                                                  
```


### On-Chain Layer: The Foundation

The on-chain layer is the system's backbone, built with robust, modular, and gas-efficient Solidity contracts.

-   **`Vault.sol`**: This is the user-facing contract, fully compliant with the **ERC-4626 Tokenized Vault Standard**. This ensures interoperability with other DeFi protocols and tooling. Users deposit assets (e.g., USDC, WETH) and receive vault shares representing their portion of the managed assets.
-   **`EulerSwapStrategy.sol`**: This contract contains the specialized logic to interact directly with EulerSwap. It receives funds from the `Vault` and executes the strategy—in this case, providing liquidity to an EulerSwap pool to earn trading fees and lending yield simultaneously.
-   **Factories (`VaultFactory.sol`, `StrategyFactory.sol`)**: To avoid the 24KB contract size limit and ensure the system is scalable, we use a dual-factory model. This separates the deployment logic for vaults and strategies, allowing us to add new, complex strategies in the future without bloating a single factory contract.

### AI Agent Layer: The Brain

The agent is the command and control center, operating autonomously to manage the on-chain contracts.

-   **Technology Stack**: Built in Python using **FastAPI** for a robust API and **LangChain** to structure the AI's decision-making process. It is powered by OpenAI's `gpt-4o-mini` model.
-   **ReAct Framework**: The agent uses the "Reasoning and Acting" (ReAct) framework. It can reason about a goal, choose a tool to execute, observe the outcome, and repeat this loop until the task is complete.
-   **Core Tools**: The agent is equipped with a set of tools that serve as its interface to the blockchain:
    -   `get_system_status`: Fetches a full report of all vault and strategy balances.
    -   `mint_test_tokens`: Mints mock assets for testing and simulation.
    -   `deposit_into_vault`: Executes a deposit transaction into the specified vault.
    -   `deploy_to_euler_strategy`: The key action. It moves capital from the vaults into the EulerSwap strategy to be put to work.
    -   `harvest_euler_strategy`: Triggers the harvesting of accumulated fees from the strategy.

### Machine Learning Layer: The Shield

This is our most innovative component—a proactive security layer that goes beyond simple audits.

-   **Purpose**: To assess the risk of any smart contract *before* deploying capital. This is crucial for a system designed to be extensible with new, third-party strategies in the future.
-   **Model**: We use an **Isolation Forest**, an unsupervised learning algorithm ideal for anomaly detection. This is a powerful choice because it doesn't require pre-labeled data of "hacks" or "scams."
-   **Methodology**: The model is trained on the on-chain transaction patterns of well-established, battle-tested protocols (e.g., Aave, Compound, Uniswap). It learns what "normal", safe contract behavior looks like across dozens of features (transaction frequency, value distribution, gas patterns, user concentration, etc.). When presented with a new strategy contract, it can generate a risk score based on how much its behavior deviates from these safe norms. The AI agent can then use this score to automatically avoid deploying funds to a contract that seems anomalous or potentially malicious.

---

## How It Works: A Complete Workflow

1.  **Deposit**: A user finds our vault and deposits 10,000 USDC, receiving vault shares in return.
2.  **AI Detection**: The autonomous agent, running 24/7, periodically calls `get_system_status`. It observes that there is now 10,000 USDC sitting idle in the `Vault`.
3.  **ML Risk Assessment**: Before acting, the agent passes the `EulerSwapStrategy` contract address to the ML model. The model analyzes its on-chain footprint and returns a low risk score, giving the agent the green light to proceed.
4.  **AI Decision & Execution**: The agent's LLM determines that the idle capital should be deployed to generate yield. It formulates a plan, selects the `deploy_to_euler_strategy` tool, and constructs the transaction to move the 10,000 USDC.
5.  **On-Chain Action**: The agent sends the signed transaction to the Unichain network. The `Vault` contract receives the call and transfers the 10,000 USDC to the `EulerSwapStrategy`.
6.  **Liquidity Provision**: The `EulerSwapStrategy` contract then interacts with EulerSwap, depositing the USDC into a liquidity pool. The funds are now actively earning trading fees and lending yield.
7.  **Automated Harvesting**: Days later, the agent observes that fees have accrued. It executes the `harvest_euler_strategy` tool. The strategy contract claims the fees and sends them back to the main `Vault`, increasing the total assets and thus the value of every user's vault shares.

---

## Technical Deep Dive

-   **Smart Contracts**: Written in **Solidity v0.8.13**. Leverages OpenZeppelin's battle-tested libraries for security (e.g., `Ownable`, `ERC4626`).
-   **AI Agent**: Built with **Python 3.10+**.
    -   **FastAPI**: Provides a clean, modern API for interacting with the agent.
    -   **LangChain**: Uses `create_react_agent` and `AgentExecutor` to power the agent's core logic loop.
    -   **LLM**: Integrated with **OpenAI's `gpt-4o-mini`** for its strong reasoning capabilities and cost-effectiveness.
-   **Machine Learning**: Uses **Scikit-learn** to implement the `IsolationForest` model and `StandardScaler` for feature normalization.

---

## Getting Started & How to Run

Follow these steps to run the entire system locally.

#### 0. Deploy the contracts

#### 1. Add your environment variables
```
UNICHAIN_RPC_URL=https://unichain-rpc.publicnode.com
UNICHAIN_CHAIN_ID=130


VAULT_ADDRESS="0xf0f994B4A8dB86A46a1eD4F12263c795b26703Ca"
VRF_STRATEGY_ADDRESS="0x959e85561b3cc2E2AE9e9764f55499525E350f56"
USDC_TOKEN_ADDRESS="0xC0933C5440c656464D1Eb1F886422bE3466B1459"
# or your own versions

AGENT_PRIVATE_KEY=

OPENAI_API_KEY=
```

#### 2. Run the AI Agent
-   Start the agent server: 
`python euler-vault-agent/unichain_vault_agent.py`


#### 3. Test the System
-   In a new terminal, test the endpoints, or deploy the frontend locally to interact:
`cd frontend`
`npm install`
`npm run dev`

---

## Future Roadmap

We are committed to building this project out far beyond the hackathon and have a clear roadmap to transition it into a production-ready, feature-rich DeFi protocol.

1.  **Production Hardening & Security**: Our highest priority is to conduct comprehensive, professional security audits of all smart contracts to ensure user funds are completely secure before a public launch.

2.  **Multi-Chain Strategy Expansion**: The vault architecture is designed to be chain-agnostic. We plan to expand by integrating new yield strategies from other leading protocols across various EVM-compatible chains, allowing the AI agent to perform true multi-protocol yield optimization.

3.  **No-Loss Lottery Feature**: A primary goal is to leverage this system to create a **no-loss lottery**. In this model, the collective yield generated by the AI-managed strategy would form the prize pool awarded to a single winner. Afterwards, all participants get their initial deposit back in full, creating a risk-free way to participate.

4.  **Enhancing the AI & ML Core**: We will continue to train our ML risk model with more diverse on-chain data. We also plan to enhance the AI agent's decision-making capabilities, enabling it to manage more complex strategies and react to market changes with greater sophistication.

5.  **Team Expansion**: We are actively looking to bring more developers on board to accelerate progress. We already know a couple of talented people who are interested in taking this project further with us, and we are excited to build out the team to realize the full potential of this AI-driven DeFi system.









Quickly overview static endpoints:
GET /health
Provides a health check of the service, confirming connection to the blockchain and that all models are loaded.

GET /status
Directly invokes the get_unichain_vault_status tool and returns the raw JSON status.


POST /mint-tokens
For testing, allows the agent to directly mints test tokens. Can take usdc_amount and weth_amount as query parameters.

POST /deposit
Directly deposits tokens to a vault. Takes token and amount as query parameters.

POST /deploy
Directly deploys USDC from the vault to the strategy. Takes amount as a query parameter.

Now, we could just hit a simple /status endpoint for raw data. But that traditional approach has a major limitation: it requires a rigid, one-to-one mapping of endpoints for every possible action, and it strictly requires both the intent and activation to exist externally. 

curl -X POST http://localhost:8000/invoke-agent \
-H "Content-Type: application/json" \
-d '{"command": "What is the current vault status?"}'
As a first simple example of our agent, our agent can be layered in to these existing requests:
The router correctly identified the intent as 'reporting' and used the simple reporting chain to give us a clean summary, without ever touching the more complex transaction agents.

But this is just the entry point.

By giving the agent access to wider context—seeing all of the contract internals, on-chain data, external sources, and crucially, our machine learning risk model—we allow the agent to create complex chains of thought.

It can now synthesize all this information to move beyond simple commands and act as a true, autonomous vault manager and yield farmer using our EulerSwap strategy.



curl -X POST http://localhost:8000/invoke-agent -H "Content-Type: application/json" -d '{"command": "Check vault status and deploy 25 USDC to strategy"}'
^^^ combine into one below


Here we show an example of a multi-step command that mixes different goals. Let's give it a real-world task: I want it to check the risk of our strategy, and only if it's safe, deposit 100 USDC and then deploy that to the strategy. This is where the Planner Agent comes in."

(What you do): Run the complex command. This is the most important part of the demo. Keep your eyes on the server logs.

Bash

curl -X POST http://localhost:8000/invoke-agent \
-H "Content-Type: application/json" \
-d '{"command": "First, assess the risk of the main strategy. If it seems safe, deposit 100 USDC into the vault and then deploy it."}'
(What you say): "Watch the logs. The first thing our system does is engage the Planner. You can see the plan it created: a clear, three-step process.

(Point to the logs): "Now, the system executes that plan step-by-step.

Step 1: It routes the 'assess risk' task to our Risk Agent.

Step 2: It routes the 'deposit' task to our firewalled Transaction Agent.

Step 3: It routes the 'deploy' task, again to the Transaction Agent.

Finally, after successfully completing the plan, it gives us a perfect summary of everything it did.


This is a delta-neutral, risk-managed strategy that would take a human hours of calculation and dozens of clicks to perform manually. Our agent does it in seconds, with one command.

But the ultimate vision for this system goes beyond just executing our commands faster. We've designing it to run in a fully autonomous mode, acting as a tireless vault manager even when we're offline.

In this mode, the agent extends to proactively scanning online data and the mempool for profitable opportunities unique to EulerSwap, like providing Just-in-Time liquidity for large incoming swaps while constantly aided by our ML risk model evaluate the precise profit versus the execution risk.
