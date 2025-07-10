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
+-------------------------+      +------------------------+      +-----------------------+
| Mathematical Analysis   |<---->|   AI Agent (Python)    |----->| On-Chain Vault        |
| (10+ Physics Frameworks)|      | (LangChain & FastAPI)   |      | (Solidity / ERC-4626) |
+-------------------------+      +------------------------+      +-----------------------+
^                               ^                              | (deploys to)
| (quantum predictions)         |                              |
|                               v                              v
+-------------------------+      +------------------------+      +-----------------------+
| ML Risk Assessment Model|      | Mathematical Optimizer |      | EulerSwapStrategy     |
| (Scikit-learn)          |      | (Real-time Physics)     |      +-----------------------+
+-------------------------+      +------------------------+              |
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
    -   `execute_mathematical_analysis`: Applies 10+ mathematical frameworks for research-level strategy optimization

### Machine Learning Layer: The Shield

This is our most innovative component—a proactive security layer that goes beyond simple audits.

-   **Purpose**: To assess the risk of any smart contract *before* deploying capital. This is crucial for a system designed to be extensible with new, third-party strategies in the future.
-   **Model**: We use an **Isolation Forest**, an unsupervised learning algorithm ideal for anomaly detection. This is a powerful choice because it doesn't require pre-labeled data of "hacks" or "scams."
-   **Methodology**: The model is trained on the on-chain transaction patterns of well-established, battle-tested protocols (e.g., Aave, Compound, Uniswap). It learns what "normal", safe contract behavior looks like across dozens of features (transaction frequency, value distribution, gas patterns, user concentration, etc.). When presented with a new strategy contract, it can generate a risk score based on how much its behavior deviates from these safe norms. The AI agent can then use this score to automatically avoid deploying funds to a contract that seems anomalous or potentially malicious.

---

### Mathematical Analysis Layer: The Oracle

- **Purpose**: To provide quantum-enhanced predictions, optimal control strategies, and information-theoretic optimization going beyond beyond traditional DeFi analytics.
- **Frameworks**: 
  - **Quantum Finance**: Harmonic oscillator price models for discrete energy level predictions
  - **Statistical Field Theory**: Liquidity action functionals for optimal flow dynamics  
  - **Optimal Control Theory**: Hamilton-Jacobi-Bellman equation solving for strategy optimization
  - **Information Theory**: Shannon entropy and Fisher information metrics for efficiency analysis
  - **Renormalization Group**: Critical behavior analysis and scale invariance detection
  - **Stochastic Calculus**: Advanced volatility modeling with jump diffusion processes
  - **Differential Geometry**: Riemannian manifold analysis of liquidity space
  - **And more**: Including algebraic topology, category theory, and information geometry
- **Real-time Analysis**: Unlike competitors using basic technical analysis, our system performs PhD-level mathematical computations in real-time, enabled by Unichain's low gas costs.

---

## How It Works: A Complete Workflow

1.  **Deposit**: A user finds our vault and deposits 10,000 USDC, receiving vault shares in return.
2.  **AI Detection**: The autonomous agent, running 24/7, periodically calls `get_system_status`. It observes that there is now 10,000 USDC sitting idle in the `Vault`.
3. **Mathematical + ML Analysis**: The agent performs two types of analysis:
   - **ML Risk Assessment**: Passes the `EulerSwapStrategy` to the anomaly detection model
   - **Mathematical Analysis**: Applies 10+ frameworks including quantum price prediction, optimal control optimization, and information theory to determine the mathematically optimal allocation
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
    -   **Mathematical Analysis Engine**: Implements **multiple advanced mathematical frameworks**:
-   **Machine Learning**: Uses **Scikit-learn** to implement the `IsolationForest` model and `StandardScaler` for feature normalization.

---

## Competition Advantage: Mathematical Sophistication

| **Our System** | **Typical DeFi Projects** |
|---|---|
| 10+ Mathematical Frameworks | 1-2 Basic Indicators |
| Quantum Finance Integration | Simple Moving Averages |
| Real-time Theoretical Physics | Rule-based Logic |
| Information-theoretic Optimization | Static Allocation Rules |
| Research-level Analysis | Basic Risk Metrics |

**Result**: While competitors use elementary math, we apply theoretical physics research to achieve superior risk-adjusted returns.


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

4. **Enhancing the AI, ML & Mathematical Core**: We will expand our mathematical framework library to 15+ theoretical physics models, continue training our ML risk model with more diverse data, and enhance the AI agent's decision-making with even more sophisticated quantum finance and field theory models.

5.  **Team Expansion**: We are actively looking to bring more developers on board to accelerate progress. We already know a couple of talented people who are interested in taking this project further with us, and we are excited to build out the team to realize the full potential of this AI-driven DeFi system.





