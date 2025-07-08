# EulerSwap Vault & Strategy System

This project provides a complete, modular, and extensible system for creating and managing token vaults that integrate with EulerSwap liquidity strategies on the Unichain network. It is designed to be a robust foundation for yield aggregation, allowing users to deposit assets and have them automatically deployed into a specified EulerSwap liquidity pool to earn trading fees.

The system is architected to be gas-efficient and secure, separating core logic into distinct, specialized contracts. This includes separate factories for creating vaults and strategies, which resolves common contract size limitations on the EVM.

## Table of Contents

1.  [System Architecture](#system-architecture)
2.  [Core Contracts](#core-contracts)
3.  [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Configuration](#configuration)
4.  [Deployment](#deployment)
5.  [How It Works: User Workflow](#how-it-works-user-workflow)
6.  [Key Unichain Integrations](#key-unichain-integrations)

---

## System Architecture

The system is composed of several key contracts that work together to provide a seamless experience from asset deposit to yield generation.

+------------------+      +--------------------+      +-----------------------+
|      User        |----->|    Vault (ERC4626) |----->|  EulerSwapStrategy    |
+------------------+      +--------------------+      +-----------------------+
        ^                       ^                             ^
        |                       |                             |
        | (deposit)             | (create)                    | (create)
        |                       |                             |
+------------------+      +--------------------+      +-----------------------+
| AutoDepositProxy |      | VaultFactorySlim   |      |    StrategyFactory    |
+------------------+      +--------------------+      +-----------------------+

1.  **Factories (`VaultFactorySlim`, `StrategyFactory`)**: These are responsible for deploying new `Vault` and `EulerSwapStrategy` contracts. Separating them ensures that the system remains scalable and avoids the `max code size` limit.
2.  **Vault (`Vault.sol`)**: The user-facing contract. It's an [ERC-4626](https://ethereum.org/en/developers/docs/standards/tokens/erc-4626/) compliant tokenized vault. Users deposit a single asset (like USDC) and receive vault shares in return. The vault is responsible for managing these assets and allocating them to one or more strategies.
3.  **Strategy (`EulerSwapStrategy.sol`)**: The "worker" contract. It contains the logic for interacting with a specific DeFi protocol—in this case, EulerSwap. It takes funds from the Vault and deploys them into a USDC/WETH liquidity pool on EulerSwap to earn fees.
4.  **Auto-Deposit Proxy (`AutoDepositProxy.sol`)**: A convenience contract that allows users or systems to simply transfer tokens to its address. The proxy will automatically deposit these tokens into a pre-configured vault on the user's behalf.

---

## Core Contracts

### `Vault.sol`

-   **ERC-4626 Compliant**: Provides a standard interface for tokenized vaults, making it compatible with other DeFi protocols and tooling.
-   **Strategy Agnostic**: Can support multiple, different strategy contracts. A manager can add or remove strategies.
-   **Access Control**: Uses a `MANAGER_ROLE` for managing strategies and an `AGENT_ROLE` for executing strategy operations like depositing funds or harvesting rewards. This separates high-level control from automated, routine operations.

### `EulerSwapStrategy.sol`

-   **Specific Logic**: Contains all the code required to interact with the EulerSwap Factory and its liquidity pools.
-   **Liquidity Provision**: Can create a new EulerSwap pool or add liquidity to an existing one.
-   **Yield Generation**: The primary goal is to hold assets in EulerSwap's liquidity pools, which generates trading fees that accrue to the value of the assets held by the strategy.
-   **Harvesting**: The `harvest` function is designed to realize the gains from the strategy and can be configured to send profits back to the vault.

### `VaultFactorySlim.sol`

-   **Lightweight**: Its only job is to deploy new `Vault` contracts.
-   **Configurable**: Allows setting default managers, agents, and a treasury address for new vaults.

### `StrategyFactory.sol`

-   **Single Purpose**: Its only job is to deploy new `EulerSwapStrategy` contracts.
-   **Decoupled**: By separating this from the `VaultFactorySlim`, we ensure neither contract becomes too large.

---

## Getting Started

### Prerequisites

-   [Node.js](https://nodejs.org/en/) (v18 or later)
-   [Yarn](https://yarnpkg.com/) or [npm](https://www.npmjs.com/)
-   A wallet with Unichain testnet funds.

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  Install dependencies:
    ```bash
    yarn install
    # or
    npm install
    ```

### Configuration

1.  Create a `.env` file in the root of the project by copying the example file:
    ```bash
    cp .env.example .env
    ```

2.  Edit the `.env` file and add your private key. This key will be used to deploy the contracts.
    ```
    PRIVATE_KEY="your_wallet_private_key_here"
    ETHERSCAN_API_KEY="your_etherscan_api_key_optional"
    ```

---

## Deployment

The project includes a comprehensive Hardhat script to deploy and configure the entire system in a single command.

1.  **Compile the contracts:**
    ```bash
    npx hardhat compile
    ```

2.  **Run the deployment script on the Unichain network:**
    ```bash
    npx hardhat run scripts/deploy.js --network unichain
    ```

The script will perform the following steps:
1.  Connect to the pre-existing Mock USDC and WETH tokens.
2.  Deploy the `VaultFactorySlim` contract.
3.  Deploy the `StrategyFactory` contract.
4.  Use the `VaultFactorySlim` to create a new `Vault` for USDC.
5.  Use the `VaultFactorySlim` to create a new `Vault` for WETH.
6.  Use the `StrategyFactory` to create a new `EulerSwapStrategy` linked to the USDC vault.
7.  Link the newly created strategy to the USDC vault.
8.  Output a summary of all deployed contract addresses.

---

## How It Works: User Workflow

Here is a typical end-to-end workflow for a user interacting with the deployed system.

### Step 1: Obtain Tokens

A user first needs the underlying asset, for example, USDC. In a test environment, they can use the `faucet` function on the `MockUSDC` contract.

### Step 2: Approve and Deposit into the Vault

The user must first approve the `Vault` contract to spend their USDC. They then call the `deposit()` function on the `Vault`.

```javascript
// Example using ethers.js
const usdc = await ethers.getContractAt("MockUSDC", usdcAddress);
const vault = await ethers.getContractAt("Vault", usdcVaultAddress);
const depositAmount = ethers.parseUnits("1000", 6); // 1000 USDC

// 1. Approve
await usdc.approve(usdcVaultAddress, depositAmount);

// 2. Deposit
await vault.deposit(depositAmount, userAddress);
In return for their deposit, the user receives vault shares (vUSDC), which represent their share of the total assets managed by the vault.

Step 3: The Vault Manager Deploys Funds to the Strategy
The vault now holds the user's USDC. A manager (or an automated agent) can now move these funds into the EulerSwapStrategy to put them to work.

JavaScript

// The data specifies how to split funds and whether to create a pool
const data = ethers.AbiCoder.defaultAbiCoder().encode(
  ["uint256", "uint256", "bool"],
  [ethers.parseUnits("500", 6), ethers.parseEther("0.2"), true]
);

// Call depositToStrategy from the vault
await vault.depositToStrategy(strategyAddress, totalAmount, data);
```

Of course. Here are the two sections you requested in separate, independent markdown boxes.

Here is the architecture diagram:

Markdown

+------------------+      +--------------------+      +-----------------------+
|      User        |----->|    Vault (ERC4626) |----->|  EulerSwapStrategy    |
+------------------+      +--------------------+      +-----------------------+
        ^                       ^                             ^
        |                       |                             |
        | (deposit)             | (create)                    | (create)
        |                       |                             |
+------------------+      +--------------------+      +-----------------------+
| AutoDepositProxy |      | VaultFactorySlim   |      |    StrategyFactory    |
+------------------+      +--------------------+      +-----------------------+
And here is the rest of the text, starting from the specified line:

Markdown

In return for their deposit, the user receives vault shares (vUSDC), which represent their share of the total assets managed by the vault.

### Step 3: The Vault Manager Deploys Funds to the Strategy

The vault now holds the user's USDC. A manager (or an automated agent) can now move these funds into the `EulerSwapStrategy` to put them to work.

```javascript
// The data specifies how to split funds and whether to create a pool
const data = ethers.AbiCoder.defaultAbiCoder().encode(
  ["uint256", "uint256", "bool"],
  [ethers.parseUnits("500", 6), ethers.parseEther("0.2"), true]
);

// Call depositToStrategy from the vault
await vault.depositToStrategy(strategyAddress, totalAmount, data);
The strategy then takes these funds and provides them as liquidity to the USDC/WETH pool on EulerSwap.
```
### Step 4: Harvesting Rewards

Over time, the liquidity provided earns trading fees. An agent can periodically call the `harvest()` function on the `Vault`.

```javascript
await vault.harvestStrategy(strategyAddress, "0x");
```

### Step 5: Withdrawing from the Vault

At any time, the user can redeem their vault shares for the underlying USDC. The amount of USDC they receive will reflect their initial deposit plus any yield earned.

```javascript
// User redeems their shares
await vault.redeem(sharesAmount, userAddress, userAddress);
```


## Key Unichain Integrations

This system is built to natively integrate with the following deployed contracts on the Unichain network:

- EulerSwap V1 Factory: 0x45b146BC07c9985589B52df651310e75C6BE066A

- USDC eVault: 0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba

- WETH eVault: 0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC

- Vault Factory: 0xEc0036beC79dBCf0279dAd3bC59F6231b9F461d9

- Strategy Factory: 0xa5752653c78D9254EB3082d8559b76a38C9E8563

- Mock USDC: 0xC0933C5440c656464D1Eb1F886422bE3466B1459

- Mock WETH: 0xf0f994B4A8dB86A46a1eD4F12263c795b26703Ca

The EulerSwapStrategy uses these addresses to create and interact with liquidity pools, depositing the underlying tokens into the correct Euler Vaults (eVaults) as required by the EulerSwap protocol.


==============================================================================================================================
------------------------------------------------------------------------------------------------------------------------------
==============================================================================================================================

# Latest Deployment Summary:

🔗 Step 1: Using Existing Mock Tokens...
   ✅ MockUSDC: 0xC0933C5440c656464D1Eb1F886422bE3466B1459
   ✅ MockWETH: 0xf0f994B4A8dB86A46a1eD4F12263c795b26703Ca

🏭 Step 2: Deploying Slim Vault Factory...
   ✅ VaultFactorySlim deployed at: 0xEc0036beC79dBCf0279dAd3bC59F6231b9F461d9

🏭 Step 3: Deploying Strategy Factory...
   ✅ StrategyFactory deployed at: 0xa5752653c78D9254EB3082d8559b76a38C9E8563

🌐 Step 4: Using Unichain eVault Addresses...
   📦 USDC eVault: 0x6eAe95ee783e4D862867C4e0E4c3f4B95AA682Ba
   💎 WETH eVault: 0x1f3134C3f3f8AdD904B9635acBeFC0eA0D0E1ffC
   ✅ eVault addresses configured from Maglev

💰 Step 5: Creating USDC Vault...
   ✅ USDC Vault created: 0xFAf4Af2Ed51cDb2967B0d204074cc2e4302F9188

💎 Step 6: Creating WETH Vault...
   ✅ WETH Vault created: 0x112b33c07deE1697d9b4A68b2BA0F66c2417635C

🎯 Step 7: Creating EulerSwap Strategy...
   ✅ Strategy created: 0x807463769044222F3b7B5F98da8d3E25e0aC44B0

⚙️  Step 8: Setting up vaults...
   ✅ Strategy added to USDC vault

🎉 EulerSwap Vault System Deployment Complete!
════════════════════════════════════════════════════════════
🏭 Vault Factory:       0xEc0036beC79dBCf0279dAd3bC59F6231b9F461d9
🏭 Strategy Factory:    0xa5752653c78D9254EB3082d8559b76a38C9E8563
💰 Mock USDC:           0xC0933C5440c656464D1Eb1F886422bE3466B1459
💎 Mock WETH:           0xf0f994B4A8dB86A46a1eD4F12263c795b26703Ca
🏦 USDC Vault:          0xFAf4Af2Ed51cDb2967B0d204074cc2e4302F9188
🏦 WETH Vault:          0x112b33c07deE1697d9b4A68b2BA0F66c2417635C
🎯 EulerSwap Strategy:   0x807463769044222F3b7B5F98da8d3E25e0aC44B0
════════════════════════════════════════════════════════════