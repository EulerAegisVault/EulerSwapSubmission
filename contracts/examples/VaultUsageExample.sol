// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "../core/VaultFactorySlim.sol";
import "../core/Vault.sol";
import "../strategies/EulerSwapStrategy.sol";
import "../mocks/MockUSDC.sol";
import "../mocks/MockWETH.sol";
import "../interfaces/IEulerSwap.sol";

/// @title Usage Example for EulerSwap Vault System on Unichain
/// @notice Complete example showing how to interact with the vault system
contract VaultUsageExample {
    VaultFactorySlim public factory;
    Vault public usdcVault;
    Vault public wethVault;
    EulerSwapStrategy public strategy;
    MockUSDC public usdc;
    MockWETH public weth;
    
    event ExampleStep(string step, uint256 amount, address user, string details);
    event PoolInteraction(string action, uint256 amountIn, uint256 amountOut);
    
    struct SystemStatus {
        uint256 userUsdcShares;
        uint256 userWethShares;
        uint256 strategyBalance;
        uint256 vaultUsdcBalance;
        uint256 vaultWethBalance;
        uint256 totalVaultValue;
        bool hasPool;
        address poolAddress;
    }
    
    constructor(
        address _factory,
        address _usdcVault,
        address _wethVault,
        address _strategy,
        address _usdc,
        address payable _weth
    ) {
        factory = VaultFactorySlim(_factory);
        usdcVault = Vault(_usdcVault);
        wethVault = Vault(_wethVault);
        strategy = EulerSwapStrategy(_strategy);
        usdc = MockUSDC(_usdc);
        weth = MockWETH(_weth);
    }
    
    /// @notice Complete example workflow for new users
    function runCompleteExample() external {
        address user = msg.sender;
        
        // Step 1: Get test tokens
        _getTestTokens(user);
        
        // Step 2: Deposit to vaults
        _depositToVaults(user);
        
        // Step 3: Deploy to EulerSwap strategy
        _deployToStrategy();
        
        // Step 4: Simulate some time passing and harvest
        _harvestRewards();
        
        // Step 5: Show final status
        SystemStatus memory status = getSystemStatus(user);
        emit ExampleStep(
            "Complete Example Finished", 
            status.totalVaultValue, 
            user, 
            status.hasPool ? "Pool created successfully" : "No pool created"
        );
    }
    
    /// @notice Step 1: Get test tokens from faucet
    function _getTestTokens(address user) internal {
        uint256 usdcAmount = 5000 * 10**6; // 5000 USDC
        uint256 wethAmount = 2 * 10**18;   // 2 WETH
        
        // Get tokens from faucet
        usdc.faucet(usdcAmount);
        weth.faucet(wethAmount);
        
        // Transfer to user if not the caller
        if (user != address(this)) {
            usdc.transfer(user, usdcAmount);
            weth.transfer(user, wethAmount);
        }
        
        emit ExampleStep(
            "Tokens Acquired", 
            usdcAmount + wethAmount, 
            user, 
            "Got test tokens from faucet"
        );
    }
    
    /// @notice Step 2: Deposit tokens to vaults
    function _depositToVaults(address user) internal {
        uint256 usdcAmount = 2000 * 10**6; // 2000 USDC
        uint256 wethAmount = 1 * 10**18;   // 1 WETH
        
        // Approve vaults
        usdc.approve(address(usdcVault), usdcAmount);
        weth.approve(address(wethVault), wethAmount);
        
        // Deposit to vaults
        uint256 usdcShares = usdcVault.deposit(usdcAmount, user);
        uint256 wethShares = wethVault.deposit(wethAmount, user);
        
        emit ExampleStep(
            "Deposited to Vaults", 
            usdcShares + wethShares, 
            user, 
            "Received vault shares"
        );
    }
    
    /// @notice Step 3: Deploy funds to EulerSwap strategy
    function _deployToStrategy() internal {
        uint256 deployAmount = 1500 * 10**6; // 1500 USDC equivalent
        
        // Encode strategy data: (amount0, amount1, createPool)
        bytes memory strategyData = abi.encode(
            1000 * 10**6,  // 1000 USDC
            0.5 * 10**18,  // 0.5 WETH  
            true           // Create pool if it doesn't exist
        );
        
        // Deploy to strategy (as vault agent)
        try usdcVault.depositToStrategy(
            address(strategy),
            deployAmount,
            strategyData
        ) {
            emit ExampleStep(
                "Deployed to Strategy", 
                deployAmount, 
                address(strategy), 
                "Funds deployed to EulerSwap strategy"
            );
        } catch Error(string memory reason) {
            emit ExampleStep(
                "Strategy Deployment Failed", 
                deployAmount, 
                address(strategy), 
                reason
            );
        }
    }
    
    /// @notice Step 4: Harvest rewards from strategy
    function _harvestRewards() internal {
        try usdcVault.harvestStrategy(address(strategy), "") {
            emit ExampleStep(
                "Harvested Rewards", 
                0, 
                address(strategy), 
                "Successfully harvested strategy rewards"
            );
        } catch Error(string memory reason) {
            emit ExampleStep(
                "Harvest Failed", 
                0, 
                address(strategy), 
                reason
            );
        }
    }
    
    /// @notice Simulate a swap through the EulerSwap pool
    function simulateSwap(
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) external returns (uint256 amountOut) {
        // Get pool info first
        (address pool,,,) = strategy.getPoolInfo();
        
        if (pool == address(0)) {
            emit PoolInteraction("Swap Failed", amountIn, 0);
            return 0;
        }
        
        // Get quote
        try IEulerSwap(pool).computeQuote(tokenIn, tokenOut, amountIn, true) returns (uint256 quote) {
            amountOut = quote;
            emit PoolInteraction("Swap Simulated", amountIn, amountOut);
        } catch {
            emit PoolInteraction("Quote Failed", amountIn, 0);
        }
    }
    
    /// @notice Get comprehensive system status
    function getSystemStatus(address user) public view returns (SystemStatus memory status) {
        status.userUsdcShares = usdcVault.balanceOf(user);
        status.userWethShares = wethVault.balanceOf(user);
        status.strategyBalance = strategy.getBalance();
        status.vaultUsdcBalance = usdc.balanceOf(address(usdcVault));
        status.vaultWethBalance = weth.balanceOf(address(wethVault));
        status.totalVaultValue = usdcVault.totalAssets() + wethVault.totalAssets();
        
        (address pool,,,) = strategy.getPoolInfo();
        status.hasPool = pool != address(0);
        status.poolAddress = pool;
    }
    
    /// @notice Get detailed strategy metrics
    function getStrategyMetrics() external view returns (
        uint256 totalDeposits0,
        uint256 totalDeposits1,
        uint256 currentBalance0,
        uint256 currentBalance1,
        uint256 lastHarvestTime,
        bool hasPool
    ) {
        return strategy.getStrategyMetrics();
    }
    
    /// @notice Advanced: Create liquidity and test swaps
    function advancedLiquidityTest() external {
        address user = msg.sender;
        
        // 1. Get larger amounts
        usdc.faucet(10000 * 10**6); // 10k USDC
        weth.faucet(5 * 10**18);    // 5 WETH
        
        // 2. Deposit to vaults
        usdc.approve(address(usdcVault), 8000 * 10**6);
        weth.approve(address(wethVault), 4 * 10**18);
        
        usdcVault.deposit(8000 * 10**6, user);
        wethVault.deposit(4 * 10**18, user);
        
        // 3. Deploy large amounts to strategy
        bytes memory strategyData = abi.encode(
            5000 * 10**6,  // 5000 USDC
            2 * 10**18,    // 2 WETH
            true           // Create pool
        );
        
        usdcVault.depositToStrategy(
            address(strategy),
            7000 * 10**6,
            strategyData
        );
        
        // 4. Test swap simulation
        this.simulateSwap(address(usdc), address(weth), 100 * 10**6);
        
        emit ExampleStep(
            "Advanced Test Complete", 
            7000 * 10**6, 
            user, 
            "Large liquidity deployment and swap test"
        );
    }
    
    /// @notice Emergency functions
    function emergencyWithdraw() external {
        try usdcVault.emergencyExit(address(strategy), "") {
            emit ExampleStep(
                "Emergency Exit", 
                0, 
                address(strategy), 
                "Emergency exit completed"
            );
        } catch Error(string memory reason) {
            emit ExampleStep(
                "Emergency Exit Failed", 
                0, 
                address(strategy), 
                reason
            );
        }
    }
    
    /// @notice Get formatted status string
    function getStatusString(address user) external view returns (string memory) {
        SystemStatus memory status = getSystemStatus(user);
        
        return string(abi.encodePacked(
            "=== VAULT SYSTEM STATUS ===\n",
            "User USDC Shares: ", _uint256ToString(status.userUsdcShares), "\n",
            "User WETH Shares: ", _uint256ToString(status.userWethShares), "\n",
            "Strategy Balance: ", _uint256ToString(status.strategyBalance), "\n",
            "Total Vault Value: ", _uint256ToString(status.totalVaultValue), "\n",
            "Pool Status: ", status.hasPool ? "Created" : "Not Created", "\n",
            "EulerSwap Integration: Active"
        ));
    }
    
    function _uint256ToString(uint256 value) internal pure returns (string memory) {
        if (value == 0) return "0";
        
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        return string(buffer);
    }
}