// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "../core/VaultFactorySlim.sol";
import "../core/StrategyFactory.sol"; // Import the new StrategyFactory
import "../core/Vault.sol";
import "../mocks/MockUSDC.sol";
import "../mocks/MockWETH.sol";
import "../core/AutoDepositProxy.sol";

/// @title Unichain Deployment Script
/// @notice Complete deployment script for EulerSwap-integrated vault system on Unichain mainnet
contract UnichainDeployment {
    // ============ Unichain Mainnet Addresses ============
    
    /// @notice EulerSwap V1 Factory
    address public constant EULER_SWAP_FACTORY = 0x45b146BC07c9985589B52df651310e75C6BE066A;
    
    /// @notice Ethereum Vault Connector (EVC)
    address public constant EVC = 0x2A1176964F5D7caE5406B627Bf6166664FE83c60;
    
    /// @notice Euler V2 Vault Factory
    address public constant EVAULT_FACTORY = 0xbAd8b5BDFB2bcbcd78Cc9f1573D3Aad6E865e752;
    
    /// @notice Protocol Config
    address public constant PROTOCOL_CONFIG = 0xdCD02E4eA8cd273498D315AD8c047305f8480656;
    
    // ============ Deployment Result Struct ============
    
    struct DeploymentResult {
        address vaultFactory;
        address strategyFactory; // Added strategy factory address
        address mockUSDC;
        address mockWETH;
        address usdcVault;
        address wethVault;
        address usdcWethStrategy;
        address autoDepositProxyUSDC;
        address autoDepositProxyWETH;
        uint256 usdcVaultId;
        uint256 wethVaultId;
        uint256 strategyId;
        DeploymentInfo info;
    }
    
    struct DeploymentInfo {
        address deployer;
        uint256 deploymentTime;
        uint256 blockNumber;
        string network;
        address treasury;
        address defaultManager;
        address defaultAgent;
    }
    
    // ============ Events ============
    
    event SystemDeployed(
        address indexed deployer,
        address vaultFactory,
        address strategyFactory,
        address usdcVault,
        address wethVault,
        address strategy
    );
    
    event ContractDeployed(string name, address contractAddress);
    
    // ============ Errors ============
    
    error InvalidParameters();
    error DeploymentFailed(string step);

    /// @notice Deploy the complete EulerSwap vault system
    function deploySystem(
        address treasury,
        address manager,
        address agent,
        address eVaultUSDC,
        address eVaultWETH
    ) external payable returns (DeploymentResult memory result) {
        if (treasury == address(0) || manager == address(0) || agent == address(0)) {
            revert InvalidParameters();
        }
        if (eVaultUSDC == address(0) || eVaultWETH == address(0)) {
            revert InvalidParameters();
        }
        
        // Store deployment info
        result.info = DeploymentInfo({
            deployer: msg.sender,
            deploymentTime: block.timestamp,
            blockNumber: block.number,
            network: "Unichain Mainnet",
            treasury: treasury,
            defaultManager: manager,
            defaultAgent: agent
        });
        
        // 1. Deploy mock tokens for testing
        result.mockUSDC = _deployMockUSDC();
        result.mockWETH = _deployMockWETH();
        
        // 2. Deploy vault factory
        result.vaultFactory = _deployVaultFactorySlim(treasury, manager, agent);

        // 3. Deploy strategy factory
        result.strategyFactory = _deployStrategyFactory();
        
        // 4. Create USDC vault
        (result.usdcVault, result.usdcVaultId) = _createUSDCVault(
            result.vaultFactory, 
            result.mockUSDC, 
            manager, 
            agent
        );
        
        // 5. Create WETH vault
        (result.wethVault, result.wethVaultId) = _createWETHVault(
            result.vaultFactory, 
            result.mockWETH, 
            manager, 
            agent
        );
        
        // 6. Create EulerSwap strategy for USDC/WETH pair using the new StrategyFactory
        (result.usdcWethStrategy, result.strategyId) = _createEulerSwapStrategy(
            result.strategyFactory, // Use the new strategy factory
            result.usdcVault,
            result.mockUSDC,
            result.mockWETH,
            eVaultUSDC,
            eVaultWETH
        );
        
        // 7. Add strategy to USDC vault
        Vault(result.usdcVault).addStrategy(result.usdcWethStrategy);
        
        // 8. Deploy auto-deposit proxies
        result.autoDepositProxyUSDC = _deployAutoDepositProxy(
            result.usdcVault,
            result.mockUSDC,
            msg.sender,
            100 * 10**6 // 100 USDC minimum
        );
        
        result.autoDepositProxyWETH = _deployAutoDepositProxy(
            result.wethVault,
            result.mockWETH,
            msg.sender,
            0.01 * 10**18 // 0.01 WETH minimum
        );
        
        // 9. Mint test tokens to deployer
        _mintTestTokens(result.mockUSDC, payable(result.mockWETH), msg.sender);
        
        emit SystemDeployed(
            msg.sender, 
            result.vaultFactory, 
            result.strategyFactory,
            result.usdcVault, 
            result.wethVault,
            result.usdcWethStrategy
        );
        
        return result;
    }
    
    // ============ Internal Deployment Functions ============
    
    function _deployMockUSDC() internal returns (address) {
        MockUSDC usdc = new MockUSDC();
        emit ContractDeployed("MockUSDC", address(usdc));
        return address(usdc);
    }
    
    function _deployMockWETH() internal returns (address) {
        MockWETH weth = new MockWETH();
        emit ContractDeployed("MockWETH", address(weth));
        return address(weth);
    }
    
    function _deployVaultFactorySlim(
        address treasury,
        address manager,
        address agent
    ) internal returns (address) {
        VaultFactorySlim factory = new VaultFactorySlim(
            manager,
            agent,
            treasury
        );
        emit ContractDeployed("VaultFactorySlim", address(factory));
        return address(factory);
    }

    function _deployStrategyFactory() internal returns (address) {
        StrategyFactory factory = new StrategyFactory();
        emit ContractDeployed("StrategyFactory", address(factory));
        return address(factory);
    }
    
    function _createUSDCVault(
        address factory,
        address usdc,
        address manager,
        address agent
    ) internal returns (address vault, uint256 vaultId) {
        (vault, vaultId) = VaultFactorySlim(factory).createVault(
            IERC20(usdc),
            "USDC Vault",
            "vUSDC",
            manager,
            agent
        );
        emit ContractDeployed("USDC Vault", vault);
    }
    
    function _createWETHVault(
        address factory,
        address weth,
        address manager,
        address agent
    ) internal returns (address vault, uint256 vaultId) {
        (vault, vaultId) = VaultFactorySlim(factory).createVault(
            IERC20(weth),
            "WETH Vault",
            "vWETH",
            manager,
            agent
        );
        emit ContractDeployed("WETH Vault", vault);
    }
    
    function _createEulerSwapStrategy(
        address factory, // This is now the strategy factory address
        address vault,
        address usdc,
        address weth,
        address eVaultUSDC,
        address eVaultWETH
    ) internal returns (address strategy, uint256 strategyId) {
        // Call createEulerSwapStrategy on the StrategyFactory contract
        (strategy, strategyId) = StrategyFactory(factory).createEulerSwapStrategy(
            vault,
            usdc,
            weth,
            eVaultUSDC,
            eVaultWETH
        );
        emit ContractDeployed("EulerSwap Strategy", strategy);
    }
    
    function _deployAutoDepositProxy(
        address vault,
        address token,
        address beneficiary,
        uint256 minAmount
    ) internal returns (address) {
        AutoDepositProxy proxy = new AutoDepositProxy(
            vault,
            token,
            beneficiary,
            minAmount
        );
        emit ContractDeployed("AutoDepositProxy", address(proxy));
        return address(proxy);
    }
    
    function _mintTestTokens(address usdc, address payable weth, address recipient) internal {
        MockUSDC(usdc).mint(recipient, 100_000 * 10**6); // 100k USDC
        MockWETH(weth).mint(recipient, 50 * 10**18); // 50 WETH
    }
}
