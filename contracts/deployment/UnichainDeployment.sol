// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "../core/VaultFactorySlim.sol";
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
        address usdcVault,
        address wethVault,
        address strategy
    );
    
    event ContractDeployed(string name, address contractAddress);
    
    // ============ Errors ============
    
    error InvalidParameters();
    error DeploymentFailed(string step);

    /// @notice Deploy the complete EulerSwap vault system
    /// @param treasury Address to receive fees
    /// @param manager Default manager for vaults
    /// @param agent Default agent for vaults
    /// @param eVaultUSDC Address of USDC Euler vault (must be provided)
    /// @param eVaultWETH Address of WETH Euler vault (must be provided)
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
        
        // 3. Create USDC vault
        (result.usdcVault, result.usdcVaultId) = _createUSDCVault(
            result.vaultFactory, 
            result.mockUSDC, 
            manager, 
            agent
        );
        
        // 4. Create WETH vault
        (result.wethVault, result.wethVaultId) = _createWETHVault(
            result.vaultFactory, 
            result.mockWETH, 
            manager, 
            agent
        );
        
        // 5. Create EulerSwap strategy for USDC/WETH pair
        (result.usdcWethStrategy, result.strategyId) = _createEulerSwapStrategy(
            result.vaultFactory,
            result.usdcVault,
            result.mockUSDC,
            result.mockWETH,
            eVaultUSDC,
            eVaultWETH
        );
        
        // 6. Add strategy to USDC vault
        Vault(result.usdcVault).addStrategy(result.usdcWethStrategy);
        
        // 7. Deploy auto-deposit proxies
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
        
        // 8. Mint test tokens to deployer
        _mintTestTokens(result.mockUSDC, payable(result.mockWETH), msg.sender);
        
        emit SystemDeployed(
            msg.sender, 
            result.vaultFactory, 
            result.usdcVault, 
            result.wethVault,
            result.usdcWethStrategy
        );
        
        return result;
    }
    
    /// @notice Deploy system with default parameters for testing
    function deployTestSystem(
        address eVaultUSDC,
        address eVaultWETH
    ) external payable returns (DeploymentResult memory) {
        return this.deploySystem{value: msg.value}(
            msg.sender, // treasury
            msg.sender, // manager
            msg.sender, // agent
            eVaultUSDC,
            eVaultWETH
        );
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
        address factory,
        address vault,
        address usdc,
        address weth,
        address eVaultUSDC,
        address eVaultWETH
    ) internal returns (address strategy, uint256 strategyId) {
        (strategy, strategyId) = VaultFactorySlim(factory).createEulerSwapStrategy(
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
        // Mint test tokens
        MockUSDC(usdc).mint(recipient, 100_000 * 10**6); // 100k USDC
        MockWETH(weth).mint(recipient, 50 * 10**18); // 50 WETH
        
        // Also mint to this contract for distribution
        MockUSDC(usdc).mint(address(this), 1_000_000 * 10**6); // 1M USDC
        MockWETH(weth).mint(address(this), 1000 * 10**18); // 1000 WETH
    }
    
    // ============ Utility Functions ============
    
    /// @notice Get deployment summary as string
    function getDeploymentSummary(DeploymentResult memory result) external pure returns (string memory) {
        return string(abi.encodePacked(
            "=== UNICHAIN DEPLOYMENT SUMMARY ===\n",
            "Network: Unichain Mainnet\n",
            "Vault Factory: ", _addressToString(result.vaultFactory), "\n",
            "USDC Vault: ", _addressToString(result.usdcVault), "\n",
            "WETH Vault: ", _addressToString(result.wethVault), "\n",
            "USDC/WETH Strategy: ", _addressToString(result.usdcWethStrategy), "\n",
            "Auto Deposit USDC: ", _addressToString(result.autoDepositProxyUSDC), "\n",
            "Auto Deposit WETH: ", _addressToString(result.autoDepositProxyWETH), "\n",
            "Mock USDC: ", _addressToString(result.mockUSDC), "\n",
            "Mock WETH: ", _addressToString(result.mockWETH), "\n",
            "=== INTEGRATION ADDRESSES ===\n",
            "EulerSwap Factory: ", _addressToString(EULER_SWAP_FACTORY), "\n",
            "EVC: ", _addressToString(EVC), "\n",
            "eVault Factory: ", _addressToString(EVAULT_FACTORY)
        ));
    }
    
    function _addressToString(address addr) internal pure returns (string memory) {
        bytes32 value = bytes32(uint256(uint160(addr)));
        bytes memory alphabet = "0123456789abcdef";
        
        bytes memory str = new bytes(42);
        str[0] = '0';
        str[1] = 'x';
        for (uint256 i = 0; i < 20; i++) {
            str[2+i*2] = alphabet[uint8(value[i + 12] >> 4)];
            str[3+i*2] = alphabet[uint8(value[i + 12] & 0x0f)];
        }
        return string(str);
    }
    
    /// @notice Faucet function for test tokens
    function faucet(address usdc, address payable weth, address recipient, uint256 usdcAmount, uint256 wethAmount) external {
        if (usdcAmount > 0) {
            MockUSDC(usdc).transfer(recipient, usdcAmount);
        }
        if (wethAmount > 0) {
            MockWETH(weth).transfer(recipient, wethAmount);
        }
    }
    
    /// @notice Get Unichain contract addresses
    function getUnichainAddresses() external pure returns (
        address eulerSwapFactory,
        address evc,
        address eVaultFactory,
        address protocolConfig
    ) {
        return (EULER_SWAP_FACTORY, EVC, EVAULT_FACTORY, PROTOCOL_CONFIG);
    }
}