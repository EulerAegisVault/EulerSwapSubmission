// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./Vault.sol";
import "../strategies/EulerSwapStrategy.sol";

/// @title Simplified Vault Factory for Unichain
/// @notice Lightweight factory for creating vaults and strategies
contract VaultFactorySlim is Ownable {
    // ============ Unichain Contract Addresses ============
    address public constant EULER_SWAP_FACTORY = 0x45b146BC07c9985589B52df651310e75C6BE066A;
    
    // ============ State Variables ============
    uint256 public vaultCounter;
    uint256 public strategyCounter;
    
    mapping(uint256 => address) public vaults;
    mapping(uint256 => address) public strategies;
    
    address public defaultManager;
    address public defaultAgent;
    address public treasury;

    // ============ Events ============
    event VaultCreated(uint256 indexed vaultId, address indexed vaultAddress, address indexed asset);
    event StrategyCreated(uint256 indexed strategyId, address indexed strategyAddress, address indexed vault);

    // ============ Constructor ============
    constructor(
        address _defaultManager,
        address _defaultAgent,
        address _treasury
    ) {
        _transferOwnership(msg.sender);
        defaultManager = _defaultManager;
        defaultAgent = _defaultAgent;
        treasury = _treasury;
    }

    // ============ Core Functions ============

    /// @notice Create a new vault
    function createVault(
        IERC20 asset,
        string memory name,
        string memory symbol,
        address manager,
        address agent
    ) external returns (address vaultAddress, uint256 vaultId) {
        address _manager = manager != address(0) ? manager : defaultManager;
        address _agent = agent != address(0) ? agent : defaultAgent;

        vaultCounter++;
        vaultId = vaultCounter;

        Vault vault = new Vault(asset, name, symbol, _manager, _agent);
        vaultAddress = address(vault);
        vaults[vaultId] = vaultAddress;

        emit VaultCreated(vaultId, vaultAddress, address(asset));
        return (vaultAddress, vaultId);
    }
    
    /// @notice Create an EulerSwap strategy
    function createEulerSwapStrategy(
        address vault,
        address token0,
        address token1,
        address eVault0,
        address eVault1
    ) external returns (address strategyAddress, uint256 strategyId) {
        strategyCounter++;
        strategyId = strategyCounter;
        
        EulerSwapStrategy strategy = new EulerSwapStrategy(
            vault,
            EULER_SWAP_FACTORY,
            token0,
            token1,
            eVault0,
            eVault1
        );
        
        strategyAddress = address(strategy);
        strategies[strategyId] = strategyAddress;
        
        emit StrategyCreated(strategyId, strategyAddress, vault);
        return (strategyAddress, strategyId);
    }
    
    /// @notice Create vault and strategy together
    function createVaultWithStrategy(
        IERC20 asset,
        string memory name,
        string memory symbol,
        address token0,
        address token1,
        address eVault0,
        address eVault1
    ) external returns (
        address vaultAddress,
        uint256 vaultId,
        address strategyAddress,
        uint256 strategyId
    ) {
        // Create vault
        (vaultAddress, vaultId) = this.createVault(asset, name, symbol, msg.sender, msg.sender);
        
        // Create strategy
        (strategyAddress, strategyId) = this.createEulerSwapStrategy(
            vaultAddress,
            token0,
            token1,
            eVault0,
            eVault1
        );
        
        return (vaultAddress, vaultId, strategyAddress, strategyId);
    }

    // ============ View Functions ============
    
    function getVaultCount() external view returns (uint256) {
        return vaultCounter;
    }
    
    function getStrategyCount() external view returns (uint256) {
        return strategyCounter;
    }

    // ============ Admin Functions ============
    
    function setTreasury(address _newTreasury) external onlyOwner {
        treasury = _newTreasury;
    }
    
    function setDefaultManager(address _newManager) external onlyOwner {
        defaultManager = _newManager;
    }
    
    function setDefaultAgent(address _newAgent) external onlyOwner {
        defaultAgent = _newAgent;
    }
}