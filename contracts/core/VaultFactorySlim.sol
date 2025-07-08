// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./Vault.sol";

/// @title Simplified Vault Factory for Unichain
/// @notice Lightweight factory for creating vaults. Strategy creation is handled by StrategyFactory.
contract VaultFactorySlim is Ownable {
    // ============ State Variables ============
    uint256 public vaultCounter;
    mapping(uint256 => address) public vaults;
    
    address public defaultManager;
    address public defaultAgent;
    address public treasury;

    // ============ Events ============
    event VaultCreated(uint256 indexed vaultId, address indexed vaultAddress, address indexed asset);

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
    
    // ============ View Functions ============
    
    function getVaultCount() external view returns (uint256) {
        return vaultCounter;
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
