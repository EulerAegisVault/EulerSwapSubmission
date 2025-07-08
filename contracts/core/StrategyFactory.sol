// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/access/Ownable.sol";
import "../strategies/EulerSwapStrategy.sol";

/// @title StrategyFactory
/// @notice A dedicated factory for creating EulerSwap strategies.
/// @dev Separated from VaultFactorySlim to reduce contract size.
contract StrategyFactory is Ownable {
    // ============ Unichain Contract Addresses ============
    address public constant EULER_SWAP_FACTORY = 0x45b146BC07c9985589B52df651310e75C6BE066A;
    
    // ============ State Variables ============
    uint256 public strategyCounter;
    mapping(uint256 => address) public strategies;

    // ============ Events ============
    event StrategyCreated(uint256 indexed strategyId, address indexed strategyAddress, address indexed vault);

    // ============ Constructor ============
    constructor() {
        _transferOwnership(msg.sender);
    }

    // ============ Core Functions ============

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

    // ============ View Functions ============

    function getStrategyCount() external view returns (uint256) {
        return strategyCounter;
    }
}
