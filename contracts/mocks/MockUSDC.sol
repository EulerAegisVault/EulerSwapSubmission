// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/// @title Mock USDC Token for Unichain Testing
/// @notice Mock USDC token for testing vault system on Unichain
contract MockUSDC is ERC20, Ownable {
    uint8 private constant DECIMALS = 6;
    uint256 private constant INITIAL_SUPPLY = 1_000_000_000 * 10 ** DECIMALS; // 1B USDC

    constructor() ERC20("USD Coin", "USDC") {
        _transferOwnership(msg.sender);
        _mint(msg.sender, INITIAL_SUPPLY);
    }

    function decimals() public pure override returns (uint8) {
        return DECIMALS;
    }

    /// @notice Mint new tokens (only owner)
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }

    /// @notice Burn tokens from caller
    function burn(uint256 amount) external {
        _burn(msg.sender, amount);
    }

    /// @notice Faucet function for testing (anyone can call)
    /// @param amount Amount to mint to caller (max 10,000 USDC)
    function faucet(uint256 amount) external {
        require(
            amount <= 10_000 * 10 ** DECIMALS,
            "MockUSDC: Max 10,000 USDC per faucet"
        );
        _mint(msg.sender, amount);
    }
    
    /// @notice Batch faucet for multiple addresses
    function batchFaucet(address[] calldata recipients, uint256 amount) external {
        require(
            amount <= 10_000 * 10 ** DECIMALS,
            "MockUSDC: Max 10,000 USDC per recipient"
        );
        
        for (uint256 i = 0; i < recipients.length; i++) {
            _mint(recipients[i], amount);
        }
    }
}