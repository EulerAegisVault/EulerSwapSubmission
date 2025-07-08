// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/// @title Mock WETH Token for Unichain Testing
/// @notice Mock WETH token for testing vault system on Unichain
contract MockWETH is ERC20, Ownable {
    uint8 private constant DECIMALS = 18;
    uint256 private constant INITIAL_SUPPLY = 100_000 * 10 ** DECIMALS; // 100k WETH

    constructor() ERC20("Wrapped Ether", "WETH") {
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
    /// @param amount Amount to mint to caller (max 10 WETH)
    function faucet(uint256 amount) external {
        require(
            amount <= 10 * 10 ** DECIMALS,
            "MockWETH: Max 10 WETH per faucet"
        );
        _mint(msg.sender, amount);
    }

    /// @notice Deposit ETH and get WETH (like real WETH)
    function deposit() external payable {
        _mint(msg.sender, msg.value);
    }

    /// @notice Withdraw WETH and get ETH (like real WETH)
    function withdraw(uint256 amount) external {
        require(
            balanceOf(msg.sender) >= amount,
            "MockWETH: Insufficient balance"
        );
        _burn(msg.sender, amount);
        payable(msg.sender).transfer(amount);
    }

    /// @notice Batch faucet for multiple addresses
    function batchFaucet(address[] calldata recipients, uint256 amount) external {
        require(
            amount <= 10 * 10 ** DECIMALS,
            "MockWETH: Max 10 WETH per recipient"
        );
        
        for (uint256 i = 0; i < recipients.length; i++) {
            _mint(recipients[i], amount);
        }
    }

    /// @notice Fallback to deposit ETH
    receive() external payable {
        _mint(msg.sender, msg.value);
    }
}