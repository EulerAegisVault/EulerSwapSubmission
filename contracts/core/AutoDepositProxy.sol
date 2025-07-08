// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "./Vault.sol";

/// @title AutoDepositProxy
/// @notice A proxy contract that automatically deposits received tokens into a vault
/// @dev This allows external systems to send tokens directly to this contract for auto-deposit
contract AutoDepositProxy {
    using SafeERC20 for IERC20;
    
    /// @notice The target vault for deposits
    Vault public immutable vault;
    
    /// @notice The token this proxy handles
    IERC20 public immutable token;
    
    /// @notice Address that should receive the vault shares
    address public immutable beneficiary;
    
    /// @notice Minimum amount required to trigger auto-deposit
    uint256 public immutable minDepositAmount;
    
    event AutoDeposit(address indexed beneficiary, uint256 tokenAmount, uint256 sharesReceived);
    event DepositFailed(uint256 amount, string reason);
    
    error InvalidVault();
    error InvalidToken();
    error InvalidBeneficiary();
    error InsufficientAmount();
    
    constructor(
        address _vault, 
        address _token, 
        address _beneficiary,
        uint256 _minDepositAmount
    ) {
        if (_vault == address(0)) revert InvalidVault();
        if (_token == address(0)) revert InvalidToken();
        if (_beneficiary == address(0)) revert InvalidBeneficiary();
        
        vault = Vault(_vault);
        token = IERC20(_token);
        beneficiary = _beneficiary;
        minDepositAmount = _minDepositAmount;
        
        // Pre-approve vault to save gas on deposits
        token.safeApprove(_vault, type(uint256).max);
    }
    
    /// @notice Automatically deposit any token balance to vault
    /// @dev Can be called by anyone, sends vault shares to beneficiary
    function autoDeposit() external {
        uint256 balance = token.balanceOf(address(this));
        if (balance >= minDepositAmount) {
            try vault.deposit(balance, beneficiary) returns (uint256 shares) {
                emit AutoDeposit(beneficiary, balance, shares);
            } catch Error(string memory reason) {
                emit DepositFailed(balance, reason);
            } catch {
                emit DepositFailed(balance, "Unknown error");
            }
        }
    }
    
    /// @notice Manual deposit with specific amount
    /// @param amount Amount to deposit
    function deposit(uint256 amount) external {
        if (amount == 0) revert InsufficientAmount();
        
        uint256 balance = token.balanceOf(address(this));
        if (balance < amount) revert InsufficientAmount();
        
        try vault.deposit(amount, beneficiary) returns (uint256 shares) {
            emit AutoDeposit(beneficiary, amount, shares);
        } catch Error(string memory reason) {
            emit DepositFailed(amount, reason);
        }
    }
    
    /// @notice Get current balance
    function getBalance() external view returns (uint256) {
        return token.balanceOf(address(this));
    }
    
    /// @notice Check if balance is sufficient for auto-deposit
    function canAutoDeposit() external view returns (bool) {
        return token.balanceOf(address(this)) >= minDepositAmount;
    }
    
    /// @notice Fallback function that triggers auto-deposit when tokens are received
    /// @dev This allows the contract to automatically deposit when tokens arrive
    fallback() external {
        uint256 balance = token.balanceOf(address(this));
        if (balance >= minDepositAmount) {
            try vault.deposit(balance, beneficiary) returns (uint256 shares) {
                emit AutoDeposit(beneficiary, balance, shares);
            } catch {
                // Silently fail to avoid reverting the token transfer
            }
        }
    }
    
    /// @notice Emergency withdrawal function (only beneficiary)
    function emergencyWithdraw() external {
        require(msg.sender == beneficiary, "Only beneficiary");
        uint256 balance = token.balanceOf(address(this));
        if (balance > 0) {
            token.safeTransfer(beneficiary, balance);
        }
    }
}