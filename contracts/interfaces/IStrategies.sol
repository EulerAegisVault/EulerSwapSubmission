// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

/**
 * @title IStrategies
 * @dev Interface for strategy contracts that interact with DeFi protocols
 */
interface IStrategies {
    // ============ Events ============
    event Deposit(uint256 amount);
    event Withdraw(uint256 amount);
    event Claim(uint256 amount);
    event VaultSet(address vault);
    event PausedState(bool isPaused);
    event Executed(uint256 amount, bytes data);
    event Harvested(bytes data);
    event EmergencyExited(uint256 balance, bytes data);
    event RewardTokenAdded(address indexed token);
    event TokensForwarded(address indexed token, uint256 amount);
    event ClaimRewardsFailed(bytes reason);

    // ============ Errors ============
    error NoVaultSet();
    error StrategyPaused();
    error DepositFailed(bytes reason);
    error WithdrawFailed(bytes reason);
    error ClaimFailed(bytes reason);
    error GetBalanceFailed(bytes reason);
    error NoUnderlyingBalance();
    error InvalidTokenAddress();
    error InvalidAmount();

    // ============ View Functions ============
    function underlyingToken() external view returns (address);
    function protocol() external view returns (address);
    function depositSelector() external view returns (bytes4);
    function withdrawSelector() external view returns (bytes4);
    function claimSelector() external view returns (bytes4);
    function getBalanceSelector() external view returns (bytes4);
    function vault() external view returns (address);
    function paused() external view returns (bool);
    function knownRewardTokens(address token) external view returns (bool);
    function rewardTokensList() external view returns (address[] memory);
    function getBalance() external view returns (uint256);
    function queryProtocol(bytes4 selector, bytes calldata params) external view returns (bytes memory);

    // ============ State-Changing Functions ============
    function setVault(address _vault) external;
    function addRewardToken(address tokenAddress) external;
    function execute(uint256 amount, bytes calldata data) external;
    function harvest(bytes calldata data) external;
    function emergencyExit(bytes calldata data) external;
    function claimRewards(bytes calldata data) external;
    function setPaused(bool _paused) external;
}