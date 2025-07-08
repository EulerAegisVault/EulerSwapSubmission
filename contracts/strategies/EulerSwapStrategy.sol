// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "../interfaces/IStrategies.sol";
import "../interfaces/IEulerSwap.sol";

/// @title EulerSwapStrategy
/// @notice Strategy for providing liquidity to EulerSwap pools on Unichain and earning fees
contract EulerSwapStrategy is IStrategies, ReentrancyGuard, AccessControl {
    using SafeERC20 for IERC20;

    // ============ Constants ============
    
    /// @notice Role for vault that can execute strategy
    bytes32 public constant VAULT_ROLE = keccak256("VAULT_ROLE");

    // ============ State Variables ============
    
    /// @notice The vault address that owns this strategy
    address public vault;
    
    /// @notice Whether the strategy is paused
    bool public paused;
    
    /// @notice The EulerSwap pool we're providing liquidity to
    address public eulerSwapPool;
    
    /// @notice The EulerSwap factory
    address public eulerSwapFactory;
    
    /// @notice The two underlying tokens
    address public token0;
    address public token1;
    
    /// @notice The two Euler vaults
    address public eVault0;
    address public eVault1;
    
    /// @notice Total amount deposited (in underlying tokens)
    uint256 public totalDeposited0;
    uint256 public totalDeposited1;
    
    /// @notice Last harvest timestamp
    uint256 public lastHarvest;
    
    /// @notice Strategy deployment timestamp
    uint256 public immutable deploymentTime;
    
    /// @notice Pool parameters for EulerSwap
    IEulerSwap.Params public poolParams;

    // ============ Events ============
    
    event PoolCreated(address indexed pool, address token0, address token1);
    event LiquidityAdded(uint256 amount0, uint256 amount1);
    event LiquidityRemoved(uint256 amount0, uint256 amount1);
    event FeesCollected(uint256 amount0, uint256 amount1);

    // ============ Errors ============
    
    error OnlyVault();
    error PoolAlreadyExists();
    error PoolNotFound();
    error InvalidTokens();

    // ============ Modifiers ============
    
    modifier onlyVault() {
        if (msg.sender != vault && !hasRole(VAULT_ROLE, msg.sender)) {
            revert OnlyVault();
        }
        _;
    }
    
    modifier whenNotPaused() {
        if (paused) revert IStrategies.StrategyPaused();
        _;
    }

    // ============ Constructor ============
    
    constructor(
        address _vault,
        address _eulerSwapFactory,
        address _token0,
        address _token1,
        address _eVault0,
        address _eVault1
    ) {
        require(_vault != address(0), "Invalid vault");
        require(_eulerSwapFactory != address(0), "Invalid factory");
        require(_token0 != address(0) && _token1 != address(0), "Invalid tokens");
        require(_eVault0 != address(0) && _eVault1 != address(0), "Invalid eVaults");
        
        vault = _vault;
        eulerSwapFactory = _eulerSwapFactory;
        token0 = _token0;
        token1 = _token1;
        eVault0 = _eVault0;
        eVault1 = _eVault1;
        
        deploymentTime = block.timestamp;
        lastHarvest = block.timestamp;
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(VAULT_ROLE, _vault);
        
        // Approve tokens for Euler vaults
        IERC20(token0).approve(_eVault0, type(uint256).max);
        IERC20(token1).approve(_eVault1, type(uint256).max);
        
        emit IStrategies.VaultSet(_vault);
    }

    // ============ External Functions ============
    
    /// @inheritdoc IStrategies
    function execute(uint256 amount, bytes calldata data) external onlyVault whenNotPaused nonReentrant {
        if (amount == 0) revert IStrategies.InvalidAmount();
        
        // Decode data to get amount distribution
        (uint256 amount0, uint256 amount1, bool createPool) = abi.decode(data, (uint256, uint256, bool));
        
        if (amount0 + amount1 != amount) revert IStrategies.InvalidAmount();
        
        // Transfer tokens from vault
        if (amount0 > 0) {
            IERC20(token0).safeTransferFrom(msg.sender, address(this), amount0);
        }
        if (amount1 > 0) {
            IERC20(token1).safeTransferFrom(msg.sender, address(this), amount1);
        }
        
        // Create pool if requested and doesn't exist
        if (createPool && eulerSwapPool == address(0)) {
            _createPool(amount0, amount1);
        }
        
        // Add liquidity to existing pool or deposit to Euler vaults directly
        if (eulerSwapPool != address(0)) {
            _addLiquidity(amount0, amount1);
        } else {
            // If no pool, deposit to Euler vaults directly
            if (amount0 > 0) {
                IEVault(eVault0).deposit(amount0, address(this));
                totalDeposited0 += amount0;
            }
            if (amount1 > 0) {
                IEVault(eVault1).deposit(amount1, address(this));
                totalDeposited1 += amount1;
            }
        }
        
        emit IStrategies.Deposit(amount);
        emit IStrategies.Executed(amount, data);
    }
    
    /// @inheritdoc IStrategies
    function harvest(bytes calldata data) external onlyVault nonReentrant {
        if (eulerSwapPool == address(0)) return;
        
        // Get current balances in Euler vaults
        uint256 currentBalance0 = IEVault(eVault0).balanceOf(address(this));
        uint256 currentBalance1 = IEVault(eVault1).balanceOf(address(this));
        
        // Convert to underlying assets
        uint256 currentAssets0 = currentBalance0 > 0 ? IEVault(eVault0).convertToAssets(currentBalance0) : 0;
        uint256 currentAssets1 = currentBalance1 > 0 ? IEVault(eVault1).convertToAssets(currentBalance1) : 0;
        
        // Calculate earned fees (simplified)
        uint256 earned0 = currentAssets0 > totalDeposited0 ? currentAssets0 - totalDeposited0 : 0;
        uint256 earned1 = currentAssets1 > totalDeposited1 ? currentAssets1 - totalDeposited1 : 0;
        
        if (earned0 > 0 || earned1 > 0) {
            // Withdraw earned amounts and send to vault
            if (earned0 > 0) {
                IEVault(eVault0).withdraw(earned0, vault, address(this));
            }
            if (earned1 > 0) {
                IEVault(eVault1).withdraw(earned1, vault, address(this));
            }
            
            lastHarvest = block.timestamp;
            emit FeesCollected(earned0, earned1);
        }
        
        emit IStrategies.Harvested(data);
    }
    
    /// @inheritdoc IStrategies
    function emergencyExit(bytes calldata data) external onlyVault nonReentrant {
        // Withdraw all funds from Euler vaults
        uint256 balance0 = IEVault(eVault0).balanceOf(address(this));
        uint256 balance1 = IEVault(eVault1).balanceOf(address(this));
        
        if (balance0 > 0) {
            uint256 assets0 = IEVault(eVault0).convertToAssets(balance0);
            IEVault(eVault0).withdraw(assets0, vault, address(this));
        }
        
        if (balance1 > 0) {
            uint256 assets1 = IEVault(eVault1).convertToAssets(balance1);
            IEVault(eVault1).withdraw(assets1, vault, address(this));
        }
        
        // Send any remaining tokens to vault
        uint256 remainingBalance0 = IERC20(token0).balanceOf(address(this));
        uint256 remainingBalance1 = IERC20(token1).balanceOf(address(this));
        
        if (remainingBalance0 > 0) {
            IERC20(token0).safeTransfer(vault, remainingBalance0);
        }
        if (remainingBalance1 > 0) {
            IERC20(token1).safeTransfer(vault, remainingBalance1);
        }
        
        totalDeposited0 = 0;
        totalDeposited1 = 0;
        
        emit IStrategies.EmergencyExited(balance0 + balance1, data);
    }
    
    /// @inheritdoc IStrategies
    function claimRewards(bytes calldata data) external onlyVault {
        // EulerSwap fees are automatically compounded
        emit IStrategies.Harvested(data);
    }
    
    /// @inheritdoc IStrategies
    function setVault(address _vault) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (_vault == address(0)) revert IStrategies.InvalidTokenAddress();
        
        vault = _vault;
        _grantRole(VAULT_ROLE, _vault);
        
        emit IStrategies.VaultSet(_vault);
    }
    
    /// @inheritdoc IStrategies
    function setPaused(bool _paused) external onlyRole(DEFAULT_ADMIN_ROLE) {
        paused = _paused;
        emit IStrategies.PausedState(_paused);
    }
    
    /// @inheritdoc IStrategies
    function addRewardToken(address tokenAddress) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (tokenAddress == address(0)) revert IStrategies.InvalidTokenAddress();
        // EulerSwap doesn't have additional reward tokens beyond trading fees
    }

    // ============ View Functions ============
    
    /// @inheritdoc IStrategies
    function underlyingToken() external view returns (address) {
        return token0; // Primary token
    }
    
    /// @inheritdoc IStrategies
    function protocol() external view returns (address) {
        return eulerSwapPool;
    }
    
    /// @inheritdoc IStrategies
    function depositSelector() external pure returns (bytes4) {
        return this.execute.selector;
    }
    
    /// @inheritdoc IStrategies
    function withdrawSelector() external pure returns (bytes4) {
        return this.emergencyExit.selector;
    }
    
    /// @inheritdoc IStrategies
    function claimSelector() external pure returns (bytes4) {
        return this.harvest.selector;
    }
    
    /// @inheritdoc IStrategies
    function getBalanceSelector() external pure returns (bytes4) {
        return this.getBalance.selector;
    }
    
    /// @inheritdoc IStrategies
    function getBalance() public view returns (uint256) {
        uint256 balance0 = IEVault(eVault0).balanceOf(address(this));
        uint256 balance1 = IEVault(eVault1).balanceOf(address(this));
        
        uint256 assets0 = balance0 > 0 ? IEVault(eVault0).convertToAssets(balance0) : 0;
        uint256 assets1 = balance1 > 0 ? IEVault(eVault1).convertToAssets(balance1) : 0;
        
        // Return total value in token0 terms (simplified)
        return assets0 + assets1;
    }
    
    /// @inheritdoc IStrategies
    function knownRewardTokens(address token) external view returns (bool) {
        return token == token0 || token == token1;
    }
    
    /// @inheritdoc IStrategies
    function rewardTokensList() external view returns (address[] memory) {
        address[] memory tokens = new address[](2);
        tokens[0] = token0;
        tokens[1] = token1;
        return tokens;
    }
    
    /// @inheritdoc IStrategies
    function queryProtocol(
        bytes4 selector,
        bytes calldata params
    ) external view returns (bytes memory) {
        if (eulerSwapPool == address(0)) return "";
        
        if (selector == IEulerSwap.getReserves.selector) {
            try IEulerSwap(eulerSwapPool).getReserves() returns (uint112 reserve0, uint112 reserve1, uint32 status) {
                return abi.encode(reserve0, reserve1, status);
            } catch {
                return "";
            }
        }
        
        if (selector == IEulerSwap.computeQuote.selector) {
            (address tokenIn, address tokenOut, uint256 amount, bool exactIn) = 
                abi.decode(params, (address, address, uint256, bool));
            try IEulerSwap(eulerSwapPool).computeQuote(tokenIn, tokenOut, amount, exactIn) returns (uint256 quote) {
                return abi.encode(quote);
            } catch {
                return "";
            }
        }
        
        return "";
    }

    // ============ Internal Functions ============
    
    /// @notice Create a new EulerSwap pool
    function _createPool(uint256 amount0, uint256 amount1) internal {
        if (eulerSwapPool != address(0)) revert PoolAlreadyExists();
        
        // Set up pool parameters
        poolParams = IEulerSwap.Params({
            vault0: eVault0,
            vault1: eVault1,
            eulerAccount: address(this),
            equilibriumReserve0: uint112(amount0),
            equilibriumReserve1: uint112(amount1),
            priceX: 1e18, // 1:1 price initially
            priceY: 1e18,
            concentrationX: 1e18, // Standard concentration
            concentrationY: 1e18,
            fee: 3000, // 0.3% fee (3000 basis points)
            protocolFee: 0,
            protocolFeeRecipient: address(0)
        });
        
        IEulerSwap.InitialState memory initialState = IEulerSwap.InitialState({
            currReserve0: uint112(amount0),
            currReserve1: uint112(amount1)
        });
        
        // Create pool through factory
        try IEulerSwapFactory(eulerSwapFactory).createPool(
            poolParams,
            initialState,
            keccak256(abi.encodePacked(address(this), block.timestamp))
        ) returns (address pool) {
            eulerSwapPool = pool;
            emit PoolCreated(pool, token0, token1);
        } catch {
            // Pool creation failed, continue with direct vault deposits
        }
    }
    
    /// @notice Add liquidity to the EulerSwap pool
    function _addLiquidity(uint256 amount0, uint256 amount1) internal {
        if (eulerSwapPool == address(0)) revert PoolNotFound();
        
        // Deposit tokens to Euler vaults
        if (amount0 > 0) {
            IEVault(eVault0).deposit(amount0, address(this));
            totalDeposited0 += amount0;
        }
        if (amount1 > 0) {
            IEVault(eVault1).deposit(amount1, address(this));
            totalDeposited1 += amount1;
        }
        
        emit LiquidityAdded(amount0, amount1);
    }
    
    /// @notice Get pool information
    function getPoolInfo() external view returns (
        address pool,
        uint256 reserve0,
        uint256 reserve1,
        uint256 totalValue
    ) {
        pool = eulerSwapPool;
        if (pool != address(0)) {
            try IEulerSwap(pool).getReserves() returns (uint112 r0, uint112 r1, uint32) {
                reserve0 = r0;
                reserve1 = r1;
            } catch {
                reserve0 = 0;
                reserve1 = 0;
            }
        }
        totalValue = getBalance();
    }
    
    /// @notice Get strategy metrics
    function getStrategyMetrics() external view returns (
        uint256 totalDeposits0,
        uint256 totalDeposits1,
        uint256 currentBalance0,
        uint256 currentBalance1,
        uint256 lastHarvestTime,
        bool hasPool
    ) {
        totalDeposits0 = totalDeposited0;
        totalDeposits1 = totalDeposited1;
        
        uint256 balance0 = IEVault(eVault0).balanceOf(address(this));
        uint256 balance1 = IEVault(eVault1).balanceOf(address(this));
        
        currentBalance0 = balance0 > 0 ? IEVault(eVault0).convertToAssets(balance0) : 0;
        currentBalance1 = balance1 > 0 ? IEVault(eVault1).convertToAssets(balance1) : 0;
        
        lastHarvestTime = lastHarvest;
        hasPool = eulerSwapPool != address(0);
    }
}