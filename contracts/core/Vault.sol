// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC4626.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "../interfaces/IStrategies.sol";

/// @title Vault Contract for EulerSwap Integration
/// @notice An ERC4626 vault that manages deposits and allocates them to EulerSwap strategies
contract Vault is Ownable, ERC4626, AccessControl, ReentrancyGuard {
    using SafeERC20 for IERC20;

    bytes32 public constant MANAGER_ROLE = keccak256("MANAGER_ROLE");
    bytes32 public constant AGENT_ROLE = keccak256("AGENT_ROLE");

    address[] public strategies;
    mapping(address => bool) public isStrategy;
    mapping(address => uint256) public strategyAllocations;
    
    uint256 public totalAllocated;

    event StrategyAdded(address indexed strategy);
    event StrategyRemoved(address indexed strategy);
    event StrategyExecuted(address indexed strategy, uint256 amount, bytes data);
    event StrategyHarvested(address indexed strategy, bytes data);
    event StrategyEmergencyExit(address indexed strategy, uint256 recovered);

    error InvalidStrategy();
    error StrategyAlreadyExists();
    error StrategyDoesNotExist();
    error ExecutionFailed();
    error InvalidAddress();
    error InsufficientBalance();
    error InvalidAmount();

    modifier onlyManager() {
        require(hasRole(MANAGER_ROLE, msg.sender), "Vault: caller is not a manager");
        _;
    }

    modifier onlyAgent() {
        require(hasRole(AGENT_ROLE, msg.sender), "Vault: caller is not an agent");
        _;
    }

    constructor(
        IERC20 _asset,
        string memory _name,
        string memory _symbol,
        address _manager,
        address _agent
    ) ERC4626(_asset) ERC20(_name, _symbol) {
        require(_manager != address(0), "Manager cannot be zero address");
        require(_agent != address(0), "Agent cannot be zero address");

        _transferOwnership(msg.sender);
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setRoleAdmin(MANAGER_ROLE, DEFAULT_ADMIN_ROLE);
        _setRoleAdmin(AGENT_ROLE, DEFAULT_ADMIN_ROLE);

        _grantRole(MANAGER_ROLE, _manager);
        _grantRole(AGENT_ROLE, _agent);
    }

    // ============ Strategy Management ============

    function addStrategy(address _strategy) external onlyManager {
        if (_strategy == address(0)) revert InvalidAddress();
        if (isStrategy[_strategy]) revert StrategyAlreadyExists();

        try IStrategies(_strategy).underlyingToken() returns (address strategyAsset) {
            require(strategyAsset == asset() || strategyAsset == address(0), "Strategy asset mismatch");
        } catch {
            revert InvalidStrategy();
        }

        isStrategy[_strategy] = true;
        strategies.push(_strategy);

        emit StrategyAdded(_strategy);
    }

    function removeStrategy(address _strategy) external onlyManager {
        if (!isStrategy[_strategy]) revert StrategyDoesNotExist();
        
        if (strategyAllocations[_strategy] > 0) {
            _emergencyExitStrategy(_strategy);
        }
        
        isStrategy[_strategy] = false;

        for (uint256 i = 0; i < strategies.length; i++) {
            if (strategies[i] == _strategy) {
                strategies[i] = strategies[strategies.length - 1];
                strategies.pop();
                break;
            }
        }
        
        delete strategyAllocations[_strategy];
        emit StrategyRemoved(_strategy);
    }

    // ============ Strategy Operations ============

    function depositToStrategy(
        address _strategy,
        uint256 _amount,
        bytes calldata _data
    ) external onlyAgent nonReentrant {
        if (!isStrategy[_strategy]) revert StrategyDoesNotExist();
        if (_amount == 0) revert InvalidAmount();
        
        uint256 availableBalance = IERC20(asset()).balanceOf(address(this));
        if (availableBalance < _amount) revert InsufficientBalance();

        IERC20(asset()).safeApprove(_strategy, _amount);
        
        try IStrategies(_strategy).execute(_amount, _data) {
            strategyAllocations[_strategy] += _amount;
            totalAllocated += _amount;
            emit StrategyExecuted(_strategy, _amount, _data);
        } catch {
            IERC20(asset()).safeApprove(_strategy, 0);
            revert ExecutionFailed();
        }
    }
    
    function harvestStrategy(
        address _strategy,
        bytes calldata _data
    ) external onlyAgent nonReentrant {
        if (!isStrategy[_strategy]) revert StrategyDoesNotExist();

        uint256 balanceBefore = IERC20(asset()).balanceOf(address(this));
        
        try IStrategies(_strategy).harvest(_data) {
            emit StrategyHarvested(_strategy, _data);
        } catch {
            emit StrategyHarvested(_strategy, _data);
        }
    }
    
    function harvestAllStrategies() external onlyAgent nonReentrant {
        for (uint256 i = 0; i < strategies.length; i++) {
            address strategy = strategies[i];
            if (strategyAllocations[strategy] > 0) {
                try IStrategies(strategy).harvest("") {
                    emit StrategyHarvested(strategy, "");
                } catch {
                    // Continue with next strategy
                }
            }
        }
    }

    function emergencyExit(address _strategy, bytes calldata _data) external onlyManager nonReentrant {
        if (!isStrategy[_strategy]) revert StrategyDoesNotExist();
        _emergencyExitStrategy(_strategy);
    }
    
    function emergencyExitAll() external onlyManager nonReentrant {
        for (uint256 i = 0; i < strategies.length; i++) {
            address strategy = strategies[i];
            if (strategyAllocations[strategy] > 0) {
                _emergencyExitStrategy(strategy);
            }
        }
    }

    function _emergencyExitStrategy(address _strategy) internal {
        uint256 balanceBefore = IERC20(asset()).balanceOf(address(this));
        
        try IStrategies(_strategy).emergencyExit("") {
            uint256 balanceAfter = IERC20(asset()).balanceOf(address(this));
            uint256 recovered = balanceAfter > balanceBefore ? balanceAfter - balanceBefore : 0;
            
            if (strategyAllocations[_strategy] > 0) {
                totalAllocated -= strategyAllocations[_strategy];
                strategyAllocations[_strategy] = 0;
            }
            
            emit StrategyEmergencyExit(_strategy, recovered);
        } catch {
            if (strategyAllocations[_strategy] > 0) {
                totalAllocated -= strategyAllocations[_strategy];
                strategyAllocations[_strategy] = 0;
            }
            emit StrategyEmergencyExit(_strategy, 0);
        }
    }

    // ============ View Functions ============

    function getStrategies() external view returns (address[] memory) {
        return strategies;
    }
    
    function getStrategyAllocation(address _strategy) external view returns (uint256) {
        return strategyAllocations[_strategy];
    }
    
    function getTotalAllocated() external view returns (uint256) {
        return totalAllocated;
    }
    
    function getAvailableBalance() external view returns (uint256) {
        return IERC20(asset()).balanceOf(address(this));
    }
    
    function getStrategyBalance(address _strategy) external view returns (uint256) {
        if (!isStrategy[_strategy]) return 0;
        
        try IStrategies(_strategy).getBalance() returns (uint256 balance) {
            return balance;
        } catch {
            return 0;
        }
    }
    
    function getTotalStrategyBalances() external view returns (uint256 total) {
        for (uint256 i = 0; i < strategies.length; i++) {
            address strategy = strategies[i];
            try IStrategies(strategy).getBalance() returns (uint256 balance) {
                total += balance;
            } catch {
                // Skip failed strategies
            }
        }
    }

    // ============ ERC4626 Overrides ============

    function totalAssets() public view override returns (uint256) {
        uint256 vaultBalance = IERC20(asset()).balanceOf(address(this));
        uint256 strategyBalances = this.getTotalStrategyBalances();
        return vaultBalance + strategyBalances;
    }

    function maxDeposit(address) public pure override returns (uint256) {
        return type(uint256).max;
    }

    function maxMint(address) public pure override returns (uint256) {
        return type(uint256).max;
    }

    function maxWithdraw(address owner) public view override returns (uint256) {
        return _convertToAssets(balanceOf(owner), Math.Rounding.Down);
    }

    function maxRedeem(address owner) public view override returns (uint256) {
        return balanceOf(owner);
    }

    // ============ Admin Functions ============
    
    function setManager(address _newManager) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_newManager != address(0), "Invalid manager");
        _grantRole(MANAGER_ROLE, _newManager);
    }
    
    function setAgent(address _newAgent) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_newAgent != address(0), "Invalid agent");
        _grantRole(AGENT_ROLE, _newAgent);
    }
    
    function revokeManager(address _manager) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _revokeRole(MANAGER_ROLE, _manager);
    }
    
    function revokeAgent(address _agent) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _revokeRole(AGENT_ROLE, _agent);
    }
}