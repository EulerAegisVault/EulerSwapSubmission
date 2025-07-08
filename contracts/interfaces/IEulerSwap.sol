// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

/// @title IEulerSwap Interface
/// @notice Interface for EulerSwap pools on Unichain
interface IEulerSwap {
    /// @dev Immutable pool parameters
    struct Params {
        address vault0;
        address vault1;
        address eulerAccount;
        uint112 equilibriumReserve0;
        uint112 equilibriumReserve1;
        uint256 priceX;
        uint256 priceY;
        uint256 concentrationX;
        uint256 concentrationY;
        uint256 fee;
        uint256 protocolFee;
        address protocolFeeRecipient;
    }

    /// @dev Starting configuration of pool storage
    struct InitialState {
        uint112 currReserve0;
        uint112 currReserve1;
    }

    function activate(InitialState calldata initialState) external;
    function getParams() external view returns (Params memory);
    function getAssets() external view returns (address asset0, address asset1);
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 status);
    function computeQuote(address tokenIn, address tokenOut, uint256 amount, bool exactIn) external view returns (uint256);
    function getLimits(address tokenIn, address tokenOut) external view returns (uint256 limitIn, uint256 limitOut);
    function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes calldata data) external;
}

/// @title IEulerSwapFactory Interface  
interface IEulerSwapFactory {
    function createPool(
        IEulerSwap.Params calldata params,
        IEulerSwap.InitialState calldata initialState,
        bytes32 salt
    ) external returns (address pool);

    function getPool(address token0, address token1) external view returns (address pool);
    function pools() external view returns (address[] memory);
    function poolByEulerAccount(address eulerAccount) external view returns (address pool);
    function eulerSwapImplementation() external view returns (address);
    function EVC() external view returns (address);
    function protocolFee() external view returns (uint256);
    function protocolFeeRecipient() external view returns (address);
    function deployPool(
        IEulerSwap.Params calldata params,
        IEulerSwap.InitialState calldata initialState,
        bytes32 salt
    ) external returns (address pool);
    function uninstallPool() external;
}

/// @title IEVC Interface
interface IEVC {
    struct BatchItem {
        address onBehalfOfAccount;
        address targetContract;
        uint256 value;
        bytes data;
    }

    function batch(BatchItem[] calldata items) external;
    function setAccountOperator(address account, address operator, bool authorized) external;
}

/// @title IEVault Interface
interface IEVault {
    function asset() external view returns (address);
    function deposit(uint256 assets, address receiver) external returns (uint256 shares);
    function withdraw(uint256 assets, address receiver, address owner) external returns (uint256 shares);
    function redeem(uint256 shares, address receiver, address owner) external returns (uint256 assets);
    function balanceOf(address account) external view returns (uint256);
    function convertToAssets(uint256 shares) external view returns (uint256);
    function convertToShares(uint256 assets) external view returns (uint256);
    function totalSupply() external view returns (uint256);
    function totalAssets() external view returns (uint256);
}