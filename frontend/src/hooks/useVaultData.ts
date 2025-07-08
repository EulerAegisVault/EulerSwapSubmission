import { useState, useEffect } from 'react';
import { useReadContract, useAccount } from 'wagmi';
import { CONTRACT_ADDRESSES } from '../config/constants';
import { formatUnits } from 'viem';

// ERC4626 Vault ABI (minimal)
const VAULT_ABI = [
  {
    name: 'totalAssets',
    type: 'function',
    inputs: [],
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
  },
  {
    name: 'totalSupply',
    type: 'function',
    inputs: [],
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
  },
  {
    name: 'balanceOf',
    type: 'function',
    inputs: [{ type: 'address' }],
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
  },
  {
    name: 'asset',
    type: 'function',
    inputs: [],
    outputs: [{ type: 'address' }],
    stateMutability: 'view',
  },
] as const;

// ERC20 ABI (minimal)
const ERC20_ABI = [
  {
    name: 'balanceOf',
    type: 'function',
    inputs: [{ type: 'address' }],
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
  },
  {
    name: 'symbol',
    type: 'function',
    inputs: [],
    outputs: [{ type: 'string' }],
    stateMutability: 'view',
  },
  {
    name: 'decimals',
    type: 'function',
    inputs: [],
    outputs: [{ type: 'uint8' }],
    stateMutability: 'view',
  },
] as const;

// Strategy ABI (minimal)
const STRATEGY_ABI = [
  {
    name: 'getBalance',
    type: 'function',
    inputs: [],
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
  },
] as const;

interface VaultStatus {
  network: string;
  agent_address: string;
  usdc_vault: {
    address: string;
    total_assets: string;
    total_shares: string;
    share_price: string;
  };
  weth_vault: {
    address: string;
    total_assets: string;
    total_shares: string;
    share_price: string;
  };
  strategy: {
    address: string;
    balance: string;
  };
  agent_balances: {
    usdc: string;
    weth: string;
  };
}

export function useVaultData() {
  const [vaultStatus, setVaultStatus] = useState<VaultStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const { address } = useAccount();

  // USDC Vault data
  const { data: usdcTotalAssets } = useReadContract({
    address: CONTRACT_ADDRESSES.USDC_VAULT as `0x${string}`,
    abi: VAULT_ABI,
    functionName: 'totalAssets',
  });

  const { data: usdcTotalSupply } = useReadContract({
    address: CONTRACT_ADDRESSES.USDC_VAULT as `0x${string}`,
    abi: VAULT_ABI,
    functionName: 'totalSupply',
  });

  // WETH Vault data
  const { data: wethTotalAssets } = useReadContract({
    address: CONTRACT_ADDRESSES.WETH_VAULT as `0x${string}`,
    abi: VAULT_ABI,
    functionName: 'totalAssets',
  });

  const { data: wethTotalSupply } = useReadContract({
    address: CONTRACT_ADDRESSES.WETH_VAULT as `0x${string}`,
    abi: VAULT_ABI,
    functionName: 'totalSupply',
  });

  // Strategy balance
  const { data: strategyBalance } = useReadContract({
    address: CONTRACT_ADDRESSES.EULERSWAP_STRATEGY as `0x${string}`,
    abi: STRATEGY_ABI,
    functionName: 'getBalance',
  });

  // User token balances (if connected)
  const { data: userUsdcBalance } = useReadContract({
    address: CONTRACT_ADDRESSES.MOCK_USDC as `0x${string}`,
    abi: ERC20_ABI,
    functionName: 'balanceOf',
    args: address ? [address] : undefined,
    query: {
      enabled: !!address,
    },
  });

  const { data: userWethBalance } = useReadContract({
    address: CONTRACT_ADDRESSES.MOCK_WETH as `0x${string}`,
    abi: ERC20_ABI,
    functionName: 'balanceOf',
    args: address ? [address] : undefined,
    query: {
      enabled: !!address,
    },
  });

  useEffect(() => {
    if (
      usdcTotalAssets !== undefined &&
      usdcTotalSupply !== undefined &&
      wethTotalAssets !== undefined &&
      wethTotalSupply !== undefined &&
      strategyBalance !== undefined
    ) {
      // Format the data
      const usdcAssets = Number(formatUnits(usdcTotalAssets, 6));
      const usdcShares = Number(formatUnits(usdcTotalSupply, 18));
      const wethAssets = Number(formatUnits(wethTotalAssets, 18));
      const wethShares = Number(formatUnits(wethTotalSupply, 18));
      const strategyBal = Number(formatUnits(strategyBalance, 6));

      const userUsdc = userUsdcBalance ? Number(formatUnits(userUsdcBalance, 6)) : 0;
      const userWeth = userWethBalance ? Number(formatUnits(userWethBalance, 18)) : 0;

      const status: VaultStatus = {
        network: 'Unichain',
        agent_address: address || '0x0000000000000000000000000000000000000000',
        usdc_vault: {
          address: CONTRACT_ADDRESSES.USDC_VAULT,
          total_assets: `${usdcAssets.toFixed(2)} USDC`,
          total_shares: usdcShares.toFixed(6),
          share_price: usdcShares > 0 ? (usdcAssets / usdcShares).toFixed(6) : '1.000000',
        },
        weth_vault: {
          address: CONTRACT_ADDRESSES.WETH_VAULT,
          total_assets: `${wethAssets.toFixed(6)} WETH`,
          total_shares: wethShares.toFixed(6),
          share_price: wethShares > 0 ? (wethAssets / wethShares).toFixed(6) : '1.000000',
        },
        strategy: {
          address: CONTRACT_ADDRESSES.EULERSWAP_STRATEGY,
          balance: `${strategyBal.toFixed(2)} USDC`,
        },
        agent_balances: {
          usdc: `${userUsdc.toFixed(2)} USDC`,
          weth: `${userWeth.toFixed(6)} WETH`,
        },
      };

      setVaultStatus(status);
      setLoading(false);
    }
  }, [
    usdcTotalAssets,
    usdcTotalSupply,
    wethTotalAssets,
    wethTotalSupply,
    strategyBalance,
    userUsdcBalance,
    userWethBalance,
    address,
  ]);

  const refreshData = () => {
    // Data will automatically refresh due to wagmi's built-in polling
    setLoading(true);
  };

  return {
    vaultStatus,
    loading,
    refreshData,
  };
}