import { useState, useEffect, useCallback } from 'react';
import { useApi } from './useApi';

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
  const { getStatus, getHealth } = useApi();

  const refreshData = useCallback(async () => {
    setLoading(true);
    try {
      const statusResponse = await getStatus();
      if (statusResponse.success && statusResponse.data?.status) {
        // Parse the status string if it's JSON
        const statusStr = statusResponse.data.status;
        const match = statusStr.match(/Unichain Vault Status: ({.*})/s);
        if (match) {
          const parsedStatus = JSON.parse(match[1]);
          setVaultStatus(parsedStatus);
        }
      }
    } catch (error) {
      console.error('Error fetching vault data:', error);
    } finally {
      setLoading(false);
    }
  }, [getStatus]);

  useEffect(() => {
    refreshData();
    const interval = setInterval(refreshData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [refreshData]);

  return {
    vaultStatus,
    loading,
    refreshData,
  };
}