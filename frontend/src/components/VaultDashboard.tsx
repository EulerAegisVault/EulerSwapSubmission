import React from 'react';
import { Vault, Activity, TrendingUp, Shield } from 'lucide-react';
import { StatusCard } from './StatusCard';
import { useVaultData } from '../hooks/useVaultData';

export function VaultDashboard() {
  const { vaultStatus, loading } = useVaultData();

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Vault className="w-6 h-6 text-blue-400" />
        <h2 className="text-2xl font-bold text-white">Vault Overview</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatusCard
          title="USDC Vault"
          value={vaultStatus?.usdc_vault?.total_assets || '0 USDC'}
          subtitle={`${vaultStatus?.usdc_vault?.total_shares || '0'} shares`}
          icon={Vault}
          loading={loading}
        />
        
        <StatusCard
          title="WETH Vault"
          value={vaultStatus?.weth_vault?.total_assets || '0 WETH'}
          subtitle={`${vaultStatus?.weth_vault?.total_shares || '0'} shares`}
          icon={Vault}
          loading={loading}
        />
        
        <StatusCard
          title="Strategy Balance"
          value={vaultStatus?.strategy?.balance || '0 USDC'}
          subtitle="EulerSwap Strategy"
          icon={TrendingUp}
          trend="up"
          loading={loading}
        />
        
        <StatusCard
          title="Agent Balance"
          value={vaultStatus?.agent_balances?.usdc || '0 USDC'}
          subtitle={`${vaultStatus?.agent_balances?.weth || '0'} WETH`}
          icon={Activity}
          loading={loading}
        />
      </div>

      {vaultStatus && (
        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-5 h-5 text-green-400" />
            <h3 className="text-lg font-semibold text-white">System Status</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-white/60 mb-1">Network</div>
              <div className="text-white font-medium">{vaultStatus.network}</div>
            </div>
            <div>
              <div className="text-white/60 mb-1">Agent Address</div>
              <div className="text-white font-mono text-xs">
                {vaultStatus.agent_address?.slice(0, 8)}...{vaultStatus.agent_address?.slice(-6)}
              </div>
            </div>
            <div>
              <div className="text-white/60 mb-1">Status</div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-green-400 font-medium">Active</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}