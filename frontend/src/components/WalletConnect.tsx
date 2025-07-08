import React from 'react';
import { useAccount, useConnect, useDisconnect } from 'wagmi';
import { Wallet, LogOut, Copy, ExternalLink } from 'lucide-react';
import toast from 'react-hot-toast';

export function WalletConnect() {
  const { address, isConnected } = useAccount();
  const { connect, connectors, isPending } = useConnect();
  const { disconnect } = useDisconnect();

  const copyAddress = () => {
    if (address) {
      navigator.clipboard.writeText(address);
      toast.success('Address copied to clipboard');
    }
  };

  const formatAddress = (addr: string) => {
    return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
  };

  if (isConnected && address) {
    return (
      <div className="flex items-center gap-2 bg-white/10 backdrop-blur-md border border-white/20 rounded-xl px-4 py-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-white font-medium">{formatAddress(address)}</span>
        </div>
        <button
          onClick={copyAddress}
          className="p-1 hover:bg-white/10 rounded-lg transition-colors"
          title="Copy address"
        >
          <Copy size={14} className="text-white/70" />
        </button>
        <a
          href={`https://unichain.org/address/${address}`}
          target="_blank"
          rel="noopener noreferrer"
          className="p-1 hover:bg-white/10 rounded-lg transition-colors"
          title="View on explorer"
        >
          <ExternalLink size={14} className="text-white/70" />
        </a>
        <button
          onClick={() => disconnect()}
          className="p-1 hover:bg-white/10 rounded-lg transition-colors text-red-400"
          title="Disconnect"
        >
          <LogOut size={14} />
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={() => connect({ connector: connectors[0] })}
      disabled={isPending || !connectors[0]}
      className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-6 py-2 rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      <Wallet size={18} />
      {isPending ? 'Connecting...' : 'Connect Wallet'}
    </button>
  );
}