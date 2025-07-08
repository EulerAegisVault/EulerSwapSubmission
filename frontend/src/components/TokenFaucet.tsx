import React, { useState } from 'react';
import { Droplets, Coins } from 'lucide-react';
import { useApi } from '../hooks/useApi';
import toast from 'react-hot-toast';

export function TokenFaucet() {
  const [usdcAmount, setUsdcAmount] = useState('1000');
  const [wethAmount, setWethAmount] = useState('1');
  const { mintTokens, loading, isBackendAvailable } = useApi();

  const handleMint = async () => {
    if (!usdcAmount || !wethAmount) {
      toast.error('Please enter amounts for both tokens');
      return;
    }

    const response = await mintTokens(usdcAmount, wethAmount);
    if (response.success) {
      toast.success('Tokens minted successfully!');
      // Reset to default values
      setUsdcAmount('1000');
      setWethAmount('1');
    }
  };

  return (
    <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <Droplets className="w-6 h-6 text-cyan-400" />
        <h2 className="text-xl font-bold text-white">Test Token Faucet</h2>
        {!isBackendAvailable && (
          <div className="px-2 py-1 bg-amber-500/20 border border-amber-500/30 rounded-lg">
            <span className="text-amber-400 text-xs font-medium">Backend Required</span>
          </div>
        )}
      </div>

      {!isBackendAvailable && (
        <div className="mb-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
          <div className="text-amber-400 text-sm font-medium mb-1">Backend Required</div>
          <div className="text-xs text-white/70">
            Token minting requires your local Python backend. Start your agent to enable this feature.
          </div>
        </div>
      )}

      <div className="space-y-4">
        <div>
          <label className="block text-white/80 text-sm font-medium mb-2">
            USDC Amount
          </label>
          <div className="relative">
            <input
              type="number"
              value={usdcAmount}
              onChange={(e) => setUsdcAmount(e.target.value)}
              className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50"
              placeholder="1000"
              min="1"
              max="10000"
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/60 text-sm">
              USDC
            </div>
          </div>
          <div className="text-xs text-white/50 mt-1">Max: 10,000 USDC per request</div>
        </div>

        <div>
          <label className="block text-white/80 text-sm font-medium mb-2">
            WETH Amount
          </label>
          <div className="relative">
            <input
              type="number"
              value={wethAmount}
              onChange={(e) => setWethAmount(e.target.value)}
              step="0.1"
              className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50"
              placeholder="1"
              min="0.1"
              max="10"
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/60 text-sm">
              WETH
            </div>
          </div>
          <div className="text-xs text-white/50 mt-1">Max: 10 WETH per request</div>
        </div>

        <button
          onClick={handleMint}
          disabled={loading || !usdcAmount || !wethAmount || !isBackendAvailable}
          className={`w-full flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white py-3 rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${
            !isBackendAvailable ? 'opacity-50' : ''
          }`}
        >
          <Coins size={18} />
          {loading ? 'Minting...' : isBackendAvailable ? 'Mint Test Tokens' : 'Backend Required'}
        </button>

        <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3">
          <div className="text-blue-400 text-sm font-medium mb-1">Contract Addresses:</div>
          <div className="space-y-1 text-xs font-mono">
            <div className="text-white/70">
              USDC: <span className="text-cyan-400">0xC093...1459</span>
            </div>
            <div className="text-white/70">
              WETH: <span className="text-cyan-400">0xf0f9...03Ca</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}