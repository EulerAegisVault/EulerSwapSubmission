import React, { useState } from 'react';
import { ArrowDownToLine, Loader2, ArrowUpFromLine } from 'lucide-react';
import { useApi } from '../hooks/useApi';
import toast from 'react-hot-toast';

export function VaultActions() {
  const [action, setAction] = useState<'deposit' | 'withdraw'>('deposit'); // NEW
  const [amount, setAmount] = useState('100');
  const [token, setToken] = useState<'usdc' | 'weth'>('usdc');
  const { depositToVault, loading, isBackendAvailable } = useApi();

  const handleAction = async () => {
    if (!amount) {
      toast.error('Please enter an amount');
      return;
    }

    if (action === 'deposit') {
      const response = await depositToVault(token, amount);
      if (response.success) {
        toast.success(`Deposited ${amount} ${token.toUpperCase()} to vault!`);
        setAmount('100');
      }
    } else {
      // Dummy withdraw logic
      await new Promise((resolve) => setTimeout(resolve, 1000));
      toast.success(`Withdrew ${amount} ${token.toUpperCase()} yield from vault!`);
      setAmount('100');
    }
  };

  return (
    <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-6">
        {action === 'deposit' ? (
          <ArrowDownToLine className="w-6 h-6 text-green-400" />
        ) : (
          <ArrowUpFromLine className="w-6 h-6 text-yellow-400" />
        )}
        <h2 className="text-xl font-bold text-white">
          {action === 'deposit' ? 'Deposit to Vault' : 'Withdraw Yield'}
        </h2>
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
            Vault operations require your local Python backend. Start your agent to enable transactions.
          </div>
        </div>
      )}

      <div className="space-y-4">
        {/* Action toggle */}
        <div>
          <div className="flex gap-2">
            <button
              onClick={() => setAction('deposit')}
              className={`flex-1 p-2 rounded-lg font-medium transition ${
                action === 'deposit'
                  ? 'bg-green-600 text-white'
                  : 'bg-white/10 text-white/60 hover:bg-white/20'
              }`}
            >
              Deposit
            </button>
            <button
              onClick={() => setAction('withdraw')}
              className={`flex-1 p-2 rounded-lg font-medium transition ${
                action === 'withdraw'
                  ? 'bg-yellow-500 text-white'
                  : 'bg-white/10 text-white/60 hover:bg-white/20'
              }`}
            >
              Withdraw Yield
            </button>
          </div>
        </div>

        {/* Token selection */}
        <div>
          <label className="block text-white/80 text-sm font-medium mb-2">
            Select Token
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => setToken('usdc')}
              className={`flex-1 p-3 rounded-xl font-medium transition-all duration-200 ${
                token === 'usdc'
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
                  : 'bg-white/10 text-white/70 hover:bg-white/20'
              }`}
            >
              USDC
            </button>
            <button
              onClick={() => setToken('weth')}
              className={`flex-1 p-3 rounded-xl font-medium transition-all duration-200 ${
                token === 'weth'
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
                  : 'bg-white/10 text-white/70 hover:bg-white/20'
              }`}
            >
              WETH
            </button>
          </div>
        </div>

        {/* Amount */}
        <div>
          <label className="block text-white/80 text-sm font-medium mb-2">
            Amount
          </label>
          <div className="relative">
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500/50"
              placeholder="100"
              min="0.1"
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/60 text-sm">
              {token.toUpperCase()}
            </div>
          </div>
        </div>

        {/* Action Button */}
        <button
          onClick={handleAction}
          disabled={loading || !amount || !isBackendAvailable}
          className={`w-full flex items-center justify-center gap-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white py-3 rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${
            !isBackendAvailable ? 'opacity-50' : ''
          }`}
        >
          {loading ? (
            <Loader2 size={18} className="animate-spin" />
          ) : action === 'deposit' ? (
            <ArrowDownToLine size={18} />
          ) : (
            <ArrowUpFromLine size={18} />
          )}
          {loading
            ? action === 'deposit'
              ? 'Depositing...'
              : 'Withdrawing...'
            : !isBackendAvailable
              ? 'Backend Required'
              : action === 'deposit'
                ? 'Deposit to Vault'
                : 'Withdraw Yield'}
        </button>
      </div>
    </div>
  );
}
