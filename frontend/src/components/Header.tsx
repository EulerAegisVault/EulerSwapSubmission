import React from 'react';
import { Shield, Zap } from 'lucide-react';
import { WalletConnect } from './WalletConnect';

export function Header() {
  return (
    <header className="bg-gradient-to-r from-slate-900 to-slate-800 border-b border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Shield className="w-8 h-8 text-blue-400" />
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                AegisVault
              </span>
            </div>
            <div className="hidden sm:flex items-center gap-2 px-3 py-1 bg-amber-500/20 border border-amber-500/30 rounded-lg">
              <Zap size={14} className="text-amber-400" />
              <span className="text-amber-400 text-sm font-medium">Unichain</span>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <WalletConnect />
          </div>
        </div>
      </div>
    </header>
  );
}