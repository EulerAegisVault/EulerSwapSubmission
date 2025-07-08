import React from 'react';
import { WagmiProvider } from 'wagmi';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { config } from './config/wagmi';
import { Header } from './components/Header';
import { VaultDashboard } from './components/VaultDashboard';
import { AIAgent } from './components/AIAgent';
import { TokenFaucet } from './components/TokenFaucet';
import { VaultActions } from './components/VaultActions';

const queryClient = new QueryClient();

function App() {
  return (
    <WagmiProvider config={config}>
      <QueryClientProvider client={queryClient}>
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
          <Header />
          
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Notice Banner */}
            <div className="mb-8 bg-amber-500/20 border border-amber-500/30 rounded-xl p-4">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-amber-400 rounded-full animate-pulse"></div>
                <div className="text-amber-400 font-medium">
                  Run frontend and backend locally to use full functionality
                </div>
              </div>
            </div>

            {/* Main Dashboard */}
            <div className="space-y-8">
              <VaultDashboard />
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <VaultActions />
                <TokenFaucet />
              </div>
              
              {/* AI Agent at the bottom */}
              <AIAgent />
            </div>
          </main>
          
          <Toaster
            position="bottom-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: 'rgba(15, 23, 42, 0.9)',
                color: 'white',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                backdropFilter: 'blur(16px)',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: 'white',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: 'white',
                },
              },
            }}
          />
        </div>
      </QueryClientProvider>
    </WagmiProvider>
  );
}

export default App;