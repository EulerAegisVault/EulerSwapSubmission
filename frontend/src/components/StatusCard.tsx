import React from 'react';
import { TrendingUp, Wallet, Activity, DivideIcon as LucideIcon } from 'lucide-react';

interface StatusCardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  loading?: boolean;
}

export function StatusCard({ title, value, subtitle, icon: Icon, trend = 'neutral', loading }: StatusCardProps) {
  const trendColors = {
    up: 'text-green-400',
    down: 'text-red-400',
    neutral: 'text-blue-400'
  };

  return (
    <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all duration-300">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg">
            <Icon className="w-5 h-5 text-blue-400" />
          </div>
          <h3 className="text-white/80 font-medium">{title}</h3>
        </div>
        {trend !== 'neutral' && (
          <TrendingUp className={`w-4 h-4 ${trendColors[trend]} ${trend === 'down' ? 'rotate-180' : ''}`} />
        )}
      </div>
      
      <div className="space-y-1">
        {loading ? (
          <div className="space-y-2">
            <div className="h-8 bg-white/10 rounded animate-pulse"></div>
            <div className="h-4 bg-white/5 rounded animate-pulse w-2/3"></div>
          </div>
        ) : (
          <>
            <div className="text-2xl font-bold text-white">{value}</div>
            {subtitle && (
              <div className="text-sm text-white/60">{subtitle}</div>
            )}
          </>
        )}
      </div>
    </div>
  );
}