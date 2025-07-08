import { useState, useCallback } from 'react';
import { API_BASE_URL } from '../config/constants';
import toast from 'react-hot-toast';

interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

export function useApi() {
  const [loading, setLoading] = useState(false);

  const makeRequest = useCallback(async <T = any>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Request failed');
      }

      return { success: true, data };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  }, []);

  const invokeAgent = useCallback(async (command: string) => {
    return makeRequest('/invoke-agent', {
      method: 'POST',
      body: JSON.stringify({ command }),
    });
  }, [makeRequest]);

  const getStatus = useCallback(async () => {
    return makeRequest('/status');
  }, [makeRequest]);

  const getHealth = useCallback(async () => {
    return makeRequest('/health');
  }, [makeRequest]);

  const mintTokens = useCallback(async (usdcAmount: string, wethAmount: string) => {
    return makeRequest('/mint-tokens', {
      method: 'POST',
      body: JSON.stringify({ usdc_amount: usdcAmount, weth_amount: wethAmount }),
    });
  }, [makeRequest]);

  const depositToVault = useCallback(async (token: string, amount: string) => {
    return makeRequest('/deposit', {
      method: 'POST',
      body: JSON.stringify({ token, amount }),
    });
  }, [makeRequest]);

  const deployToStrategy = useCallback(async (amount: string) => {
    return makeRequest('/deploy', {
      method: 'POST',
      body: JSON.stringify({ amount }),
    });
  }, [makeRequest]);

  return {
    loading,
    invokeAgent,
    getStatus,
    getHealth,
    mintTokens,
    depositToVault,
    deployToStrategy,
  };
}