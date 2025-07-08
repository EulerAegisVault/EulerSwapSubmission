import { useState, useCallback } from 'react';
import toast from 'react-hot-toast';

interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

export function useApi() {
  const [loading, setLoading] = useState(false);

  // Check if we're in development and backend is available
  const isLocalDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const API_BASE_URL = isLocalDevelopment ? 'http://localhost:8000' : null;

  const makeRequest = useCallback(async <T = any>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> => {
    if (!API_BASE_URL) {
      toast.error('Backend not available. Please run the backend locally for full functionality.');
      return { success: false, error: 'Backend not available' };
    }

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
  }, [API_BASE_URL]);

  const invokeAgent = useCallback(async (command: string) => {
    if (!API_BASE_URL) {
      toast.error('AI Agent requires local backend. Please run your Python agent locally.');
      return { success: false, error: 'Backend not available' };
    }
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
    if (!API_BASE_URL) {
      toast.error('Token minting requires local backend. Please run your Python agent locally.');
      return { success: false, error: 'Backend not available' };
    }
    return makeRequest('/mint-tokens', {
      method: 'POST',
      body: JSON.stringify({ usdc_amount: usdcAmount, weth_amount: wethAmount }),
    });
  }, [makeRequest]);

  const depositToVault = useCallback(async (token: string, amount: string) => {
    if (!API_BASE_URL) {
      toast.error('Vault operations require local backend. Please run your Python agent locally.');
      return { success: false, error: 'Backend not available' };
    }
    return makeRequest('/deposit', {
      method: 'POST',
      body: JSON.stringify({ token, amount }),
    });
  }, [makeRequest]);

  const deployToStrategy = useCallback(async (amount: string) => {
    if (!API_BASE_URL) {
      toast.error('Strategy operations require local backend. Please run your Python agent locally.');
      return { success: false, error: 'Backend not available' };
    }
    return makeRequest('/deploy', {
      method: 'POST',
      body: JSON.stringify({ amount }),
    });
  }, [makeRequest]);

  return {
    loading,
    isBackendAvailable: !!API_BASE_URL,
    invokeAgent,
    getStatus,
    getHealth,
    mintTokens,
    depositToVault,
    deployToStrategy,
  };
}