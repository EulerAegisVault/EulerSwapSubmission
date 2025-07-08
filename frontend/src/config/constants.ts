export const UNICHAIN_CONFIG = {
  chainId: 130,
  name: 'Unichain',
  currency: 'ETH',
  rpcUrl: 'https://unichain-rpc.publicnode.com',
  blockExplorer: 'https://unichain.org',
} as const;

export const CONTRACT_ADDRESSES = {
  VAULT_FACTORY: '0xEc0036beC79dBCf0279dAd3bC59F6231b9F461d9',
  STRATEGY_FACTORY: '0xa5752653c78D9254EB3082d8559b76a38C9E8563',
  MOCK_USDC: '0xC0933C5440c656464D1Eb1F886422bE3466B1459',
  MOCK_WETH: '0xf0f994B4A8dB86A46a1eD4F12263c795b26703Ca',
  USDC_VAULT: '0xFAf4Af2Ed51cDb2967B0d204074cc2e4302F9188',
  WETH_VAULT: '0x112b33c07deE1697d9b4A68b2BA0F66c2417635C',
  EULERSWAP_STRATEGY: '0x807463769044222F3b7B5F98da8d3E25e0aC44B0',
} as const;

export const API_BASE_URL = 'http://localhost:8000';

export const UNICHAIN_CHAIN = {
  id: 130,
  name: 'Unichain',
  network: 'unichain',
  nativeCurrency: {
    decimals: 18,
    name: 'Ether',
    symbol: 'ETH',
  },
  rpcUrls: {
    public: { http: ['https://unichain-rpc.publicnode.com'] },
    default: { http: ['https://unichain-rpc.publicnode.com'] },
  },
  blockExplorers: {
    default: { name: 'Unichain Explorer', url: 'https://unichain.org' },
  },
} as const;