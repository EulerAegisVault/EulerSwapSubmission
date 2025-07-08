import { createConfig, http } from 'wagmi';
import { injected } from 'wagmi/connectors';
import { UNICHAIN_CHAIN } from './constants';

export const config = createConfig({
  chains: [UNICHAIN_CHAIN],
  connectors: [
    injected(),
  ],
  transports: {
    [UNICHAIN_CHAIN.id]: http(),
  },
});