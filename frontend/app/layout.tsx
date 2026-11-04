import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Poker AI',
  description: 'Play self-play poker agents, inspect their policies, and explore exact game-theoretic evaluations.'
};

export default function Layout({ children }: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body>{children}</body></html>;
}
