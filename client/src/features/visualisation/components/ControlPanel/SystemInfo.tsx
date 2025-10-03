/**
 * System Info Display Component
 */

import React from 'react';

interface SystemInfoProps {
  showStats: boolean;
  enableBloom: boolean;
}

export const SystemInfo: React.FC<SystemInfoProps> = ({ showStats, enableBloom }) => {
  return (
    <div className="flex justify-between text-xs opacity-80 pb-3 mb-4 border-b border-white/15">
      <div>Stats: {showStats ? '✓ ON' : '✗ OFF'}</div>
      <div>Bloom: {enableBloom ? '✓ ON' : '✗ OFF'}</div>
    </div>
  );
};
