/**
 * Control Panel Header Component
 */

import React from 'react';
import { X } from 'lucide-react';
import { AutoBalanceIndicator } from '../AutoBalanceIndicator';
import { VoiceStatusIndicator } from '../../../../components/VoiceStatusIndicator';

interface ControlPanelHeaderProps {
  onClose: () => void;
}

export const ControlPanelHeader: React.FC<ControlPanelHeaderProps> = ({ onClose }) => {
  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 bg-green-500 rounded-full shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
        <span className="text-sm font-semibold">Control Center</span>
        <AutoBalanceIndicator />
      </div>

      <div className="flex items-center gap-2">
        <VoiceStatusIndicator className="ml-auto" />
        <button
          onClick={onClose}
          className="w-6 h-6 flex items-center justify-center rounded bg-white/10 hover:bg-white/20 border border-white/20 transition-colors"
          title="Fold panel"
        >
          <X size={14} />
        </button>
      </div>
    </div>
  );
};
