

import React, { useEffect } from 'react';
import { X, LogOut } from 'lucide-react';
import { AutoBalanceIndicator } from '../AutoBalanceIndicator';
import { VoiceStatusIndicator } from '../../../../components/VoiceStatusIndicator';
import { AuthGatedVoiceButton } from '../../../../components/AuthGatedVoiceButton';
import { useVoiceInteraction } from '../../../../hooks/useVoiceInteraction';
import { useNostrAuth } from '../../../../hooks/useNostrAuth';

interface ControlPanelHeaderProps {
  onClose: () => void;
}

export const ControlPanelHeader: React.FC<ControlPanelHeaderProps> = ({ onClose }) => {
  const { toggleListening } = useVoiceInteraction();
  const { authenticated, logout } = useNostrAuth();

  // Listen for SpacePilot button 3 voice toggle event
  useEffect(() => {
    const handler = () => { toggleListening(); };
    window.addEventListener('spacepilot:voice-toggle', handler);
    return () => window.removeEventListener('spacepilot:voice-toggle', handler);
  }, [toggleListening]);

  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 bg-green-500 rounded-full shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
        <span className="text-sm font-semibold">Control Center</span>
        <AutoBalanceIndicator />
      </div>

      <div className="flex items-center gap-2">
        <VoiceStatusIndicator className="ml-auto" />
        <AuthGatedVoiceButton size="sm" variant="ghost" />
        {authenticated && (
          <button
            onClick={logout}
            className="w-6 h-6 flex items-center justify-center rounded bg-white/10 hover:bg-white/20 border border-white/20 transition-colors"
            title="Logout"
          >
            <LogOut size={14} />
          </button>
        )}
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
