import React from 'react';
import { VoiceIndicator, VoiceIndicatorProps } from './VoiceIndicator';
import { useSettingsStore } from '../store/settingsStore';
import { nostrAuth } from '../services/nostrAuthService';

export interface AuthGatedVoiceIndicatorProps extends VoiceIndicatorProps {
  onAuthRequired?: () => void;
}

export const AuthGatedVoiceIndicator: React.FC<AuthGatedVoiceIndicatorProps> = ({
  onAuthRequired,
  ...voiceIndicatorProps
}) => {
  const authenticated = useSettingsStore(state => state.authenticated);
  const authEnabled = useSettingsStore(state => state.settings?.auth?.enabled);

  
  if (!authEnabled) {
    return <VoiceIndicator {...voiceIndicatorProps} />;
  }

  
  if (!authenticated || !nostrAuth.isAuthenticated()) {
    return (
      <div 
        className={`${voiceIndicatorProps.className || ''} text-muted-foreground italic cursor-pointer`}
        onClick={() => {
          if (onAuthRequired) {
            onAuthRequired();
          } else {
            alert('Please authenticate with Nostr to use voice features');
          }
        }}
      >
        Voice features require Nostr authentication
      </div>
    );
  }

  
  return <VoiceIndicator {...voiceIndicatorProps} />;
};