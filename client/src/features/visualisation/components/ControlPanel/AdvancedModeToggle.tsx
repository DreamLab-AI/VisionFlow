import React from 'react';
import { Settings2, Lock, Unlock } from 'lucide-react';
import { useControlPanelContext } from '../../../settings/components/control-panel-context';
import { useSettingsStore } from '../../../../store/settingsStore';
import { createLogger } from '../../../../utils/loggerConfig';

const logger = createLogger('AdvancedModeToggle');

interface AdvancedModeToggleProps {
  compact?: boolean;
}

export const AdvancedModeToggle: React.FC<AdvancedModeToggleProps> = ({
  compact = false
}) => {
  const { advancedMode, toggleAdvancedMode } = useControlPanelContext();
  const isPowerUser = useSettingsStore(state => state.isPowerUser);
  const isAuthenticated = useSettingsStore(state => state.user !== null);

  const canToggle = isAuthenticated || !advancedMode; // Can always disable, need auth to enable

  const handleToggle = () => {
    if (!advancedMode && !isAuthenticated) {
      // Attempting to enable advanced mode without auth - show message
      // In production this would trigger login flow
      logger.warn('Advanced mode requires Nostr authentication for write access');
    }
    toggleAdvancedMode();
  };

  if (compact) {
    return (
      <button
        onClick={handleToggle}
        title={advancedMode ? 'Switch to Basic Mode' : 'Switch to Advanced Mode'}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '32px',
          height: '32px',
          borderRadius: '4px',
          border: `1px solid ${advancedMode ? 'rgba(168,85,247,0.5)' : 'rgba(255,255,255,0.15)'}`,
          background: advancedMode
            ? 'linear-gradient(135deg, rgba(168,85,247,0.2), rgba(139,92,246,0.1))'
            : 'rgba(255,255,255,0.05)',
          cursor: 'pointer',
          transition: 'all 0.2s',
          color: advancedMode ? '#a855f7' : 'rgba(255,255,255,0.6)'
        }}
      >
        <Settings2 size={14} />
      </button>
    );
  }

  return (
    <div style={{
      background: advancedMode
        ? 'linear-gradient(135deg, rgba(168,85,247,0.15), rgba(139,92,246,0.1))'
        : 'rgba(255,255,255,0.05)',
      border: `1px solid ${advancedMode ? 'rgba(168,85,247,0.3)' : 'rgba(255,255,255,0.15)'}`,
      borderRadius: '4px',
      padding: '6px 8px',
      marginBottom: '6px'
    }}>
      <button
        onClick={handleToggle}
        disabled={!canToggle && advancedMode}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          background: 'none',
          border: 'none',
          cursor: canToggle ? 'pointer' : 'not-allowed',
          padding: '4px',
          borderRadius: '3px',
          transition: 'all 0.2s'
        }}
      >
        {/* Icon */}
        <div style={{
          width: '28px',
          height: '28px',
          borderRadius: '4px',
          background: advancedMode
            ? 'linear-gradient(135deg, #a855f7, #8b5cf6)'
            : 'rgba(255,255,255,0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.2s'
        }}>
          <Settings2 size={14} style={{ color: advancedMode ? 'white' : 'rgba(255,255,255,0.6)' }} />
        </div>

        {/* Label */}
        <div style={{ flex: 1, textAlign: 'left' }}>
          <div style={{
            fontSize: '10px',
            fontWeight: '600',
            color: advancedMode ? '#a855f7' : 'rgba(255,255,255,0.8)'
          }}>
            {advancedMode ? 'Advanced Mode' : 'Basic Mode'}
          </div>
          <div style={{
            fontSize: '8px',
            color: 'rgba(255,255,255,0.5)'
          }}>
            {advancedMode ? 'All settings visible' : 'Essential settings only'}
          </div>
        </div>

        {/* Auth indicator */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}>
          {isAuthenticated ? (
            <Unlock size={12} style={{ color: '#22c55e' }} />
          ) : (
            <Lock size={12} style={{ color: 'rgba(255,255,255,0.4)' }} />
          )}
        </div>

        {/* Toggle switch */}
        <div style={{
          width: '36px',
          height: '18px',
          borderRadius: '9px',
          background: advancedMode ? '#a855f7' : 'rgba(255,255,255,0.2)',
          position: 'relative',
          transition: 'all 0.2s',
          flexShrink: 0
        }}>
          <div style={{
            width: '14px',
            height: '14px',
            borderRadius: '50%',
            background: 'white',
            position: 'absolute',
            top: '2px',
            left: advancedMode ? '20px' : '2px',
            transition: 'left 0.2s',
            boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
          }} />
        </div>
      </button>

      {/* Power user badge */}
      {isPowerUser && (
        <div style={{
          marginTop: '6px',
          padding: '3px 6px',
          background: 'rgba(251,191,36,0.1)',
          border: '1px solid rgba(251,191,36,0.3)',
          borderRadius: '3px',
          fontSize: '8px',
          color: '#fbbf24',
          textAlign: 'center',
          fontWeight: '600'
        }}>
          Power User - Full write access enabled
        </div>
      )}

      {/* Auth hint for non-authenticated users */}
      {!isAuthenticated && advancedMode && (
        <div style={{
          marginTop: '6px',
          padding: '3px 6px',
          background: 'rgba(245,158,11,0.1)',
          border: '1px solid rgba(245,158,11,0.3)',
          borderRadius: '3px',
          fontSize: '8px',
          color: '#f59e0b',
          textAlign: 'center'
        }}>
          Connect Nostr for write access
        </div>
      )}
    </div>
  );
};
