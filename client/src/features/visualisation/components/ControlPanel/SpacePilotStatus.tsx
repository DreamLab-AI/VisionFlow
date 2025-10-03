/**
 * SpacePilot Status Component
 */

import React from 'react';
import { Puzzle } from 'lucide-react';

interface SpacePilotStatusProps {
  webHidAvailable: boolean;
  spacePilotConnected: boolean;
  spacePilotButtons: string[];
  onConnect: () => void;
}

export const SpacePilotStatus: React.FC<SpacePilotStatusProps> = ({
  webHidAvailable,
  spacePilotConnected,
  onConnect
}) => {
  return (
    <div style={{
      marginBottom: '6px',
      paddingBottom: '6px',
      borderBottom: '1px solid rgba(255,255,255,0.15)'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        marginBottom: '4px',
        fontSize: '10px',
        fontWeight: '600'
      }}>
        <Puzzle size={12} />
        SpacePilot
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px' }}>
        {!webHidAvailable ? (
          <span style={{ color: '#f87171', fontSize: '9px' }}>WebHID not available</span>
        ) : spacePilotConnected ? (
          <>
            <div style={{
              width: '5px',
              height: '5px',
              borderRadius: '50%',
              background: '#22c55e',
              boxShadow: '0 0 4px rgba(34,197,94,0.6)'
            }} />
            <span style={{ color: '#22c55e', fontWeight: '500', fontSize: '10px' }}>Connected</span>
          </>
        ) : (
          <>
            <div style={{
              width: '5px',
              height: '5px',
              borderRadius: '50%',
              background: '#f87171',
              boxShadow: '0 0 4px rgba(248,113,113,0.6)'
            }} />
            <button
              onClick={onConnect}
              style={{
                background: 'linear-gradient(to right, #3b82f6, #2563eb)',
                color: 'white',
                padding: '3px 8px',
                borderRadius: '3px',
                fontSize: '9px',
                fontWeight: '500',
                border: 'none',
                cursor: 'pointer',
                transition: 'transform 0.2s'
              }}
            >
              Connect
            </button>
          </>
        )}
      </div>
    </div>
  );
};
