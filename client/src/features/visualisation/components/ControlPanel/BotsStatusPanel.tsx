/**
 * Bots Status Panel Component
 */

import React, { useState } from 'react';
import { Zap } from 'lucide-react';
import { MultiAgentInitializationPrompt } from '../../../bots/components';
import { AgentTelemetryStream } from '../../../bots/components/AgentTelemetryStream';
import { unifiedApiClient } from '../../../../services/api/UnifiedApiClient';
import { botsWebSocketIntegration } from '../../../bots/services/BotsWebSocketIntegration';
import { useBotsData } from '../../../bots/contexts/BotsDataContext';
import type { BotsData } from './types';

interface BotsStatusPanelProps {
  botsData?: BotsData;
}

export const BotsStatusPanel: React.FC<BotsStatusPanelProps> = ({ botsData }) => {
  const [showMultiAgentPrompt, setShowMultiAgentPrompt] = useState(false);
  const { updateBotsData } = useBotsData();

  if (!botsData) return null;

  const handleDisconnect = async () => {
    try {
      const response = await unifiedApiClient.post('/bots/disconnect-multi-agent');
      if (response.status >= 200 && response.status < 300) {
        botsWebSocketIntegration.clearAgents();
        updateBotsData({
          nodeCount: 0,
          edgeCount: 0,
          tokenCount: 0,
          mcpConnected: false,
          dataSource: 'disconnected',
          agents: [],
          edges: []
        });
      }
    } catch (error) {
      // Silently fail - connection may already be closed
    }
  };

  return (
    <>
      <div style={{
        marginBottom: '6px',
        paddingBottom: '6px',
        borderBottom: '1px solid rgba(255,255,255,0.15)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          marginBottom: '6px',
          color: '#fbbf24',
          fontWeight: '600',
          fontSize: '10px'
        }}>
          <Zap size={12} />
          VisionFlow ({botsData.dataSource.toUpperCase()})
        </div>

        {botsData.nodeCount === 0 ? (
          <div style={{ textAlign: 'center', padding: '6px 0' }}>
            <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>
              No active multi-agent
            </div>
            <button
              onClick={() => setShowMultiAgentPrompt(true)}
              style={{
                background: 'linear-gradient(to right, #fbbf24, #f59e0b)',
                color: 'black',
                padding: '4px 10px',
                borderRadius: '3px',
                fontSize: '10px',
                fontWeight: '600',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              Initialize multi-agent
            </button>
          </div>
        ) : (
          <>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '4px',
              fontSize: '10px',
              marginBottom: '6px'
            }}>
              <div style={{
                textAlign: 'center',
                padding: '4px',
                background: 'rgba(255,255,255,0.05)',
                borderRadius: '3px'
              }}>
                <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '9px' }}>Agents</div>
                <div style={{ color: '#fbbf24', fontWeight: '600' }}>{botsData.nodeCount}</div>
              </div>
              <div style={{
                textAlign: 'center',
                padding: '4px',
                background: 'rgba(255,255,255,0.05)',
                borderRadius: '3px'
              }}>
                <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '9px' }}>Links</div>
                <div style={{ color: '#fbbf24', fontWeight: '600' }}>{botsData.edgeCount}</div>
              </div>
              <div style={{
                textAlign: 'center',
                padding: '4px',
                background: 'rgba(255,255,255,0.05)',
                borderRadius: '3px'
              }}>
                <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '9px' }}>Tokens</div>
                <div style={{ color: '#f59e0b', fontWeight: '600', fontSize: '9px' }}>
                  {botsData.tokenCount.toLocaleString()}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '4px' }}>
              <button
                onClick={() => setShowMultiAgentPrompt(true)}
                style={{
                  flex: 1,
                  background: 'linear-gradient(to right, #22c55e, #16a34a)',
                  color: 'white',
                  padding: '4px 8px',
                  borderRadius: '3px',
                  fontSize: '10px',
                  fontWeight: '600',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                New Task
              </button>
              <button
                onClick={handleDisconnect}
                style={{
                  flex: 1,
                  background: 'linear-gradient(to right, #ef4444, #dc2626)',
                  color: 'white',
                  padding: '4px 8px',
                  borderRadius: '3px',
                  fontSize: '10px',
                  fontWeight: '600',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                Disconnect
              </button>
            </div>
          </>
        )}
      </div>

      {/* Telemetry Stream - Always visible when agents are active */}
      {botsData.nodeCount > 0 && (
        <AgentTelemetryStream />
      )}

      {showMultiAgentPrompt && (
        <MultiAgentInitializationPrompt
          onClose={() => setShowMultiAgentPrompt(false)}
          onInitialized={() => setShowMultiAgentPrompt(false)}
        />
      )}
    </>
  );
};
