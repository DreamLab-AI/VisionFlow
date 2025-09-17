import { useEffect, useState } from 'react';
import { botsWebSocketIntegration } from '../services/BotsWebSocketIntegration';
import { createLogger } from '../../../utils/logger';
import { agentTelemetry } from '../../../telemetry/AgentTelemetry';
import { useTelemetry } from '../../../telemetry/useTelemetry';

const logger = createLogger('useBotsWebSocketIntegration');

/**
 * Hook to manage BotsWebSocket integration lifecycle
 * Ensures the integration service is properly initialized and cleaned up
 */
export function useBotsWebSocketIntegration() {
  const telemetry = useTelemetry('useBotsWebSocketIntegration');
  const [connectionStatus, setConnectionStatus] = useState({
    mcp: false,
    logseq: false,
    overall: false
  });

  useEffect(() => {
    logger.info('Initializing bots WebSocket integration');

    // Listen for connection status changes
    const unsubMcp = botsWebSocketIntegration.on('mcp-connected', ({ connected }) => {
      setConnectionStatus(prev => ({ ...prev, mcp: connected }));

      // Log MCP connection changes
      agentTelemetry.logAgentAction('websocket', 'mcp', connected ? 'connected' : 'disconnected');
    });

    const unsubLogseq = botsWebSocketIntegration.on('logseq-connected', ({ connected }) => {
      setConnectionStatus(prev => ({ ...prev, logseq: connected }));

      // Log Logseq connection changes
      agentTelemetry.logAgentAction('websocket', 'logseq', connected ? 'connected' : 'disconnected');
    });

    // Update overall status
    const updateOverall = setInterval(() => {
      const status = botsWebSocketIntegration.getConnectionStatus();
      setConnectionStatus({
        mcp: status.mcp,
        logseq: status.logseq,
        overall: status.overall
      });
    }, 2000);

    // Request initial data once connected
    const checkAndRequestData = setInterval(() => {
      const status = botsWebSocketIntegration.getConnectionStatus();
      if (status.overall) {
        agentTelemetry.logAgentAction('websocket', 'hook', 'requesting_initial_data');
        botsWebSocketIntegration.requestInitialData()
          .then(() => {
            logger.info('Initial data requested');
            agentTelemetry.logAgentAction('websocket', 'hook', 'initial_data_success');
          })
          .catch(err => {
            logger.error('Failed to request initial data:', err);
            agentTelemetry.logAgentAction('websocket', 'hook', 'initial_data_failed', { error: err.message });
          });
        clearInterval(checkAndRequestData);
      }
    }, 1000);

    return () => {
      unsubMcp();
      unsubLogseq();
      clearInterval(updateOverall);
      clearInterval(checkAndRequestData);

      // Log hook cleanup
      agentTelemetry.logAgentAction('websocket', 'hook', 'cleanup');

      // Note: We don't disconnect here as the service is a singleton
      // and might be used by other components
    };
  }, []);

  return connectionStatus;
}