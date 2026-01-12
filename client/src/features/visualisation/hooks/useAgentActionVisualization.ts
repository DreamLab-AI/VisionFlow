/**
 * useAgentActionVisualization Hook
 *
 * Integrates WebSocket agent action events with the visualization layer.
 * Automatically subscribes to 'agent-action' events and feeds them to
 * the ActionConnections system.
 */

import { useEffect, useCallback, useMemo } from 'react';
import { useWebSocketStore } from '@/store/websocketStore';
import { useActionConnections, UseActionConnectionsOptions } from './useActionConnections';
import { AgentActionEvent } from '@/services/BinaryWebSocketProtocol';
import { graphDataManager } from '@/features/graph/managers/graphDataManager';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('useAgentActionVisualization');

export interface UseAgentActionVisualizationOptions extends Omit<UseActionConnectionsOptions, 'getNodePosition'> {
  /** Enable the visualization (default: true) */
  enabled?: boolean;
  /** Log debug info */
  debug?: boolean;
}

export const useAgentActionVisualization = (options: UseAgentActionVisualizationOptions = {}) => {
  const { enabled = true, debug = false, ...connectionOptions } = options;

  // Position resolver using graphDataManager
  const getNodePosition = useCallback((nodeId: number): { x: number; y: number; z: number } | null => {
    try {
      // Look up string ID from numeric ID
      const stringId = graphDataManager.reverseNodeIds.get(nodeId);
      if (!stringId) {
        if (debug) logger.debug(`No string ID for numeric ID ${nodeId}`);
        return null;
      }

      // Get node from graph data synchronously if cached
      const cachedData = graphDataManager.getCachedGraphData();
      if (cachedData) {
        const node = cachedData.nodes.find(n => n.id === stringId);
        if (node?.position) {
          return node.position;
        }
      }

      return null;
    } catch (error) {
      logger.error('Error getting node position:', error);
      return null;
    }
  }, [debug]);

  // Initialize action connections with position resolver
  const actionConnections = useActionConnections({
    ...connectionOptions,
    getNodePosition,
  });

  // Subscribe to WebSocket events
  const wsOn = useWebSocketStore(state => state.on);

  useEffect(() => {
    if (!enabled) return;

    // Subscribe to agent-action events
    const unsubscribe = wsOn('agent-action', (data: unknown) => {
      const actions = data as AgentActionEvent[];
      if (Array.isArray(actions) && actions.length > 0) {
        actionConnections.addActions(actions);

        if (debug) {
          logger.debug(`Received ${actions.length} agent actions`);
        }
      }
    });

    return unsubscribe;
  }, [enabled, wsOn, actionConnections, debug]);

  // Periodically update positions (when nodes move)
  useEffect(() => {
    if (!enabled || actionConnections.activeCount === 0) return;

    const interval = setInterval(() => {
      actionConnections.updatePositions();
    }, 100); // Update every 100ms while connections are active

    return () => clearInterval(interval);
  }, [enabled, actionConnections]);

  return {
    ...actionConnections,
    enabled,
  };
};

export default useAgentActionVisualization;
