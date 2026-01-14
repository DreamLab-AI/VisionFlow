/**
 * useAgentActionVisualization Hook
 *
 * Integrates WebSocket agent action events with the visualization layer.
 * Automatically subscribes to 'agent-action' events and feeds them to
 * the ActionConnections system.
 *
 * Phase 2b: Updated to work with enhanced useActionConnections hook.
 */

import { useEffect, useCallback, useMemo } from 'react';
import * as THREE from 'three';
import { useWebSocketStore } from '@/store/websocketStore';
import { useActionConnections, UseActionConnectionsOptions, ActionConnection } from './useActionConnections';
import { AgentActionEvent } from '@/services/BinaryWebSocketProtocol';
import { graphDataManager } from '@/features/graph/managers/graphDataManager';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('useAgentActionVisualization');

export interface UseAgentActionVisualizationOptions extends Omit<UseActionConnectionsOptions, 'getNodePosition' | 'autoSubscribe'> {
  /** Enable the visualization (default: true) */
  enabled?: boolean;
  /** Log debug info */
  debug?: boolean;
  /** VR mode - applies performance optimizations */
  vrMode?: boolean;
}

export interface UseAgentActionVisualizationReturn {
  connections: ActionConnection[];
  addAction: (event: AgentActionEvent) => void;
  addActions: (events: AgentActionEvent[]) => void;
  clearAll: () => void;
  updatePositions: () => void;
  updateConnections: () => void;
  activeCount: number;
  enabled: boolean;
  vrMode: boolean;
}

export const useAgentActionVisualization = (
  options: UseAgentActionVisualizationOptions = {}
): UseAgentActionVisualizationReturn => {
  const { enabled = true, debug = false, vrMode = false, ...connectionOptions } = options;

  // Apply VR-specific optimizations
  const vrOptimizedOptions = useMemo(() => {
    if (!vrMode) return connectionOptions;

    return {
      ...connectionOptions,
      // Reduce max connections for Quest 3 @ 72fps
      maxConnections: Math.min(connectionOptions.maxConnections ?? 50, 20),
      // Faster animation for responsiveness
      baseDuration: Math.min(connectionOptions.baseDuration ?? 500, 400),
    };
  }, [vrMode, connectionOptions]);

  // Position resolver using graphDataManager - returns THREE.Vector3 compatible object
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

  // Initialize action connections with position resolver (legacy interface)
  const actionConnections = useActionConnections({
    ...vrOptimizedOptions,
    getNodePosition,
    autoSubscribe: false, // We handle subscription here for fine-grained control
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
  }, [enabled, wsOn, actionConnections.addActions, debug]);

  // Periodically update positions (when nodes move)
  useEffect(() => {
    if (!enabled || actionConnections.activeCount === 0) return;

    const interval = setInterval(() => {
      actionConnections.updatePositions();
    }, 100); // Update every 100ms while connections are active

    return () => clearInterval(interval);
  }, [enabled, actionConnections.activeCount, actionConnections.updatePositions]);

  return {
    connections: actionConnections.connections,
    addAction: actionConnections.addAction,
    addActions: actionConnections.addActions,
    clearAll: actionConnections.clearAll,
    updatePositions: actionConnections.updatePositions,
    updateConnections: actionConnections.updateConnections,
    activeCount: actionConnections.activeCount,
    enabled,
    vrMode,
  };
};

export default useAgentActionVisualization;
