/**
 * useControlCenterStatus Hook
 *
 * Provides system status information for the unified Control Center component.
 * Integrates with existing hooks to provide WebSocket, MCP, and metadata status.
 */

import { useMemo } from 'react';
import { useHybridSystemStatus } from './useHybridSystemStatus';

export interface ControlCenterStatus {
  websocketStatus: 'connected' | 'connecting' | 'disconnected';
  mcpConnected: boolean;
  metadataStatus: 'loaded' | 'loading' | 'error' | 'none';
  nodeCount: number;
  edgeCount: number;
  isFullyConnected: boolean;
  systemHealth: 'healthy' | 'degraded' | 'critical' | 'unknown';
}

export interface UseControlCenterStatusOptions {
  graphData?: { nodes: any[]; edges: any[] };
  pollingInterval?: number;
}

export function useControlCenterStatus(options: UseControlCenterStatusOptions = {}): ControlCenterStatus {
  const { graphData } = options;

  // Use the hybrid system status hook for connection info
  const {
    isConnected,
    isMcpAvailable,
    status: systemStatus,
  } = useHybridSystemStatus({
    pollingInterval: options.pollingInterval || 30000,
    enableWebSocket: true,
    enableHealthChecks: true,
  });

  // Derive websocket status
  const websocketStatus = useMemo(() => {
    if (isConnected) return 'connected';
    if (systemStatus.dockerHealth === 'degraded') return 'connecting';
    return 'disconnected';
  }, [isConnected, systemStatus.dockerHealth]);

  // Derive metadata status from graph data presence
  const metadataStatus = useMemo(() => {
    if (!graphData) return 'none';
    if (graphData.nodes && graphData.nodes.length > 0) return 'loaded';
    return 'loading';
  }, [graphData]);

  // Get node/edge counts
  const nodeCount = graphData?.nodes?.length || 0;
  const edgeCount = graphData?.edges?.length || 0;

  // Determine if fully connected
  const isFullyConnected = useMemo(() => {
    return websocketStatus === 'connected' &&
           metadataStatus === 'loaded' &&
           nodeCount > 0;
  }, [websocketStatus, metadataStatus, nodeCount]);

  // Map system health
  const systemHealth = useMemo(() => {
    return systemStatus.systemStatus || 'unknown';
  }, [systemStatus.systemStatus]);

  return {
    websocketStatus,
    mcpConnected: isMcpAvailable,
    metadataStatus,
    nodeCount,
    edgeCount,
    isFullyConnected,
    systemHealth,
  };
}

export default useControlCenterStatus;
