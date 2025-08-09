import { useEffect, useRef, useCallback } from 'react';
import { webSocketService } from '../../../services/WebSocketService';
import { createLogger } from '../../../utils/logger';
import { parseBinaryNodeData, isAgentNode, getActualNodeId, BinaryNodeData } from '../../../types/binaryProtocol';
import * as THREE from 'three';

const logger = createLogger('useBotsBinaryUpdates');

interface BotsBinaryUpdateOptions {
  enabled: boolean;
  onPositionUpdate?: (agentNodes: BinaryNodeData[]) => void;
}

export function useBotsBinaryUpdates({
  enabled,
  onPositionUpdate
}: BotsBinaryUpdateOptions) {
  const agentNodesRef = useRef<BinaryNodeData[]>([]);
  const nodeIdMapRef = useRef<Map<number, number>>(new Map());

  // Handle binary updates from WebSocket
  const handleBinaryData = useCallback((data: ArrayBuffer) => {
    try {
      // Parse binary data using the standard protocol
      const allNodes = parseBinaryNodeData(data);
      
      // Filter for agent nodes only
      const agentNodes = allNodes.filter(node => isAgentNode(node.nodeId));
      
      if (agentNodes.length === 0) {
        return; // No agent nodes in this update
      }

      logger.debug(`Processing bots binary update: ${agentNodes.length} agent nodes`);

      // Update our refs
      agentNodesRef.current = agentNodes;
      
      // Update node ID mapping
      nodeIdMapRef.current.clear();
      agentNodes.forEach((node, index) => {
        const actualNodeId = getActualNodeId(node.nodeId);
        nodeIdMapRef.current.set(actualNodeId, index);
      });

      // Notify callback with the parsed agent nodes
      if (onPositionUpdate) {
        onPositionUpdate(agentNodes);
      }

      logger.debug(`Updated ${agentNodes.length} agent node positions from binary protocol`);
    } catch (error) {
      logger.error('Error processing bots binary data:', error);
    }
  }, [onPositionUpdate]);

  // Request bots positions periodically
  const requestBotsPositions = useCallback(() => {
    if (webSocketService.isReady()) {
      webSocketService.sendMessage('requestBotsPositions');
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    // Subscribe to bots-position-update events from WebSocketService
    const unsubscribe = webSocketService.on('bots-position-update', handleBinaryData);

    // Request initial bots positions
    requestBotsPositions();

    // Request updates periodically
    const interval = setInterval(requestBotsPositions, 1000); // Every second

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [enabled, handleBinaryData, requestBotsPositions]);

  return {
    agentNodes: agentNodesRef.current,
    nodeIdMap: nodeIdMapRef.current,
    requestUpdate: requestBotsPositions
  };
}