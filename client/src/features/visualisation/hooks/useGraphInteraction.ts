import { useCallback, useRef, useState, useEffect } from 'react';
import { throttle } from 'lodash';
import { createLogger } from '../../../utils/loggerConfig';
import { debugState } from '../../../utils/clientDebugState';
import { graphDataManager } from '../../graph/managers/graphDataManager';
import { BinaryNodeData } from '../../../types/binaryProtocol';

const logger = createLogger('useGraphInteraction');

export interface GraphInteractionState {
  hasActiveInteractions: boolean;
  interactionCount: number;
  lastInteractionTime: number;
  isUserInteracting: boolean; // Master flag for any user interaction
}

export interface UseGraphInteractionOptions {
  positionUpdateThrottleMs?: number;
  interactionTimeoutMs?: number;
  onInteractionStateChange?: (isInteracting: boolean) => void;
}

export const useGraphInteraction = (options: UseGraphInteractionOptions = {}) => {
  const {
    positionUpdateThrottleMs = 100,
    interactionTimeoutMs = 500,
    onInteractionStateChange
  } = options;

  const [interactionState, setInteractionState] = useState<GraphInteractionState>({
    hasActiveInteractions: false,
    interactionCount: 0,
    lastInteractionTime: 0,
    isUserInteracting: false
  });

  // Refs for tracking active interactions
  const activeInteractionsRef = useRef(new Set<string>());
  const interactionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastPositionSentRef = useRef<Map<string, number>>(new Map());

  // Throttled function for sending position updates
  const throttledSendPositions = useRef(
    throttle(async () => {
      // Only send if there are active interactions
      if (activeInteractionsRef.current.size === 0) {
        return;
      }

      try {
        // Get current graph data
        const graphData = await graphDataManager.getGraphData();

        // Create position updates for nodes that have been interacted with recently
        const updates: Array<{
          nodeId: number;
          position: { x: number; y: number; z: number };
          velocity?: { x: number; y: number; z: number };
        }> = [];

        const now = Date.now();

        for (const nodeId of activeInteractionsRef.current) {
          const node = graphData.nodes.find(n => n.id === nodeId);
          if (node && node.position) {
            const numericId = graphDataManager.nodeIdMap.get(nodeId);
            if (numericId !== undefined) {
              const lastSent = lastPositionSentRef.current.get(nodeId) || 0;

              // Only send if enough time has passed since last update for this node
              if (now - lastSent >= positionUpdateThrottleMs) {
                updates.push({
                  nodeId: numericId,
                  position: {
                    x: node.position.x,
                    y: node.position.y,
                    z: node.position.z
                  },
                  velocity: node.metadata?.velocity as { x: number; y: number; z: number } || { x: 0, y: 0, z: 0 }
                });

                lastPositionSentRef.current.set(nodeId, now);
              }
            }
          }
        }

        // Send updates if we have any
        if (updates.length > 0 && graphDataManager.webSocketService) {
          graphDataManager.webSocketService.sendNodePositionUpdates(updates);

          if (debugState.isEnabled()) {
            logger.debug(`Sent position updates for ${updates.length} nodes during interaction`);
          }
        }
      } catch (error) {
        logger.error('Error sending position updates during interaction:', error);
      }
    }, positionUpdateThrottleMs)
  ).current;

  // Start interaction for a specific node
  const startInteraction = useCallback((nodeId: string) => {
    activeInteractionsRef.current.add(nodeId);

    const now = Date.now();
    const newInteractionCount = activeInteractionsRef.current.size;
    const wasInteracting = interactionState.isUserInteracting;

    setInteractionState(prev => ({
      ...prev,
      hasActiveInteractions: true,
      interactionCount: newInteractionCount,
      lastInteractionTime: now,
      isUserInteracting: true
    }));

    // Clear any existing timeout
    if (interactionTimeoutRef.current) {
      clearTimeout(interactionTimeoutRef.current);
      interactionTimeoutRef.current = null;
    }

    // Notify GraphDataManager if this is the first interaction
    if (!wasInteracting) {
      graphDataManager.setUserInteracting(true);

      if (onInteractionStateChange) {
        onInteractionStateChange(true);
      }
    }

    if (debugState.isEnabled()) {
      logger.debug(`Started interaction for node ${nodeId}. Active interactions: ${newInteractionCount}`);
    }
  }, [interactionState.isUserInteracting, onInteractionStateChange]);

  // End interaction for a specific node
  const endInteraction = useCallback((nodeId: string | null) => {
    if (!nodeId) return;

    activeInteractionsRef.current.delete(nodeId);
    lastPositionSentRef.current.delete(nodeId);

    const newInteractionCount = activeInteractionsRef.current.size;
    const hasInteractions = newInteractionCount > 0;

    setInteractionState(prev => ({
      ...prev,
      hasActiveInteractions: hasInteractions,
      interactionCount: newInteractionCount,
      isUserInteracting: hasInteractions
    }));

    // If no more active interactions, set a timeout before fully ending
    if (!hasInteractions) {
      interactionTimeoutRef.current = setTimeout(() => {
        // Double-check that there are still no interactions
        if (activeInteractionsRef.current.size === 0) {
          setInteractionState(prev => ({
            ...prev,
            isUserInteracting: false
          }));

          // Notify GraphDataManager that interactions have ended
          graphDataManager.setUserInteracting(false);

          if (onInteractionStateChange) {
            onInteractionStateChange(false);
          }

          if (debugState.isEnabled()) {
            logger.debug('All interactions ended');
          }
        }
        interactionTimeoutRef.current = null;
      }, interactionTimeoutMs);
    }

    if (debugState.isEnabled()) {
      logger.debug(`Ended interaction for node ${nodeId}. Active interactions: ${newInteractionCount}`);
    }
  }, [interactionTimeoutMs, onInteractionStateChange]);

  // Update position during interaction
  const updateNodePosition = useCallback((nodeId: string, position: { x: number; y: number; z: number }) => {
    // Only update if this node is actively being interacted with
    if (!activeInteractionsRef.current.has(nodeId)) {
      return;
    }

    // Update the last interaction time
    setInteractionState(prev => ({
      ...prev,
      lastInteractionTime: Date.now()
    }));

    // Trigger throttled position sending
    throttledSendPositions();
  }, [throttledSendPositions]);

  // Check if should send position updates to WebSocket
  const shouldSendPositionUpdates = useCallback(() => {
    return activeInteractionsRef.current.size > 0;
  }, []);

  // Force send positions immediately (useful for drag end)
  const flushPositionUpdates = useCallback(async () => {
    if (activeInteractionsRef.current.size > 0) {
      throttledSendPositions.flush();

      // Also flush the WebSocket queue if available
      if (graphDataManager.webSocketService) {
        await graphDataManager.webSocketService.flushPositionUpdates();
      }
    }
  }, [throttledSendPositions]);

  // Get list of actively interacting nodes
  const getActiveNodes = useCallback(() => {
    return Array.from(activeInteractionsRef.current);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (interactionTimeoutRef.current) {
        clearTimeout(interactionTimeoutRef.current);
      }
      throttledSendPositions.cancel();
      activeInteractionsRef.current.clear();
      lastPositionSentRef.current.clear();
    };
  }, [throttledSendPositions]);

  return {
    interactionState,
    startInteraction,
    endInteraction,
    updateNodePosition,
    shouldSendPositionUpdates,
    flushPositionUpdates,
    getActiveNodes
  };
};

export default useGraphInteraction;