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
  isUserInteracting: boolean; 
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

  
  const activeInteractionsRef = useRef(new Set<string>());
  const interactionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastPositionSentRef = useRef<Map<string, number>>(new Map());
  const lastInteractionTimeRef = useRef(Date.now());

  
  const throttledSendPositions = useRef(
    throttle(async () => {
      
      if (activeInteractionsRef.current.size === 0) {
        return;
      }

      try {
        
        const graphData = await graphDataManager.getGraphData();

        
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


        if (updates.length > 0 && graphDataManager.webSocketService) {
          (graphDataManager.webSocketService as unknown as { sendNodePositionUpdates?: (updates: unknown[]) => void }).sendNodePositionUpdates?.(updates);

          if (debugState.isEnabled()) {
            logger.debug(`Sent position updates for ${updates.length} nodes during interaction`);
          }
        }
      } catch (error) {
        logger.error('Error sending position updates during interaction:', error);
      }
    }, positionUpdateThrottleMs)
  ).current;

  
  const startInteraction = useCallback((nodeId: string) => {
    activeInteractionsRef.current.add(nodeId);

    const now = Date.now();
    const newInteractionCount = activeInteractionsRef.current.size;
    const wasInteracting = interactionState.isUserInteracting;

    lastInteractionTimeRef.current = now;
    setInteractionState(prev => ({
      ...prev,
      hasActiveInteractions: true,
      interactionCount: newInteractionCount,
      lastInteractionTime: now,
      isUserInteracting: true
    }));

    
    if (interactionTimeoutRef.current) {
      clearTimeout(interactionTimeoutRef.current);
      interactionTimeoutRef.current = null;
    }

    
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

    
    if (!hasInteractions) {
      interactionTimeoutRef.current = setTimeout(() => {
        
        if (activeInteractionsRef.current.size === 0) {
          setInteractionState(prev => ({
            ...prev,
            isUserInteracting: false
          }));

          
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

  
  const updateNodePosition = useCallback((nodeId: string, position: { x: number; y: number; z: number }) => {
    
    if (!activeInteractionsRef.current.has(nodeId)) {
      return;
    }

    
    lastInteractionTimeRef.current = Date.now();


    throttledSendPositions();
  }, [throttledSendPositions]);

  
  const shouldSendPositionUpdates = useCallback(() => {
    return activeInteractionsRef.current.size > 0;
  }, []);

  
  const flushPositionUpdates = useCallback(async () => {
    if (activeInteractionsRef.current.size > 0) {
      throttledSendPositions.flush();


      if (graphDataManager.webSocketService) {
        await (graphDataManager.webSocketService as unknown as { flushPositionUpdates?: () => Promise<void> }).flushPositionUpdates?.();
      }
    }
  }, [throttledSendPositions]);

  
  const getActiveNodes = useCallback(() => {
    return Array.from(activeInteractionsRef.current);
  }, []);

  
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
    lastInteractionTimeRef,
    startInteraction,
    endInteraction,
    updateNodePosition,
    shouldSendPositionUpdates,
    flushPositionUpdates,
    getActiveNodes
  };
};

export default useGraphInteraction;