import { useCallback, useRef, useState, useEffect } from 'react';
import { throttle } from 'lodash';
import { createLogger } from '../../../utils/loggerConfig';
import { debugState } from '../../../utils/clientDebugState';

const logger = createLogger('useNodeInteraction');

export interface NodeInteractionState {
  isInteracting: boolean;
  isDragging: boolean;
  isClicking: boolean;
  nodeId: string | null;
  instanceId: number | null;
  lastInteractionTime: number;
}

export interface UseNodeInteractionOptions {
  throttleMs?: number;
  onInteractionStart?: (nodeId: string, instanceId: number) => void;
  onInteractionEnd?: (nodeId: string | null, instanceId: number | null) => void;
  onPositionUpdate?: (nodeId: string, position: { x: number; y: number; z: number }) => void;
  dragThreshold?: number;
}

export const useNodeInteraction = (options: UseNodeInteractionOptions = {}) => {
  const {
    throttleMs = 100,
    onInteractionStart,
    onInteractionEnd,
    onPositionUpdate,
    dragThreshold = 5
  } = options;

  const [interactionState, setInteractionState] = useState<NodeInteractionState>({
    isInteracting: false,
    isDragging: false,
    isClicking: false,
    nodeId: null,
    instanceId: null,
    lastInteractionTime: 0
  });

  
  const interactionDataRef = useRef({
    isInteracting: false,
    isDragging: false,
    isClicking: false,
    nodeId: null as string | null,
    instanceId: null as number | null,
    startPointerPos: { x: 0, y: 0 },
    startTime: 0,
    lastUpdateTime: 0
  });

  
  const interactionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  
  const throttledPositionUpdate = useRef(
    throttle((nodeId: string, position: { x: number; y: number; z: number }) => {
      if (interactionDataRef.current.isInteracting && onPositionUpdate) {
        onPositionUpdate(nodeId, position);
        if (debugState.isEnabled()) {
          logger.debug(`Throttled position update for node ${nodeId}`, position);
        }
      }
    }, throttleMs)
  ).current;

  
  const startInteraction = useCallback((nodeId: string, instanceId: number, pointerPos?: { x: number; y: number }) => {
    const now = Date.now();

    
    if (interactionTimeoutRef.current) {
      clearTimeout(interactionTimeoutRef.current);
      interactionTimeoutRef.current = null;
    }

    
    interactionDataRef.current = {
      ...interactionDataRef.current,
      isInteracting: true,
      isClicking: true,
      nodeId,
      instanceId,
      startPointerPos: pointerPos || { x: 0, y: 0 },
      startTime: now,
      lastUpdateTime: now
    };

    
    setInteractionState(prev => ({
      ...prev,
      isInteracting: true,
      isClicking: true,
      nodeId,
      instanceId,
      lastInteractionTime: now
    }));

    
    if (onInteractionStart) {
      onInteractionStart(nodeId, instanceId);
    }

    if (debugState.isEnabled()) {
      logger.debug(`Started interaction with node ${nodeId} (instanceId: ${instanceId})`);
    }
  }, [onInteractionStart]);

  
  const startDrag = useCallback((pointerPos: { x: number; y: number }) => {
    const data = interactionDataRef.current;

    if (!data.isInteracting || data.isDragging) return;

    
    const distance = Math.sqrt(
      Math.pow(pointerPos.x - data.startPointerPos.x, 2) +
      Math.pow(pointerPos.y - data.startPointerPos.y, 2)
    );

    if (distance > dragThreshold) {
      
      interactionDataRef.current.isDragging = true;
      interactionDataRef.current.isClicking = false;

      setInteractionState(prev => ({
        ...prev,
        isDragging: true,
        isClicking: false
      }));

      if (debugState.isEnabled()) {
        logger.debug(`Started dragging node ${data.nodeId} (distance: ${distance.toFixed(2)})`);
      }
    }
  }, [dragThreshold]);

  
  const updatePosition = useCallback((position: { x: number; y: number; z: number }) => {
    const data = interactionDataRef.current;

    if (!data.isInteracting || !data.nodeId) return;

    const now = Date.now();
    data.lastUpdateTime = now;

    
    throttledPositionUpdate(data.nodeId, position);

    
    setInteractionState(prev => ({
      ...prev,
      lastInteractionTime: now
    }));
  }, [throttledPositionUpdate]);

  
  const endInteraction = useCallback(() => {
    const data = interactionDataRef.current;
    const previousNodeId = data.nodeId;
    const previousInstanceId = data.instanceId;

    
    interactionDataRef.current = {
      ...interactionDataRef.current,
      isInteracting: false,
      isDragging: false,
      isClicking: false,
      nodeId: null,
      instanceId: null
    };

    
    setInteractionState(prev => ({
      ...prev,
      isInteracting: false,
      isDragging: false,
      isClicking: false,
      nodeId: null,
      instanceId: null
    }));

    
    
    interactionTimeoutRef.current = setTimeout(() => {
      interactionTimeoutRef.current = null;

      
      if (onInteractionEnd) {
        onInteractionEnd(previousNodeId, previousInstanceId);
      }

      if (debugState.isEnabled()) {
        logger.debug(`Ended interaction with node ${previousNodeId}`);
      }
    }, 50);
  }, [onInteractionEnd]);

  
  const isActivelyInteracting = useCallback(() => {
    const data = interactionDataRef.current;
    return data.isInteracting && (data.isDragging || data.isClicking);
  }, []);

  
  const shouldSendUpdates = useCallback(() => {
    const data = interactionDataRef.current;
    const now = Date.now();

    
    return data.isInteracting &&
           data.isDragging && 
           (now - data.lastUpdateTime) >= throttleMs;
  }, [throttleMs]);

  
  useEffect(() => {
    return () => {
      if (interactionTimeoutRef.current) {
        clearTimeout(interactionTimeoutRef.current);
      }
      throttledPositionUpdate.cancel();
    };
  }, [throttledPositionUpdate]);

  return {
    interactionState,
    startInteraction,
    startDrag,
    updatePosition,
    endInteraction,
    isActivelyInteracting,
    shouldSendUpdates
  };
};

export default useNodeInteraction;