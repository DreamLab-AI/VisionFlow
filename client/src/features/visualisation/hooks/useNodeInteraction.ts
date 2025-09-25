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

  // Refs for tracking interaction data
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

  // Interaction timeout ref
  const interactionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Throttled position update function
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

  // Start interaction
  const startInteraction = useCallback((nodeId: string, instanceId: number, pointerPos?: { x: number; y: number }) => {
    const now = Date.now();

    // Clear any existing interaction timeout
    if (interactionTimeoutRef.current) {
      clearTimeout(interactionTimeoutRef.current);
      interactionTimeoutRef.current = null;
    }

    // Update refs
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

    // Update state
    setInteractionState(prev => ({
      ...prev,
      isInteracting: true,
      isClicking: true,
      nodeId,
      instanceId,
      lastInteractionTime: now
    }));

    // Notify parent
    if (onInteractionStart) {
      onInteractionStart(nodeId, instanceId);
    }

    if (debugState.isEnabled()) {
      logger.debug(`Started interaction with node ${nodeId} (instanceId: ${instanceId})`);
    }
  }, [onInteractionStart]);

  // Start drag (upgrade from click to drag)
  const startDrag = useCallback((pointerPos: { x: number; y: number }) => {
    const data = interactionDataRef.current;

    if (!data.isInteracting || data.isDragging) return;

    // Check drag threshold
    const distance = Math.sqrt(
      Math.pow(pointerPos.x - data.startPointerPos.x, 2) +
      Math.pow(pointerPos.y - data.startPointerPos.y, 2)
    );

    if (distance > dragThreshold) {
      // Upgrade to drag
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

  // Update position during interaction
  const updatePosition = useCallback((position: { x: number; y: number; z: number }) => {
    const data = interactionDataRef.current;

    if (!data.isInteracting || !data.nodeId) return;

    const now = Date.now();
    data.lastUpdateTime = now;

    // Use throttled position update
    throttledPositionUpdate(data.nodeId, position);

    // Update interaction time in state
    setInteractionState(prev => ({
      ...prev,
      lastInteractionTime: now
    }));
  }, [throttledPositionUpdate]);

  // End interaction
  const endInteraction = useCallback(() => {
    const data = interactionDataRef.current;
    const previousNodeId = data.nodeId;
    const previousInstanceId = data.instanceId;

    // Reset refs
    interactionDataRef.current = {
      ...interactionDataRef.current,
      isInteracting: false,
      isDragging: false,
      isClicking: false,
      nodeId: null,
      instanceId: null
    };

    // Update state
    setInteractionState(prev => ({
      ...prev,
      isInteracting: false,
      isDragging: false,
      isClicking: false,
      nodeId: null,
      instanceId: null
    }));

    // Set a brief timeout before completely ending interaction
    // This helps with rapid click sequences
    interactionTimeoutRef.current = setTimeout(() => {
      interactionTimeoutRef.current = null;

      // Notify parent
      if (onInteractionEnd) {
        onInteractionEnd(previousNodeId, previousInstanceId);
      }

      if (debugState.isEnabled()) {
        logger.debug(`Ended interaction with node ${previousNodeId}`);
      }
    }, 50);
  }, [onInteractionEnd]);

  // Check if actively interacting (used for sending position updates)
  const isActivelyInteracting = useCallback(() => {
    const data = interactionDataRef.current;
    return data.isInteracting && (data.isDragging || data.isClicking);
  }, []);

  // Check if should send position updates
  const shouldSendUpdates = useCallback(() => {
    const data = interactionDataRef.current;
    const now = Date.now();

    // Only send if actively interacting and not too frequent
    return data.isInteracting &&
           data.isDragging && // Only during drag, not just click
           (now - data.lastUpdateTime) >= throttleMs;
  }, [throttleMs]);

  // Cleanup on unmount
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