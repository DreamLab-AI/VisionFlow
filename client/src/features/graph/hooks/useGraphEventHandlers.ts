import { useCallback, useRef, useEffect } from 'react';
import { ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import { throttle } from 'lodash';
import { graphDataManager, type GraphData, type Node } from '../managers/graphDataManager';
import { graphWorkerProxy } from '../managers/graphWorkerProxy';
import { createBinaryNodeData, BinaryNodeData } from '../../../types/binaryProtocol';
import { createLogger } from '../../../utils/loggerConfig';
import { debugState } from '../../../utils/clientDebugState';
import { useGraphInteraction } from '../../visualisation/hooks/useGraphInteraction';

const logger = createLogger('useGraphEventHandlers');

const DRAG_THRESHOLD = 5; 
const BASE_SPHERE_RADIUS = 0.5;
const POSITION_UPDATE_THROTTLE_MS = 100; 

// Helper function to slugify node labels
const slugifyNodeLabel = (label: string): string => {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
};

export const useGraphEventHandlers = (
  meshRef: React.RefObject<THREE.InstancedMesh>,
  dragDataRef: React.MutableRefObject<any>,
  setDragState: React.Dispatch<React.SetStateAction<{ nodeId: string | null; instanceId: number | null }>>,
  graphData: any,
  camera: THREE.Camera,
  size: { width: number; height: number },
  settings: any,
  setGraphData: React.Dispatch<React.SetStateAction<any>>,
  onDragStateChange?: (isDragging: boolean) => void,
  onNodeSelect?: (nodeId: string | null) => void
) => {
  
  const {
    startInteraction,
    endInteraction,
    updateNodePosition,
    shouldSendPositionUpdates,
    flushPositionUpdates
  } = useGraphInteraction({
    positionUpdateThrottleMs: POSITION_UPDATE_THROTTLE_MS,
    onInteractionStateChange: onDragStateChange
  });

  
  const throttledWebSocketUpdate = useRef(
    throttle((nodeId: string, position: { x: number; y: number; z: number }) => {

      if (shouldSendPositionUpdates()) {
        const numericId = graphDataManager.nodeIdMap.get(nodeId);
        if (numericId !== undefined && graphDataManager.webSocketService?.isReady()) {
          // Send server-side drag update (JSON) for pin-at-position + fast-settle
          (graphDataManager.webSocketService as unknown as { sendMessage: (type: string, data?: unknown) => void }).sendMessage('nodeDragUpdate', {
            nodeId: numericId,
            position,
            timestamp: Date.now()
          });

          // Also send legacy binary position update for backwards compatibility
          const update = {
            nodeId: numericId,
            position,
            velocity: { x: 0, y: 0, z: 0 }
          };
          if ('sendNodePositionUpdates' in graphDataManager.webSocketService!) {
            (graphDataManager.webSocketService as unknown as { sendNodePositionUpdates: (updates: unknown[]) => void }).sendNodePositionUpdates([update]);
          }

          if (debugState.isEnabled()) {
            logger.debug(`Throttled WebSocket update for node ${nodeId}`, position);
          }
        }
      }
    }, POSITION_UPDATE_THROTTLE_MS)
  ).current;

  const handlePointerDown = useCallback((event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    if (!meshRef.current) return;

    const instanceId = event.instanceId;
    if (instanceId === undefined || instanceId < 0 || instanceId >= graphData.nodes.length) return;

    const node = graphData.nodes[instanceId];
    if (!node || !node.position) return;

    // CRITICAL: Disable OrbitControls IMMEDIATELY on pointer down
    // This prevents OrbitControls from capturing subsequent move events
    // before React state updates propagate
    if (onDragStateChange) {
      onDragStateChange(true);
    }

    dragDataRef.current = {
      ...dragDataRef.current,
      pointerDown: true,
      nodeId: node.id,
      instanceId: instanceId,
      startPointerPos: new THREE.Vector2(event.nativeEvent.offsetX, event.nativeEvent.offsetY),
      startTime: Date.now(),
      startNodePos3D: new THREE.Vector3(node.position.x, node.position.y, node.position.z),
      currentNodePos3D: new THREE.Vector3(node.position.x, node.position.y, node.position.z),
    };

    // Capture pointer to ensure we receive all subsequent pointer events
    // even if pointer moves outside the mesh
    const target = event.nativeEvent.target as Element;
    if (target && 'setPointerCapture' in target) {
      try {
        target.setPointerCapture(event.nativeEvent.pointerId);
      } catch (e) {
        // Pointer capture may fail in some browsers/contexts
      }
    }

    startInteraction(node.id);

    if (debugState.isEnabled()) {
      logger.debug(`Started interaction tracking for node ${node.id}`);
    }
  }, [graphData.nodes, meshRef, dragDataRef, startInteraction, onDragStateChange]);

  const handlePointerMove = useCallback((event: ThreeEvent<PointerEvent>) => {
    const drag = dragDataRef.current;
    if (!drag.pointerDown) return;

    
    if (!drag.isDragging) {
      const currentPos = new THREE.Vector2(event.nativeEvent.offsetX, event.nativeEvent.offsetY);
      const distance = currentPos.distanceTo(drag.startPointerPos);

      if (distance > DRAG_THRESHOLD) {
        drag.isDragging = true;
        setDragState({ nodeId: drag.nodeId, instanceId: drag.instanceId });

        const numericId = graphDataManager.nodeIdMap.get(drag.nodeId!);
        if (numericId !== undefined) {
          graphWorkerProxy.pinNode(numericId);

          // Notify server of drag start so it can pin the node server-side
          if (graphDataManager.webSocketService?.isReady()) {
            (graphDataManager.webSocketService as unknown as { sendMessage: (type: string, data?: unknown) => void }).sendMessage('nodeDragStart', {
              nodeId: numericId,
              position: {
                x: drag.startNodePos3D.x,
                y: drag.startNodePos3D.y,
                z: drag.startNodePos3D.z
              }
            });
          }
        }
        if (debugState.isEnabled()) {
          logger.debug(`Drag started on node ${drag.nodeId}`);
        }
      }
    }

    
    if (drag.isDragging) {
      event.stopPropagation();

      
      

      
      const cameraDirection = new THREE.Vector3();
      camera.getWorldDirection(cameraDirection);
      const planeNormal = cameraDirection.clone().normalize();

      
      
      const plane = new THREE.Plane(planeNormal, -planeNormal.dot(drag.startNodePos3D));

      
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(event.pointer, camera);

      
      const intersection = new THREE.Vector3();
      const intersectionFound = raycaster.ray.intersectPlane(plane, intersection);

      if (intersectionFound && intersection) {
        const numericId = graphDataManager.nodeIdMap.get(drag.nodeId!);
        if (numericId !== undefined) {
          graphWorkerProxy.updateUserDrivenNodePosition(numericId, intersection);
        }

        drag.currentNodePos3D.copy(intersection);

        // GemNodes reads dragDataRef.current.currentNodePos3D directly in useFrame,
        // so we do NOT need to call setGraphData or setMatrixAt here.
        // This avoids expensive React re-renders during drag (was ~16ms/frame).

        updateNodePosition(drag.nodeId!, {
          x: drag.currentNodePos3D.x,
          y: drag.currentNodePos3D.y,
          z: drag.currentNodePos3D.z
        });

        
        throttledWebSocketUpdate(drag.nodeId!, {
          x: drag.currentNodePos3D.x,
          y: drag.currentNodePos3D.y,
          z: drag.currentNodePos3D.z
        });
      }
    }
  }, [camera, settings?.visualisation?.nodes?.nodeSize, meshRef, dragDataRef, setDragState, setGraphData, updateNodePosition, throttledWebSocketUpdate]);

  const handlePointerUp = useCallback((event?: ThreeEvent<PointerEvent>) => {
    const drag = dragDataRef.current;
    if (!drag.pointerDown) {
      return;
    }

    // Release pointer capture if we captured it
    if (event?.nativeEvent) {
      const target = event.nativeEvent.target as Element;
      if (target && 'releasePointerCapture' in target) {
        try {
          target.releasePointerCapture(event.nativeEvent.pointerId);
        } catch (e) {
          // Pointer may not have been captured
        }
      }
    }

    if (drag.isDragging) {
      if (debugState.isEnabled()) logger.debug(`Drag ended for node ${drag.nodeId}`);

      const numericId = graphDataManager.nodeIdMap.get(drag.nodeId!);
      if (numericId !== undefined) {
        graphWorkerProxy.unpinNode(numericId);
        flushPositionUpdates();

        // Notify server of drag end so it can unpin the node and run final settle
        if (graphDataManager.webSocketService?.isReady()) {
          (graphDataManager.webSocketService as unknown as { sendMessage: (type: string, data?: unknown) => void }).sendMessage('nodeDragEnd', {
            nodeId: numericId
          });
        }
      }
    } else {
      // Click action (not a drag) — select this node to highlight adjacent edges
      if (drag.nodeId) {
        if (onNodeSelect) {
          onNodeSelect(drag.nodeId);
        }
        if (debugState.isEnabled()) logger.debug(`Selected node ${drag.nodeId}`);
      }
    }

    endInteraction(drag.nodeId);

    // CRITICAL: Re-enable OrbitControls by signaling drag is complete
    if (onDragStateChange) {
      onDragStateChange(false);
    }

    dragDataRef.current.pointerDown = false;
    dragDataRef.current.isDragging = false;
    dragDataRef.current.nodeId = null;
    dragDataRef.current.instanceId = null;
    dragDataRef.current.pendingUpdate = null;
    setDragState({ nodeId: null, instanceId: null });

    if (debugState.isEnabled()) {
      logger.debug(`Ended interaction tracking for node ${drag.nodeId}`);
    }
  }, [graphData.nodes, dragDataRef, setDragState, endInteraction, flushPositionUpdates, onDragStateChange, onNodeSelect]);

  // Global pointer up listener as safety net
  // Ensures drag ends even if pointer is released outside the canvas
  useEffect(() => {
    const handleGlobalPointerUp = () => {
      if (dragDataRef.current.pointerDown) {
        handlePointerUp();
      }
    };

    // Add listener on window to catch all pointer up events
    window.addEventListener('pointerup', handleGlobalPointerUp);
    window.addEventListener('pointercancel', handleGlobalPointerUp);

    return () => {
      window.removeEventListener('pointerup', handleGlobalPointerUp);
      window.removeEventListener('pointercancel', handleGlobalPointerUp);
    };
  }, [handlePointerUp]);

  return {
    handlePointerDown,
    handlePointerMove,
    handlePointerUp
  };
};