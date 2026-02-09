import { useCallback, useEffect } from 'react';
import { ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import { throttle } from 'lodash';
import { graphDataManager, type GraphData, type Node } from '../managers/graphDataManager';
import { graphWorkerProxy } from '../managers/graphWorkerProxy';
import { createBinaryNodeData, BinaryNodeData } from '../../../types/binaryProtocol';
import { createLogger } from '../../../utils/loggerConfig';
import { debugState } from '../../../utils/clientDebugState';
// Removed iframeCommunication import - now using direct window.open
import { useGraphInteraction } from '../../visualisation/hooks/useGraphInteraction';

const logger = createLogger('GraphManager');

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

export const createEventHandlers = (
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

  
  const throttledWebSocketUpdate = useCallback(
    throttle((nodeId: string, position: { x: number; y: number; z: number }) => {
      
      if (shouldSendPositionUpdates()) {
        const numericId = graphDataManager.nodeIdMap.get(nodeId);
        if (numericId !== undefined && graphDataManager.webSocketService?.isReady()) {
          const update = {
            nodeId: numericId,
            position,
            velocity: { x: 0, y: 0, z: 0 }
          };
          if ('sendNodePositionUpdates' in graphDataManager.webSocketService) {
            (graphDataManager.webSocketService as any).sendNodePositionUpdates([update]);
          }

          if (debugState.isEnabled()) {
            logger.debug(`Throttled WebSocket update for node ${nodeId}`, position);
          }
        }
      }
    }, POSITION_UPDATE_THROTTLE_MS),
    [shouldSendPositionUpdates]
  );
  const handlePointerDown = useCallback((event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    if (!meshRef.current) return;

    const instanceId = event.instanceId;
    if (instanceId === undefined || instanceId < 0 || instanceId >= graphData.nodes.length) return;

    const node = graphData.nodes[instanceId];
    if (!node || !node.position) return;

    
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

    
    startInteraction(node.id);

    if (debugState.isEnabled()) {
      logger.debug(`Started interaction tracking for node ${node.id}`);
    }

    if (debugState.isEnabled()) {
      logger.debug(`Pointer down on node ${node.id}`);
    }
  }, [graphData.nodes, meshRef, dragDataRef, onDragStateChange]);

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

        
        const nodeSize = settings?.visualisation?.nodes?.nodeSize || 0.01;
        const scale = nodeSize / BASE_SPHERE_RADIUS;
        const tempMatrix = new THREE.Matrix4();
        tempMatrix.makeScale(scale, scale, scale);
        tempMatrix.setPosition(drag.currentNodePos3D);
        if (meshRef.current) {
          meshRef.current.setMatrixAt(drag.instanceId!, tempMatrix);
          meshRef.current.instanceMatrix.needsUpdate = true;
        }

        
        setGraphData((prev: GraphData) => ({
          ...prev,
          nodes: prev.nodes.map((node: Node, idx: number) =>
            idx === drag.instanceId
              ? { ...node, position: { x: drag.currentNodePos3D.x, y: drag.currentNodePos3D.y, z: drag.currentNodePos3D.z } }
              : node
          )
        }));

        
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
  }, [camera, settings?.visualisation?.nodes?.nodeSize, meshRef, dragDataRef, setDragState, setGraphData]);

  const handlePointerUp = useCallback(() => {
    const drag = dragDataRef.current;
    if (!drag.pointerDown) {
      return;
    }

    if (drag.isDragging) {
      
      if (debugState.isEnabled()) logger.debug(`Drag ended for node ${drag.nodeId}`);

      const numericId = graphDataManager.nodeIdMap.get(drag.nodeId!);
      if (numericId !== undefined) {
        graphWorkerProxy.unpinNode(numericId);

        
        flushPositionUpdates();
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

    
    dragDataRef.current.pointerDown = false;
    dragDataRef.current.isDragging = false;
    dragDataRef.current.nodeId = null;
    dragDataRef.current.instanceId = null;
    dragDataRef.current.pendingUpdate = null;
    setDragState({ nodeId: null, instanceId: null });

    if (debugState.isEnabled()) {
      logger.debug(`Ended interaction tracking for node ${drag.nodeId}`);
    }
  }, [graphData.nodes, dragDataRef, setDragState, endInteraction, flushPositionUpdates, onNodeSelect]);

  
  useEffect(() => {
    const handleGlobalPointerUp = () => {
      if (dragDataRef.current.pointerDown || dragDataRef.current.isDragging) {
        handlePointerUp();
      }
    };

    
    window.addEventListener('pointerup', handleGlobalPointerUp);
    window.addEventListener('pointercancel', handleGlobalPointerUp);
    
    return () => {
      window.removeEventListener('pointerup', handleGlobalPointerUp);
      window.removeEventListener('pointercancel', handleGlobalPointerUp);
    };
  }, [handlePointerUp, dragDataRef]);

  return {
    handlePointerDown,
    handlePointerMove,
    handlePointerUp
  };
};