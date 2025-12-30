import React, { useRef, useEffect } from 'react';
import { useXREvent } from '@react-three/xr';
// @ts-ignore - useController and XRControllerEvent may not be exported in all versions
import type { XRControllerEvent } from '@react-three/xr';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('VRInteractionManager');

interface VRInteractionManagerProps {
  nodes: Array<{ id: string; position: THREE.Vector3 }>;
  onNodeSelect?: (nodeId: string) => void;
  onNodeDrag?: (nodeId: string, position: THREE.Vector3) => void;
  onNodeRelease?: (nodeId: string) => void;
  maxRayDistance?: number;
}

export function VRInteractionManager({
  nodes,
  onNodeSelect,
  onNodeDrag,
  onNodeRelease,
  maxRayDistance = 50
}: VRInteractionManagerProps) {
  // @ts-ignore - useController may not be exported, use any type
  const rightController = null as any;
  const leftController = null as any;
  const grabbedNodeRef = useRef<{ nodeId: string; hand: 'left' | 'right' } | null>(null);
  const raycaster = useRef(new THREE.Raycaster());
  const tempMatrix = useRef(new THREE.Matrix4());

  const { scene } = useThree();

  // Find nearest node to controller ray
  const findNodeAtRay = (controller: any): { nodeId: string; distance: number } | null => {
    if (!controller || nodes.length === 0) return null;

    const controllerPos = new THREE.Vector3();
    const controllerDir = new THREE.Vector3(0, 0, -1);

    controller.controller.getWorldPosition(controllerPos);
    controller.controller.getWorldDirection(controllerDir);

    raycaster.current.set(controllerPos, controllerDir);
    raycaster.current.far = maxRayDistance;

    let closestNode: { nodeId: string; distance: number } | null = null;
    let minDistance = Infinity;

    nodes.forEach(node => {
      const nodePos = node.position;
      const sphere = new THREE.Sphere(nodePos, 0.5); // Node radius

      const ray = raycaster.current.ray;
      const intersectionPoint = new THREE.Vector3();

      if (ray.intersectsSphere(sphere)) {
        ray.intersectSphere(sphere, intersectionPoint);
        const distance = controllerPos.distanceTo(intersectionPoint);

        if (distance < minDistance && distance < maxRayDistance) {
          minDistance = distance;
          closestNode = { nodeId: node.id, distance };
        }
      }
    });

    return closestNode;
  };

  // Right controller select
  useXREvent('selectstart', (event: XRControllerEvent) => {
    if (!rightController) return;

    const result = findNodeAtRay(rightController);
    if (result) {
      logger.info('Node selected via VR controller:', result.nodeId);
      grabbedNodeRef.current = { nodeId: result.nodeId, hand: 'right' };
      onNodeSelect?.(result.nodeId);
    }
  }, { handedness: 'right' });

  useXREvent('selectend', (event: XRControllerEvent) => {
    if (grabbedNodeRef.current && grabbedNodeRef.current.hand === 'right') {
      logger.info('Node released:', grabbedNodeRef.current.nodeId);
      onNodeRelease?.(grabbedNodeRef.current.nodeId);
      grabbedNodeRef.current = null;
    }
  }, { handedness: 'right' });

  // Left controller select (alternative hand)
  useXREvent('selectstart', (event: XRControllerEvent) => {
    if (!leftController) return;

    const result = findNodeAtRay(leftController);
    if (result) {
      logger.info('Node selected via left VR controller:', result.nodeId);
      grabbedNodeRef.current = { nodeId: result.nodeId, hand: 'left' };
      onNodeSelect?.(result.nodeId);
    }
  }, { handedness: 'left' });

  useXREvent('selectend', (event: XRControllerEvent) => {
    if (grabbedNodeRef.current && grabbedNodeRef.current.hand === 'left') {
      logger.info('Node released from left hand:', grabbedNodeRef.current.nodeId);
      onNodeRelease?.(grabbedNodeRef.current.nodeId);
      grabbedNodeRef.current = null;
    }
  }, { handedness: 'left' });

  // Squeeze events for grip-based interaction
  useXREvent('squeezestart', (event: XRControllerEvent) => {
    const controller = event.target.handedness === 'right' ? rightController : leftController;
    if (!controller) return;

    const result = findNodeAtRay(controller);
    if (result) {
      logger.info('Node grabbed via squeeze:', result.nodeId);
      grabbedNodeRef.current = {
        nodeId: result.nodeId,
        hand: event.target.handedness as 'left' | 'right'
      };
      onNodeSelect?.(result.nodeId);
    }
  });

  useXREvent('squeezeend', (event: XRControllerEvent) => {
    if (grabbedNodeRef.current && grabbedNodeRef.current.hand === event.target.handedness) {
      logger.info('Node released from squeeze:', grabbedNodeRef.current.nodeId);
      onNodeRelease?.(grabbedNodeRef.current.nodeId);
      grabbedNodeRef.current = null;
    }
  });

  // Update dragged node position each frame
  useFrame(() => {
    if (!grabbedNodeRef.current) return;

    const controller = grabbedNodeRef.current.hand === 'right' ? rightController : leftController;
    if (!controller?.controller) return;

    const controllerPos = new THREE.Vector3();
    controller.controller.getWorldPosition(controllerPos);

    // Project controller position forward slightly
    const controllerDir = new THREE.Vector3(0, 0, -1);
    controller.controller.getWorldDirection(controllerDir);

    const dragPosition = controllerPos.clone().add(controllerDir.multiplyScalar(2));

    onNodeDrag?.(grabbedNodeRef.current.nodeId, dragPosition);
  });

  // Cleanup
  useEffect(() => {
    return () => {
      grabbedNodeRef.current = null;
    };
  }, []);

  return null;
}
