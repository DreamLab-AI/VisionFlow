import React, { useRef, useEffect } from 'react';
import { useXREvent } from '@react-three/xr';
// @ts-ignore - useController and XRControllerEvent may not be exported in all versions
import type { XRControllerEvent } from '@react-three/xr';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { createLogger } from '../../utils/loggerConfig';
import { graphWorkerProxy } from '../../features/graph/managers/graphWorkerProxy';
import { graphDataManager } from '../../features/graph/managers/graphDataManager';

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
  // XR controllers - accessed via refs updated from XR frame data
  const rightControllerRef = useRef<THREE.Group | null>(null);
  const leftControllerRef = useRef<THREE.Group | null>(null);
  const grabbedNodeRef = useRef<{ nodeId: string; hand: 'left' | 'right' } | null>(null);
  const raycaster = useRef(new THREE.Raycaster());
  const tempMatrix = useRef(new THREE.Matrix4());

  const { scene } = useThree();

  // Find nearest node to controller ray
  const findNodeAtRay = (controller: THREE.Group | null): { nodeId: string; distance: number } | null => {
    if (!controller || nodes.length === 0) return null;

    const controllerPos = new THREE.Vector3();
    const controllerDir = new THREE.Vector3(0, 0, -1);

    controller.getWorldPosition(controllerPos);
    controller.getWorldDirection(controllerDir);

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
    if (!rightControllerRef.current) return;

    const result = findNodeAtRay(rightControllerRef.current);
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
    if (!leftControllerRef.current) return;

    const result = findNodeAtRay(leftControllerRef.current);
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
    const controller = event.target.handedness === 'right' ? rightControllerRef.current : leftControllerRef.current;
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
  useFrame((state) => {
    // Update controller refs from XR session input sources
    const xrManager = state.gl.xr as unknown as { getSession?: () => XRSession | null; getFrame?: () => XRFrame | null; getReferenceSpace?: () => XRReferenceSpace | null } | undefined;
    const session = xrManager?.getSession?.();
    if (session) {
      const inputSources = session.inputSources;
      for (const source of inputSources) {
        const frame = xrManager?.getFrame?.();
        const refSpace = xrManager?.getReferenceSpace?.();
        if (!frame || !refSpace || !source.gripSpace) continue;

        const pose = frame.getPose(source.gripSpace, refSpace);
        if (!pose) continue;

        const targetRef = source.handedness === 'right' ? rightControllerRef : leftControllerRef;
        if (!targetRef.current) {
          targetRef.current = new THREE.Group();
        }
        targetRef.current.position.set(
          pose.transform.position.x,
          pose.transform.position.y,
          pose.transform.position.z
        );
        const q = pose.transform.orientation;
        targetRef.current.quaternion.set(q.x, q.y, q.z, q.w);
      }
    }

    if (!grabbedNodeRef.current) return;

    const controller = grabbedNodeRef.current.hand === 'right' ? rightControllerRef.current : leftControllerRef.current;
    if (!controller) return;

    const controllerPos = new THREE.Vector3();
    controller.getWorldPosition(controllerPos);

    // Project controller position forward slightly
    const controllerDir = new THREE.Vector3(0, 0, -1);
    controller.getWorldDirection(controllerDir);

    const dragPosition = controllerPos.clone().add(controllerDir.multiplyScalar(2));

    // Send drag position directly to the worker proxy with numeric ID
    const numericId = graphDataManager.nodeIdMap.get(grabbedNodeRef.current.nodeId);
    if (numericId !== undefined) {
      graphWorkerProxy.updateUserDrivenNodePosition(numericId, dragPosition);
    }

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
