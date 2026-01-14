/**
 * useVRHandTracking Hook
 *
 * Manages hand tracking state for VR action connection preview.
 * Detects potential targets within reach of the hand/controller.
 *
 * Features:
 * - Tracks both hands/controllers
 * - Detects nodes within reach radius
 * - Provides preview line endpoints
 * - Haptic feedback integration
 *
 * @see VRActionConnectionsLayer for rendering
 */

import { useRef, useCallback, useMemo, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

export interface HandState {
  position: THREE.Vector3;
  direction: THREE.Vector3;
  isTracking: boolean;
  isPointing: boolean;
  pinchStrength: number;
}

export interface TargetNode {
  id: string;
  position: THREE.Vector3;
  type?: string;
}

export interface VRHandTrackingConfig {
  /** Maximum ray distance for target detection (meters) */
  maxRayDistance?: number;
  /** Radius around target for hit detection (meters) */
  targetRadius?: number;
  /** Minimum pinch strength to activate (0-1) */
  activationThreshold?: number;
  /** Enable haptic feedback */
  enableHaptics?: boolean;
}

const DEFAULT_CONFIG: Required<VRHandTrackingConfig> = {
  maxRayDistance: 30,
  targetRadius: 1.0,
  activationThreshold: 0.7,
  enableHaptics: true,
};

export interface VRHandTrackingResult {
  /** Current state of primary (right) hand */
  primaryHand: HandState | null;
  /** Current state of secondary (left) hand */
  secondaryHand: HandState | null;
  /** Currently targeted node (if any) */
  targetedNode: TargetNode | null;
  /** Preview line start position (hand) */
  previewStart: THREE.Vector3 | null;
  /** Preview line end position (target or ray end) */
  previewEnd: THREE.Vector3 | null;
  /** Whether preview should be shown */
  showPreview: boolean;
  /** Color for preview line */
  previewColor: string;
  /** Manually update hand state (for external sources) */
  updateHandState: (hand: 'primary' | 'secondary', state: Partial<HandState>) => void;
  /** Set available target nodes */
  setTargetNodes: (nodes: TargetNode[]) => void;
  /** Trigger haptic feedback */
  triggerHaptic: (hand: 'primary' | 'secondary', intensity: number, duration: number) => void;
}

/**
 * Hook for VR hand tracking and target detection
 */
export const useVRHandTracking = (
  config: VRHandTrackingConfig = {}
): VRHandTrackingResult => {
  const settings = useMemo(
    () => ({ ...DEFAULT_CONFIG, ...config }),
    [config]
  );

  const { gl } = useThree();

  // Hand state references
  const primaryHandRef = useRef<HandState>({
    position: new THREE.Vector3(),
    direction: new THREE.Vector3(0, 0, -1),
    isTracking: false,
    isPointing: false,
    pinchStrength: 0,
  });

  const secondaryHandRef = useRef<HandState>({
    position: new THREE.Vector3(),
    direction: new THREE.Vector3(0, 0, -1),
    isTracking: false,
    isPointing: false,
    pinchStrength: 0,
  });

  // Target nodes
  const targetNodesRef = useRef<TargetNode[]>([]);

  // Detected target
  const [targetedNode, setTargetedNode] = useState<TargetNode | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [previewStart, setPreviewStart] = useState<THREE.Vector3 | null>(null);
  const [previewEnd, setPreviewEnd] = useState<THREE.Vector3 | null>(null);

  // Raycaster for target detection
  const raycaster = useRef(new THREE.Raycaster());

  // Colors for different states
  const previewColor = useMemo(() => {
    if (targetedNode) return '#00ff88'; // Locked on target
    return '#00ffff'; // Searching
  }, [targetedNode]);

  /**
   * Update hand state from external source
   */
  const updateHandState = useCallback(
    (hand: 'primary' | 'secondary', state: Partial<HandState>) => {
      const ref = hand === 'primary' ? primaryHandRef : secondaryHandRef;

      if (state.position) ref.current.position.copy(state.position);
      if (state.direction) ref.current.direction.copy(state.direction);
      if (state.isTracking !== undefined) ref.current.isTracking = state.isTracking;
      if (state.isPointing !== undefined) ref.current.isPointing = state.isPointing;
      if (state.pinchStrength !== undefined)
        ref.current.pinchStrength = state.pinchStrength;
    },
    []
  );

  /**
   * Set target nodes for detection
   */
  const setTargetNodes = useCallback((nodes: TargetNode[]) => {
    targetNodesRef.current = nodes;
  }, []);

  /**
   * Trigger haptic feedback
   */
  const triggerHaptic = useCallback(
    (hand: 'primary' | 'secondary', intensity: number, duration: number) => {
      if (!settings.enableHaptics) return;

      // Access XR session if available
      const session = (gl.xr as any)?.getSession?.();
      if (!session) return;

      const inputSources = session.inputSources;
      if (!inputSources) return;

      const handedness = hand === 'primary' ? 'right' : 'left';
      const source = Array.from(inputSources).find(
        (s: any) => s.handedness === handedness
      ) as any;

      if (source?.gamepad?.hapticActuators?.[0]) {
        source.gamepad.hapticActuators[0].pulse(intensity, duration);
      }
    },
    [gl, settings.enableHaptics]
  );

  /**
   * Find target node along ray
   */
  const findTargetAlongRay = useCallback(
    (origin: THREE.Vector3, direction: THREE.Vector3): TargetNode | null => {
      raycaster.current.set(origin, direction);
      raycaster.current.far = settings.maxRayDistance;

      let closestNode: TargetNode | null = null;
      let closestDistance = Infinity;

      for (const node of targetNodesRef.current) {
        const sphere = new THREE.Sphere(node.position, settings.targetRadius);
        const intersectionPoint = new THREE.Vector3();

        // Check if ray intersects sphere
        const intersects = raycaster.current.ray.intersectSphere(sphere, intersectionPoint);
        if (intersects) {
          const distance = origin.distanceTo(intersectionPoint);

          if (distance < closestDistance && distance < settings.maxRayDistance) {
            closestDistance = distance;
            closestNode = node;
          }
        }
      }

      return closestNode;
    },
    [settings.maxRayDistance, settings.targetRadius]
  );

  /**
   * Update targeting each frame
   */
  useFrame(() => {
    const primary = primaryHandRef.current;

    // Only show preview when hand is tracking and pointing
    const shouldShowPreview = primary.isTracking && primary.isPointing;
    setShowPreview(shouldShowPreview);

    if (!shouldShowPreview) {
      setTargetedNode(null);
      setPreviewStart(null);
      setPreviewEnd(null);
      return;
    }

    // Update preview start
    setPreviewStart(primary.position.clone());

    // Find target
    const target = findTargetAlongRay(primary.position, primary.direction);

    if (target) {
      setTargetedNode(target);
      setPreviewEnd(target.position.clone());

      // Haptic feedback on target acquisition
      if (!targetedNode || targetedNode.id !== target.id) {
        triggerHaptic('primary', 0.3, 50);
      }
    } else {
      setTargetedNode(null);

      // Calculate ray endpoint
      const rayEnd = primary.position
        .clone()
        .add(primary.direction.clone().multiplyScalar(settings.maxRayDistance));
      setPreviewEnd(rayEnd);
    }
  });

  return {
    primaryHand: primaryHandRef.current.isTracking ? { ...primaryHandRef.current } : null,
    secondaryHand: secondaryHandRef.current.isTracking ? { ...secondaryHandRef.current } : null,
    targetedNode,
    previewStart,
    previewEnd,
    showPreview,
    previewColor,
    updateHandState,
    setTargetNodes,
    triggerHaptic,
  };
};

/**
 * Convert XR controller state to hand state
 */
export function xrControllerToHandState(
  controller: any,
  gamepad: Gamepad | null
): Partial<HandState> {
  const position = new THREE.Vector3();
  const direction = new THREE.Vector3(0, 0, -1);

  if (controller) {
    controller.getWorldPosition(position);
    controller.getWorldDirection(direction);
  }

  const isPointing = gamepad
    ? gamepad.buttons[0]?.pressed || gamepad.buttons[1]?.pressed
    : false;

  const pinchStrength = gamepad
    ? Math.max(
        gamepad.buttons[0]?.value || 0,
        gamepad.buttons[1]?.value || 0
      )
    : 0;

  return {
    position,
    direction,
    isTracking: !!controller,
    isPointing,
    pinchStrength,
  };
}

/**
 * Extract nodes from agent data for targeting
 */
export function agentsToTargetNodes(
  agents: Array<{
    id: string;
    position?: { x: number; y: number; z: number };
    type?: string;
  }>
): TargetNode[] {
  return agents
    .filter((a) => a.position)
    .map((a) => ({
      id: a.id,
      position: new THREE.Vector3(a.position!.x, a.position!.y, a.position!.z),
      type: a.type,
    }));
}

export default useVRHandTracking;
