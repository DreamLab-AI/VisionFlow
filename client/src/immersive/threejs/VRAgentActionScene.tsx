/**
 * VRAgentActionScene
 *
 * Complete VR scene for agent action visualization.
 * Integrates VRActionConnectionsLayer with hand tracking and XR controls.
 *
 * Usage:
 * ```tsx
 * <Canvas>
 *   <XR store={xrStore}>
 *     <VRAgentActionScene agents={agents} />
 *   </XR>
 * </Canvas>
 * ```
 *
 * Performance targets:
 * - Quest 3: 72fps stable
 * - Max 20 active connections
 * - LOD-based geometry reduction
 */

import React, { useMemo, useEffect, useCallback } from 'react';
import { useXREvent } from '@react-three/xr';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { VRActionConnectionsLayer } from './VRActionConnectionsLayer';
import { useActionConnections, ActionConnection } from '../../features/visualisation/hooks/useActionConnections';
import { useAgentActionVisualization } from '../../features/visualisation/hooks/useAgentActionVisualization';
import { useVRHandTracking, agentsToTargetNodes, TargetNode } from '../hooks/useVRHandTracking';
import { useVRConnectionsLOD, calculateOptimalThresholds } from '../hooks/useVRConnectionsLOD';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('VRAgentActionScene');

interface AgentData {
  id: string;
  type: string;
  position?: { x: number; y: number; z: number };
  status?: 'active' | 'idle' | 'error' | 'warning';
}

interface VRAgentActionSceneProps {
  /** Agent data for targeting */
  agents?: AgentData[];
  /** Maximum connections (auto-scales for performance) */
  maxConnections?: number;
  /** Base animation duration (ms) */
  baseDuration?: number;
  /** Enable hand tracking interaction */
  enableHandTracking?: boolean;
  /** Show performance overlay */
  showStats?: boolean;
  /** Callback when agent is targeted */
  onAgentTargeted?: (agentId: string | null) => void;
  /** Callback when agent is selected (pinch/trigger) */
  onAgentSelected?: (agentId: string) => void;
  /** Enable debug visualization */
  debug?: boolean;
}

export const VRAgentActionScene: React.FC<VRAgentActionSceneProps> = ({
  agents = [],
  maxConnections = 20,
  baseDuration = 500,
  enableHandTracking = true,
  showStats = false,
  onAgentTargeted,
  onAgentSelected,
  debug = false,
}) => {
  const { gl, camera } = useThree();

  // Calculate optimal LOD thresholds
  const lodConfig = useMemo(
    () => calculateOptimalThresholds(72, maxConnections),
    [maxConnections]
  );

  // LOD management
  const { updateCameraPosition, getCacheStats } = useVRConnectionsLOD(lodConfig);

  // Action visualization hook (VR mode)
  const { connections, activeCount } = useAgentActionVisualization({
    enabled: true,
    maxConnections,
    baseDuration,
    vrMode: true,
    debug,
  });

  // Convert agents to target nodes
  const targetNodes = useMemo(() => agentsToTargetNodes(agents), [agents]);

  // Hand tracking
  const {
    previewStart,
    previewEnd,
    showPreview,
    previewColor,
    targetedNode,
    setTargetNodes,
    updateHandState,
    triggerHaptic,
  } = useVRHandTracking({
    maxRayDistance: 30,
    targetRadius: 1.5,
    enableHaptics: true,
  });

  // Update target nodes when agents change
  useEffect(() => {
    setTargetNodes(targetNodes);
  }, [targetNodes, setTargetNodes]);

  // Notify parent of targeted agent
  useEffect(() => {
    onAgentTargeted?.(targetedNode?.id || null);
  }, [targetedNode, onAgentTargeted]);

  // Handle XR controller input
  useXREvent('selectstart', (event) => {
    // Access handedness from the inputSource data
    const inputSource = event.data as XRInputSource;
    if (targetedNode && inputSource?.handedness === 'right') {
      logger.info('Agent selected via VR controller:', targetedNode.id);
      triggerHaptic('primary', 0.8, 100);
      onAgentSelected?.(targetedNode.id);
    }
  });

  // Update camera position for LOD each frame
  useFrame(() => {
    updateCameraPosition(camera.position);

    // Update hand tracking from XR session
    const session = (gl.xr as unknown as { getSession?: () => XRSession | null })?.getSession?.();
    if (session && enableHandTracking) {
      updateHandTrackingFromSession(session, updateHandState);
    }
  });

  // Calculate opacity based on active count
  const opacity = useMemo(() => {
    if (activeCount > 18) return 0.6;
    if (activeCount > 12) return 0.8;
    return 1.0;
  }, [activeCount]);

  // Convert Vector3 to the expected format
  const handPos = useMemo(() => {
    if (!previewStart) return null;
    return previewStart.clone();
  }, [previewStart]);

  const targetPos = useMemo(() => {
    if (!previewEnd) return null;
    return previewEnd.clone();
  }, [previewEnd]);

  return (
    <group name="vr-agent-action-scene">
      {/* Main action connections layer */}
      <VRActionConnectionsLayer
        connections={connections}
        opacity={opacity}
        showHandPreview={showPreview && enableHandTracking}
        handPosition={handPos}
        previewTarget={targetPos}
        previewColor={previewColor}
      />

      {/* Target highlight when agent is being targeted */}
      {targetedNode && (
        <VRTargetHighlight
          position={targetedNode.position}
          color={previewColor}
        />
      )}

      {/* Performance stats (debug) */}
      {showStats && (
        <VRPerformanceStats
          activeConnections={activeCount}
          lodCacheSize={getCacheStats().size}
        />
      )}
    </group>
  );
};

/**
 * Highlight ring around targeted agent
 */
const VRTargetHighlight: React.FC<{
  position: THREE.Vector3;
  color: string;
}> = ({ position, color }) => {
  const ringRef = React.useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ringRef.current) {
      // Rotate slowly
      ringRef.current.rotation.z = state.clock.elapsedTime * 0.5;

      // Pulse scale
      const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
      ringRef.current.scale.setScalar(scale);
    }
  });

  return (
    <group position={position}>
      {/* Outer ring */}
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.8, 2.2, 32]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.4}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Inner glow */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.2, 1.8, 32]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.2}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
};

/**
 * VR-visible performance stats (positioned in 3D space)
 */
const VRPerformanceStats: React.FC<{
  activeConnections: number;
  lodCacheSize: number;
}> = ({ activeConnections, lodCacheSize }) => {
  const { camera } = useThree();
  const groupRef = React.useRef<THREE.Group>(null);

  // Position stats panel in front of camera
  useFrame(() => {
    if (groupRef.current) {
      const offset = new THREE.Vector3(0, -0.3, -1);
      offset.applyQuaternion(camera.quaternion);
      groupRef.current.position.copy(camera.position).add(offset);
      groupRef.current.quaternion.copy(camera.quaternion);
    }
  });

  return (
    <group ref={groupRef}>
      {/* Background panel */}
      <mesh position={[0, 0, 0.01]}>
        <planeGeometry args={[0.4, 0.15]} />
        <meshBasicMaterial color="#000000" transparent opacity={0.7} />
      </mesh>

      {/* Stats text would go here - using simple geometry for now */}
      <mesh position={[-0.15, 0.03, 0]}>
        <planeGeometry args={[0.02 * activeConnections, 0.03]} />
        <meshBasicMaterial color="#00ff88" />
      </mesh>

      <mesh position={[-0.15, -0.03, 0]}>
        <planeGeometry args={[0.001 * lodCacheSize, 0.03]} />
        <meshBasicMaterial color="#ffaa00" />
      </mesh>
    </group>
  );
};

/**
 * Update hand tracking state from XR session
 */
function updateHandTrackingFromSession(
  session: XRSession,
  updateHandState: (hand: 'primary' | 'secondary', state: any) => void
) {
  const inputSources = session.inputSources;
  if (!inputSources) return;

  for (const source of Array.from(inputSources) as XRInputSource[]) {
    const hand = source.handedness === 'right' ? 'primary' : 'secondary';

    // Try to get hand tracking data
    if (source.hand) {
      // Full hand tracking (Quest hand tracking)
      const indexTip = source.hand.get('index-finger-tip');
      if (indexTip) {
        // Note: Would need XRFrame to get actual pose
        updateHandState(hand, {
          isTracking: true,
          isPointing: true,
        });
      }
    } else if (source.gamepad) {
      // Controller tracking
      const isPointing =
        source.gamepad.buttons[0]?.pressed ||
        source.gamepad.buttons[1]?.pressed;

      updateHandState(hand, {
        isTracking: true,
        isPointing,
        pinchStrength: Math.max(
          source.gamepad.buttons[0]?.value || 0,
          source.gamepad.buttons[1]?.value || 0
        ),
      });
    }
  }
}

export default VRAgentActionScene;
