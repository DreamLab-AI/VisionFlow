/**
 * WebXRScene
 *
 * Unified 3D visualization scene with WebXR support.
 * Automatically detects VR mode and applies appropriate optimizations.
 *
 * Desktop Mode:
 * - Full geometry detail (40 curve segments)
 * - Glow effects on particles
 * - Up to 50 concurrent connections
 *
 * VR Mode (Quest 3):
 * - Simplified geometry (LOD-based)
 * - No glow effects
 * - Max 20 connections for 72fps
 * - Hand tracking preview
 * - Haptic feedback on action events
 *
 * LOD Thresholds:
 * - < 5m: High detail
 * - 5-15m: Medium detail
 * - 15-30m: Low detail
 * - > 30m: Culled
 */

import React, { Suspense, useMemo, useCallback, useRef, useEffect } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { createXRStore, XR, useXREvent } from '@react-three/xr';
import * as THREE from 'three';

import { ActionConnectionsLayer } from './components/ActionConnectionsLayer';
import { VRActionConnectionsLayer } from '../../immersive/threejs/VRActionConnectionsLayer';
import { useAgentActionVisualization } from './hooks/useAgentActionVisualization';
import {
  useVRConnectionsLOD,
  calculateOptimalThresholds,
  LODLevel,
} from '../../immersive/hooks/useVRConnectionsLOD';
import {
  useVRHandTracking,
  agentsToTargetNodes,
  TargetNode,
} from '../../immersive/hooks/useVRHandTracking';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('WebXRScene');

// Create XR store singleton
const xrStore = createXRStore({
  hand: true,
  controller: true,
});

/** Agent data for VR targeting */
interface AgentData {
  id: string;
  type?: string;
  position?: { x: number; y: number; z: number };
  status?: 'active' | 'idle' | 'error' | 'warning';
}

interface WebXRSceneProps {
  /** Agent data for visualization and VR targeting */
  agents?: AgentData[];
  /** Enable action connections visualization */
  enableActionConnections?: boolean;
  /** Maximum concurrent connections */
  maxConnections?: number;
  /** Base animation duration (ms) */
  baseDuration?: number;
  /** Show performance stats (debug) */
  showStats?: boolean;
  /** Enable debug mode */
  debug?: boolean;
  /** Callback when VR session starts */
  onVRSessionStart?: () => void;
  /** Callback when VR session ends */
  onVRSessionEnd?: () => void;
  /** Callback when agent is targeted in VR */
  onAgentTargeted?: (agentId: string | null) => void;
  /** Callback when agent is selected in VR (pinch/trigger) */
  onAgentSelected?: (agentId: string) => void;
  /** Custom children to render in scene */
  children?: React.ReactNode;
}

/**
 * Main WebXR Scene component.
 * Wrap your 3D content and get automatic VR support.
 */
export const WebXRScene: React.FC<WebXRSceneProps> = ({
  agents = [],
  enableActionConnections = true,
  maxConnections = 50,
  baseDuration = 500,
  showStats = false,
  debug = false,
  onVRSessionStart,
  onVRSessionEnd,
  onAgentTargeted,
  onAgentSelected,
  children,
}) => {
  const [isVRSupported, setIsVRSupported] = React.useState<boolean | null>(null);
  const [isInVR, setIsInVR] = React.useState(false);

  // Check VR support on mount
  useEffect(() => {
    if (navigator.xr) {
      navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
        setIsVRSupported(supported);
        logger.info(`WebXR VR support: ${supported}`);
      });
    } else {
      setIsVRSupported(false);
      logger.info('WebXR not available');
    }
  }, []);

  // Track VR session state
  useEffect(() => {
    const handleSessionStart = () => {
      setIsInVR(true);
      logger.info('VR session started');
      onVRSessionStart?.();
    };

    const handleSessionEnd = () => {
      setIsInVR(false);
      logger.info('VR session ended');
      onVRSessionEnd?.();
    };

    // Subscribe to XR store state changes
    const unsubscribe = xrStore.subscribe((state) => {
      if (state.session && !isInVR) {
        handleSessionStart();
      } else if (!state.session && isInVR) {
        handleSessionEnd();
      }
    });

    return unsubscribe;
  }, [isInVR, onVRSessionStart, onVRSessionEnd]);

  const handleEnterVR = useCallback(async () => {
    try {
      await xrStore.enterVR();
    } catch (error) {
      logger.error('Failed to enter VR:', error);
    }
  }, []);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* VR Entry Button */}
      {isVRSupported && !isInVR && (
        <button
          onClick={handleEnterVR}
          style={{
            position: 'absolute',
            bottom: 20,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '12px 24px',
            fontSize: 16,
            fontWeight: 600,
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: 8,
            cursor: 'pointer',
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            transition: 'background-color 0.2s',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#45a049')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#4CAF50')}
        >
          Enter VR
        </button>
      )}

      {/* VR Active Indicator */}
      {isInVR && (
        <div
          style={{
            position: 'absolute',
            top: 20,
            right: 20,
            padding: '8px 16px',
            backgroundColor: 'rgba(76, 175, 80, 0.9)',
            color: 'white',
            borderRadius: 4,
            fontSize: 14,
            fontWeight: 600,
            zIndex: 1000,
          }}
        >
          VR Mode Active
        </div>
      )}

      {/* Three.js Canvas with XR */}
      <Canvas
        gl={{
          antialias: !isInVR, // Disable AA in VR for performance
          alpha: false,
          powerPreference: 'high-performance',
        }}
        camera={{ position: [0, 1.6, 3], fov: 70 }}
      >
        <XR store={xrStore}>
          <Suspense fallback={null}>
            {/* Lighting */}
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={0.5} />

            {/* Action Connections - switches between desktop and VR mode */}
            {enableActionConnections && (
              <ActionConnectionsScene
                agents={agents}
                isVRMode={isInVR}
                maxConnections={maxConnections}
                baseDuration={baseDuration}
                showStats={showStats}
                debug={debug}
                onAgentTargeted={onAgentTargeted}
                onAgentSelected={onAgentSelected}
              />
            )}

            {/* Custom content */}
            {children}
          </Suspense>
        </XR>
      </Canvas>
    </div>
  );
};

/**
 * Internal component handling VR/Desktop mode switching for connections.
 */
const ActionConnectionsScene: React.FC<{
  agents: AgentData[];
  isVRMode: boolean;
  maxConnections: number;
  baseDuration: number;
  showStats: boolean;
  debug: boolean;
  onAgentTargeted?: (agentId: string | null) => void;
  onAgentSelected?: (agentId: string) => void;
}> = ({
  agents,
  isVRMode,
  maxConnections,
  baseDuration,
  showStats,
  debug,
  onAgentTargeted,
  onAgentSelected,
}) => {
  const { camera, gl } = useThree();

  // VR-specific: Calculate optimal LOD thresholds
  const lodConfig = useMemo(
    () => (isVRMode ? calculateOptimalThresholds(72, Math.min(maxConnections, 20)) : null),
    [isVRMode, maxConnections]
  );

  // LOD management for VR
  const { updateCameraPosition, getLODLevel, getCacheStats } = useVRConnectionsLOD(
    lodConfig ?? {}
  );

  // Action visualization hook
  const { connections, activeCount } = useAgentActionVisualization({
    enabled: true,
    maxConnections: isVRMode ? Math.min(maxConnections, 20) : maxConnections,
    baseDuration: isVRMode ? Math.min(baseDuration, 400) : baseDuration,
    vrMode: isVRMode,
    debug,
  });

  // VR hand tracking
  const targetNodes = useMemo(() => agentsToTargetNodes(agents), [agents]);
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
    enableHaptics: isVRMode,
  });

  // Update target nodes when agents change
  useEffect(() => {
    if (isVRMode) {
      setTargetNodes(targetNodes);
    }
  }, [isVRMode, targetNodes, setTargetNodes]);

  // Notify parent of targeted agent
  useEffect(() => {
    if (isVRMode) {
      onAgentTargeted?.(targetedNode?.id || null);
    }
  }, [isVRMode, targetedNode, onAgentTargeted]);

  // Handle XR controller selection
  useXREvent('selectstart', (event) => {
    if (!isVRMode) return;

    const inputSource = event.data as XRInputSource;
    if (targetedNode && inputSource?.handedness === 'right') {
      logger.info('Agent selected via VR controller:', targetedNode.id);
      triggerHaptic('primary', 0.8, 100);
      onAgentSelected?.(targetedNode.id);
    }
  });

  // Update each frame
  useFrame(() => {
    if (isVRMode) {
      // Update camera position for LOD
      updateCameraPosition(camera.position);

      // Update hand tracking from XR session
      const session = (gl.xr as any)?.getSession?.();
      if (session) {
        updateHandTrackingFromSession(session, updateHandState);
      }
    }
  });

  // Calculate opacity based on active count
  const opacity = useMemo(() => {
    if (isVRMode) {
      if (activeCount > 18) return 0.6;
      if (activeCount > 12) return 0.8;
      return 1.0;
    }
    if (activeCount > 40) return 0.6;
    if (activeCount > 30) return 0.8;
    return 1.0;
  }, [isVRMode, activeCount]);

  if (connections.length === 0 && !showPreview) return null;

  // Render VR-optimized or desktop version
  if (isVRMode) {
    return (
      <group name="vr-action-connections-scene">
        <VRActionConnectionsLayer
          connections={connections}
          opacity={opacity}
          showHandPreview={showPreview}
          handPosition={previewStart}
          previewTarget={previewEnd}
          previewColor={previewColor}
        />

        {/* Target highlight when agent is being targeted */}
        {targetedNode && (
          <VRTargetHighlight position={targetedNode.position} color={previewColor} />
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
  }

  // Desktop mode
  return (
    <group name="desktop-action-connections-scene">
      <ActionConnectionsLayer
        connections={connections}
        vrMode={false}
        opacity={opacity}
        lineWidth={2}
      />
    </group>
  );
};

/**
 * Highlight ring around targeted agent in VR.
 */
const VRTargetHighlight: React.FC<{
  position: THREE.Vector3;
  color: string;
}> = ({ position, color }) => {
  const ringRef = useRef<THREE.Mesh>(null);

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
 * VR-visible performance stats positioned in 3D space.
 */
const VRPerformanceStats: React.FC<{
  activeConnections: number;
  lodCacheSize: number;
}> = ({ activeConnections, lodCacheSize }) => {
  const { camera } = useThree();
  const groupRef = useRef<THREE.Group>(null);

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

      {/* Connection bar */}
      <mesh position={[-0.15, 0.03, 0]}>
        <planeGeometry args={[Math.min(0.02 * activeConnections, 0.3), 0.03]} />
        <meshBasicMaterial color="#00ff88" />
      </mesh>

      {/* LOD cache bar */}
      <mesh position={[-0.15, -0.03, 0]}>
        <planeGeometry args={[Math.min(0.001 * lodCacheSize, 0.3), 0.03]} />
        <meshBasicMaterial color="#ffaa00" />
      </mesh>
    </group>
  );
};

/**
 * Update hand tracking state from XR session.
 */
function updateHandTrackingFromSession(
  session: XRSession,
  updateHandState: (hand: 'primary' | 'secondary', state: any) => void
) {
  const inputSources = session.inputSources;
  if (!inputSources) return;

  for (const source of Array.from(inputSources) as XRInputSource[]) {
    const hand = source.handedness === 'right' ? 'primary' : 'secondary';

    if (source.hand) {
      // Full hand tracking (Quest hand tracking)
      updateHandState(hand, {
        isTracking: true,
        isPointing: true,
      });
    } else if (source.gamepad) {
      // Controller tracking
      const isPointing =
        source.gamepad.buttons[0]?.pressed || source.gamepad.buttons[1]?.pressed;

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

/** Export XR store for external control */
export { xrStore };

export default WebXRScene;
