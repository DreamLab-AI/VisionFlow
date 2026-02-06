/**
 * VRActionConnectionsLayer
 *
 * WebXR/VR-optimized version of ActionConnectionsLayer for Meta Quest 3.
 * Targets 72fps with aggressive performance optimizations.
 *
 * Optimizations:
 * - InstancedMesh for particles (single draw call for all particles)
 * - LOD: Reduced geometry at distance
 * - Max 20 active connections in VR mode
 * - Simplified shaders (no bloom/glow effects)
 * - Billboarded particles for stereoscopic correctness
 * - Hand tracking connection preview
 *
 * @see ActionConnectionsLayer for desktop version
 */

import React, { useRef, useMemo, useCallback, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { ActionConnection } from '../../features/visualisation/hooks/useActionConnections';
import { useVRConnectionsLOD, LODLevel } from '../hooks/useVRConnectionsLOD';

/** Maximum connections for Quest 3 @ 72fps */
const VR_MAX_CONNECTIONS = 20;

const _tempColor = new THREE.Color();

/** Geometry segments by LOD level */
const LOD_SEGMENTS: Record<LODLevel, { curve: number; sphere: number }> = {
  high: { curve: 24, sphere: 12 },
  medium: { curve: 16, sphere: 8 },
  low: { curve: 8, sphere: 6 },
  culled: { curve: 0, sphere: 0 },
};

/** Phase timing boundaries */
const PHASE_BOUNDS = {
  spawnEnd: 0.2,
  travelEnd: 0.8,
  impactEnd: 0.9,
  fadeEnd: 1.0,
};

interface VRActionConnectionsLayerProps {
  connections: ActionConnection[];
  /** Global opacity multiplier */
  opacity?: number;
  /** Enable hand tracking preview */
  showHandPreview?: boolean;
  /** Hand/controller position for preview line */
  handPosition?: THREE.Vector3 | null;
  /** Target position for preview line */
  previewTarget?: THREE.Vector3 | null;
  /** Preview line color */
  previewColor?: string;
}

/**
 * Main VR-optimized ActionConnections component.
 * Uses InstancedMesh for particles and simplified line geometry.
 */
export const VRActionConnectionsLayer: React.FC<VRActionConnectionsLayerProps> = ({
  connections,
  opacity = 1.0,
  showHandPreview = false,
  handPosition = null,
  previewTarget = null,
  previewColor = '#00ffff',
}) => {
  const { camera } = useThree();

  // Limit connections for VR performance
  const activeConnections = useMemo(() => {
    if (connections.length <= VR_MAX_CONNECTIONS) return connections;
    // Prioritize newest connections
    return connections.slice(-VR_MAX_CONNECTIONS);
  }, [connections]);

  // LOD management
  const { getLODLevel, updateCameraPosition } = useVRConnectionsLOD();

  // Update camera position for LOD calculations
  useFrame(() => {
    updateCameraPosition(camera.position);
  });

  if (activeConnections.length === 0 && !showHandPreview) return null;

  return (
    <group name="vr-action-connections-layer">
      {/* Instanced particles for all connections */}
      <VRConnectionParticles
        connections={activeConnections}
        opacity={opacity}
        getLODLevel={getLODLevel}
      />

      {/* Connection lines */}
      {activeConnections.map((conn) => (
        <VRConnectionLine
          key={conn.id}
          connection={conn}
          opacity={opacity}
          getLODLevel={getLODLevel}
        />
      ))}

      {/* Hand tracking preview */}
      {showHandPreview && handPosition && previewTarget && (
        <VRConnectionPreviewLine
          start={handPosition}
          end={previewTarget}
          color={previewColor}
          opacity={opacity * 0.6}
        />
      )}
    </group>
  );
};

/**
 * Instanced particles for all active connections.
 * Single draw call for massive performance gain.
 */
const VRConnectionParticles: React.FC<{
  connections: ActionConnection[];
  opacity: number;
  getLODLevel: (position: THREE.Vector3) => LODLevel;
}> = ({ connections, opacity, getLODLevel }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const materialRef = useRef<THREE.MeshBasicMaterial>(null!);
  const maxInstances = VR_MAX_CONNECTIONS;

  const geometry = useMemo(() => new THREE.SphereGeometry(0.15, 8, 8), []);
  const material = useMemo(() => {
    const mat = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.9 * opacity,
      depthWrite: false,
    });
    materialRef.current = mat;
    return mat;
  }, []);

  // Dummy object for matrix calculations
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const colorArray = useMemo(
    () => new Float32Array(maxInstances * 3),
    [maxInstances]
  );

  useFrame(() => {
    if (!meshRef.current) return;

    materialRef.current.opacity = 0.9 * opacity;

    for (let i = 0; i < maxInstances; i++) {
      if (i < connections.length) {
        const conn = connections[i];
        const position = calculateParticlePosition(conn);

        const lod = getLODLevel(position);
        if (lod === 'culled') {
          dummy.scale.setScalar(0);
        } else {
          const scale = calculateParticleScale(conn);
          dummy.position.copy(position);
          dummy.scale.setScalar(scale);
        }

        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);

        _tempColor.set(conn.color);
        colorArray[i * 3] = _tempColor.r;
        colorArray[i * 3 + 1] = _tempColor.g;
        colorArray[i * 3 + 2] = _tempColor.b;
      } else {
        // Hide unused instances
        dummy.scale.setScalar(0);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);
      }
    }

    meshRef.current.instanceMatrix.needsUpdate = true;

    // Update instance colors
    const colorAttr = meshRef.current.geometry.getAttribute('instanceColor');
    if (colorAttr) {
      (colorAttr as THREE.BufferAttribute).set(colorArray);
      colorAttr.needsUpdate = true;
    }
  });

  // Add instance color attribute
  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.geometry.setAttribute(
        'instanceColor',
        new THREE.InstancedBufferAttribute(colorArray, 3)
      );
      // Enable vertex colors on material
      (meshRef.current.material as THREE.MeshBasicMaterial).vertexColors = true;
    }
  }, [colorArray]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, maxInstances]}
      frustumCulled={true}
    />
  );
};

/**
 * Calculate particle position along connection curve
 */
function calculateParticlePosition(conn: ActionConnection): THREE.Vector3 {
  const sourcePos = conn.sourcePosition
    ? new THREE.Vector3(conn.sourcePosition.x, conn.sourcePosition.y, conn.sourcePosition.z)
    : new THREE.Vector3(Math.sin(conn.sourceAgentId * 1337) * 15, 0, 0);

  const targetPos = conn.targetPosition
    ? new THREE.Vector3(conn.targetPosition.x, conn.targetPosition.y, conn.targetPosition.z)
    : new THREE.Vector3(Math.sin(conn.targetNodeId * 2749) * 20, 5, 0);

  const progress = conn.progress;

  if (conn.phase === 'spawn') {
    return sourcePos.clone();
  }

  if (conn.phase === 'travel') {
    const travelProgress =
      (progress - PHASE_BOUNDS.spawnEnd) /
      (PHASE_BOUNDS.travelEnd - PHASE_BOUNDS.spawnEnd);

    // Simple lerp for VR (no expensive bezier)
    const midPoint = new THREE.Vector3()
      .addVectors(sourcePos, targetPos)
      .multiplyScalar(0.5);
    midPoint.y += sourcePos.distanceTo(targetPos) * 0.15;

    if (travelProgress < 0.5) {
      return sourcePos.clone().lerp(midPoint, travelProgress * 2);
    } else {
      return midPoint.clone().lerp(targetPos, (travelProgress - 0.5) * 2);
    }
  }

  return targetPos.clone();
}

/**
 * Calculate particle scale based on phase
 */
function calculateParticleScale(conn: ActionConnection): number {
  const { progress, phase } = conn;

  switch (phase) {
    case 'spawn':
      return progress / PHASE_BOUNDS.spawnEnd;
    case 'travel':
      return 1.0;
    case 'impact':
      return 0.5;
    case 'fade': {
      const fadeProgress =
        (progress - PHASE_BOUNDS.impactEnd) /
        (PHASE_BOUNDS.fadeEnd - PHASE_BOUNDS.impactEnd);
      return 0.5 * (1 - fadeProgress);
    }
    default:
      return 1.0;
  }
}

/**
 * Single VR-optimized connection line.
 * Uses simplified geometry based on LOD.
 */
const VRConnectionLine: React.FC<{
  connection: ActionConnection;
  opacity: number;
  getLODLevel: (position: THREE.Vector3) => LODLevel;
}> = ({ connection, opacity, getLODLevel }) => {
  const lineRef = useRef<THREE.Line | null>(null);

  // Calculate positions
  const { sourcePos, targetPos, midPoint } = useMemo(() => {
    const src = connection.sourcePosition
      ? new THREE.Vector3(
          connection.sourcePosition.x,
          connection.sourcePosition.y,
          connection.sourcePosition.z
        )
      : new THREE.Vector3(Math.sin(connection.sourceAgentId * 1337) * 15, 0, 0);

    const tgt = connection.targetPosition
      ? new THREE.Vector3(
          connection.targetPosition.x,
          connection.targetPosition.y,
          connection.targetPosition.z
        )
      : new THREE.Vector3(Math.sin(connection.targetNodeId * 2749) * 20, 5, 0);

    const mid = new THREE.Vector3().addVectors(src, tgt).multiplyScalar(0.5);
    mid.y += src.distanceTo(tgt) * 0.15;

    return { sourcePos: src, targetPos: tgt, midPoint: mid };
  }, [connection.sourcePosition, connection.targetPosition, connection.sourceAgentId, connection.targetNodeId]);

  // Get LOD level for this connection
  const lod = useMemo(() => {
    return getLODLevel(midPoint);
  }, [getLODLevel, midPoint]);

  // Generate curve points based on LOD
  const geometry = useMemo(() => {
    if (lod === 'culled') return null;

    const segments = LOD_SEGMENTS[lod].curve;
    const curve = new THREE.QuadraticBezierCurve3(sourcePos, midPoint, targetPos);
    const points = curve.getPoints(segments);

    return new THREE.BufferGeometry().setFromPoints(points);
  }, [sourcePos, midPoint, targetPos, lod]);

  // Update line opacity based on phase
  useFrame(() => {
    if (!lineRef.current) return;

    const { progress, phase } = connection;
    let lineOpacity = 1.0;

    switch (phase) {
      case 'spawn':
        lineOpacity = progress / PHASE_BOUNDS.spawnEnd;
        break;
      case 'travel':
        lineOpacity = 1.0;
        break;
      case 'impact':
        lineOpacity = 1.0;
        break;
      case 'fade': {
        const fadeProgress =
          (progress - PHASE_BOUNDS.impactEnd) /
          (PHASE_BOUNDS.fadeEnd - PHASE_BOUNDS.impactEnd);
        lineOpacity = 1 - fadeProgress;
        break;
      }
    }

    const material = lineRef.current.material as THREE.LineBasicMaterial;
    material.opacity = lineOpacity * opacity;
  });

  // Create line object with material
  const lineObject = useMemo(() => {
    if (!geometry) return null;

    const material = new THREE.LineBasicMaterial({
      color: connection.color,
      transparent: true,
      opacity: opacity,
      linewidth: 1,
      depthWrite: false,
    });

    return new THREE.Line(geometry, material);
  }, [geometry, connection.color, opacity]);

  // Store ref for frame updates
  useEffect(() => {
    if (lineObject) {
      lineRef.current = lineObject;
    }
    return () => {
      if (lineObject) {
        lineObject.geometry.dispose();
        (lineObject.material as THREE.Material).dispose();
      }
    };
  }, [lineObject]);

  if (!lineObject || lod === 'culled') return null;

  return <primitive object={lineObject} />;
};

/**
 * Hand tracking preview line.
 * Shows potential connection from controller/hand to target.
 */
const VRConnectionPreviewLine: React.FC<{
  start: THREE.Vector3;
  end: THREE.Vector3;
  color: string;
  opacity: number;
}> = ({ start, end, color, opacity }) => {
  const lineRef = useRef<THREE.Line | null>(null);

  // Simple straight line for preview (no curve overhead)
  const geometry = useMemo(() => {
    const points = [start.clone(), end.clone()];
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [start, end]);

  // Create line object
  const lineObject = useMemo(() => {
    const material = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      linewidth: 2,
      depthWrite: false,
    });

    return new THREE.Line(geometry, material);
  }, [geometry, color, opacity]);

  // Store ref for frame updates
  useEffect(() => {
    lineRef.current = lineObject;
    return () => {
      lineObject.geometry.dispose();
      (lineObject.material as THREE.Material).dispose();
    };
  }, [lineObject]);

  // Pulsing animation for preview
  useFrame((state) => {
    if (!lineRef.current) return;

    const pulse = 0.5 + Math.sin(state.clock.elapsedTime * 4) * 0.3;
    const material = lineRef.current.material as THREE.LineBasicMaterial;
    material.opacity = opacity * pulse;
  });

  return <primitive object={lineObject} />;
};

/**
 * Impact ring for VR.
 * Simplified ring geometry without bloom.
 */
export const VRImpactRing: React.FC<{
  position: THREE.Vector3;
  color: string;
  scale: number;
  opacity: number;
}> = ({ position, color, scale, opacity }) => {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (ringRef.current) {
      ringRef.current.rotation.x = Math.PI / 2;
      ringRef.current.lookAt(0, 0, 0); // Face center for visibility
    }
  });

  if (scale <= 0) return null;

  return (
    <mesh ref={ringRef} position={position} scale={[scale, scale, scale]}>
      <ringGeometry args={[0.3, 0.8, 16]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity * 0.5}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
};

export default VRActionConnectionsLayer;
