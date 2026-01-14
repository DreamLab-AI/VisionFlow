/**
 * ActionConnectionsLayer
 *
 * Renders ephemeral animated connections between agent nodes and data nodes
 * showing real-time agent-to-data interactions.
 *
 * Visual Design:
 * - Bezier curve from agent to target
 * - Animated particle traveling along the path
 * - Color coded by action type (query=blue, update=yellow, create=green, delete=red, link=purple, transform=cyan)
 * - Impact burst effect at target
 *
 * Performance:
 * - VR mode uses simplified geometry for Quest 3 @ 72fps
 * - LOD: Reduces detail at distance
 * - Max 50 concurrent connections
 */

import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { ActionConnection } from '../hooks/useActionConnections';
import { AgentActionType } from '@/services/BinaryWebSocketProtocol';

interface ActionConnectionsLayerProps {
  connections: ActionConnection[];
  /** Enable VR-optimized rendering */
  vrMode?: boolean;
  /** Global opacity multiplier */
  opacity?: number;
  /** Line width for connections */
  lineWidth?: number;
}

/**
 * Phase timing boundaries (cumulative percentages of total 500ms duration)
 * - spawn:  0.0-0.2 (100ms) - Line fades in, particle grows at source
 * - travel: 0.2-0.8 (300ms) - Particle travels along bezier curve
 * - impact: 0.8-1.0 (100ms) - Burst effect at target, then fade out
 *
 * Note: Original spec had separate impact (50ms) and fade (50ms),
 * combined here for smoother visual transition.
 */
const PHASE_BOUNDS = {
  spawnEnd: 0.2,    // 100ms / 500ms = 0.2
  travelEnd: 0.8,   // 300ms / 500ms = 0.6, cumulative = 0.8
  impactEnd: 1.0,   // Combined impact+fade = 100ms, cumulative = 1.0
  fadeEnd: 1.0,     // Kept for compatibility
};

export const ActionConnectionsLayer: React.FC<ActionConnectionsLayerProps> = ({
  connections,
  vrMode = false,
  opacity = 1.0,
  lineWidth = 2,
}) => {
  if (connections.length === 0) return null;

  return (
    <group name="action-connections-layer">
      {connections.map((conn) => (
        <ActionConnectionLine
          key={conn.id}
          connection={conn}
          vrMode={vrMode}
          opacity={opacity}
          lineWidth={lineWidth}
        />
      ))}
    </group>
  );
};

/** Single animated action connection */
const ActionConnectionLine: React.FC<{
  connection: ActionConnection;
  vrMode: boolean;
  opacity: number;
  lineWidth: number;
}> = ({ connection, vrMode, opacity, lineWidth }) => {
  const lineRef = useRef<THREE.Line>(null);
  const particleRef = useRef<THREE.Mesh>(null);
  const impactRef = useRef<THREE.Mesh>(null);

  // Fallback positions if not provided
  const sourcePos = useMemo(() => {
    if (connection.sourcePosition) {
      return new THREE.Vector3(
        connection.sourcePosition.x,
        connection.sourcePosition.y,
        connection.sourcePosition.z
      );
    }
    // Fallback: generate position from agent ID (deterministic)
    const hash = connection.sourceAgentId * 1337;
    return new THREE.Vector3(
      Math.sin(hash) * 15,
      Math.cos(hash * 0.7) * 10,
      Math.sin(hash * 0.3) * 15
    );
  }, [connection.sourcePosition, connection.sourceAgentId]);

  const targetPos = useMemo(() => {
    if (connection.targetPosition) {
      return new THREE.Vector3(
        connection.targetPosition.x,
        connection.targetPosition.y,
        connection.targetPosition.z
      );
    }
    // Fallback: generate position from target ID
    const hash = connection.targetNodeId * 2749;
    return new THREE.Vector3(
      Math.sin(hash) * 20,
      Math.cos(hash * 0.5) * 15,
      Math.sin(hash * 0.8) * 20
    );
  }, [connection.targetPosition, connection.targetNodeId]);

  // Generate bezier curve points
  const { curve, points } = useMemo(() => {
    // Control point creates arc effect
    const midPoint = new THREE.Vector3().addVectors(sourcePos, targetPos).multiplyScalar(0.5);
    const distance = sourcePos.distanceTo(targetPos);

    // Perpendicular offset for curve
    const direction = new THREE.Vector3().subVectors(targetPos, sourcePos).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const perpendicular = new THREE.Vector3().crossVectors(direction, up).normalize();

    // Offset based on action type for visual variety
    // Use _actionTypeEnum if available (new API), otherwise derive from string
    const actionTypeIndex = connection._actionTypeEnum ??
      (['query', 'update', 'create', 'delete', 'link', 'transform'].indexOf(connection.actionType as string) || 0);
    const offsetAmount = distance * 0.3 * (1 + (actionTypeIndex * 0.1));
    midPoint.add(perpendicular.multiplyScalar(offsetAmount));
    midPoint.y += distance * 0.15; // Slight upward arc

    const bezier = new THREE.QuadraticBezierCurve3(sourcePos, midPoint, targetPos);
    const curvePoints = bezier.getPoints(vrMode ? 20 : 40);

    return { curve: bezier, points: curvePoints };
  }, [sourcePos, targetPos, connection.actionType, vrMode]);

  // Line geometry
  const lineGeometry = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [points]);

  // Calculate particle position along curve
  const particlePosition = useMemo(() => {
    if (connection.phase === 'spawn') {
      // Particle grows at source during spawn
      return sourcePos.clone();
    }
    if (connection.phase === 'travel') {
      // Particle travels along curve
      const travelProgress = (connection.progress - PHASE_BOUNDS.spawnEnd) /
        (PHASE_BOUNDS.travelEnd - PHASE_BOUNDS.spawnEnd);
      return curve.getPoint(travelProgress);
    }
    // Impact/fade: particle at target
    return targetPos.clone();
  }, [connection.progress, connection.phase, curve, sourcePos, targetPos]);

  // Calculate visual properties based on phase
  const visuals = useMemo(() => {
    const { progress, phase, color } = connection;

    let lineOpacity = 1.0;
    let particleScale = 1.0;
    let impactScale = 0;

    switch (phase) {
      case 'spawn':
        // Line fades in, particle grows (0-100ms)
        lineOpacity = progress / PHASE_BOUNDS.spawnEnd;
        particleScale = progress / PHASE_BOUNDS.spawnEnd;
        break;
      case 'travel':
        // Full visibility during travel (100-400ms)
        lineOpacity = 1.0;
        particleScale = 1.0;
        break;
      case 'impact':
      case 'fade': {
        // Combined impact + fade phase (400-500ms)
        // First half: burst expands, second half: everything fades
        const impactProgress = (progress - PHASE_BOUNDS.travelEnd) / (PHASE_BOUNDS.impactEnd - PHASE_BOUNDS.travelEnd);

        if (impactProgress < 0.5) {
          // Impact burst expansion (first 50ms)
          lineOpacity = 1.0;
          particleScale = 0.5;
          impactScale = impactProgress * 2; // 0 -> 1 over first half
        } else {
          // Fade out (last 50ms)
          const fadeProgress = (impactProgress - 0.5) * 2; // 0 -> 1 over second half
          lineOpacity = 1 - fadeProgress;
          particleScale = 0.5 * (1 - fadeProgress);
          impactScale = 1 - fadeProgress;
        }
        break;
      }
    }

    return {
      lineOpacity: lineOpacity * opacity,
      particleScale,
      impactScale,
      color: new THREE.Color(color),
    };
  }, [connection, opacity]);

  // Animate line opacity
  useFrame(() => {
    if (lineRef.current) {
      const material = lineRef.current.material as THREE.LineBasicMaterial;
      material.opacity = visuals.lineOpacity;
    }
    if (particleRef.current) {
      particleRef.current.position.copy(particlePosition);
      particleRef.current.scale.setScalar(visuals.particleScale * (vrMode ? 0.3 : 0.5));
    }
    if (impactRef.current && visuals.impactScale > 0) {
      impactRef.current.scale.setScalar(visuals.impactScale * 2);
      const mat = impactRef.current.material as THREE.MeshBasicMaterial;
      mat.opacity = visuals.impactScale * 0.5 * opacity;
    }
  });

  return (
    <group>
      {/* Connection line */}
      <primitive
        ref={lineRef}
        object={new THREE.Line(
          lineGeometry,
          new THREE.LineBasicMaterial({
            color: visuals.color,
            transparent: true,
            opacity: visuals.lineOpacity,
            linewidth: lineWidth,
          })
        )}
      />

      {/* Traveling particle */}
      <mesh ref={particleRef} position={particlePosition}>
        <sphereGeometry args={[vrMode ? 0.2 : 0.4, vrMode ? 8 : 16, vrMode ? 8 : 16]} />
        <meshBasicMaterial
          color={visuals.color}
          transparent
          opacity={0.9 * opacity}
        />
      </mesh>

      {/* Glow around particle */}
      {!vrMode && (
        <mesh position={particlePosition}>
          <sphereGeometry args={[0.8, 12, 12]} />
          <meshBasicMaterial
            color={visuals.color}
            transparent
            opacity={0.3 * opacity * visuals.particleScale}
            side={THREE.BackSide}
          />
        </mesh>
      )}

      {/* Impact burst at target */}
      {visuals.impactScale > 0 && (
        <mesh ref={impactRef} position={targetPos}>
          <ringGeometry args={[0.5, 2, vrMode ? 16 : 32]} />
          <meshBasicMaterial
            color={visuals.color}
            transparent
            opacity={visuals.impactScale * 0.5 * opacity}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}
    </group>
  );
};

/** Statistics component for debugging */
export const ActionConnectionsStats: React.FC<{
  connections: ActionConnection[];
}> = ({ connections }) => {
  const stats = useMemo(() => {
    const byType: Record<string, number> = {};
    for (const conn of connections) {
      // Support both new string type and legacy enum
      const typeName = typeof conn.actionType === 'string'
        ? conn.actionType
        : AgentActionType[conn._actionTypeEnum ?? conn.actionType] || 'Unknown';
      byType[typeName] = (byType[typeName] || 0) + 1;
    }
    return byType;
  }, [connections]);

  return (
    <div style={{
      position: 'absolute',
      bottom: 10,
      left: 10,
      background: 'rgba(0,0,0,0.7)',
      color: 'white',
      padding: '8px 12px',
      borderRadius: 4,
      fontSize: 12,
      fontFamily: 'monospace',
    }}>
      <div>Active Actions: {connections.length}</div>
      {Object.entries(stats).map(([type, count]) => (
        <div key={type} style={{ opacity: 0.8 }}>
          {type}: {count}
        </div>
      ))}
    </div>
  );
};

export default ActionConnectionsLayer;
