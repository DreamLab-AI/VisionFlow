/**
 * Property edge visualization component
 */

import { useMemo } from 'react';
import { Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import { useGraphStore } from '../../../stores/useGraphStore';
import { useUIStore } from '../../../stores/useUIStore';
import type { Edge } from '../../../types/graph';

interface PropertyEdgeProps {
  edge: Edge;
}

export function PropertyEdge({ edge }: PropertyEdgeProps) {
  const { nodes, selectedEdge } = useGraphStore();
  const { settings } = useUIStore();

  const isSelected = selectedEdge === edge.id;

  // Get source and target nodes
  const sourceNode = nodes.get(edge.source);
  const targetNode = nodes.get(edge.target);

  // Calculate line points
  const points = useMemo(() => {
    if (!sourceNode || !targetNode) return null;

    // Safely check if positions exist and are valid
    if (!sourceNode.position || !targetNode.position) return null;

    // Validate positions are valid numbers (prevent NaN errors during initialization)
    const sx = sourceNode.position.x;
    const sy = sourceNode.position.y;
    const sz = sourceNode.position.z;
    const tx = targetNode.position.x;
    const ty = targetNode.position.y;
    const tz = targetNode.position.z;

    if (
      typeof sx !== 'number' || typeof sy !== 'number' || typeof sz !== 'number' ||
      typeof tx !== 'number' || typeof ty !== 'number' || typeof tz !== 'number' ||
      isNaN(sx) || isNaN(sy) || isNaN(sz) ||
      isNaN(tx) || isNaN(ty) || isNaN(tz)
    ) {
      return null;
    }

    return [
      new THREE.Vector3(sx, sy, sz),
      new THREE.Vector3(tx, ty, tz)
    ];
  }, [sourceNode, targetNode]);

  // Calculate midpoint for label
  const midpoint = useMemo(() => {
    if (!points) return null;
    return new THREE.Vector3(
      (points[0].x + points[1].x) / 2,
      (points[0].y + points[1].y) / 2,
      (points[0].z + points[1].z) / 2 + 1
    );
  }, [points]);

  // Calculate arrow direction for directed edges
  const arrowDirection = useMemo(() => {
    if (!points) return null;
    const direction = new THREE.Vector3()
      .subVectors(points[1], points[0])
      .normalize();
    return direction;
  }, [points]);

  // Edge styling based on type
  const getEdgeColor = () => {
    if (isSelected) return '#67bc0f';
    return '#ff0000'; // Red for all edges as requested
  };

  const getLineWidth = () => {
    const base = 0.5; // Thinner lines
    return base * settings.edgeWidth;
  };

  // Get line dash pattern for different edge types
  const getLineDash = () => {
    switch (edge.type) {
      case 'disjoint':
        return [5, 5];  // Dashed
      default:
        return undefined;
    }
  };

  if (!points || !sourceNode || !targetNode || !midpoint || !arrowDirection) return null;

  return (
    <group>
      {/* Main edge line */}
      <Line
        points={points}
        color={getEdgeColor()}
        lineWidth={getLineWidth()}
        transparent
        opacity={0.5} // 50% alpha
        dashed={!!getLineDash()}
        dashSize={getLineDash()?.[0]}
        gapSize={getLineDash()?.[1]}
      />

      {/* Edge label */}
      {settings.showLabels && edge.label && (
        <Text
          position={midpoint}
          fontSize={10}
          color="#ffffff"
          outlineWidth={0.1}
          outlineColor="#000000"
          renderOrder={10}
          anchorX="center"
          anchorY="middle"
          maxWidth={200}
          textAlign="center"
        >
          {edge.label}
        </Text>
      )}

      {/* Arrow indicator for directed edges */}
      {edge.type !== 'disjoint' && (
        <mesh
          position={[
            targetNode.position.x - arrowDirection.x * 25,
            targetNode.position.y - arrowDirection.y * 25,
            targetNode.position.z
          ]}
          rotation={[0, 0, Math.atan2(arrowDirection.y, arrowDirection.x)]}
        >
          <coneGeometry args={[5, 10, 8]} />
          <meshBasicMaterial
            color={getEdgeColor()}
            transparent
            opacity={0.5}
          />
        </mesh>
      )}

      {/* Selection highlight */}
      {isSelected && (
        <Line
          points={points}
          color="#67bc0f"
          lineWidth={getLineWidth() + 4}
          transparent
          opacity={0.3}
        />
      )}
    </group>
  );
}
