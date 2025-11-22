/**
 * Class node visualization component
 */

import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFrame, useThree } from '@react-three/fiber';
import { Circle, Text } from '@react-three/drei';
import { useGesture } from '@use-gesture/react';
import * as THREE from 'three';
import { useGraphStore } from '../../../stores/useGraphStore';
import { useUIStore } from '../../../stores/useUIStore';
import type { Node } from '../../../types/graph';

interface ClassNodeProps {
  node: Node;
  onDragStart?: () => void;
  onDragEnd?: () => void;
}

export function ClassNode({ node, onDragStart, onDragEnd }: ClassNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const [dragging, setDragging] = useState(false);
  const navigate = useNavigate();
  const { camera, size } = useThree();

  const { selectNode, hoverNode, selectedNode, updateNodePosition } = useGraphStore();
  const { settings, viewport } = useUIStore();

  const isSelected = selectedNode === node.id;
  const is3D = viewport.mode === '3d';

  // Calculate node radius based on properties
  const instanceCount = node.properties?.instances || 0;
  const baseRadius = 1.6; // Further reduced (8 / 5 = 1.6)
  const radius = instanceCount > 0
    ? Math.sqrt(instanceCount) * 0.1 + baseRadius
    : baseRadius;

  const scaledRadius = radius * settings.nodeScale;

  // Node colors
  const getNodeColor = () => {
    if (dragging) return '#ff6b6b';   // Red when dragging
    if (isSelected) return '#67bc0f'; // Green when selected
    if (hovered) return '#8cd0f0';    // Light blue when hovered
    return '#aaccee';                  // Default blue
  };

  // Drag gesture handler
  const bind = useGesture({
    onDrag: ({ delta: [dx, dy], first, last }) => {
      if (first) {
        setDragging(true);
        useUIStore.getState().setDragging(true);
        onDragStart?.();
      }

      if (meshRef.current) {
        // Convert screen delta to world space
        const scaleFactor = (camera.position.z / size.height) * 2;
        const worldDx = dx * scaleFactor;
        const worldDy = -dy * scaleFactor;

        // Update position
        updateNodePosition(node.id, [
          node.position.x + worldDx,
          node.position.y + worldDy,
          node.position.z
        ]);
      }

      if (last) {
        setDragging(false);
        useUIStore.getState().setDragging(false);
        onDragEnd?.();
      }
    },
    onClick: (e: any) => {
      if (e.event.detail === 2 || e.event.metaKey || e.event.ctrlKey) {
        // Use term_id if available, fallback to label
        const pageId = node.properties?.term_id || node.label;
        navigate(`/page/${encodeURIComponent(pageId)}`);
      } else {
        selectNode(isSelected ? null : node.id);
      }
    }
  });

  const handlePointerOver = (e: any) => {
    e.stopPropagation();
    setHovered(true);
    hoverNode(node.id);
    document.body.style.cursor = dragging ? 'grabbing' : 'grab';
  };

  const handlePointerOut = () => {
    if (!dragging) {
      setHovered(false);
      hoverNode(null);
      document.body.style.cursor = 'auto';
    }
  };

  // Smooth position interpolation
  useFrame(() => {
    if (meshRef.current && !node.pinned) {
      const targetPos = new THREE.Vector3(
        node.position.x,
        node.position.y,
        node.position.z
      );

      meshRef.current.position.lerp(targetPos, 0.1);
    }
  });

  return (
    <group>
      {/* Node circle */}
      <Circle
        ref={meshRef}
        args={[scaledRadius, 32]}
        position={[node.position.x, node.position.y, node.position.z]}
        {...bind()}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
      >
        <meshBasicMaterial
          color={getNodeColor()}
          transparent
          opacity={node.opacity || 1}
        />
      </Circle>

      {/* Node border */}
      <Circle
        args={[scaledRadius + 2, 32]}
        position={[node.position.x, node.position.y, node.position.z - 0.1]}
      >
        <meshBasicMaterial
          color={isSelected ? '#67bc0f' : '#333'}
          transparent
          opacity={isSelected ? 0.5 : 0.2}
          side={THREE.DoubleSide}
        />
      </Circle>

      {/* Node label */}
      {settings.showLabels && (
        <Text
          position={[
            node.position.x,
            node.position.y - scaledRadius - 3,
            node.position.z + 1
          ]}
          fontSize={1.4}
          color="#ffffff"
          outlineWidth={0.1}
          outlineColor="#000000"
          renderOrder={10}
          anchorX="center"
          anchorY="middle"
          maxWidth={scaledRadius * 4}
          textAlign="center"
        >
          {node.label}
        </Text>
      )}

      {/* Instance count indicator */}
      {settings.showNodeDetails && instanceCount > 0 && (
        <Text
          position={[node.position.x, node.position.y, node.position.z + 2]}
          fontSize={1.0}
          color="#ffffff"
          outlineWidth={0.05}
          outlineColor="#000000"
          anchorX="center"
          anchorY="middle"
        >
          {instanceCount}
        </Text>
      )}

      {/* Selection ring */}
      {isSelected && (
        <Circle
          args={[scaledRadius + 8, 32]}
          position={[node.position.x, node.position.y, node.position.z - 0.2]}
        >
          <meshBasicMaterial
            color="#67bc0f"
            transparent
            opacity={0.3}
            side={THREE.DoubleSide}
          />
        </Circle>
      )}

      {/* Hover effect */}
      {hovered && !isSelected && (
        <Circle
          args={[scaledRadius + 5, 32]}
          position={[node.position.x, node.position.y, node.position.z - 0.15]}
        >
          <meshBasicMaterial
            color="#8cd0f0"
            transparent
            opacity={0.2}
            side={THREE.DoubleSide}
          />
        </Circle>
      )}
    </group>
  );
}
