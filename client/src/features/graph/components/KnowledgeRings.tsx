import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface KnowledgeRingsProps {
  nodes: Array<{ id: string; metadata?: any; position?: { x: number; y: number; z: number } }>;
  graphMode: 'knowledge_graph' | 'ontology' | 'agent';
  perNodeVisualModeMap: Map<string, string>;
  nodePositionsRef: React.RefObject<Float32Array | null>;
  nodeIdToIndexMap: Map<string, number>;
  connectionCountMap: Map<string, number>;
  edges: any[];
  hierarchyMap: Map<string, any>;
  settings: any;
}

const RING_COLOR = '#4FC3F7';
const RING_OPACITY = 0.6;
const BASE_SCALE = 1.5;
const ROTATION_SPEED = 0.5;

export const KnowledgeRings: React.FC<KnowledgeRingsProps> = ({
  nodes,
  graphMode,
  perNodeVisualModeMap,
  nodePositionsRef,
  nodeIdToIndexMap,
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  // Pre-allocated objects to avoid GC pressure
  const tempMatrix = useRef(new THREE.Matrix4());
  const tempPosition = useRef(new THREE.Vector3());
  const tempQuaternion = useRef(new THREE.Quaternion());
  const tempScale = useRef(new THREE.Vector3());
  const tempEuler = useRef(new THREE.Euler());

  const knowledgeNodes = useMemo(() => {
    return nodes.filter((node) => {
      const mode = perNodeVisualModeMap.get(node.id);
      // Only show rings for nodes positively identified as knowledge_graph.
      // Nodes without an explicit visual mode tag should NOT get rings —
      // this prevents ontology/agent nodes from getting rings when
      // the global graphMode happens to be 'knowledge_graph'.
      return mode === 'knowledge_graph';
    });
  }, [nodes, perNodeVisualModeMap]);

  const geometry = useMemo(() => {
    return new THREE.TorusGeometry(1.2, 0.03, 32, 64);
  }, []);

  const material = useMemo(() => {
    return new THREE.MeshBasicMaterial({
      color: RING_COLOR,
      transparent: true,
      opacity: RING_OPACITY,
      depthWrite: false,
      toneMapped: false,
    });
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      geometry.dispose();
      material.dispose();
    };
  }, [geometry, material]);

  // Register with bloom layers
  useEffect(() => {
    const mesh = meshRef.current;
    if (mesh) {
      if (!mesh.layers) {
        mesh.layers = new THREE.Layers();
      }
      mesh.layers.set(0);
      mesh.layers.enable(1);
    }
  }, [knowledgeNodes.length]);

  useFrame((state) => {
    const mesh = meshRef.current;
    const positions = nodePositionsRef.current;
    if (!mesh || !positions) return;

    const elapsed = state.clock.elapsedTime;
    const mat4 = tempMatrix.current;
    const pos = tempPosition.current;
    const quat = tempQuaternion.current;
    const scl = tempScale.current;
    const euler = tempEuler.current;

    for (let i = 0; i < knowledgeNodes.length; i++) {
      const node = knowledgeNodes[i];
      const idx = nodeIdToIndexMap.get(node.id);

      if (idx === undefined) continue;

      const offset = idx * 3;
      pos.set(positions[offset], positions[offset + 1], positions[offset + 2]);

      // Per-ring tilt using node index for variety
      const tiltOffset = (i * 0.4) % (Math.PI * 2);
      euler.set(
        Math.sin(tiltOffset) * 0.3,
        elapsed * ROTATION_SPEED + tiltOffset,
        Math.cos(tiltOffset) * 0.2,
      );
      quat.setFromEuler(euler);

      scl.set(BASE_SCALE, BASE_SCALE, BASE_SCALE);

      mat4.compose(pos, quat, scl);
      mesh.setMatrixAt(i, mat4);
    }

    mesh.instanceMatrix.needsUpdate = true;
    mesh.count = knowledgeNodes.length;
  });

  if (knowledgeNodes.length === 0) {
    return null;
  }

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, knowledgeNodes.length]}
      frustumCulled={false}
      renderOrder={4}
    />
  );
};
