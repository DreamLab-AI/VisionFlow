import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import {
  createAgentCapsuleMaterial,
  createAgentCapsuleGeometry,
} from '../../../rendering/materials/AgentCapsuleMaterial';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AgentGemsProps {
  agents: Array<{
    id: string;
    position?: { x: number; y: number; z: number };
    status?: string;
    agentType?: string;
    workload?: number;
  }>;
  connections?: Array<{
    source: string;
    target: string;
  }>;
}

// ---------------------------------------------------------------------------
// Status color palette
// ---------------------------------------------------------------------------

const STATUS_COLORS: Record<string, THREE.Color> = {
  active: new THREE.Color('#2ECC71'),
  busy:   new THREE.Color('#F39C12'),
  idle:   new THREE.Color('#95A5A6'),
  error:  new THREE.Color('#E74C3C'),
  queen:  new THREE.Color('#FFD700'),
};

const DEFAULT_COLOR = STATUS_COLORS.active;

function statusToColor(status: string | undefined): THREE.Color {
  if (!status) return DEFAULT_COLOR;
  return STATUS_COLORS[status] ?? DEFAULT_COLOR;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const _tmpMat = new THREE.Matrix4();
const _tmpPos = new THREE.Vector3();
const _tmpScale = new THREE.Vector3();
const _identityQuat = new THREE.Quaternion(); // Identity quaternion for compose

export const AgentGems: React.FC<AgentGemsProps> = ({ agents }) => {
  const maxCount = Math.max(agents.length, 1);
  const meshRef = useRef<THREE.InstancedMesh | null>(null);

  const { mesh, uniforms, geometry, matResult } = useMemo(() => {
    const geo = createAgentCapsuleGeometry();
    const result = createAgentCapsuleMaterial();
    const inst = new THREE.InstancedMesh(geo, result.material, maxCount);
    inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    inst.frustumCulled = false;

    // Initialize instance colors
    const col = new THREE.Color(0.18, 0.8, 0.44);
    for (let i = 0; i < maxCount; i++) {
      inst.setColorAt(i, col);
    }
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true;

    meshRef.current = inst;
    return { mesh: inst, uniforms: result.uniforms, geometry: geo, matResult: result };
  }, [maxCount]);

  // Bloom layer
  useEffect(() => {
    mesh.layers.enable(1);
  }, [mesh]);

  // Cleanup: dispose geometry, material, and instanced mesh GPU resources
  useEffect(() => {
    return () => {
      geometry.dispose();
      matResult.material.dispose();
      if (meshRef.current) {
        meshRef.current.dispose();
      }
    };
  }, [geometry, matResult]);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    uniforms.time.value = t;

    // Counting loop instead of filter() to avoid per-frame allocation
    let activeCount = 0;
    for (const a of agents) {
      if (a.status === 'active' || a.status === 'busy') activeCount++;
    }
    uniforms.activityLevel.value = agents.length > 0 ? activeCount / agents.length : 0;

    // Subtle emissive pulse based on activity
    const pulse = Math.pow((Math.sin(t * Math.PI) + 1) * 0.5, 4);
    const currentMat = mesh.material as THREE.MeshPhysicalMaterial;
    if (currentMat.emissiveIntensity !== undefined) {
      currentMat.emissiveIntensity = 0.15 + pulse * uniforms.activityLevel.value * 0.2;
    }

    for (let i = 0; i < agents.length; i++) {
      const agent = agents[i];
      const { x, y, z } = agent.position ?? { x: 0, y: 0, z: 0 };
      const workload = agent.workload ?? 0;
      const s = 1 + workload * 0.5;

      _tmpPos.set(x, y, z);
      _tmpScale.set(s, s, s);
      _tmpMat.compose(_tmpPos, _identityQuat, _tmpScale);
      mesh.setMatrixAt(i, _tmpMat);

      const col = statusToColor(agent.status);
      mesh.setColorAt(i, col);
    }

    mesh.count = agents.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  if (agents.length === 0) return null;

  return <primitive object={mesh} />;
};

export default AgentGems;
