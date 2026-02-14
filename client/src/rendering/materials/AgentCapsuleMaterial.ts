import * as THREE from 'three';
import { isWebGPURenderer } from '../rendererFactory';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AgentCapsuleMaterialResult {
  material: THREE.Material;
  uniforms: {
    time: { value: number };
    glowStrength: { value: number };
    activityLevel: { value: number };
  };
  ready: Promise<void>;
}

// ---------------------------------------------------------------------------
// Shared agent capsule material — works on both renderers.
// ---------------------------------------------------------------------------

export function createAgentCapsuleMaterial(): AgentCapsuleMaterialResult {
  const uniforms = { time: { value: 0 }, glowStrength: { value: 1.0 }, activityLevel: { value: 1.0 } };

  const material = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.82, 0.95, 0.86),
    ior: 1.58,
    transmission: isWebGPURenderer ? 0 : 0.5,
    thickness: isWebGPURenderer ? 0 : 0.4,
    roughness: 0.15,
    metalness: 0.0,
    clearcoat: 0.8,
    clearcoatRoughness: 0.1,
    transparent: true,
    opacity: isWebGPURenderer ? 0.7 : 0.85,
    side: THREE.DoubleSide,
    depthWrite: true,
    emissive: new THREE.Color(0.08, 0.4, 0.2),
    emissiveIntensity: 0.25,
    iridescence: isWebGPURenderer ? 0.25 : 0.15,
    iridescenceIOR: 1.5,
    iridescenceThicknessRange: [80, 300] as [number, number],
    ...(isWebGPURenderer ? {
      sheen: 0.4,
      sheenRoughness: 0.2,
      sheenColor: new THREE.Color(0.3, 0.8, 0.5),
      envMapIntensity: 1.8,
      specularIntensity: 1.0,
      specularColor: new THREE.Color(0.8, 1.0, 0.85),
    } : {}),
  });

  // TSL DISABLED: Adding colorNode/emissiveNode/opacityNode to MeshPhysicalMaterial
  // triggers shader recompilation that breaks InstancedMesh draw calls on WebGPU r182.
  // Visual quality achieved through standard PBR properties + per-frame emissive
  // modulation in GemNodes useFrame instead.
  const ready = Promise.resolve();

  return { material, uniforms, ready };
}

export function createAgentCapsuleGeometry(): THREE.CapsuleGeometry {
  return new THREE.CapsuleGeometry(0.3, 0.6, 8, 16);
}
