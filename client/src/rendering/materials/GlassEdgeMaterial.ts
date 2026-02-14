import * as THREE from 'three';
import { isWebGPURenderer } from '../rendererFactory';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GlassEdgeMaterialResult {
  material: THREE.Material;
  uniforms: { time: { value: number }; flowSpeed: { value: number } };
  ready: Promise<void>;
}

// ---------------------------------------------------------------------------
// Shared glass edge material — works on both renderers.
// Edges stay thin and subtle; depth-write off so they don't z-fight with nodes.
// ---------------------------------------------------------------------------

export function createGlassEdgeMaterial(): GlassEdgeMaterialResult {
  const uniforms = { time: { value: 0 }, flowSpeed: { value: 0.5 } };

  const material = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.7, 0.85, 1.0),
    ior: 1.5,
    transmission: isWebGPURenderer ? 0 : 0.7,
    thickness: isWebGPURenderer ? 0 : 0.15,
    roughness: 0.05,
    metalness: 0.0,
    transparent: true,
    opacity: isWebGPURenderer ? 0.6 : 0.4,
    side: THREE.DoubleSide,
    depthWrite: false, // Edges behind nodes
    emissive: new THREE.Color(isWebGPURenderer ? 0.2 : 0.1, isWebGPURenderer ? 0.3 : 0.18, isWebGPURenderer ? 0.55 : 0.3),
    emissiveIntensity: isWebGPURenderer ? 0.8 : 0.3,
    iridescence: isWebGPURenderer ? 0.2 : 0.1,
    iridescenceIOR: 1.3,
    iridescenceThicknessRange: [100, 250] as [number, number],
    ...(isWebGPURenderer ? {
      sheen: 0.3,
      sheenRoughness: 0.1,
      sheenColor: new THREE.Color(0.4, 0.6, 1.0),
      specularIntensity: 0.8,
      specularColor: new THREE.Color(0.7, 0.85, 1.0),
    } : {}),
  });

  // TSL DISABLED: Adding emissiveNode/opacityNode to MeshPhysicalMaterial triggers
  // shader recompilation that breaks InstancedMesh draw calls on WebGPU r182.
  // Visual quality achieved through standard PBR properties + per-frame emissive
  // modulation in GlassEdges useFrame instead.
  const ready = Promise.resolve();

  return { material, uniforms, ready };
}

/**
 * Creates a unit-length cylinder geometry for glass tube edges.
 * Stretch to actual edge length via instance/object matrix.
 */
export function createGlassEdgeGeometry(radius: number = 0.03): THREE.CylinderGeometry {
  return new THREE.CylinderGeometry(radius, radius, 1, 8);
}
