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
    opacity: isWebGPURenderer ? 0.35 : 0.4,
    side: THREE.DoubleSide,
    depthWrite: false, // Edges behind nodes
    emissive: new THREE.Color(0.1, 0.18, 0.3),
    emissiveIntensity: 0.2,
    ...(isWebGPURenderer ? {
      sheen: 0.3,
      sheenRoughness: 0.1,
      sheenColor: new THREE.Color(0.4, 0.6, 1.0),
      specularIntensity: 0.8,
      specularColor: new THREE.Color(0.7, 0.85, 1.0),
    } : {}),
  });

  // TSL Fresnel upgrade for WebGPU (onBeforeCompile injects GLSL which WebGPU ignores)
  const ready = isWebGPURenderer
    ? import('three/tsl').then((tsl: any) => {
        const { float, mix, pow, dot, normalize: tslNorm, normalView, positionView, oneMinus, saturate } = tsl;
        const vDir = tslNorm(positionView.negate());
        const nDotV = saturate(dot(normalView, vDir));
        const fresnel = pow(oneMinus(nDotV), float(3.0));
        (material as any).opacityNode = mix(float(0.12), float(0.5), fresnel);
        (material as any).needsUpdate = true;
      }).catch((err: any) => console.warn('[GlassEdgeMaterial] TSL upgrade failed:', err))
    : Promise.resolve();

  return { material, uniforms, ready };
}

/**
 * Creates a unit-length cylinder geometry for glass tube edges.
 * Stretch to actual edge length via instance/object matrix.
 */
export function createGlassEdgeGeometry(radius: number = 0.03): THREE.CylinderGeometry {
  return new THREE.CylinderGeometry(radius, radius, 1, 8);
}
