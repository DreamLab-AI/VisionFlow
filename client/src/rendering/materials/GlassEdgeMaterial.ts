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

export function createGlassEdgeMaterial(baseColor?: string | THREE.Color): GlassEdgeMaterialResult {
  const uniforms = { time: { value: 0 }, flowSpeed: { value: 0.5 } };

  const resolvedColor = baseColor
    ? (baseColor instanceof THREE.Color ? baseColor : new THREE.Color(baseColor))
    : new THREE.Color(0.35, 0.55, 0.85);

  const material = new THREE.MeshPhysicalMaterial({
    color: resolvedColor,
    ior: 1.5,
    transmission: isWebGPURenderer ? 0 : 0.7,
    thickness: isWebGPURenderer ? 0 : 0.15,
    roughness: 0.05,
    metalness: 0.0,
    transparent: true,
    opacity: isWebGPURenderer ? 0.6 : 0.4,
    side: THREE.DoubleSide,
    depthWrite: true,
    polygonOffset: true,
    polygonOffsetFactor: 1, // Push edges slightly behind nodes at connection points
    polygonOffsetUnits: 1,
    // Derive emissive from the resolved base color (30% intensity) so edges
    // glow in their own hue rather than a fixed blue.
    emissive: resolvedColor.clone().multiplyScalar(isWebGPURenderer ? 0.5 : 0.3),
    emissiveIntensity: isWebGPURenderer ? 0.6 : 0.3,
    iridescence: isWebGPURenderer ? 0.2 : 0.1,
    iridescenceIOR: 1.3,
    iridescenceThicknessRange: [100, 250] as [number, number],
    ...(isWebGPURenderer ? {
      sheen: 0.3,
      sheenRoughness: 0.1,
      sheenColor: resolvedColor.clone().multiplyScalar(0.7),
      specularIntensity: 0.8,
      specularColor: resolvedColor.clone().lerp(new THREE.Color(1, 1, 1), 0.3),
    } : {}),
  });

  // TSL ENABLED (r183+) with PBR fallback — Fresnel emissive and opacity nodes
  // are applied asynchronously on WebGPU; standard PBR + per-frame emissive
  // modulation in GlassEdges useFrame remains the active fallback path.
  const ready = (async () => {
    if (!isWebGPURenderer) return;
    try {
      const { float, vec3, normalize, positionView, normalView, dot, saturate, pow, oneMinus } = await import('three/tsl') as any;
      const viewDir = normalize(positionView.negate());
      const fresnel = pow(oneMinus(saturate(dot(normalView, viewDir))), float(3.0));
      const emissiveNode = vec3(float(0.1), float(0.15), float(0.3)).mul(fresnel);
      const opacityNode = float(0.3).add(fresnel.mul(0.5));
      (material as any).emissiveNode = emissiveNode;
      (material as any).opacityNode = opacityNode;
      (material as any).needsUpdate = true;
      console.log('[GlassEdgeMaterial] TSL nodes enabled (r183+)');
    } catch (err) {
      console.warn('[GlassEdgeMaterial] TSL upgrade failed, using PBR fallback:', err);
    }
  })();

  return { material, uniforms, ready };
}

/**
 * Creates a unit-length cylinder geometry for glass tube edges.
 * Stretch to actual edge length via instance/object matrix.
 */
export function createGlassEdgeGeometry(radius: number = 0.03): THREE.CylinderGeometry {
  return new THREE.CylinderGeometry(radius, radius, 1, 8);
}
