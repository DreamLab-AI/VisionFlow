import * as THREE from 'three';
import { isWebGPURenderer } from '../rendererFactory';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GemMaterialResult {
  material: THREE.Material;
  uniforms: {
    time: { value: number };
    glowStrength: { value: number };
  };
  ready: Promise<void>;
}

// ---------------------------------------------------------------------------
// Base gem material — MeshPhysicalMaterial that works on both renderers.
// WebGL gets transmission; WebGPU uses sheen + Fresnel + emissive instead.
// This is always available as the synchronous fallback.
// ---------------------------------------------------------------------------

export function createGemNodeMaterial(): GemMaterialResult {
  const uniforms = { time: { value: 0 }, glowStrength: { value: 1.5 } };
  console.log('[GemNodeMaterial] creating, isWebGPURenderer=', isWebGPURenderer);

  // WebGPU: mid-tone crystalline base so iridescence has reflected light to modulate.
  // Emissive kept subtle — the per-instance TSL emissiveNode provides the glow.
  // WebGL: classic glass gem with transmission.
  const material = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(isWebGPURenderer ? 0.35 : 0.88, isWebGPURenderer ? 0.38 : 0.92, isWebGPURenderer ? 0.5 : 1.0),
    ior: 2.42,
    transmission: isWebGPURenderer ? 0 : 0.6,
    thickness: isWebGPURenderer ? 0 : 0.5,
    roughness: isWebGPURenderer ? 0.08 : 0.08,
    metalness: isWebGPURenderer ? 0.15 : 0.0,
    clearcoat: 1.0,
    clearcoatRoughness: 0.02,
    transparent: true,
    opacity: isWebGPURenderer ? 0.7 : 0.85,
    side: THREE.DoubleSide,
    depthWrite: true,
    emissive: new THREE.Color(isWebGPURenderer ? 0.03 : 0.15, isWebGPURenderer ? 0.04 : 0.18, isWebGPURenderer ? 0.1 : 0.3),
    emissiveIntensity: isWebGPURenderer ? 0.4 : 0.3,
    iridescence: isWebGPURenderer ? 1.0 : 0.3,
    iridescenceIOR: isWebGPURenderer ? 1.8 : 1.3,
    iridescenceThicknessRange: [100, 600] as [number, number],
    ...(isWebGPURenderer ? {
      sheen: 0.2,
      sheenRoughness: 0.15,
      sheenColor: new THREE.Color(0.4, 0.5, 0.8),
      envMapIntensity: 1.8,
      specularIntensity: 1.0,
      specularColor: new THREE.Color(0.85, 0.9, 1.0),
    } : {}),
  });

  // TSL Fresnel upgrade deferred to createTslGemMaterial (metadata version).
  // The initial Fresnel-only upgrade was conflicting with the metadata TSL pass,
  // and the double-needsUpdate can break shader compilation on WebGPU InstancedMesh.
  const ready = Promise.resolve();

  return { material, uniforms, ready };
}

// ---------------------------------------------------------------------------
// TSL metadata-driven material (WebGPU only)
//
// Reads per-instance metadata from a DataTexture (Nx1, RGBA, Float):
//   .r = quality   (0-1)  → emissive glow brightness
//   .g = authority  (0-1)  → pulse speed + base opacity
//   .b = connections (0-1) → emissive warmth (blue → orange)
//   .a = recency   (0-1)  → overall vibrancy
//
// Texture sampled via instanceIndex — avoids InstancedBufferAttribute which
// causes drawIndexed(Infinity) crash in WebGPU backend.
// instanceColor is read for per-node tinting.
// No backdropNode/viewportSharedTexture — avoids the transmission crash.
// ---------------------------------------------------------------------------

/**
 * Augment an existing MeshPhysicalMaterial with TSL metadata-driven emissive
 * and opacity nodes.  This follows the same pattern as GlassEdgeMaterial:
 * add TSL nodes to the EXISTING material rather than creating a new
 * MeshPhysicalNodeMaterial — the latter silently fails with InstancedMesh
 * on WebGPU.
 *
 * Per-instance color is handled by the standard material's native
 * instanceColor support (setColorAt) — no colorNode override needed.
 */
export async function createTslGemMaterial(
  material: THREE.MeshPhysicalMaterial,
  metadataTexture: THREE.DataTexture,
  instanceCount: number,
): Promise<boolean> {
  if (!isWebGPURenderer) return false;

  try {
    const tslMod = await import('three/tsl') as any;

    const {
      float, vec2, vec3,
      mix, pow, sin, add, sub,
      dot, normalize, oneMinus, saturate, fract,
      time, instanceIndex,
      normalView, positionView,
      texture: tslTexture,
    } = tslMod;

    // --- Per-instance metadata via DataTexture ---
    const texW = float(instanceCount);
    const texU = float(instanceIndex).add(0.5).div(texW);
    const meta = tslTexture(metadataTexture, vec2(texU, float(0.5)));
    const quality = meta.x;
    const authority = meta.y;
    const connections = meta.z;
    const recency = meta.w;

    // --- Per-instance unique phase ---
    const rawIndex = float(instanceIndex);
    const phase = fract(sin(rawIndex.mul(43758.5453))).mul(6.2831);

    // --- Fresnel rim lighting ---
    const viewDir = normalize(positionView.negate());
    const nDotV = saturate(dot(normalView, viewDir));
    const fresnel = pow(oneMinus(nDotV), float(3.0));

    // --- Authority-driven pulse ---
    const pulseSpeed = mix(float(0.8), float(3.0), authority);
    const pulse = sin(time.mul(pulseSpeed).add(phase)).mul(0.5).add(0.5);

    // --- Quality drives emissive brightness ---
    const qualityBrightness = mix(float(0.3), float(0.8), quality);
    const recencyBoost = mix(float(0.5), float(1.0), recency);
    const warmShift = connections.mul(0.25);

    const baseEmissive = vec3(
      add(float(0.25), warmShift),
      float(0.30),
      sub(float(0.50), warmShift.mul(0.5)),
    );

    const emissiveNode = baseEmissive
      .mul(qualityBrightness)
      .mul(mix(float(0.4), float(1.0), pulse))
      .mul(recencyBoost);

    // --- Opacity: Fresnel rim + authority-based solidity ---
    const baseAlpha = mix(float(0.55), float(0.85), authority);
    const opacityNode = mix(baseAlpha, float(0.95), fresnel);

    // Add TSL nodes to the EXISTING material (GlassEdges pattern).
    // Do NOT set colorNode — the standard material reads instanceColor natively.
    (material as any).emissiveNode = emissiveNode;
    (material as any).opacityNode = opacityNode;
    (material as any).needsUpdate = true;

    console.log('[GemNodeMaterial] TSL metadata nodes applied to existing material');
    return true;
  } catch (err) {
    console.warn('[GemNodeMaterial] TSL metadata upgrade failed:', err);
    return false;
  }
}

// ---------------------------------------------------------------------------
// Disposal helper
// ---------------------------------------------------------------------------

export function disposeGemMaterial(result: GemMaterialResult): void {
  if (result.material) {
    result.material.dispose();
  }
}

// ---------------------------------------------------------------------------
// Geometry helper — faceted gem (detail=2 for finer facets)
// ---------------------------------------------------------------------------

export function createGemGeometry(): THREE.IcosahedronGeometry {
  return new THREE.IcosahedronGeometry(0.5, 2);
}
