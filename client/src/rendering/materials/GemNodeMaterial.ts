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

  const material = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.88, 0.92, 1.0),
    ior: 2.42,
    transmission: isWebGPURenderer ? 0 : 0.6,
    thickness: isWebGPURenderer ? 0 : 0.5,
    roughness: 0.08,
    metalness: 0.0,
    clearcoat: 1.0,
    clearcoatRoughness: 0.03,
    transparent: true,
    opacity: isWebGPURenderer ? 0.75 : 0.85,
    side: THREE.DoubleSide,
    depthWrite: true,
    emissive: new THREE.Color(0.15, 0.18, 0.3),
    emissiveIntensity: 0.3,
    ...(isWebGPURenderer ? {
      sheen: 0.5,
      sheenRoughness: 0.15,
      sheenColor: new THREE.Color(0.6, 0.7, 1.0),
      envMapIntensity: 2.0,
      specularIntensity: 1.2,
      specularColor: new THREE.Color(1.0, 1.0, 1.0),
    } : {}),
  });

  // TSL Fresnel upgrade for WebGPU (onBeforeCompile injects GLSL which WebGPU ignores)
  const ready = isWebGPURenderer
    ? import('three/tsl').then((tsl: any) => {
        const { float, mix, pow, abs: tslAbs, dot, normalize: tslNorm, normalView, positionView, oneMinus, saturate } = tsl;
        const vDir = tslNorm(positionView.negate());
        const nDotV = saturate(dot(normalView, vDir));
        const fresnel = pow(oneMinus(nDotV), float(3.0));
        (material as any).opacityNode = mix(float(0.35), float(0.9), fresnel);
        (material as any).needsUpdate = true;
      }).catch((err: any) => console.warn('[GemNodeMaterial] TSL upgrade failed:', err))
    : Promise.resolve();

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

export async function createTslGemMaterial(
  metadataTexture: THREE.DataTexture,
  instanceCount: number,
): Promise<THREE.Material | null> {
  if (!isWebGPURenderer) return null;

  try {
    const [webgpuMod, tslMod] = await Promise.all([
      import('three/webgpu') as any,
      import('three/tsl') as any,
    ]);

    const MeshPhysicalNodeMaterial = webgpuMod.MeshPhysicalNodeMaterial;
    if (!MeshPhysicalNodeMaterial) throw new Error('MeshPhysicalNodeMaterial not found');

    const {
      float, vec2, vec3,
      mix, pow, sin, add, sub,
      dot, normalize, oneMinus, saturate, fract,
      time, instanceIndex, instanceColor,
      normalView, positionView,
      texture: tslTexture,
    } = tslMod;

    // --- Per-instance metadata via DataTexture (avoids InstancedBufferAttribute drawIndexed crash) ---
    const texW = float(instanceCount);
    const texU = float(instanceIndex).add(0.5).div(texW);
    const meta = tslTexture(metadataTexture, vec2(texU, float(0.5)));
    const quality = meta.x;       // 0-1
    const authority = meta.y;     // 0-1
    const connections = meta.z;   // 0-1
    const recency = meta.w;       // 0-1

    // --- Per-instance unique phase (pseudo-hash via sin) ---
    const rawIndex = float(instanceIndex);
    const phase = fract(sin(rawIndex.mul(43758.5453))).mul(6.2831);

    // --- Fresnel rim lighting ---
    const viewDir = normalize(positionView.negate());
    const nDotV = saturate(dot(normalView, viewDir));
    const fresnel = pow(oneMinus(nDotV), float(3.0));

    // --- Authority-driven pulse speed (slow for low, fast for high) ---
    const pulseSpeed = mix(float(0.8), float(3.0), authority);
    const pulse = sin(time.mul(pulseSpeed).add(phase)).mul(0.5).add(0.5);

    // --- Quality drives emissive brightness ---
    const qualityBrightness = mix(float(0.08), float(0.5), quality);

    // --- Recency drives overall vibrancy ---
    const recencyBoost = mix(float(0.5), float(1.0), recency);

    // --- Connection density shifts emissive toward warm ---
    const warmShift = connections.mul(0.25);

    // Base emissive: blue-white, warmer with more connections
    const baseEmissive = vec3(
      add(float(0.12), warmShift),          // R: warmer
      float(0.15),                           // G: steady
      sub(float(0.28), warmShift.mul(0.5)), // B: cooler
    );

    // Final emissive = base * quality * pulse * recency
    const emissiveNode = baseEmissive
      .mul(qualityBrightness)
      .mul(mix(float(0.4), float(1.0), pulse))
      .mul(recencyBoost);

    // --- Opacity: Fresnel rim + authority-based solidity ---
    const baseAlpha = mix(float(0.35), float(0.55), authority);
    const opacityNode = mix(baseAlpha, float(0.92), fresnel);

    // --- Color: instanceColor tint + Fresnel rim brightening ---
    const instCol = instanceColor; // per-node tint from computeColor()
    const rimWhite = vec3(1.0, 1.0, 1.0);
    const colorNode = mix(instCol, rimWhite, fresnel.mul(0.35));

    // --- Assemble the material ---
    const mat = new MeshPhysicalNodeMaterial();
    mat.colorNode = colorNode;
    mat.emissiveNode = emissiveNode;
    mat.opacityNode = opacityNode;

    mat.ior = 2.42;
    mat.roughness = 0.08;
    mat.metalness = 0.0;
    mat.clearcoat = 1.0;
    mat.clearcoatRoughness = 0.03;
    mat.transparent = true;
    mat.side = THREE.DoubleSide;
    mat.depthWrite = true;

    console.log('[GemNodeMaterial] TSL metadata material ready');
    return mat;
  } catch (err) {
    console.warn('[GemNodeMaterial] TSL material unavailable:', err);
    return null;
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
