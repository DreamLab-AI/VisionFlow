import * as THREE from 'three';
import { isWebGPURenderer } from '../rendererFactory';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CrystalOrbMaterialResult {
  material: THREE.Material;
  uniforms: {
    time: { value: number };
    glowStrength: { value: number };
    pulseSpeed: { value: number };
  };
  ready: Promise<void>;
}

// ---------------------------------------------------------------------------
// Shared crystal orb material — works on both renderers.
// ---------------------------------------------------------------------------

export function createCrystalOrbMaterial(): CrystalOrbMaterialResult {
  const uniforms = { time: { value: 0 }, glowStrength: { value: 1.2 }, pulseSpeed: { value: 0.8 } };

  const material = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.78, 0.78, 1.0),
    ior: 1.77,
    transmission: isWebGPURenderer ? 0 : 0.8,
    thickness: isWebGPURenderer ? 0 : 0.6,
    roughness: 0.12,
    metalness: 0.0,
    clearcoat: 0.8,
    clearcoatRoughness: 0.05,
    transparent: true,
    opacity: isWebGPURenderer ? 0.7 : 0.9,
    side: THREE.DoubleSide,
    depthWrite: true,
    emissive: new THREE.Color(0.12, 0.12, 0.25),
    emissiveIntensity: 0.3,
    iridescence: isWebGPURenderer ? 0.35 : 0.25,
    iridescenceIOR: 1.4,
    iridescenceThicknessRange: [120, 350] as [number, number],
    ...(isWebGPURenderer ? {
      sheen: 0.4,
      sheenRoughness: 0.2,
      sheenColor: new THREE.Color(0.5, 0.5, 1.0),
      envMapIntensity: 2.0,
      specularIntensity: 1.0,
      specularColor: new THREE.Color(0.85, 0.85, 1.0),
    } : {
      sheen: 0.3,
      sheenRoughness: 0.2,
    }),
  });

  // TSL Fresnel + depth-pulsing emissive upgrade for WebGPU
  const ready = isWebGPURenderer
    ? import('three/tsl').then((tsl: any) => {
        const {
          float, vec3, mix, pow, sin, add,
          dot, normalize: tslNorm, normalView, positionView, positionLocal,
          oneMinus, saturate, time, vertexColor,
        } = tsl;

        // Fresnel rim lighting — cosmic nebula glow at grazing angles
        const vDir = tslNorm(positionView.negate());
        const nDotV = saturate(dot(normalView, vDir));
        const fresnel = pow(oneMinus(nDotV), float(3.0));
        (material as any).opacityNode = mix(float(0.3), float(0.88), fresnel);

        // Depth-pulse emissive: ontology nodes pulse slowly with a cosmic spectrum
        // driven by the pulseSpeed uniform (deeper nodes pulse slower)
        const pulse = sin(time.mul(float(uniforms.pulseSpeed.value)).add(positionLocal.y.mul(2.0))).mul(0.5).add(0.5);
        const baseEmissive = vec3(float(0.12), float(0.12), float(0.3));
        const warmEmissive = vec3(float(0.25), float(0.15), float(0.35));
        const emissiveNode = mix(baseEmissive, warmEmissive, pulse).mul(float(uniforms.glowStrength.value));
        (material as any).emissiveNode = emissiveNode;

        // Per-instance color via vertexColor (instanceColor buffer) + Fresnel brightening
        if (vertexColor) {
          const rimWhite = vec3(1.0, 1.0, 1.0);
          (material as any).colorNode = mix(vertexColor, rimWhite, fresnel.mul(0.25));
        }

        (material as any).needsUpdate = true;
      }).catch((err: any) => console.warn('[CrystalOrbMaterial] TSL upgrade failed:', err))
    : Promise.resolve();

  return { material, uniforms, ready };
}

export function createCrystalOrbGeometry(): THREE.SphereGeometry {
  return new THREE.SphereGeometry(0.5, 32, 32);
}
