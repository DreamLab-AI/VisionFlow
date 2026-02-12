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

  // TSL Fresnel + bioluminescent pulse upgrade for WebGPU
  const ready = isWebGPURenderer
    ? import('three/tsl').then((tsl: any) => {
        const {
          float, vec3, mix, pow, sin,
          dot, normalize: tslNorm, normalView, positionView, positionLocal,
          oneMinus, saturate, time, vertexColor,
        } = tsl;

        // Fresnel rim lighting — bioluminescent green glow at grazing angles
        const vDir = tslNorm(positionView.negate());
        const nDotV = saturate(dot(normalView, vDir));
        const fresnel = pow(oneMinus(nDotV), float(3.0));
        (material as any).opacityNode = mix(float(0.3), float(0.85), fresnel);

        // Bioluminescent heartbeat pulse driven by activityLevel:
        // Active agents pulse faster and brighter, idle agents dim and slow.
        // Phase offset from positionLocal.y creates a traveling wave effect.
        const heartbeat = pow(
          sin(time.mul(float(2.0 * uniforms.activityLevel.value)).add(positionLocal.y.mul(3.0))).mul(0.5).add(0.5),
          float(2.0),
        );
        const baseGreen = vec3(float(0.06), float(0.3), float(0.15));
        const peakGreen = vec3(float(0.15), float(0.6), float(0.3));
        const emissiveNode = mix(baseGreen, peakGreen, heartbeat).mul(float(uniforms.glowStrength.value));
        (material as any).emissiveNode = emissiveNode;

        // Per-instance color via vertexColor (instanceColor buffer) + Fresnel rim
        if (vertexColor) {
          const rimGreen = vec3(0.7, 1.0, 0.8);
          (material as any).colorNode = mix(vertexColor, rimGreen, fresnel.mul(0.3));
        }

        (material as any).needsUpdate = true;
      }).catch((err: any) => console.warn('[AgentCapsuleMaterial] TSL upgrade failed:', err))
    : Promise.resolve();

  return { material, uniforms, ready };
}

export function createAgentCapsuleGeometry(): THREE.CapsuleGeometry {
  return new THREE.CapsuleGeometry(0.3, 0.6, 8, 16);
}
