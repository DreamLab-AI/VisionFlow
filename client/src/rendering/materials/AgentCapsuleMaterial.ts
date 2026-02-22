import * as THREE from 'three';
import { isWebGPURenderer } from '../rendererFactory';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('AgentCapsuleMaterial');

// ---------------------------------------------------------------------------
// TSL node-augmented material interface
// ---------------------------------------------------------------------------

/** Runtime-augmented material properties added by Three.js TSL (node-based shading). */
interface TSLNodeProperties {
  emissiveNode: unknown;
  opacityNode: unknown;
  needsUpdate: boolean;
}

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

  // TSL ENABLED (r183+) with PBR fallback — Fresnel emissive and opacity nodes
  // are applied asynchronously on WebGPU; standard PBR + per-frame emissive
  // modulation in GemNodes useFrame remains the active fallback path.
  const ready = (async () => {
    if (!isWebGPURenderer) return;
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Three.js TSL module exports are complex node builder types with no stable public API
      const { float, vec3, normalize, positionView, normalView, dot, saturate, pow, oneMinus, sin, time } = await import('three/tsl') as any;
      const viewDir = normalize(positionView.negate());
      const fresnel = pow(oneMinus(saturate(dot(normalView, viewDir))), float(3.0));
      const pulse = sin(time.mul(float(1.0))).mul(0.5).add(0.5);
      const emissiveNode = vec3(float(0.08), float(0.4), float(0.2)).mul(fresnel).mul(pulse.mul(0.3).add(0.7));
      const opacityNode = float(0.5).add(fresnel.mul(0.45));
      const augmented = material as unknown as TSLNodeProperties;
      augmented.emissiveNode = emissiveNode;
      augmented.opacityNode = opacityNode;
      augmented.needsUpdate = true;
      logger.info('TSL nodes enabled (r183+)');
    } catch (err) {
      logger.warn('TSL upgrade failed, using PBR fallback:', err);
    }
  })();

  return { material, uniforms, ready };
}

export function createAgentCapsuleGeometry(): THREE.CapsuleGeometry {
  return new THREE.CapsuleGeometry(0.3, 0.6, 8, 16);
}
