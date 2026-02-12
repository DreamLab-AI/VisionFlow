/**
 * WasmSceneEffects
 *
 * React Three Fiber component that renders WASM-driven ambient background
 * effects for the knowledge graph visualization:
 *
 * 1. Particle field: subtle drifting points with noise-based motion (WASM)
 * 2. Atmosphere plane: procedural nebula texture as far background (WASM)
 * 3. JS fallback: if WASM fails to load, renders lightweight JS particles
 *
 * All heavy computation runs in WASM. This component only transfers
 * typed array views to Three.js buffer attributes each frame.
 *
 * Performance contract:
 *   - 2-3 draw calls maximum
 *   - All Float32Arrays pre-allocated in useMemo
 *   - Zero per-frame GC pressure (reused typed array views from WASM)
 *   - renderOrder -10/-20 so everything draws behind nodes
 */

import React, { useMemo, useRef, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useWasmSceneEffects } from '../../../hooks/useWasmSceneEffects';
import { isWebGPURenderer } from '../../../rendering/rendererFactory';

// Pre-allocated temp vector for atmosphere direction (avoids per-frame GC)
const _tempAtmDir = new THREE.Vector3();

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface WasmSceneEffectsProps {
  enabled?: boolean;
  /** Number of ambient particles (default 256, max 512). */
  particleCount?: number;
  /** Number of energy wisps (default 48, max 128). */
  wispCount?: number;
  /** Whether energy wisps are enabled (default true). */
  wispsEnabled?: boolean;
  /** Wisp drift speed multiplier (default 1.0). */
  wispDriftSpeed?: number;
  /** Atmosphere texture resolution (default 128). */
  atmosphereResolution?: number;
  /** Whether atmosphere/fog is enabled (default true). */
  atmosphereEnabled?: boolean;
  /** Overall intensity 0-1 (maps to opacity). */
  intensity?: number;
  /** Particle drift speed multiplier (0-2, default 0.5). */
  particleDrift?: number;
}

// ---------------------------------------------------------------------------
// GLSL shaders for WASM-driven particles
// ---------------------------------------------------------------------------
const PARTICLE_VERTEX = /* glsl */ `
  attribute float aOpacity;
  attribute float aSize;
  varying float vOpacity;

  void main() {
    vOpacity = aOpacity;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (200.0 / -mvPosition.z);
    gl_PointSize = clamp(gl_PointSize, 0.5, 8.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const PARTICLE_FRAGMENT = /* glsl */ `
  varying float vOpacity;

  void main() {
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);
    float alpha = smoothstep(0.5, 0.15, dist) * vOpacity;
    vec3 color = mix(
      vec3(0.4, 0.5, 0.9),
      vec3(0.7, 0.75, 1.0),
      smoothstep(0.3, 0.0, dist)
    );
    if (alpha < 0.001) discard;
    gl_FragColor = vec4(color, alpha);
  }
`;

// ---------------------------------------------------------------------------
// Fallback constants (JS-only path)
// ---------------------------------------------------------------------------
const FALLBACK_COUNT = 256;
const FALLBACK_RADIUS = 120;
const FALLBACK_DRIFT = 0.15;

function hashNoise(x: number, y: number, seed: number): number {
  let h = (seed * 374761393 + x * 668265263 + y * 1274126177) | 0;
  h = ((h ^ (h >> 13)) * 1103515245) | 0;
  return ((h & 0x7fffffff) / 0x7fffffff) * 2 - 1;
}

// ---------------------------------------------------------------------------
// GLSL for fallback fog plane
// ---------------------------------------------------------------------------
const fogVertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fogFragmentShader = /* glsl */ `
uniform float uTime;
uniform float uOpacity;
uniform vec3 uColorDeep;
uniform vec3 uColorLight;
varying vec2 vUv;

float hash(vec2 p) {
  float h = dot(p, vec2(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
  float v = 0.0; float a = 0.5;
  for (int i = 0; i < 3; i++) { v += a * noise(p); p *= 2.0; a *= 0.5; }
  return v;
}
void main() {
  float t = uTime * 0.03;
  float n = fbm(vUv * 3.0 + t);
  float wisp = smoothstep(0.35, 0.65, n);
  vec3 col = mix(uColorDeep, uColorLight, wisp);
  float vig = 1.0 - length(vUv - 0.5) * 1.2;
  vig = clamp(vig, 0.0, 1.0);
  gl_FragColor = vec4(col, uOpacity * vig * 0.5);
}
`;

// ---------------------------------------------------------------------------
// WASM-powered particles sub-component
// ---------------------------------------------------------------------------
interface WasmParticlePointsProps {
  particles: NonNullable<ReturnType<typeof useWasmSceneEffects>['particles']>;
  update: ReturnType<typeof useWasmSceneEffects>['update'];
  opacity: number;
  count: number;
}

const WasmParticlePoints: React.FC<WasmParticlePointsProps> = ({
  particles,
  update,
  opacity,
  count,
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  const geometryRef = useRef<THREE.BufferGeometry>(null);

  const { posAttr, opacityAttr, sizeAttr } = useMemo(() => {
    return {
      posAttr: new THREE.BufferAttribute(new Float32Array(count * 3), 3),
      opacityAttr: new THREE.BufferAttribute(new Float32Array(count), 1),
      sizeAttr: new THREE.BufferAttribute(new Float32Array(count), 1),
    };
  }, [count]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: PARTICLE_VERTEX,
        fragmentShader: PARTICLE_FRAGMENT,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  useEffect(() => () => { material.dispose(); }, [material]);

  useFrame(({ camera }, delta) => {
    const dt = Math.min(delta, 0.05);
    const cam = camera.position;
    update(dt, cam.x, cam.y, cam.z);

    const geom = geometryRef.current;
    if (!geom) return;

    const wasmPositions = particles.getPositions();
    const wasmOpacities = particles.getOpacities();
    const wasmSizes = particles.getSizes();

    const posArray = geom.attributes.position.array as Float32Array;
    const opaArray = (geom.attributes.aOpacity as THREE.BufferAttribute).array as Float32Array;
    const sizeArray = (geom.attributes.aSize as THREE.BufferAttribute).array as Float32Array;

    posArray.set(wasmPositions);
    opaArray.set(wasmOpacities);
    sizeArray.set(wasmSizes);

    if (opacity < 1.0) {
      for (let i = 0; i < opaArray.length; i++) {
        opaArray[i] *= opacity;
      }
    }

    (geom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (geom.attributes.aOpacity as THREE.BufferAttribute).needsUpdate = true;
    (geom.attributes.aSize as THREE.BufferAttribute).needsUpdate = true;
  });

  return (
    <points ref={pointsRef} frustumCulled={false} renderOrder={-10}>
      <bufferGeometry ref={geometryRef}>
        <bufferAttribute attach="attributes-position" args={[posAttr.array, 3]} />
        <bufferAttribute attach="attributes-aOpacity" args={[opacityAttr.array, 1]} />
        <bufferAttribute attach="attributes-aSize" args={[sizeAttr.array, 1]} />
      </bufferGeometry>
      <primitive object={material} attach="material" />
    </points>
  );
};

// ---------------------------------------------------------------------------
// GLSL shaders for WASM-driven energy wisps (hue-shifting glow)
// ---------------------------------------------------------------------------
const WISP_VERTEX = /* glsl */ `
  attribute float aOpacity;
  attribute float aSize;
  attribute float aHue;
  varying float vOpacity;
  varying float vHue;

  void main() {
    vOpacity = aOpacity;
    vHue = aHue;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (250.0 / -mvPosition.z);
    gl_PointSize = clamp(gl_PointSize, 1.0, 16.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const WISP_FRAGMENT = /* glsl */ `
  varying float vOpacity;
  varying float vHue;

  // Simple HSL to RGB (hue only, high saturation, medium lightness)
  vec3 hsl2rgb(float h) {
    vec3 rgb = clamp(
      abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0,
      0.0, 1.0
    );
    return 0.3 + 0.7 * rgb; // Boost brightness
  }

  void main() {
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);

    // Soft radial glow with bright core
    float core = smoothstep(0.5, 0.0, dist);
    float glow = smoothstep(0.5, 0.1, dist);
    float alpha = (core * 0.6 + glow * 0.4) * vOpacity;

    // Hue-shifted color with warm bias
    vec3 wispColor = hsl2rgb(vHue * 0.3 + 0.55); // blue-purple range
    vec3 coreColor = mix(wispColor, vec3(1.0), core * 0.3);

    if (alpha < 0.001) discard;
    gl_FragColor = vec4(coreColor, alpha);
  }
`;

// ---------------------------------------------------------------------------
// WASM-powered energy wisps sub-component
// ---------------------------------------------------------------------------
interface WasmWispPointsProps {
  wisps: NonNullable<ReturnType<typeof useWasmSceneEffects>['wisps']>;
  opacity: number;
  count: number;
}

const WasmWispPoints: React.FC<WasmWispPointsProps> = ({
  wisps,
  opacity,
  count,
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  const geometryRef = useRef<THREE.BufferGeometry>(null);

  const { posAttr, opacityAttr, sizeAttr, hueAttr } = useMemo(() => {
    return {
      posAttr: new THREE.BufferAttribute(new Float32Array(count * 3), 3),
      opacityAttr: new THREE.BufferAttribute(new Float32Array(count), 1),
      sizeAttr: new THREE.BufferAttribute(new Float32Array(count), 1),
      hueAttr: new THREE.BufferAttribute(new Float32Array(count), 1),
    };
  }, [count]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: WISP_VERTEX,
        fragmentShader: WISP_FRAGMENT,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  useEffect(() => () => { material.dispose(); }, [material]);

  useFrame(() => {
    const geom = geometryRef.current;
    if (!geom) return;

    const wasmPositions = wisps.getPositions();
    const wasmOpacities = wisps.getOpacities();
    const wasmSizes = wisps.getSizes();
    const wasmHues = wisps.getHues();

    const posArray = geom.attributes.position.array as Float32Array;
    const opaArray = (geom.attributes.aOpacity as THREE.BufferAttribute).array as Float32Array;
    const sizeArray = (geom.attributes.aSize as THREE.BufferAttribute).array as Float32Array;
    const hueArray = (geom.attributes.aHue as THREE.BufferAttribute).array as Float32Array;

    posArray.set(wasmPositions);
    opaArray.set(wasmOpacities);
    sizeArray.set(wasmSizes);
    hueArray.set(wasmHues);

    if (opacity < 1.0) {
      for (let i = 0; i < opaArray.length; i++) {
        opaArray[i] *= opacity;
      }
    }

    (geom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (geom.attributes.aOpacity as THREE.BufferAttribute).needsUpdate = true;
    (geom.attributes.aSize as THREE.BufferAttribute).needsUpdate = true;
    (geom.attributes.aHue as THREE.BufferAttribute).needsUpdate = true;
  });

  return (
    <points ref={pointsRef} frustumCulled={false} renderOrder={-5}>
      <bufferGeometry ref={geometryRef}>
        <bufferAttribute attach="attributes-position" args={[posAttr.array, 3]} />
        <bufferAttribute attach="attributes-aOpacity" args={[opacityAttr.array, 1]} />
        <bufferAttribute attach="attributes-aSize" args={[sizeAttr.array, 1]} />
        <bufferAttribute attach="attributes-aHue" args={[hueAttr.array, 1]} />
      </bufferGeometry>
      <primitive object={material} attach="material" />
    </points>
  );
};

// ---------------------------------------------------------------------------
// WASM-powered atmosphere sub-component
// ---------------------------------------------------------------------------
interface WasmAtmosphereProps {
  atmosphere: NonNullable<ReturnType<typeof useWasmSceneEffects>['atmosphere']>;
  opacity: number;
}

const WasmAtmosphereBackground: React.FC<WasmAtmosphereProps> = ({
  atmosphere,
  opacity,
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera } = useThree();

  const texture = useMemo(() => {
    const w = atmosphere.width;
    const h = atmosphere.height;
    const tex = new THREE.DataTexture(
      new Uint8Array(w * h * 4),
      w, h,
      THREE.RGBAFormat,
    );
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }, [atmosphere]);

  useEffect(() => () => { texture.dispose(); }, [texture]);

  useFrame((_state, delta) => {
    const dt = Math.min(delta, 0.05);
    atmosphere.update(dt);

    const wasmPixels = atmosphere.getPixels();
    const texData = texture.image.data as Uint8Array;
    texData.set(wasmPixels);
    texture.needsUpdate = true;

    const mesh = meshRef.current;
    if (mesh) {
      _tempAtmDir.set(0, 0, -1).applyQuaternion(camera.quaternion);
      mesh.position.copy(camera.position).add(_tempAtmDir.multiplyScalar(90));
      mesh.quaternion.copy(camera.quaternion);
    }
  });

  return (
    <mesh ref={meshRef} renderOrder={-20} frustumCulled={false}>
      <planeGeometry args={[200, 200]} />
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={opacity * 0.6}
        depthWrite={false}
        side={THREE.FrontSide}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
};

// ---------------------------------------------------------------------------
// JS Fallback: particles (no WASM needed)
// ---------------------------------------------------------------------------
const FallbackParticles: React.FC<{ opacity: number }> = React.memo(({ opacity }) => {
  const pointsRef = useRef<THREE.Points>(null);

  const { positions, colors, baseSpeeds } = useMemo(() => {
    const pos = new Float32Array(FALLBACK_COUNT * 3);
    const col = new Float32Array(FALLBACK_COUNT * 3);
    const spd = new Float32Array(FALLBACK_COUNT * 3);
    const c1 = new THREE.Color('#1a1a4e');
    const c2 = new THREE.Color('#c8d8ff');
    const tmp = new THREE.Color();

    for (let i = 0; i < FALLBACK_COUNT; i++) {
      const phi = hashNoise(i, 0, 42) * Math.PI;
      const theta = hashNoise(i, 1, 42) * Math.PI * 2;
      const r = (0.3 + 0.7 * Math.abs(hashNoise(i, 2, 42))) * FALLBACK_RADIUS;
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);

      const t = Math.abs(hashNoise(i, 4, 42));
      tmp.copy(c1).lerp(c2, t);
      col[i * 3] = tmp.r;
      col[i * 3 + 1] = tmp.g;
      col[i * 3 + 2] = tmp.b;

      spd[i * 3] = 0.5 + Math.abs(hashNoise(i, 5, 42));
      spd[i * 3 + 1] = 0.5 + Math.abs(hashNoise(i, 6, 42));
      spd[i * 3 + 2] = 0.5 + Math.abs(hashNoise(i, 7, 42));
    }
    return { positions: pos, colors: col, baseSpeeds: spd };
  }, []);

  const basePositions = useMemo(() => new Float32Array(positions), [positions]);

  useFrame(({ clock }) => {
    const pts = pointsRef.current;
    if (!pts) return;
    const posAttr = pts.geometry.getAttribute('position') as THREE.BufferAttribute;
    const arr = posAttr.array as Float32Array;
    const t = clock.elapsedTime * FALLBACK_DRIFT;

    for (let i = 0; i < FALLBACK_COUNT; i++) {
      const i3 = i * 3;
      arr[i3] = basePositions[i3] + Math.sin(t * baseSpeeds[i3] + i * 0.7) * 2.0;
      arr[i3 + 1] = basePositions[i3 + 1] + Math.sin(t * baseSpeeds[i3 + 1] + i * 1.3) * 1.5;
      arr[i3 + 2] = basePositions[i3 + 2] + Math.cos(t * baseSpeeds[i3 + 2] + i * 0.9) * 2.0;
    }
    posAttr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef} renderOrder={-10}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial
        vertexColors
        transparent
        opacity={Math.min(opacity, 0.08)}
        sizeAttenuation
        size={1.5}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
});
FallbackParticles.displayName = 'FallbackParticles';

// ---------------------------------------------------------------------------
// JS Fallback: fog plane
// ---------------------------------------------------------------------------
const FallbackFogPlane: React.FC<{ opacity: number }> = React.memo(({ opacity }) => {
  const matRef = useRef<THREE.ShaderMaterial>(null);
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uOpacity: { value: Math.min(opacity, 0.03) },
      uColorDeep: { value: new THREE.Color('#0a0a1e') },
      uColorLight: { value: new THREE.Color('#12122e') },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  useFrame(({ clock }) => {
    if (matRef.current) {
      matRef.current.uniforms.uTime.value = clock.elapsedTime;
      matRef.current.uniforms.uOpacity.value = Math.min(opacity, 0.03);
    }
  });

  return (
    <mesh position={[0, 0, -80]} renderOrder={-20}>
      <planeGeometry args={[300, 300]} />
      <shaderMaterial
        ref={matRef}
        vertexShader={fogVertexShader}
        fragmentShader={fogFragmentShader}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
});
FallbackFogPlane.displayName = 'FallbackFogPlane';

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
const WasmSceneEffects: React.FC<WasmSceneEffectsProps> = ({
  enabled = true,
  particleCount = 256,
  wispCount = 48,
  wispsEnabled = true,
  wispDriftSpeed = 1.0,
  atmosphereResolution = 128,
  atmosphereEnabled = true,
  intensity = 0.3,
  particleDrift: _particleDrift = 0.5,
}) => {
  const { ready, failed, particles, atmosphere, wisps, update } = useWasmSceneEffects({
    particleCount,
    wispCount: wispsEnabled ? wispCount : 0,
    atmosphereWidth: atmosphereResolution,
    atmosphereHeight: atmosphereResolution,
    enabled,
  });

  // Apply drift speed setting to wisps
  useEffect(() => {
    if (wisps) {
      wisps.setDriftSpeed(wispDriftSpeed);
    }
  }, [wisps, wispDriftSpeed]);

  if (!enabled) return null;

  const clamped = Math.max(0, Math.min(1, intensity));

  // WASM path: fully driven by Rust noise + particle/wisp simulation
  // Note: WASM particle/wisp shaders use raw GLSL (gl_PointSize, gl_PointCoord, gl_FragColor)
  // which is incompatible with WebGPURenderer. Force fallback path on WebGPU.
  if (ready && particles && !isWebGPURenderer) {
    return (
      <group name="wasm-scene-effects">
        <WasmParticlePoints
          particles={particles}
          update={update}
          opacity={clamped}
          count={particleCount}
        />
        {atmosphereEnabled && atmosphere && (
          <WasmAtmosphereBackground
            atmosphere={atmosphere}
            opacity={clamped}
          />
        )}
        {wispsEnabled && wisps && (
          <WasmWispPoints
            wisps={wisps}
            opacity={clamped}
            count={wispCount}
          />
        )}
      </group>
    );
  }

  // JS fallback: lightweight hash-noise particles + standard material fog
  // Used when WASM fails OR when WebGPU is active (raw GLSL not supported).
  // FallbackParticles uses PointsMaterial (standard, auto-converts via TSL).
  // FallbackFogPlane uses ShaderMaterial with GLSL — skip on WebGPU.
  if (failed || !ready || isWebGPURenderer) {
    const particleOpacity = 0.02 + clamped * 0.06;
    const fogOpacity = 0.01 + clamped * 0.04;

    return (
      <group name="wasm-scene-effects-fallback" renderOrder={-1}>
        <FallbackParticles opacity={particleOpacity} />
        {atmosphereEnabled && !isWebGPURenderer && <FallbackFogPlane opacity={fogOpacity} />}
      </group>
    );
  }

  return null;
};

export default WasmSceneEffects;
