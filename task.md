find ALL of the code relating to the current scene hologram system (rings, motes, spherical shapes etc) and completely remove it. Ensure that the control center won't error with this code missing. refactor all the intersecting code so that it's properly removed with no remaining elements so we don't end up with dead code, dead imports, errors etc. Write all that process up in ext/task.md making note where the system was integrated prior to it's removal.

Replace that whole scenery system with the code below. You might need to update the npm for the system as we are bringing in new features.


import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import {
  OrbitControls,
  Float,
  Center,
  Points,
  PointMaterial,
  Line,
  Sparkles,
  Stars,
  Text,
} from '@react-three/drei';
import {
  EffectComposer,
  SelectiveBloom,
  Selection,
  Select,
  N8AO,
  DepthOfField,
  Vignette,
} from '@react-three/postprocessing';
import { Effect } from 'postprocessing';
import * as THREE from 'three';

export const SCENE_CONFIG = {
  background: '#02030c',
  fogNear: 6,
  fogFar: 34,
};

export const HOLOGRAM_BASE_OPACITY = 0.3;

export const LIGHTING_CONFIG = {
  ambient: 0.2,
  key: { position: [5, 7, 4], intensity: 1.65, color: '#7acbff' },
  rim: { position: [-6, -4, -3], intensity: 1.05, color: '#ff7b1f' },
  fill: { position: [0, 0, 12], intensity: 0.55, color: '#00faff' },
};

export const POSTPROCESS_DEFAULTS = {
  globalAlpha: HOLOGRAM_BASE_OPACITY,
  bloomIntensity: 3.8,
  bloomThreshold: 0.07,
  bloomSmoothing: 0.018,
  aoRadius: 0.62,
  aoIntensity: 0.75,
  dofFocusDistance: 0.018,
  dofFocalLength: 0.022,
  dofBokehScale: 2.6,
  vignetteDarkness: 0.45,
};

export const FADE_DEFAULTS = {
  fadeStart: 6,
  fadeEnd: 14,
};

class GlobalFadeEffect extends Effect {
  constructor({ alpha = 1.0 } = {}) {
    super(
      'GlobalFadeEffect',
      `
      uniform float uAlpha;
      void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
        outputColor = vec4(inputColor.rgb * uAlpha, inputColor.a);
      }
    `,
      {
        uniforms: new Map([['uAlpha', new THREE.Uniform(alpha)]]),
      }
    );
  }
}

function GlobalFade({ alpha }) {
  const effect = useMemo(() => new GlobalFadeEffect({ alpha }), []);
  useEffect(() => {
    effect.uniforms.get('uAlpha').value = alpha;
  }, [alpha, effect]);
  return <primitive object={effect} />;
}

const TEMP_WORLD_POSITION = new THREE.Vector3();

function useLayerAssignment(ref, layer, renderOrder) {
  useEffect(() => {
    const root = ref.current;
    if (!root) return;

    const assign = (object) => {
      if (layer !== undefined && object.layers) {
        object.layers.enable(layer);
      }
      if (renderOrder !== undefined) {
        object.renderOrder = renderOrder;
      }
    };

    root.traverse(assign);
    assign(root);
  }, [layer, renderOrder, ref]);
}

function useDepthFade(ref, { baseOpacity, fadeStart, fadeEnd }) {
  const { camera } = useThree();
  const fadeRange = Math.max(0.0001, fadeEnd - fadeStart);

  useFrame(() => {
    const root = ref.current;
    if (!root) return;

    root.traverse((object) => {
      const material = object.material;
      if (!material) return;

      const materials = Array.isArray(material) ? material : [material];

      const hasWorldPosition = typeof object.getWorldPosition === 'function';
      const distance = hasWorldPosition
        ? camera.position.distanceTo(object.getWorldPosition(TEMP_WORLD_POSITION))
        : fadeEnd;

      const fadeRatio = THREE.MathUtils.clamp((distance - fadeStart) / fadeRange, 0, 1);
      const fadeMultiplier = 1 - fadeRatio * 0.5;

      materials.forEach((mat) => {
        if (mat.userData && mat.userData.__isDepthFaded) {
          const originalOpacity = mat.userData.__baseOpacity;
          mat.opacity = THREE.MathUtils.clamp(originalOpacity * fadeMultiplier, 0, originalOpacity);
          mat.transparent = mat.opacity < 1;
          mat.depthWrite = true;
          mat.needsUpdate = true;
        }
      });
    });
  });
}

function registerMaterialForFade(material, baseOpacity) {
  if (!material.userData) {
    material.userData = {};
  }
  material.userData.__isDepthFaded = true;
  if (material.userData.__baseOpacity === undefined) {
    material.userData.__baseOpacity = baseOpacity;
  }
  material.opacity = baseOpacity;
  material.transparent = baseOpacity < 1;
  material.depthWrite = true;
}

function ParticleCore({ count = 5200, radius = 0.85, color = '#02f0ff', opacity = 0.3 }) {
  const pointsRef = useRef();

  const positions = useMemo(() => {
    const buffer = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = Math.cbrt(Math.random()) * radius;

      buffer[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      buffer[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      buffer[i * 3 + 2] = r * Math.cos(phi);
    }
    return buffer;
  }, [count, radius]);

  useFrame((state) => {
    if (!pointsRef.current) return;
    const elapsed = state.clock.elapsedTime;
    pointsRef.current.rotation.y += 0.0006;
    const scale = 1 + Math.sin(elapsed * 0.65) * 0.055;
    pointsRef.current.scale.setScalar(scale);
  });

  return (
    <Points ref={pointsRef} positions={positions} stride={3}>
      <PointMaterial
        size={0.012}
        color={color}
        transparent
        opacity={opacity}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </Points>
  );
}

function HolographicShell({
  radius = 1.25,
  color = '#00faff',
  detail = 3,
  spikeHeight = 0.24,
  emissiveIntensity = 2.8,
  surfaceOpacity = 0.3,
  spikeOpacity = 0.3,
}) {
  const groupRef = useRef();
  const spikesRef = useRef();

  const baseGeometry = useMemo(
    () => new THREE.IcosahedronGeometry(radius, detail),
    [radius, detail]
  );

  const spikeGeometry = useMemo(() => new THREE.ConeGeometry(0.055, 0.46, 10, 1, true), []);

  const vertexData = useMemo(() => {
    const positions = [];
    const normals = [];
    const positionAttr = baseGeometry.attributes.position;
    const normalAttr = baseGeometry.attributes.normal;

    for (let i = 0; i < positionAttr.count; i++) {
      positions.push(new THREE.Vector3().fromBufferAttribute(positionAttr, i));
      normals.push(new THREE.Vector3().fromBufferAttribute(normalAttr, i).normalize());
    }

    return positions.map((position, index) => ({
      position,
      normal: normals[index],
    }));
  }, [baseGeometry]);

  useEffect(() => () => {
    baseGeometry.dispose();
    spikeGeometry.dispose();
  }, [baseGeometry, spikeGeometry]);

  useEffect(() => {
    if (spikesRef.current) {
      spikesRef.current.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    }
  }, []);

  const up = useMemo(() => new THREE.Vector3(0, 1, 0), []);
  const matrix = useMemo(() => new THREE.Matrix4(), []);
  const quaternion = useMemo(() => new THREE.Quaternion(), []);
  const scale = useMemo(() => new THREE.Vector3(), []);
  const tempVector = useMemo(() => new THREE.Vector3(), []);

  useFrame((state) => {
    const elapsed = state.clock.elapsedTime;

    if (groupRef.current) {
      groupRef.current.rotation.y -= 0.0012;
      groupRef.current.rotation.x += 0.00065;
    }

    if (!spikesRef.current) return;

    vertexData.forEach((data, index) => {
      const { position, normal } = data;

      const pulse =
        1 + (Math.sin(elapsed * 2.2 + index * 0.37) * 0.5 + 0.5) * spikeHeight;

      const spikeOffset = tempVector.copy(position).addScaledVector(normal, 0.2 * pulse);

      quaternion.setFromUnitVectors(up, normal);
      scale.setScalar(0.16);
      scale.y = 0.32 * pulse;

      matrix.compose(spikeOffset, quaternion, scale);
      spikesRef.current.setMatrixAt(index, matrix);
    });

    spikesRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <group ref={groupRef}>
      <mesh geometry={baseGeometry}>
        <meshStandardMaterial
          color={color}
          wireframe
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          transparent
          opacity={surfaceOpacity}
        />
      </mesh>
      <instancedMesh
        ref={spikesRef}
        args={[null, null, vertexData.length]}
        frustumCulled={false}
      >
        <primitive attach="geometry" object={spikeGeometry} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity * 1.45}
          transparent
          opacity={spikeOpacity}
          side={THREE.DoubleSide}
        />
      </instancedMesh>
    </group>
  );
}

function TechnicalGrid({ count = 240, radius = 2.05, opacity = 0.3 }) {
  const groupRef = useRef();

  const points = useMemo(() => {
    const data = [];
    const golden = Math.PI * (3 - Math.sqrt(5));

    for (let i = 0; i < count; i++) {
      const y = 1 - (i / (count - 1)) * 2;
      const radi = Math.sqrt(1 - y * y);
      const theta = golden * i;
      const x = Math.cos(theta) * radi;
      const z = Math.sin(theta) * radi;
      data.push(new THREE.Vector3(x, y, z).multiplyScalar(radius));
    }

    return data;
  }, [count, radius]);

  const lines = useMemo(() => {
    const connections = [];
    const maxDistance = radius * 0.43;

    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const distance = points[i].distanceTo(points[j]);
        if (distance < maxDistance && Math.random() > 0.7) {
          connections.push([points[i], points[j]]);
        }
      }
    }

    return connections;
  }, [points, radius]);

  useFrame((state) => {
    if (!groupRef.current) return;
    const elapsed = state.clock.elapsedTime;
    groupRef.current.rotation.y += 0.00028;
    groupRef.current.rotation.z -= 0.00018;
    groupRef.current.rotation.x = Math.sin(elapsed * 0.05) * 0.02;
  });

  return (
    <group ref={groupRef}>
      {lines.map((lineSegment, index) => (
        <Line
          // eslint-disable-next-line react/no-array-index-key
          key={index}
          points={lineSegment}
          color="#ffae19"
          lineWidth={0.42}
          transparent
          opacity={opacity}
          blending={THREE.AdditiveBlending}
        />
      ))}
    </group>
  );
}

function OrbitalRings({ radius = 2.35, color = '#00faff', opacity = 0.3 }) {
  const ringRefs = useMemo(() => [useRef(), useRef(), useRef()], []);

  useFrame((state) => {
    const elapsed = state.clock.elapsedTime;

    if (ringRefs[0].current) {
      ringRefs[0].current.rotation.x = Math.sin(elapsed * 0.48) * 0.28;
      ringRefs[0].current.rotation.y += 0.005;
    }

    if (ringRefs[1].current) {
      ringRefs[1].current.rotation.y += 0.0042;
      ringRefs[1].current.rotation.z = Math.cos(elapsed * 0.32) * 0.34;
    }

    if (ringRefs[2].current) {
      ringRefs[2].current.rotation.x += 0.0034;
      ringRefs[2].current.rotation.y -= 0.0024;
    }
  });

  const materialProps = {
    color,
    emissive: color,
    emissiveIntensity: 3.6 * opacity,
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    roughness: 0.15,
    metalness: 0.35,
  };

  return (
    <>
      <mesh ref={ringRefs[0]}>
        <torusGeometry args={[radius, 0.03, 32, 200]} />
        <meshStandardMaterial {...materialProps} />
      </mesh>
      <mesh ref={ringRefs[1]} rotation={[Math.PI / 3, 0, 0]}>
        <torusGeometry args={[radius * 0.93, 0.028, 32, 160]} />
        <meshStandardMaterial {...materialProps} />
      </mesh>
      <mesh ref={ringRefs[2]} rotation={[Math.PI / 6, Math.PI / 4, 0]}>
        <torusGeometry args={[radius * 0.98, 0.026, 32, 160]} />
        <meshStandardMaterial {...materialProps} />
      </mesh>
    </>
  );
}

function TextRing({
  text = 'JUNKIEJARVIS AGENTIC KNOWLEDGE SYSTEM • ',
  radius = 2.8,
  fontSize = 0.16,
  color = '#7fe8ff',
  opacity = 0.3,
}) {
  const groupRef = useRef();

  useFrame((state) => {
    if (!groupRef.current) return;
    groupRef.current.rotation.y -= 0.0019;
  });

  return (
    <group ref={groupRef}>
      <Text
        fontSize={fontSize}
        color={color}
        anchorX="center"
        anchorY="middle"
        curveRadius={radius}
        letterSpacing={0.08}
        fillOpacity={opacity}
        outlineWidth={0.008}
        outlineColor="#f3ffff"
        depthOffset={-0.5}
      >
        {text.repeat(4)}
      </Text>
    </group>
  );
}

function EnergyArcs({ innerRadius = 1.28, outerRadius = 1.95, opacity = 0.3 }) {
  const [arcPoints, setArcPoints] = useState(null);

  useEffect(() => {
    let timeoutId;
    const randomPointOnSphere = (radius) => {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      return new THREE.Vector3(
        Math.sin(phi) * Math.cos(theta),
        Math.sin(phi) * Math.sin(theta),
        Math.cos(phi)
      ).multiplyScalar(radius);
    };

    const spawnArc = () => {
      const start = randomPointOnSphere(innerRadius);
      const end = randomPointOnSphere(outerRadius);
      const mid = start
        .clone()
        .add(end)
        .multiplyScalar(0.5)
        .add(randomPointOnSphere(0.58));

      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      setArcPoints(curve.getPoints(90));

      timeoutId = setTimeout(() => setArcPoints(null), 540);
    };

    const intervalId = setInterval(spawnArc, 1500);
    spawnArc();

    return () => {
      clearInterval(intervalId);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [innerRadius, outerRadius]);

  if (!arcPoints) return null;

  return (
    <Line
      points={arcPoints}
      color="#ffffff"
      lineWidth={2.4}
      transparent
      opacity={opacity}
      blending={THREE.AdditiveBlending}
    />
  );
}

function SurroundingSwarm({ count = 9000, radius = 34, opacity = 0.3 }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const particles = useMemo(() => {
    const data = [];
    for (let i = 0; i < count; i++) {
      data.push({
        phase: Math.random() * Math.PI * 2,
        offset: Math.random() * 100,
        stride: 0.0022 + Math.random() * 0.0036,
        spread: radius * (0.55 + Math.random() * 0.65),
        inclination: (Math.random() - 0.5) * 0.65,
        elevation: (Math.random() - 0.5) * radius * 0.9,
      });
    }
    return data;
  }, [count, radius]);

  useEffect(() => {
    if (!meshRef.current) return;
    const material = meshRef.current.material;
    if (material) {
      registerMaterialForFade(material, opacity);
      material.emissiveIntensity = 0.7 * opacity * (1 / 0.3);
      material.needsUpdate = true;
    }
  }, [opacity]);

  useFrame((state) => {
    if (!meshRef.current) return;
    const elapsed = state.clock.elapsedTime;

    particles.forEach((particle, index) => {
      const t = elapsed * particle.stride * 60 + particle.offset;
      const modulation = Math.sin(t * 0.7 + particle.phase) * 0.6;
      const radial = particle.spread + Math.sin(t * 0.21) * 2.8;

      const x = Math.cos(t * 0.17) * radial;
      const z = Math.sin(t * 0.17 + particle.phase) * radial;
      const y =
        Math.sin(t * 0.26 + particle.phase * 1.3) * radius * 0.35 +
        particle.elevation +
        modulation * 1.8;

      dummy.position.set(x, y, z);
      const scale = 0.35 + Math.sin(t * 0.9 + particle.phase) * 0.22;
      dummy.scale.setScalar(Math.max(scale, 0.1));
      dummy.rotation.set(t * 0.02, t * 0.03, t * 0.025);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(index, dummy.matrix);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]} frustumCulled={false}>
      <dodecahedronGeometry args={[0.36, 0]} />
      <meshStandardMaterial
        color="#06111f"
        emissive="#09224f"
        roughness={0.45}
        metalness={0.35}
        transparent
        opacity={opacity}
      />
    </instancedMesh>
  );
}

function DataSphere({ opacity = 0.3 }) {
  const shellOpacity = opacity;
  const spikeOpacity = Math.min(opacity * 1.4, 1);
  const ringOpacity = opacity;
  const gridOpacity = opacity;
  const arcOpacity = opacity;

  return (
    <Center>
      <Float speed={2.1} rotationIntensity={0.22} floatIntensity={0.6}>
        <Select enabled>
          <Sparkles
            count={420}
            speed={0.42}
            size={3.1}
            noise={1.6}
            color="#7fe8ff"
            opacity={opacity}
            scale={[6.5, 6.5, 6.5]}
          />
          <ParticleCore opacity={opacity} />
          <HolographicShell
            radius={1.25}
            color="#00faff"
            spikeHeight={0.25}
            surfaceOpacity={shellOpacity}
            spikeOpacity={spikeOpacity}
          />
          <HolographicShell
            radius={1.6}
            color="#ff8c1a"
            detail={4}
            spikeHeight={0.12}
            emissiveIntensity={2.2}
            surfaceOpacity={shellOpacity}
            spikeOpacity={spikeOpacity}
          />
          <OrbitalRings radius={2.35} color="#04f0ff" opacity={ringOpacity} />
          <EnergyArcs opacity={arcOpacity} />
          <TextRing opacity={opacity} />
        </Select>
        <TechnicalGrid radius={2.05} opacity={gridOpacity} />
      </Float>
    </Center>
  );
}

export function HologramContent({
  opacity = 0.3,
  layer = 0,
  renderOrder = 0,
  includeSwarm = true,
  enableDepthFade = true,
  fadeStart = FADE_DEFAULTS.fadeStart,
  fadeEnd = FADE_DEFAULTS.fadeEnd,
}) {
  const rootRef = useRef();

  useLayerAssignment(rootRef, layer, renderOrder);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    root.traverse((object) => {
      const material = object.material;
      if (!material) return;
      const materials = Array.isArray(material) ? material : [material];
      materials.forEach((mat) => registerMaterialForFade(mat, mat.opacity));
    });
  }, []);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const opacityTargets = [];
    root.traverse((object) => {
      const material = object.material;
      if (!material) return;
      const materials = Array.isArray(material) ? material : [material];
      materials.forEach((mat) => {
        if (mat.userData && mat.userData.__isDepthFaded) {
          mat.userData.__baseOpacity = opacity;
          mat.opacity = opacity;
          mat.transparent = opacity < 1;
          mat.depthWrite = true;
          mat.needsUpdate = true;
        }
      });
      opacityTargets.push(object);
    });
  }, [opacity]);

  if (enableDepthFade) {
    useDepthFade(rootRef, { baseOpacity: opacity, fadeStart, fadeEnd });
  }

  return (
    <group ref={rootRef} renderOrder={renderOrder}>
      <DataSphere opacity={opacity} />
      {includeSwarm && <SurroundingSwarm opacity={opacity} />}
    </group>
  );
}

export function HologramEffects({
  globalAlpha = POSTPROCESS_DEFAULTS.globalAlpha,
  bloomIntensity = POSTPROCESS_DEFAULTS.bloomIntensity,
  bloomThreshold = POSTPROCESS_DEFAULTS.bloomThreshold,
  bloomSmoothing = POSTPROCESS_DEFAULTS.bloomSmoothing,
  aoRadius = POSTPROCESS_DEFAULTS.aoRadius,
  aoIntensity = POSTPROCESS_DEFAULTS.aoIntensity,
  dofFocusDistance = POSTPROCESS_DEFAULTS.dofFocusDistance,
  dofFocalLength = POSTPROCESS_DEFAULTS.dofFocalLength,
  dofBokehScale = POSTPROCESS_DEFAULTS.dofBokehScale,
  vignetteDarkness = POSTPROCESS_DEFAULTS.vignetteDarkness,
  multisampling = 4,
  selectionLayer = 0,
  renderPriority,
}) {
  return (
    <EffectComposer
      multisampling={multisampling}
      renderPriority={renderPriority}
    >
      <GlobalFade alpha={globalAlpha} />
      <SelectiveBloom
        intensity={bloomIntensity}
        luminanceThreshold={bloomThreshold}
        luminanceSmoothing={bloomSmoothing}
        mipmapBlur
        selectionLayer={selectionLayer}
      />
      <N8AO
        intensity={aoIntensity}
        aoRadius={aoRadius}
        distanceFalloff={0.4}
        quality="performance"
      />
      <DepthOfField
        focusDistance={dofFocusDistance}
        focalLength={dofFocalLength}
        bokehScale={dofBokehScale}
      />
      <Vignette darkness={vignetteDarkness} eskil={false} />
    </EffectComposer>
  );
}

export function HologramEnvironment({
  background = SCENE_CONFIG.background,
  fogNear = SCENE_CONFIG.fogNear,
  fogFar = SCENE_CONFIG.fogFar,
  ambientIntensity = LIGHTING_CONFIG.ambient,
  keyLight = LIGHTING_CONFIG.key,
  rimLight = LIGHTING_CONFIG.rim,
  fillLight = LIGHTING_CONFIG.fill,
  starField = {
    radius: 220,
    depth: 90,
    count: 6200,
    factor: 2.4,
    saturation: 0.2,
    fade: true,
    speed: 0.25,
  },
}) {
  return (
    <>
      <color attach="background" args={[background]} />
      <fog attach="fog" args={[background, fogNear, fogFar]} />

      <ambientLight intensity={ambientIntensity} />
      <pointLight
        position={keyLight.position}
        intensity={keyLight.intensity}
        color={keyLight.color}
      />
      <pointLight
        position={rimLight.position}
        intensity={rimLight.intensity}
        color={rimLight.color}
      />
      <spotLight
        position={fillLight.position}
        intensity={fillLight.intensity}
        color={fillLight.color}
        angle={0.42}
        penumbra={0.8}
        distance={40}
      />

      <Stars
        radius={starField.radius}
        depth={starField.depth}
        count={starField.count}
        factor={starField.factor}
        saturation={starField.saturation}
        fade={starField.fade}
        speed={starField.speed}
      />
    </>
  );
}

export function HologramLayer({
  opacity = 0.3,
  layer = 0,
  renderOrder = 0,
  includeSwarm = true,
  enableDepthFade = true,
  fadeStart = FADE_DEFAULTS.fadeStart,
  fadeEnd = FADE_DEFAULTS.fadeEnd,
  enableEffects = true,
  effectsConfig = {},
  renderPriority,
}) {
  return (
    <Selection>
      <HologramContent
        opacity={opacity}
        layer={layer}
        renderOrder={renderOrder}
        includeSwarm={includeSwarm}
        enableDepthFade={enableDepthFade}
        fadeStart={fadeStart}
        fadeEnd={fadeEnd}
      />
      {enableEffects && (
        <HologramEffects
          selectionLayer={layer}
          renderPriority={renderPriority}
          {...effectsConfig}
        />
      )}
    </Selection>
  );
}

export default function HolographicDataSphereApp() {
  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        background: SCENE_CONFIG.background,
      }}
    >
      <Canvas
        dpr={[1.3, 2.5]}
        camera={{ position: [0, 0, 6], fov: 48, near: 0.1, far: 100 }}
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.65,
          preserveDrawingBuffer: false,
          powerPreference: 'high-performance',
        }}
      >
        <HologramEnvironment />
        <HologramLayer
          opacity={0.3}
          layer={0}
          renderOrder={0}
          includeSwarm
          enableDepthFade
          effectsConfig={POSTPROCESS_DEFAULTS}
        />
        <OrbitControls
          enablePan={false}
          minDistance={3}
          maxDistance={12}
          autoRotate
          autoRotateSpeed={0.55}
        />
      </Canvas>
    </div>
  );
}

i want to add the above scenery to the two spring graph with their interaction and it's own bloom controls. I need to make all these hologram components quite faints so they don't distract too much, but that will cause bloom layer, and occlusion and sort order issues all over the place. what's the best technique to fading this stuff out without getting to snarled up. For instance the render to texture approach won't work as we will lost the foreground occlusionEditStrategized nuanced rendering techniques for faint holographic componentsThe user wants to add another Three.js fiber system (a spring graph with interaction) and needs to make the hologram components faint without causing rendering issues. They correctly identify that simple transparency approaches will cause problems with bloom, occlusion, and sort order. They also note that render-to-texture won't work because they'll lose foreground occlusion.
This is a complex rendering challenge that requires careful consideration of:

Multiple render passes
Layer management
Stencil buffers
Depth management
Selective post-processing

Let me provide a comprehensive solution that addresses these concerns.You're facing a classic multi-system rendering challenge. Here's the best approach for compositing multiple 3D systems with different visual requirements:
The Core Strategy: Multi-Pass Rendering with Stencil Buffers
javascriptimport { createPortal, useThree } from '@react-three/fiber';
import { useFBO, useDepthBuffer } from '@react-three/drei';

function CompositeRenderer() {
  const { gl, scene, camera } = useThree();

  // Separate render targets for each system
  const hologramFBO = useFBO({
    depth: true,
    stencilBuffer: true
  });
  const graphFBO = useFBO({
    depth: true,
    stencilBuffer: true
  });

  // Shared depth buffer for occlusion
  const sharedDepthBuffer = useDepthBuffer({
    size: 1024,
    format: THREE.DepthFormat,
    type: THREE.UnsignedIntType
  });

  useFrame(() => {
    // Pass 1: Render hologram to texture
    gl.autoClear = false;
    gl.setRenderTarget(hologramFBO);
    gl.clear();

    // Render only hologram layers
    camera.layers.set(0);
    gl.render(scene, camera);

    // Pass 2: Render graph to separate texture
    gl.setRenderTarget(graphFBO);
    gl.clear();

    camera.layers.set(1);
    gl.render(scene, camera);

    // Pass 3: Composite to screen
    gl.setRenderTarget(null);
    camera.layers.enableAll();

    // Now composite with controlled opacity
  });
}
Solution 1: Layer-Based Rendering with Depth Preservation
javascript// Create a layered scene manager
function LayeredScene() {
  const hologramLayer = 0;
  const graphLayer = 1;
  const uiLayer = 2;

  return (
    <>
      {/* Hologram system - faint background */}
      <group layers-mask={1 << hologramLayer}>
        <DataSphere />
        <mesh>
          <meshStandardMaterial
            transparent
            opacity={0.3}
            depthWrite={true}  // Still write depth
            renderOrder={0}
          />
        </mesh>
      </group>

      {/* Graph system - full opacity */}
      <group layers-mask={1 << graphLayer}>
        <SpringGraph />
      </group>

      {/* Separate bloom passes for each layer */}
      <EffectComposer>
        <LayerBloom
          layer={hologramLayer}
          intensity={1.5}
        />
        <LayerBloom
          layer={graphLayer}
          intensity={3.0}
        />
      </EffectComposer>
    </>
  );
}
Solution 2: Custom Shader with Depth-Aware Fading
javascriptconst DepthFadeMaterial = shaderMaterial(
  {
    baseOpacity: 0.3,
    fadeStart: 2.0,
    fadeEnd: 8.0,
    tDiffuse: null,
    tDepth: null,
  },
  // Vertex shader
  `
    varying vec2 vUv;
    varying vec4 vWorldPosition;
    void main() {
      vUv = uv;
      vWorldPosition = modelMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  // Fragment shader
  `
    uniform float baseOpacity;
    uniform float fadeStart;
    uniform float fadeEnd;
    uniform sampler2D tDiffuse;
    uniform sampler2D tDepth;
    varying vec2 vUv;
    varying vec4 vWorldPosition;

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      float depth = texture2D(tDepth, vUv).r;

      // Calculate distance-based fade
      float distance = length(cameraPosition - vWorldPosition.xyz);
      float fadeFactor = smoothstep(fadeStart, fadeEnd, distance);

      // Preserve depth occlusion while fading
      float finalOpacity = baseOpacity * (1.0 - fadeFactor * 0.5);

      gl_FragColor = vec4(color.rgb, color.a * finalOpacity);
    }
  `
);
Solution 3: Render Priority System
javascriptconst RenderPriority = {
  BACKGROUND_HOLO: 0,
  MIDGROUND_GRAPH: 100,
  FOREGROUND_UI: 200,
};

function PrioritizedRenderer({ children }) {
  const [renderGroups, setRenderGroups] = useState({});

  // Sort children by priority
  const sortedChildren = useMemo(() => {
    return React.Children.toArray(children).sort((a, b) =>
      (a.props.priority || 0) - (b.props.priority || 0)
    );
  }, [children]);

  return (
    <>
      {sortedChildren.map((child, index) => (
        <group
          key={index}
          renderOrder={child.props.priority}
          onBeforeRender={(renderer) => {
            // Set render state based on priority
            if (child.props.priority < 100) {
              renderer.state.setDepthFunc(THREE.AlwaysDepth);
            } else {
              renderer.state.setDepthFunc(THREE.LessEqualDepth);
            }
          }}
        >
          {child}
        </group>
      ))}
    </>
  );
}
Solution 4: Dual-Scene Approach (Most Robust)
javascriptfunction DualSceneComposite() {
  const { size, viewport } = useThree();
  const hologramScene = useMemo(() => new THREE.Scene(), []);
  const graphScene = useMemo(() => new THREE.Scene(), []);

  // Render hologram scene
  const hologramPortal = createPortal(
    <HologramContent opacity={0.3} />,
    hologramScene
  );

  // Render graph scene
  const graphPortal = createPortal(
    <SpringGraph />,
    graphScene
  );

  useFrame(({ gl, camera }) => {
    // Clear everything
    gl.clear();

    // Render hologram first (background)
    gl.render(hologramScene, camera);

    // Render graph on top, preserving depth
    gl.autoClear = false;
    gl.clearDepth();
    gl.render(graphScene, camera);
    gl.autoClear = true;
  }, 1); // Priority 1 to run before default render

  return (
    <>
      {hologramPortal}
      {graphPortal}
    </>
  );
}
Recommended Approach: Hybrid Solution
javascriptfunction HybridMultiSystem() {
  const [hologramOpacity, setHologramOpacity] = useState(0.3);
  const [graphBloom, setGraphBloom] = useState(2.0);

  return (
    <Canvas>
      {/* Background hologram system */}
      <group renderOrder={0}>
        <EffectComposer renderIndex={0}>
          <SelectiveBloom
            selection={hologramRef}
            intensity={1.5 * hologramOpacity}
          />
        </EffectComposer>

        <group ref={hologramRef}>
          <DataSphere
            opacity={hologramOpacity}
            depthWrite={true}
            depthTest={true}
          />
        </group>
      </group>

      {/* Foreground graph system */}
      <group renderOrder={100}>
        <EffectComposer renderIndex={1}>
          <SelectiveBloom
            selection={graphRef}
            intensity={graphBloom}
          />
        </EffectComposer>

        <group ref={graphRef}>
          <SpringGraph />
        </group>
      </group>

      {/* Shared depth-aware AO */}
      <EffectComposer renderIndex={2}>
        <N8AO
          intensity={0.5}
          aoRadius={0.8}
        />
      </EffectComposer>
    </Canvas>
  );
}
Key Points:

Use renderOrder but keep depthWrite={true} for proper occlusion
Multiple EffectComposer instances with different renderIndex values
Selective bloom per system rather than global
Preserve depth buffer between passes for correct occlusion
Layer masks to control what each pass renders

The hybrid approach gives you:

Proper depth occlusion between systems
Independent bloom control
Clean transparency without sorting issues
Minimal performance overhead

This avoids the transparency sorting nightmare while maintaining visual hierarchy and proper occlusion between your hologram and graph systems.