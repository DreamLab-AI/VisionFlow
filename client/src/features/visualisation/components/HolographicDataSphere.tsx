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
  bloomIntensity: 76,
  bloomThreshold: 0.07,
  bloomSmoothing: 0.36,
  aoRadius: 124,
  aoIntensity: 0.75,
  dofFocusDistance: 3.6,
  dofFocalLength: 4.4,
  dofBokehScale: 520,
  vignetteDarkness: 0.45,
};

export const FADE_DEFAULTS = {
  fadeStart: 1200,
  fadeEnd: 2800,
};

class GlobalFadeEffect extends Effect {
  constructor({ alpha = 1.0 } = {}) {
    super(
      'GlobalFadeEffect',
      `
      uniform float uAlpha;
      void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
        // Properly handle alpha channel multiplication
        outputColor = vec4(inputColor.rgb, inputColor.a * uAlpha);
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
          // Don't force depthWrite for transparent materials
          mat.depthWrite = mat.opacity >= 0.99;
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
  // Always set the base opacity, don't check if undefined
  material.userData.__baseOpacity = baseOpacity;
  material.opacity = baseOpacity;
  material.transparent = true; // Always transparent for proper blending
  // Never use depthWrite for transparent hologram materials
  material.depthWrite = false;
  material.needsUpdate = true;
}

function ParticleCore({ count = 5200, radius = 170, color = '#02f0ff', opacity = 0.3 }) {
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
        size={2.4}
        color={color}
        transparent
        opacity={opacity}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.NormalBlending}
      />
    </Points>
  );
}

function HolographicShell({
  radius = 250,
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

  const spikeGeometry = useMemo(() => new THREE.ConeGeometry(2.2, 18.4, 10, 1, true), []);

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

      const spikeOffset = tempVector.copy(position).addScaledVector(normal, 8 * pulse);

      quaternion.setFromUnitVectors(up, normal);
      scale.setScalar(1);
      scale.y = 1 * pulse;

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

function TechnicalGrid({ count = 240, radius = 410, opacity = 0.3 }) {
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
          blending={THREE.NormalBlending}
        />
      ))}
    </group>
  );
}

function OrbitalRings({ radius = 470, color = '#00faff', opacity = 0.3 }) {
  const ringRef0 = useRef();
  const ringRef1 = useRef();
  const ringRef2 = useRef();

  useFrame((state) => {
    const elapsed = state.clock.elapsedTime;

    if (ringRef0.current) {
      ringRef0.current.rotation.x = Math.sin(elapsed * 0.48) * 0.28;
      ringRef0.current.rotation.y += 0.005;
    }

    if (ringRef1.current) {
      ringRef1.current.rotation.y += 0.0042;
      ringRef1.current.rotation.z = Math.cos(elapsed * 0.32) * 0.34;
    }

    if (ringRef2.current) {
      ringRef2.current.rotation.x += 0.0034;
      ringRef2.current.rotation.y -= 0.0024;
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
      <mesh ref={ringRef0}>
        <torusGeometry args={[radius, 6, 32, 200]} />
        <meshStandardMaterial {...materialProps} />
      </mesh>
      <mesh ref={ringRef1} rotation={[Math.PI / 3, 0, 0]}>
        <torusGeometry args={[radius * 0.93, 5.6, 32, 160]} />
        <meshStandardMaterial {...materialProps} />
      </mesh>
      <mesh ref={ringRef2} rotation={[Math.PI / 6, Math.PI / 4, 0]}>
        <torusGeometry args={[radius * 0.98, 5.2, 32, 160]} />
        <meshStandardMaterial {...materialProps} />
      </mesh>
    </>
  );
}

function TextRing({
  text = 'JUNKIEJARVIS AGENTIC KNOWLEDGE SYSTEM • ',
  radius = 560,
  fontSize = 32,
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
        letterSpacing={16}
        fillOpacity={opacity}
        outlineWidth={1.6}
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
      blending={THREE.NormalBlending}
    />
  );
}

function SurroundingSwarm({ count = 9000, radius = 6800, opacity = 0.3 }) {
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
      <dodecahedronGeometry args={[72, 0]} />
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
            size={620}
            noise={320}
            color="#7fe8ff"
            opacity={opacity}
            scale={[1300, 1300, 1300]}
          />
          <ParticleCore opacity={opacity} />
          <HolographicShell
            radius={250}
            color="#00faff"
            spikeHeight={0.25}
            surfaceOpacity={shellOpacity}
            spikeOpacity={spikeOpacity}
          />
          <HolographicShell
            radius={320}
            color="#ff8c1a"
            detail={4}
            spikeHeight={0.12}
            emissiveIntensity={2.2}
            surfaceOpacity={shellOpacity}
            spikeOpacity={spikeOpacity}
          />
          <OrbitalRings radius={470} color="#04f0ff" opacity={ringOpacity} />
          <EnergyArcs opacity={arcOpacity} />
          <TextRing opacity={opacity} />
        </Select>
        <TechnicalGrid radius={410} opacity={gridOpacity} />
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
    // Delay material registration to ensure all materials are initialized
    const timeoutId = setTimeout(() => {
      root.traverse((object) => {
        const material = object.material;
        if (!material) return;
        const materials = Array.isArray(material) ? material : [material];
        materials.forEach((mat) => {
          // Use the provided opacity as base, not the material's current opacity
          registerMaterialForFade(mat, opacity);
        });
      });
    }, 100);
    return () => clearTimeout(timeoutId);
  }, [opacity]);

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
          // Don't force depthWrite for transparent materials
          mat.depthWrite = opacity >= 0.99;
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