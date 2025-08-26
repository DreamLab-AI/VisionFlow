// World-class hologram ambient effects with diffuse rendering
import React, { useRef, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useSettingsStore } from '@/store/settingsStore';
import { registerEnvObject, unregisterEnvObject } from '../hooks/bloomRegistry';
// import { useBloomStrength } from '../../graph/contexts/BloomContext'; // Removed - bloom managed via settings
import { DiffuseEffectsIntegration } from '@/rendering/DiffuseEffectsIntegration';
import { HologramManager } from '../renderers/HologramManager';

// Quantum field shader with advanced visuals
const quantumFieldShader = {
  vertexShader: `
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec2 vUv;
    uniform float time;
    
    void main() {
      vPosition = position;
      vNormal = normal;
      vUv = uv;
      
      // Quantum fluctuation
      vec3 pos = position;
      float quantum = sin(position.x * 20.0 + time * 2.0) * 
                     sin(position.y * 20.0 - time * 1.5) * 
                     sin(position.z * 20.0 + time) * 0.02;
      pos += normal * quantum;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `,
  fragmentShader: `
    uniform float time;
    uniform vec3 color1;
    uniform vec3 color2;
    uniform float opacity;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec2 vUv;
    
    // Simplex noise for organic effects
    float noise(vec3 p) {
      return fract(sin(dot(p, vec3(12.9898, 78.233, 45.543))) * 43758.5453);
    }
    
    void main() {
      // Energy field calculation
      float field = 0.0;
      for(float i = 1.0; i <= 3.0; i++) {
        float scale = pow(2.0, i);
        field += (1.0 / scale) * noise(vPosition * scale + time * 0.1 * i);
      }
      
      // Holographic interference pattern
      float interference = sin(vPosition.x * 50.0 + time * 2.0) * 
                          sin(vPosition.y * 50.0 - time * 1.5) * 
                          sin(vPosition.z * 50.0 + time);
      interference = smoothstep(-0.5, 0.5, interference);
      
      // Rim lighting
      vec3 viewDir = normalize(cameraPosition - vPosition);
      float rim = 1.0 - abs(dot(viewDir, vNormal));
      rim = pow(rim, 2.0);
      
      // Color mixing
      vec3 color = mix(color1, color2, field);
      color += vec3(0.0, 1.0, 1.0) * interference * 0.3;
      color += vec3(1.0) * rim * 0.5;
      
      // Pulsing opacity
      float pulse = sin(time * 3.0) * 0.2 + 0.8;
      float finalOpacity = opacity * pulse * (0.3 + field * 0.7);
      
      gl_FragColor = vec4(color, finalOpacity);
    }
  `
};

// Holographic rings with particle systems
const HolographicRing: React.FC<{
  radius: number;
  thickness: number;
  rotationSpeed: number;
  color: THREE.Color;
}> = ({ radius, thickness, rotationSpeed, color }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const particlesRef = useRef<THREE.Points>(null);
  
  // Create particle geometry
  const particles = useMemo(() => {
    const count = 200;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const r = radius + (Math.random() - 0.5) * thickness;
      
      positions[i * 3] = Math.cos(angle) * r;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 2;
      positions[i * 3 + 2] = Math.sin(angle) * r;
      
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    return geometry;
  }, [radius, thickness, color]);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += rotationSpeed * 0.01;
      meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime) * 0.1;
    }
    if (particlesRef.current) {
      particlesRef.current.rotation.y -= rotationSpeed * 0.005;
    }
  });
  
  return (
    <group>
      <mesh ref={meshRef}>
        <torusGeometry args={[radius, thickness, 8, 64]} />
        <meshBasicMaterial
          color={color}
          wireframe
          transparent
          opacity={0.6}
          depthWrite={false}
        />
      </mesh>
      <points ref={particlesRef} geometry={particles}>
        <pointsMaterial
          size={0.5}
          vertexColors
          transparent
          opacity={0.8}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>
    </group>
  );
};

// Main world-class hologram system - memoized to prevent unnecessary re-renders
export const WorldClassHologram: React.FC<{
  enabled?: boolean;
  position?: [number, number, number];
  useDiffuseEffects?: boolean;
}> = React.memo(({ enabled = true, position = [0, 0, 0], useDiffuseEffects = true }) => {
  const settings = useSettingsStore((state) => state.settings);
  // Handle both snake_case and camelCase field names
  const envBloomStrength = (settings?.visualisation?.bloom as any)?.environment_bloom_strength ?? 
                           settings?.visualisation?.bloom?.environmentBloomStrength ?? 0.5;
  const sphereRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const groupRef = useRef<THREE.Group>(null);
  
  const hologramSettings = settings?.visualisation?.hologram;
  const isEnabled = enabled && (settings?.visualisation?.graphs?.logseq?.nodes?.enableHologram || 
                               (settings?.visualisation?.graphs?.logseq?.nodes as any)?.enable_hologram || false);
  
  // Removed debug logging to reduce console noise
  
  const uniforms = useMemo(() => ({
    time: { value: 0 },
    color1: { value: new THREE.Color(hologramSettings?.ringColor || '#00ffff') },
    color2: { value: new THREE.Color('#ff00ff') },
    opacity: { value: hologramSettings?.ringOpacity || 0.4 }
  }), [hologramSettings]);
  
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime;
    }
    if (sphereRef.current) {
      sphereRef.current.rotation.x = state.clock.elapsedTime * 0.05;
      sphereRef.current.rotation.y = state.clock.elapsedTime * 0.03;
    }
  });
  
  if (!isEnabled) {
    return null;
  }

  // Ensure hologram content renders on env bloom layer (1) and register for selective bloom
  React.useEffect(() => {
    const obj = groupRef.current as any;
    if (obj) {
      obj.layers.enable(1);
      registerEnvObject(obj);
    }
    return () => {
      if (obj) unregisterEnvObject(obj);
    };
  }, []);
  
  // Wrap with diffuse effects if enabled
  const hologramContent = (
    <group ref={groupRef} position={new THREE.Vector3(...position)}>
      {/* Use new HologramManager with diffuse effects */}
      <HologramManager 
        position={[0, 0, 0]}
        isXRMode={false}
        useDiffuseEffects={useDiffuseEffects}
      />
      
      {/* Quantum field sphere - fallback for custom settings */}
      {!useDiffuseEffects && (
        <mesh ref={sphereRef}>
          <icosahedronGeometry args={[hologramSettings?.sphereSizes?.[0] || 40, 4]} />
          <meshBasicMaterial
            color={hologramSettings?.ringColor || '#00ffff'}
            wireframe
            transparent
            opacity={hologramSettings?.ringOpacity || 0.4}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      )}
      
      {/* Multiple holographic rings */}
      {hologramSettings?.ringCount && Array.from({ length: hologramSettings.ringCount }, (_, i) => (
        <HolographicRing
          key={i}
          radius={30 + i * 20}  // Original values
          thickness={2}         // Original values
          rotationSpeed={1 + i * 0.3}
          color={new THREE.Color(hologramSettings.ringColor || '#00ffff')}
        />
      ))}
      
      {/* Buckminster sphere */}
      {hologramSettings?.enableBuckminster && (
        <mesh>
          <icosahedronGeometry args={[hologramSettings.buckminsterSize || 50, 2]} />
          <meshPhysicalMaterial
            color="#00ffff"
            emissive="#00ffff"
            emissiveIntensity={0.3 * envBloomStrength}
            transparent
            opacity={hologramSettings.buckminsterOpacity || 0.3}
            wireframe
            roughness={0}
            metalness={1}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      )}
      
      {/* Geodesic sphere */}
      {hologramSettings?.enableGeodesic && (
        <mesh rotation={[0, Math.PI / 4, 0]}>
          <dodecahedronGeometry args={[hologramSettings.geodesicSize || 60]} />
          <meshPhysicalMaterial
            color="#ff00ff"
            emissive="#ff00ff"
            emissiveIntensity={0.2 * envBloomStrength}
            transparent
            opacity={hologramSettings.geodesicOpacity || 0.25}
            wireframe
            roughness={0}
            metalness={1}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      )}
    </group>
  );

  // Wrap with diffuse effects integration
  return useDiffuseEffects ? (
    <DiffuseEffectsIntegration
      enableDiffuseEffects={true}
      enableBloomForGraphs={false}
    >
      {hologramContent}
    </DiffuseEffectsIntegration>
  ) : (
    hologramContent
  );
});

// Energy field particles - memoized to prevent unnecessary re-renders
export const EnergyFieldParticles: React.FC<{
  count?: number;
  bounds?: number;
  color?: string;
}> = React.memo(({ count = 1000, bounds = 200, color = '#00ffff' }) => {
  const pointsRef = useRef<THREE.Points>(null);
  
  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const c = new THREE.Color(color);
    
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * bounds;
      pos[i * 3 + 1] = (Math.random() - 0.5) * bounds;
      pos[i * 3 + 2] = (Math.random() - 0.5) * bounds;
      
      const intensity = Math.random();
      col[i * 3] = c.r * intensity;
      col[i * 3 + 1] = c.g * intensity;
      col[i * 3 + 2] = c.b * intensity;
    }
    
    return [pos, col];
  }, [count, bounds, color]);

  React.useEffect(() => {
    const obj = pointsRef.current as any;
    if (obj) {
      obj.layers.enable(1);
      registerEnvObject(obj);
    }
    return () => {
      if (obj) unregisterEnvObject(obj);
    };
  }, []);
  
  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.02;
      
      // Update particle positions for flow effect
      const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
      for (let i = 0; i < count; i++) {
        const idx = i * 3 + 1; // Y position
        positions[idx] += Math.sin(state.clock.elapsedTime + i * 0.1) * 0.1;
        
        // Wrap around
        if (positions[idx] > bounds / 2) positions[idx] = -bounds / 2;
        if (positions[idx] < -bounds / 2) positions[idx] = bounds / 2;
      }
      pointsRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });
  
  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={1}
        vertexColors
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  );
});

export default WorldClassHologram;