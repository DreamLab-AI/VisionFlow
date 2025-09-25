import React, { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useSettingsStore } from '@/store/settingsStore';
import { registerEnvObject, unregisterEnvObject } from '../hooks/bloomRegistry';
import { HologramManager } from '../renderers/HologramManager';

/**
 * CONSOLIDATED HOLOGRAM ENVIRONMENT
 * 
 * This component combines the best features from WorldClassHologram and HologramMotes
 * to provide a unified holographic environment for the graph visualization.
 * 
 * Key Features:
 * - Unified rendering on Layer 2 (environment bloom layer)
 * - Particle systems for ambient effects
 * - Holographic rings and spheres
 * - Energy field particles
 * - World-class visual effects with performance optimization
 */

interface HologramEnvironmentProps {
  enabled?: boolean;
  position?: [number, number, number];
  useDiffuseEffects?: boolean;
}

// Holographic particle ring component
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
    const count = 100;
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

  // Ensure all elements are on Layer 2 for environment glow
  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.layers.set(0); // Base layer for rendering
      meshRef.current.layers.enable(2); // Also on environment glow layer
    }
    if (particlesRef.current) {
      particlesRef.current.layers.set(0); // Base layer for rendering
      particlesRef.current.layers.enable(2); // Also on environment glow layer
    }
  }, []);
  
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
          toneMapped={false}
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

// Ring of motes circling around
const MotesRing: React.FC<{
  radius?: number;
  count?: number;
  color?: string | THREE.Color;
  size?: number;
  speed?: number;
  opacity?: number;
  height?: number;
}> = ({
  radius = 10,
  count = 40,
  color = '#00ffff',
  size = 0.4,
  speed = 0.3,
  opacity = 0.8,
  height = 8
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  
  // Create mote positions in a ring formation
  const [geometry, material] = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    const c = new THREE.Color(color);
    
    for (let i = 0; i < count; i++) {
      // Distribute around a ring with some variation
      const angle = (i / count) * Math.PI * 2;
      const r = radius + (Math.random() - 0.5) * 4;
      
      positions[i * 3] = Math.cos(angle) * r;
      positions[i * 3 + 1] = (Math.random() - 0.5) * height;
      positions[i * 3 + 2] = Math.sin(angle) * r;
      
      // Color with intensity variation
      const intensity = 0.5 + Math.random() * 0.5;
      colors[i * 3] = c.r * intensity;
      colors[i * 3 + 1] = c.g * intensity;
      colors[i * 3 + 2] = c.b * intensity;
    }
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const mat = new THREE.PointsMaterial({
      size: size,
      transparent: true,
      opacity: opacity,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
      toneMapped: false
    });
    
    return [geo, mat];
  }, [radius, count, color, size, opacity, height]);
  
  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += speed * 0.01;
      
      // Subtle vertical motion
      const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
      for (let i = 0; i < count; i++) {
        const idx = i * 3 + 1;
        positions[idx] += Math.sin(state.clock.elapsedTime * 2 + i * 0.1) * 0.01;
      }
      pointsRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  // Ensure motes are on Layer 2 for environment glow
  useEffect(() => {
    if (pointsRef.current) {
      pointsRef.current.layers.set(0); // Base layer for rendering
      pointsRef.current.layers.enable(2); // Also on environment glow layer
      registerEnvObject(pointsRef.current);
    }
    return () => {
      if (pointsRef.current) {
        unregisterEnvObject(pointsRef.current);
      }
    };
  }, []);
  
  return <points ref={pointsRef} geometry={geometry} material={material} />;
};

// Energy field particles for ambient atmosphere
const EnergyFieldParticles: React.FC<{
  count?: number;
  bounds?: number;
  color?: string;
}> = React.memo(({ count = 200, bounds = 400, color = '#00ffff' }) => {
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

  // Ensure particles are on Layer 2 for environment glow
  useEffect(() => {
    if (pointsRef.current) {
      pointsRef.current.layers.set(0); // Base layer for rendering
      pointsRef.current.layers.enable(2); // Also on environment glow layer
      registerEnvObject(pointsRef.current);
    }
    return () => {
      if (pointsRef.current) {
        unregisterEnvObject(pointsRef.current);
      }
    };
  }, []);
  
  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.02;
      
      // Update particle positions for flow effect
      const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
      for (let i = 0; i < count; i++) {
        const idx = i * 3 + 1;
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
        toneMapped={false}
      />
    </points>
  );
});

// Main consolidated hologram environment component
export const HologramEnvironment: React.FC<HologramEnvironmentProps> = React.memo(({ 
  enabled = true, 
  position = [0, 0, 0], 
  useDiffuseEffects = true 
}) => {
  const settings = useSettingsStore((state) => state.settings);
  const groupRef = useRef<THREE.Group>(null);
  
  const hologramSettings = settings?.visualisation?.hologram;
  const isEnabled = enabled && (settings?.visualisation?.graphs?.logseq?.nodes?.enableHologram || 
                               (settings?.visualisation?.graphs?.logseq?.nodes as any)?.enable_hologram || false);
  
  // Ensure all hologram elements are on Layer 2 for environment glow
  useEffect(() => {
    if (!isEnabled) return;
    
    const group = groupRef.current;
    if (group) {
      group.layers.set(0); // Base layer for rendering
      group.layers.enable(2); // Also on environment glow layer
      registerEnvObject(group);
      
      // Traverse all children and ensure they're on Layer 2
      group.traverse((child: THREE.Object3D) => {
        if (child.layers) {
          child.layers.set(0); // Base layer for rendering
          child.layers.enable(2); // Also on environment glow layer
        }
      });
    }
    
    return () => {
      if (group) {
        unregisterEnvObject(group);
      }
    };
  }, [isEnabled]);
  
  if (!isEnabled) {
    return null;
  }
  
  return (
    <group ref={groupRef} position={new THREE.Vector3(...position)}>
      {/* Core hologram manager with traditional rings and spheres */}
      <HologramManager 
        position={[0, 0, 0]}
        isXRMode={false}
        useDiffuseEffects={useDiffuseEffects}
      />
      
      {/* Holographic rings with particle systems */}
      {hologramSettings?.ringCount && Array.from({ length: Math.floor(hologramSettings.ringCount) }, (_, i) => (
        <HolographicRing
          key={`holo-ring-${i}`}
          radius={20 + i * 15}
          thickness={2}
          rotationSpeed={1 + i * 0.3}
          color={new THREE.Color(hologramSettings.ringColor || '#00ffff')}
        />
      ))}
      
      {/* Motes rings for ambient atmosphere */}
      {[15, 25, 35].map((radius, i) => (
        <MotesRing
          key={`motes-${i}`}
          radius={radius}
          count={30 + i * 10}
          color={hologramSettings?.ringColor || '#00ffff'}
          size={0.3 + i * 0.1}
          speed={0.2 + i * 0.1}
          opacity={0.6 - i * 0.1}
          height={6 + i * 2}
        />
      ))}
      
      {/* Energy field particles for wide-area ambiance */}
      <EnergyFieldParticles
        count={150}
        bounds={300}
        color={hologramSettings?.ringColor || '#00ffff'}
      />
      
      {/* Buckminster sphere if enabled */}
      {hologramSettings?.enableBuckminster && (
        <mesh>
          <icosahedronGeometry args={[hologramSettings.buckminsterSize || 50, 2]} />
          <meshPhysicalMaterial
            color="#00ffff"
            emissive="#00ffff"
            emissiveIntensity={0.3}
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
      
      {/* Geodesic sphere if enabled */}
      {hologramSettings?.enableGeodesic && (
        <mesh rotation={[0, Math.PI / 4, 0]}>
          <dodecahedronGeometry args={[hologramSettings.geodesicSize || 60]} />
          <meshPhysicalMaterial
            color="#ff00ff"
            emissive="#ff00ff"
            emissiveIntensity={0.2}
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
});

HologramEnvironment.displayName = 'HologramEnvironment';
EnergyFieldParticles.displayName = 'EnergyFieldParticles';

export default HologramEnvironment;
export { EnergyFieldParticles };