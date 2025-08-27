import React, { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
// DiffuseMoteMaterial doesn't exist, using standard Three.js materials
import { registerEnvObject, unregisterEnvObject } from '../hooks/bloomRegistry';

interface MotesRingProps {
  radius?: number;
  count?: number;
  color?: string | THREE.Color;
  size?: number;
  speed?: number;
  opacity?: number;
  density?: number;
  height?: number;
}

// Ring of motes circling around
export const MotesRing: React.FC<MotesRingProps> = ({
  radius = 10,  // Smaller rings
  count = 40,   // Reduced count
  color = '#00ffff',
  size = 0.4,   // Smaller motes
  speed = 0.3,
  opacity = 0.8,
  density = 0.7,
  height = 8    // Smaller height
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  const materialRef = useRef<DiffuseMoteMaterial>(null);
  
  // Create mote positions in a ring formation
  const [geometry, material] = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const randoms = new Float32Array(count);
    
    const c = new THREE.Color(color);
    
    for (let i = 0; i < count; i++) {
      // Distribute around a ring with some variation
      const angle = (i / count) * Math.PI * 2;
      const r = radius + (Math.random() - 0.5) * 4; // Slight radius variation
      
      positions[i * 3] = Math.cos(angle) * r;
      positions[i * 3 + 1] = (Math.random() - 0.5) * height; // Vertical spread
      positions[i * 3 + 2] = Math.sin(angle) * r;
      
      // Color with intensity variation
      const intensity = 0.5 + Math.random() * 0.5;
      colors[i * 3] = c.r * intensity;
      colors[i * 3 + 1] = c.g * intensity;
      colors[i * 3 + 2] = c.b * intensity;
      
      // Random value for animation
      randoms[i] = Math.random();
    }
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('random', new THREE.BufferAttribute(randoms, 1));
    
    // Use standard point material with additive blending for glow effect
    const mat = new THREE.PointsMaterial({
      size: size,
      vertexColors: true,
      transparent: true,
      opacity: opacity,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true
    });
    
    return [geo, mat];
  }, [count, radius, height, color, size, opacity, density, speed]);
  
  // Enable proper layers for bloom
  React.useEffect(() => {
    const points = pointsRef.current;
    if (points) {
      (points as any).layers.set(0);  // Base layer for rendering
      (points as any).layers.enable(2); // Layer 2 for hologram/glow bloom
      registerEnvObject(points as any);
    }
    
    return () => {
      if (points) {
        unregisterEnvObject(points as any);
      }
      material.dispose();
      geometry.dispose();
    };
  }, [material, geometry]);
  
  // Animate motes
  useFrame((state) => {
    if (pointsRef.current) {
      // Rotate the entire ring
      pointsRef.current.rotation.y += speed * 0.01;
      
      // Update individual mote positions for organic movement
      const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
      const randoms = pointsRef.current.geometry.attributes.random.array as Float32Array;
      
      for (let i = 0; i < count; i++) {
        const idx = i * 3;
        const random = randoms[i];
        
        // Vertical bobbing
        positions[idx + 1] += Math.sin(state.clock.elapsedTime * 2 + random * 10) * 0.02;
        
        // Keep within bounds
        if (positions[idx + 1] > height / 2) positions[idx + 1] = -height / 2;
        if (positions[idx + 1] < -height / 2) positions[idx + 1] = height / 2;
      }
      
      pointsRef.current.geometry.attributes.position.needsUpdate = true;
    }
    
    // Update point material color if needed
    if (pointsRef.current && pointsRef.current.material) {
      const mat = pointsRef.current.material as THREE.PointsMaterial;
      // Color is handled through vertex colors, update if needed
    }
  });
  
  return (
    <points ref={pointsRef} geometry={geometry} material={material} />
  );
};

// Glitter/sparkle effect particles
export const GlitterField: React.FC<{
  count?: number;
  bounds?: number;
  color?: string;
  sparkleSpeed?: number;
}> = ({
  count = 80,   // Reduced count
  bounds = 30,  // Smaller bounds
  color = '#ffffff',
  sparkleSpeed = 3.0
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  
  // Enable bloom for glitter
  useEffect(() => {
    const points = pointsRef.current;
    if (points) {
      (points as any).layers.set(0);
      (points as any).layers.enable(2); // Hologram/glow layer
      registerEnvObject(points as any);
    }
    
    return () => {
      if (points) {
        unregisterEnvObject(points as any);
      }
    };
  }, []);
  
  const [positions, colors, sizes] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const siz = new Float32Array(count);
    const c = new THREE.Color(color);
    
    for (let i = 0; i < count; i++) {
      // Random distribution in space
      pos[i * 3] = (Math.random() - 0.5) * bounds;
      pos[i * 3 + 1] = (Math.random() - 0.5) * bounds;
      pos[i * 3 + 2] = (Math.random() - 0.5) * bounds;
      
      // Varying intensity for sparkle
      const intensity = Math.random();
      col[i * 3] = c.r * intensity;
      col[i * 3 + 1] = c.g * intensity;
      col[i * 3 + 2] = c.b * intensity;
      
      // Random sizes for variety
      siz[i] = Math.random() * 0.5 + 0.1;
    }
    
    return [pos, col, siz];
  }, [count, bounds, color]);
  
  // Store color ref for animation
  const colorRef = useRef(new THREE.Color(color));
  
  // Update color when it changes
  useEffect(() => {
    colorRef.current = new THREE.Color(color);
  }, [color]);
  
  // Animate sparkles
  useFrame((state) => {
    if (pointsRef.current) {
      const colors = pointsRef.current.geometry.attributes.color.array as Float32Array;
      const c = colorRef.current;
      
      for (let i = 0; i < count; i++) {
        // Sparkle effect - random flashing
        const sparkle = Math.sin(state.clock.elapsedTime * sparkleSpeed + i * 0.1) * 0.5 + 0.5;
        const flash = Math.random() > 0.98 ? 1 : sparkle;
        
        colors[i * 3] = c.r * flash;
        colors[i * 3 + 1] = c.g * flash;
        colors[i * 3 + 2] = c.b * flash;
      }
      
      pointsRef.current.geometry.attributes.color.needsUpdate = true;
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.05;
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
        <bufferAttribute
          attach="attributes-size"
          count={count}
          array={sizes}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.3}
        vertexColors
        transparent
        opacity={0.9}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  );
};

// Combined hologram environment with all effects
export const HologramEnvironment: React.FC<{
  enabled?: boolean;
  color?: string;
  position?: [number, number, number];
}> = ({
  enabled = true,
  color = '#00ffff',
  position = [0, 0, 0]
}) => {
  const groupRef = useRef<THREE.Group>(null);
  
  // Enable bloom for the entire environment
  useEffect(() => {
    const group = groupRef.current;
    if (group && enabled) {
      group.traverse((child: any) => {
        if (child.layers) {
          child.layers.set(0);
          child.layers.enable(2); // Hologram/glow layer
        }
      });
      registerEnvObject(group as any);
    }
    
    return () => {
      if (group) {
        unregisterEnvObject(group as any);
      }
    };
  }, [enabled]);
  
  if (!enabled) return null;
  
  return (
    <group ref={groupRef} position={position}>
      {/* Ring of circling motes - smaller */}
      <MotesRing
        radius={12}
        count={40}
        color={color}
        size={0.4}
        speed={0.3}
        opacity={0.8}
        height={8}
      />
      
      {/* Second ring at different height and radius - smaller */}
      <MotesRing
        radius={20}
        count={30}
        color={color}
        size={0.3}
        speed={-0.2}
        opacity={0.6}
        height={12}
      />
      
      {/* Glitter/sparkle field - smaller */}
      <GlitterField
        count={80}
        bounds={40}
        color={color}
        sparkleSpeed={3.0}
      />
    </group>
  );
};