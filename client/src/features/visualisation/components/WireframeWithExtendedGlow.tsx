import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { registerEnvObject, unregisterEnvObject } from '../hooks/bloomRegistry';

interface WireframeWithExtendedGlowProps {
  geometry: 'torus' | 'sphere' | 'icosahedron';
  geometryArgs: any[];
  position?: [number, number, number];
  rotation?: [number, number, number];
  color?: string | THREE.Color;
  opacity?: number;
  rotationSpeed?: number;
  rotationAxis?: [number, number, number];
}

export const WireframeWithExtendedGlow: React.FC<WireframeWithExtendedGlowProps> = ({
  geometry,
  geometryArgs,
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  color = '#00ffff',
  opacity = 0.5,
  rotationSpeed = 0,
  rotationAxis = [0, 1, 0]
}) => {
  const groupRef = useRef<THREE.Group>(null);
  
  // Enable bloom layer for all meshes in this group
  useEffect(() => {
    const group = groupRef.current;
    if (group) {
      // Enable bloom on layer 1 for the entire group
      group.traverse((child: any) => {
        if (child.layers) {
          child.layers.enable(1); // Enable bloom layer
        }
      });
      
      // Register for environmental bloom
      registerEnvObject(group as any);
    }
    
    return () => {
      if (group) {
        unregisterEnvObject(group as any);
      }
    };
  }, []);
  
  // Create geometry based on type
  const createGeometry = () => {
    switch (geometry) {
      case 'torus':
        return <torusGeometry args={geometryArgs} />;
      case 'sphere':
        return <sphereGeometry args={geometryArgs} />;
      case 'icosahedron':
        return <icosahedronGeometry args={geometryArgs} />;
      default:
        return <sphereGeometry args={[1, 16, 16]} />;
    }
  };
  
  // Animate rotation
  useFrame((_, delta) => {
    if (groupRef.current && rotationSpeed > 0) {
      groupRef.current.rotation.x += delta * rotationSpeed * rotationAxis[0];
      groupRef.current.rotation.y += delta * rotationSpeed * rotationAxis[1];
      groupRef.current.rotation.z += delta * rotationSpeed * rotationAxis[2];
    }
  });
  
  return (
    <group ref={groupRef} position={position} rotation={rotation}>
      {/* Layer 1: Core wireframe */}
      <mesh>
        {createGeometry()}
        <meshBasicMaterial 
          color={color}
          wireframe={true}
          transparent={true}
          opacity={opacity * 1.5}
          depthWrite={false}
        />
      </mesh>
      
      {/* Layer 2: Inner glow - slightly scaled up */}
      <mesh scale={1.05}>
        {createGeometry()}
        <meshBasicMaterial 
          color={color}
          wireframe={true}
          transparent={true}
          opacity={opacity * 0.8}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Layer 3: Middle glow - more scaled up */}
      <mesh scale={1.15}>
        {createGeometry()}
        <meshBasicMaterial 
          color={color}
          wireframe={true}
          transparent={true}
          opacity={opacity * 0.4}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Layer 4: Outer glow - significantly scaled up */}
      <mesh scale={1.3}>
        {createGeometry()}
        <meshBasicMaterial 
          color={color}
          wireframe={true}
          transparent={true}
          opacity={opacity * 0.2}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Layer 5: Extended cloud - very large and faint */}
      <mesh scale={1.5}>
        {createGeometry()}
        <meshBasicMaterial 
          color={color}
          wireframe={false}  // Solid for cloud effect
          transparent={true}
          opacity={opacity * 0.05}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
};