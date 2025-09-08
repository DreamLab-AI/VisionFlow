import React, { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
// Using standard THREE.js materials instead of custom cloud materials

interface WireframeCloudMeshProps {
  geometry: 'torus' | 'sphere' | 'icosahedron';
  geometryArgs?: any[];
  position?: [number, number, number];
  rotation?: [number, number, number];
  scale?: number | [number, number, number];
  color?: string | THREE.Color;
  wireframeColor?: string | THREE.Color;
  opacity?: number;
  wireframeOpacity?: number;
  cloudExtension?: number;
  blurRadius?: number;
  glowIntensity?: number;
  rotationSpeed?: number;
  rotationAxis?: [number, number, number];
}

export const WireframeCloudMesh: React.FC<WireframeCloudMeshProps> = ({
  geometry = 'torus',
  geometryArgs = [],
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  scale = 1,
  color = '#00ffff',
  wireframeColor,
  opacity = 0.3,
  wireframeOpacity = 0.8,
  cloudExtension = 10.0,
  blurRadius = 15.0,
  glowIntensity = 2.0,
  rotationSpeed = 0,
  rotationAxis = [0, 1, 0]
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshBasicMaterial>();

  // Create geometry
  const geometryObj = useMemo(() => {
    let geo: THREE.BufferGeometry;
    
    switch (geometry) {
      case 'torus':
        geo = new THREE.TorusGeometry(...(geometryArgs.length ? geometryArgs : [1, 0.4, 8, 16]));
        break;
      case 'sphere':
        geo = new THREE.SphereGeometry(...(geometryArgs.length ? geometryArgs : [1, 16, 16]));
        break;
      case 'icosahedron':
        geo = new THREE.IcosahedronGeometry(...(geometryArgs.length ? geometryArgs : [1, 2]));
        break;
      default:
        geo = new THREE.SphereGeometry(1, 16, 16);
    }

    return geo;
  }, [geometry, geometryArgs]);

  // Create material using standard THREE.js material with emissive properties
  const material = useMemo(() => {
    const c = new THREE.Color(color);
    return new THREE.MeshBasicMaterial({
      color: c,
      emissive: c,
      emissiveIntensity: glowIntensity * 0.3,
      transparent: true,
      opacity: opacity,
      wireframe: true,
      toneMapped: false
    });
  }, [color, opacity, glowIntensity]);

  // Store material ref
  useEffect(() => {
    materialRef.current = material;
    return () => {
      material.dispose();
    };
  }, [material]);

  // Animate
  useFrame((state, delta) => {
    if (meshRef.current && rotationSpeed > 0) {
      meshRef.current.rotation.x += delta * rotationSpeed * rotationAxis[0];
      meshRef.current.rotation.y += delta * rotationSpeed * rotationAxis[1];
      meshRef.current.rotation.z += delta * rotationSpeed * rotationAxis[2];
    }

    // Standard material doesn't need time updates
  });

  return (
    <mesh 
      ref={meshRef}
      position={position}
      rotation={rotation}
      scale={scale}
      geometry={geometryObj}
      material={material}
    />
  );
};

// Compound component for creating multi-layer cloud effects
export const MultiLayerWireframeCloud: React.FC<{
  geometry: 'torus' | 'sphere' | 'icosahedron';
  geometryArgs?: any[];
  position?: [number, number, number];
  color?: string | THREE.Color;
  layers?: number;
  rotationSpeed?: number;
}> = ({
  geometry,
  geometryArgs,
  position = [0, 0, 0],
  color = '#00ffff',
  layers = 3,
  rotationSpeed = 0.5
}) => {
  return (
    <group position={position}>
      {Array.from({ length: layers }, (_, i) => {
        const layerScale = 1 + i * 0.1;  // Each layer slightly larger
        const layerOpacity = 0.3 / layers;  // Distribute opacity
        const extension = 5 + i * 5;  // Each layer extends further
        
        return (
          <WireframeCloudMesh
            key={i}
            geometry={geometry}
            geometryArgs={geometryArgs}
            scale={layerScale}
            color={color}
            wireframeOpacity={i === 0 ? 0.8 : 0}  // Only first layer shows wireframe
            opacity={layerOpacity}
            cloudExtension={extension}
            blurRadius={10 + i * 10}
            glowIntensity={2 - i * 0.5}
            rotationSpeed={rotationSpeed * (1 - i * 0.2)}
            rotationAxis={[
              Math.sin(i * Math.PI / 3),
              Math.cos(i * Math.PI / 3),
              0.5
            ]}
          />
        );
      })}
    </group>
  );
};