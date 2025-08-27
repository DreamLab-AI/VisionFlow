import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useSettingsStore } from '@/store/settingsStore';
import { createLogger } from '@/utils/logger';
import { HologramMaterial } from './materials/HologramMaterial';
import { 
  DiffuseWireframeMaterial, 
  DiffuseHologramRingMaterial, 
  DiffuseMoteMaterial 
} from '@/rendering/DiffuseWireframeMaterial';
import { BloomHologramMaterial, BloomStandardMaterial } from '../shaders/BloomHologramMaterial';

const logger = createLogger('HologramManager');

// Component for an individual hologram ring with diffuse effects
export const HologramRing: React.FC<{
  size?: number;
  color?: string | THREE.Color | number;
  opacity?: number;
  rotationAxis?: readonly [number, number, number];
  rotationSpeed?: number;
  segments?: number;
  useDiffuseEffects?: boolean;
}> = ({
  size = 1,
  color = '#00ffff',
  opacity = 0.7,
  rotationAxis = [0, 1, 0],
  rotationSpeed = 0.5,
  segments = 64,
  useDiffuseEffects = true
}) => {
  const materialRef = useRef<DiffuseHologramRingMaterial | THREE.MeshBasicMaterial>();
  const [rotation, setRotation] = useState<[number, number, number]>([0, 0, 0]);

  // Create material with bloom effect
  const material = React.useMemo(() => {
    if (useDiffuseEffects) {
      // Use our new bloom shader for heavy blur effect
      return new BloomHologramMaterial({
        color: color,
        opacity: opacity,
        glowRadius: 1.5,
        glowIntensity: 2.0,
        blurAmount: 4.0
      });
    } else {
      // Fallback to standard emissive material
      return new BloomStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 2.0,
        opacity: opacity,
        wireframe: true
      });
    }
  }, [color, opacity, useDiffuseEffects]);

  // Animate ring rotation
  useFrame((state, delta) => {
    if (rotationSpeed > 0) {
      setRotation(prev => [
        prev[0] + delta * rotationSpeed * rotationAxis[0],
        prev[1] + delta * rotationSpeed * rotationAxis[1],
        prev[2] + delta * rotationSpeed * rotationAxis[2]
      ]);
    }

    // Update material time for animation
    if (material instanceof BloomHologramMaterial) {
      material.updateTime(state.clock.elapsedTime);
    }
  });

  // Clean up material
  useEffect(() => {
    materialRef.current = material;
    return () => {
      material?.dispose();
    };
  }, [material]);

  return (
    <mesh rotation={rotation}>
      <torusGeometry args={[size, 0.5, 8, 16]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
};

// Component for a hologram sphere with diffuse effects
export const HologramSphere: React.FC<{
  size?: number;
  color?: string | THREE.Color | number;
  opacity?: number;
  detail?: number;
  wireframe?: boolean;
  rotationSpeed?: number;
  useDiffuseEffects?: boolean;
}> = ({
  size = 1,
  color = '#00ffff',
  opacity = 0.5,
  detail = 1,
  wireframe = true,
  rotationSpeed = 0.2,
  useDiffuseEffects = true
}) => {
  const materialRef = useRef<DiffuseWireframeMaterial | THREE.MeshBasicMaterial>();
  const [rotationY, setRotationY] = useState(0);

  // Create material - use basic material for now, shader needs debugging
  const material = React.useMemo(() => {
    // Always use basic material until we fix the shader
    return new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: opacity,
      wireframe: true,
      toneMapped: false
    });
  }, [color, opacity]);

  // Animate sphere rotation
  useFrame((_, delta) => {
    if (rotationSpeed > 0) {
      setRotationY(prev => prev + delta * rotationSpeed);
    }

    // Update material time for animation
    if (useDiffuseEffects && material instanceof DiffuseWireframeMaterial) {
      material.updateTime(performance.now() * 0.001);
    }
  });

  // Clean up material
  useEffect(() => {
    materialRef.current = material;
    return () => {
      material?.dispose();
    };
  }, [material]);

  return (
    <mesh rotation={[0, rotationY, 0]}>
      <icosahedronGeometry args={[size, detail]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
};

// Main HologramManager component that manages multiple hologram elements
export const HologramManager: React.FC<{
  position?: readonly [number, number, number];
  isXRMode?: boolean;
  useDiffuseEffects?: boolean;
}> = ({
  position = [0, 0, 0],
  isXRMode = false,
  useDiffuseEffects = true
}) => {
  const settings = useSettingsStore(state => state.settings?.visualisation?.hologram);
  const groupRef = useRef<THREE.Group>(null);

  // Parse sphere sizes from settings
  const sphereSizes: number[] = React.useMemo(() => {
    const sizesSetting: unknown = settings?.sphereSizes;

    if (typeof sizesSetting === 'string') {
      const strSetting = sizesSetting as string;
      return strSetting.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    } else if (Array.isArray(sizesSetting)) {
      const arrSetting = sizesSetting as unknown[];
      return arrSetting.filter((n): n is number => typeof n === 'number' && !isNaN(n));
    }
    return [40, 80];
  }, [settings?.sphereSizes]);

  // Setup group for diffuse effects (no bloom layers needed)
  useEffect(() => {
    const group = groupRef.current;
    if (group && !useDiffuseEffects) {
      // Only set bloom layers if NOT using diffuse effects (fallback)
      (group as any).layers.set(0);
      (group as any).layers.enable(1);

      (group as any).traverse((child: any) => {
        if (child.layers) {
          child.layers.set(0);
          child.layers.enable(1);
        }
      });
    }
  }, [useDiffuseEffects]);

  const quality = isXRMode ? 'high' : 'medium';
  const color: string | number = settings?.ringColor || '#00ffff'; // Use ringColor instead of color
  const opacity = settings?.ringOpacity !== undefined ? settings.ringOpacity : 0.7;
  const rotationSpeed = settings?.ringRotationSpeed !== undefined ? settings.ringRotationSpeed : 0.5;
  const enableTriangleSphere = settings?.enableTriangleSphere !== false;
  const triangleSphereSize = settings?.triangleSphereSize || 60;
  const triangleSphereOpacity = settings?.triangleSphereOpacity || 0.3;
  
  // Ensure we have at least some spheres to render - making them 4x smaller
  const defaultSizes = [10, 20]; // Was [40, 80], now 4x smaller
  const finalSphereSizes = sphereSizes.length > 0 ? sphereSizes.map(s => s / 4) : defaultSizes;

  return (
    <group ref={groupRef} position={position as any}>
      {/* Render rings with simple materials */}
      {finalSphereSizes.map((size, index) => (
        <HologramRing
          key={`ring-${index}`}
          size={size}  // Don't divide by 100 - use actual size
          color={color}
          opacity={opacity}
          rotationAxis={[
            Math.cos(index * Math.PI / 3),
            Math.sin(index * Math.PI / 3),
            0.5
          ]}
          rotationSpeed={rotationSpeed * (0.8 + index * 0.2)}
          segments={quality === 'high' ? 64 : 32}
          useDiffuseEffects={useDiffuseEffects}  // Re-enable diffuse effects
        />
      ))}

      {/* Render triangle sphere with diffuse material */}
      {enableTriangleSphere && (
        <HologramSphere
          size={triangleSphereSize}  // Don't divide by 100 - use actual size
          color={color}
          opacity={triangleSphereOpacity}
          detail={quality === 'high' ? 2 : 1}
          wireframe={true}
          useDiffuseEffects={useDiffuseEffects}  // Re-enable diffuse effects
        />
      )}
    </group>
  );
};

// A composite hologram component for easy use
export const Hologram: React.FC<{
  position?: readonly [number, number, number];
  color?: string | THREE.Color | number;
  size?: number;
  useDiffuseEffects?: boolean;
  children?: React.ReactNode;
}> = ({
  position = [0, 0, 0],
  color = '#00ffff',
  size = 1,
  useDiffuseEffects = true,
  children
}) => {
  return (
    <group position={position as any} scale={size}>
      {children}
      <HologramManager useDiffuseEffects={useDiffuseEffects} />
    </group>
  );
};

// Class-based wrapper for non-React usage (for backward compatibility)
// Using 'any' types to avoid TypeScript errors with THREE.js
export class HologramManagerClass {
  private scene: any; // THREE.Scene
  private group: any; // THREE.Group
  private ringInstances: any[] = []; // THREE.Mesh[]
  private sphereInstances: any[] = []; // THREE.Mesh[]
  private isXRMode: boolean = false;
  private settings: any;
  private useDiffuseEffects: boolean = true;

  constructor(scene: any, settings: any, useDiffuseEffects = true) {
    this.scene = scene;
    this.settings = settings;
    this.group = new THREE.Group();
    this.useDiffuseEffects = useDiffuseEffects;

    // Enable bloom layer only if not using diffuse effects (fallback)
    if (!this.useDiffuseEffects) {
      this.group.layers.set(0);
      this.group.layers.enable(1);
    }

    this.createHolograms();
    this.scene.add(this.group);
  }

  private createHolograms() {
    // Clear existing holograms
    const group = this.group;
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);

      // Handle geometry and material disposal
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach((m: any) => m && m.dispose());
        } else {
          child.material.dispose();
        }
      }
    }

    this.ringInstances = [];
    this.sphereInstances = [];

    // Quality based on XR mode
    const quality = this.isXRMode ? 'high' : 'medium';
    const segments = quality === 'high' ? 64 : 32;

    // Extract settings
    const hologramSettings = this.settings?.visualisation?.hologram || {};
    const color = hologramSettings.color || 0x00ffff;
    const opacity = hologramSettings.ringOpacity !== undefined ? hologramSettings.ringOpacity : 0.7;
    const sphereSizes = Array.isArray(hologramSettings.sphereSizes)
      ? hologramSettings.sphereSizes
      : [40, 80];

    // Create ring instances with diffuse effects
    sphereSizes.forEach((size, index) => {
      const geometry = new (THREE as any).RingGeometry(size * 0.8 / 100, size / 100, segments);
      
      let material: any;
      if (this.useDiffuseEffects) {
        material = new DiffuseHologramRingMaterial({
          color: color,
          opacity: opacity,
          glowIntensity: 0.9
        });
      } else {
        material = new (THREE as any).MeshBasicMaterial({
          color: color,
          transparent: true,
          opacity: opacity,
          side: (THREE as any).DoubleSide,
          depthWrite: false
        });
      }

      const ring = new (THREE as any).Mesh(geometry, material);

      // Set rotation based on index
      ring.rotation.x = Math.PI / 3 * index;
      ring.rotation.y = Math.PI / 6 * index;

      // Enable bloom layer only if not using diffuse effects
      if (!this.useDiffuseEffects) {
        ring.layers.set(0);
        ring.layers.enable(1);
      }

      this.ringInstances.push(ring);
      group.add(ring);
    });

    // Create triangle sphere if enabled
    if (hologramSettings.enableTriangleSphere) {
      const size = hologramSettings.triangleSphereSize || 60;
      const sphereOpacity = hologramSettings.triangleSphereOpacity || 0.3;
      const detail = quality === 'high' ? 2 : 1;

      const geometry = new (THREE as any).IcosahedronGeometry(size / 100, detail);
      
      let material: any;
      if (this.useDiffuseEffects) {
        material = new DiffuseWireframeMaterial({
          color: color,
          opacity: sphereOpacity,
          glowIntensity: 0.7,
          diffuseRadius: 1.5,
          animated: true
        });
      } else {
        material = new (THREE as any).MeshBasicMaterial({
          color: color,
          wireframe: true,
          transparent: true,
          opacity: sphereOpacity,
          side: (THREE as any).DoubleSide,
          depthWrite: false
        });
      }

      const sphere = new (THREE as any).Mesh(geometry, material);

      // Enable bloom layer only if not using diffuse effects
      if (!this.useDiffuseEffects) {
        sphere.layers.set(0);
        sphere.layers.enable(1);
      }

      this.sphereInstances.push(sphere);
      group.add(sphere);
    }
  }

  setXRMode(enabled: boolean) {
    this.isXRMode = enabled;
    this.createHolograms();
  }

  update(deltaTime: number) {
    const currentTime = performance.now() * 0.001;
    const rotationSpeed = this.settings?.visualisation?.hologram?.ringRotationSpeed || 0.5;

    // Update ring rotations and materials
    this.ringInstances.forEach((ring: any, index: number) => {
      const speed = rotationSpeed * (1.0 + index * 0.2);
      ring.rotation.y += deltaTime * speed;

      // Update diffuse material time if using diffuse effects
      if (this.useDiffuseEffects && ring.material instanceof DiffuseHologramRingMaterial) {
        ring.material.updateTime(currentTime);
      }
    });

    // Update sphere rotations and materials
    this.sphereInstances.forEach((sphere: any) => {
      sphere.rotation.y += deltaTime * rotationSpeed * 0.5;

      // Update diffuse material time if using diffuse effects
      if (this.useDiffuseEffects && sphere.material instanceof DiffuseWireframeMaterial) {
        sphere.material.updateTime(currentTime);
      }
    });
  }

  updateSettings(newSettings: any) {
    this.settings = newSettings;
    this.createHolograms();
  }

  getGroup() {
    return this.group;
  }

  dispose() {
    this.scene.remove(this.group);

    // Dispose geometries and materials
    this.ringInstances.forEach((ring: any) => {
      if (ring.geometry) ring.geometry.dispose();
      if (ring.material) {
        if (Array.isArray(ring.material)) {
          ring.material.forEach((m: any) => m && m.dispose());
        } else {
          ring.material.dispose();
        }
      }
    });

    this.sphereInstances.forEach((sphere: any) => {
      if (sphere.geometry) sphere.geometry.dispose();
      if (sphere.material) {
        if (Array.isArray(sphere.material)) {
          sphere.material.forEach((m: any) => m && m.dispose());
        } else {
          sphere.material.dispose();
        }
      }
    });

    this.ringInstances = [];
    this.sphereInstances = [];
  }
}

export default HologramManager;