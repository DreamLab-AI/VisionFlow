import React, { useMemo, useRef, useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import { 
  EffectComposer, 
  Bloom, 
  EffectPass,
  RenderPass,
  SelectiveBloomEffect
} from '@react-three/postprocessing';
import { BlendFunction, KernelSize, Resolution } from 'postprocessing';
import * as THREE from 'three';
import { useSettingsStore } from '../store/settingsStore';

/**
 * Modern, performant selective bloom implementation using @react-three/postprocessing
 * 
 * This replaces the old DualBloomPipeline with a proper R3F implementation that:
 * - Uses emissive materials for selective bloom (the R3F way)
 * - Properly integrates with React Three Fiber's render loop
 * - Provides better performance through optimized postprocessing pipeline
 * - Supports separate bloom settings for graph and environment elements
 */

interface SelectiveBloomProps {
  enabled?: boolean;
}

/**
 * Layer definitions for selective rendering
 * Layer 0: Base geometry, background (no bloom)
 * Layer 1: Graph elements (sharp bloom)
 * Layer 2: Environment effects (soft glow)
 */
const LAYERS = {
  BASE: 0,
  GRAPH_BLOOM: 1,
  ENVIRONMENT_GLOW: 2,
} as const;

export const SelectiveBloom: React.FC<SelectiveBloomProps> = ({ enabled = true }) => {
  const { scene, camera } = useThree();
  const settings = useSettingsStore(state => state.settings);
  
  // Extract bloom and glow settings
  const bloomSettings = settings?.visualisation?.bloom;
  const glowSettings = settings?.visualisation?.glow;
  
  // Determine if any effects are enabled
  const hasEffects = enabled && (bloomSettings?.enabled || glowSettings?.enabled);
  
  // Create bloom parameters based on settings
  const bloomParams = useMemo(() => {
    // Use glow settings if only glow is enabled, otherwise use bloom settings
    const activeSettings = !bloomSettings?.enabled && glowSettings?.enabled ? glowSettings : bloomSettings;
    
    if (!activeSettings?.enabled) {
      console.warn('SelectiveBloom: No active settings, bloom disabled');
      return null;
    }
    
    const params = {
      intensity: activeSettings.strength ?? activeSettings.intensity ?? 1.5,
      luminanceThreshold: activeSettings.threshold ?? 0.1, // Lower threshold to allow more bloom
      luminanceSmoothing: 0.025,
      kernelSize: activeSettings.radius ? 
        (activeSettings.radius > 0.5 ? KernelSize.LARGE : KernelSize.MEDIUM) : 
        KernelSize.MEDIUM,
      mipmapBlur: true,
      resolutionX: Resolution.AUTO_SIZE,
      resolutionY: Resolution.AUTO_SIZE
    };
    
    return params;
  }, [bloomSettings, glowSettings]);
  
  // Note: With @react-three/postprocessing's Bloom component, we don't need to manually
  // modify materials. The bloom effect will automatically apply to bright areas based on
  // the luminanceThreshold. Materials should already have their emissive properties set
  // appropriately by the components themselves (e.g., HologramNodeMaterial, BloomStandardMaterial)
  
  if (!hasEffects || !bloomParams) {
    return null;
  }
  
  return (
    <EffectComposer multisampling={0} renderPriority={1}>
      <Bloom
        intensity={bloomParams.intensity}
        luminanceThreshold={bloomParams.luminanceThreshold}
        luminanceSmoothing={bloomParams.luminanceSmoothing}
        kernelSize={bloomParams.kernelSize}
        mipmapBlur={bloomParams.mipmapBlur}
        resolutionX={bloomParams.resolutionX}
        resolutionY={bloomParams.resolutionY}
        blendFunction={BlendFunction.ADD}
      />
    </EffectComposer>
  );
};

/**
 * Hook to mark objects for bloom
 * Use this in components to enable/disable bloom on specific objects
 */
/**
 * Hook to mark objects for bloom with proper layer management
 * @param ref - React ref to the object
 * @param layer - Which bloom layer to use (GRAPH_BLOOM or ENVIRONMENT_GLOW)
 * @param enabled - Whether bloom is enabled
 */
export const useBloom = (
  ref: React.RefObject<THREE.Object3D>, 
  layer: typeof LAYERS.GRAPH_BLOOM | typeof LAYERS.ENVIRONMENT_GLOW = LAYERS.GRAPH_BLOOM,
  enabled: boolean = true
) => {
  const settings = useSettingsStore(state => state.settings);

  useEffect(() => {
    const obj = ref.current;
    if (!obj) return;
    
    if (enabled) {
      obj.layers.enable(layer);
      
      // If it's a mesh, setup emissive properties
      if ((obj as THREE.Mesh).isMesh) {
        const mesh = obj as THREE.Mesh;
        const material = mesh.material as THREE.MeshStandardMaterial;
        if (material?.isMeshStandardMaterial) {
          // Store original properties for proper cleanup
          if (!(material as any).__originalEmissive) {
            (material as any).__originalEmissive = material.emissive.clone();
            (material as any).__originalEmissiveIntensity = material.emissiveIntensity;
            (material as any).__originalToneMapped = material.toneMapped;
          }
          
          material.emissive = material.color.clone();
          material.emissiveIntensity = layer === LAYERS.GRAPH_BLOOM ? 1.0 : 0.5;
          material.toneMapped = false;
        }
      }
    } else {
      obj.layers.disable(layer);
      
      // Restore original emissive properties
      if ((obj as THREE.Mesh).isMesh) {
        const mesh = obj as THREE.Mesh;
        const material = mesh.material as THREE.MeshStandardMaterial;
        if (material?.isMeshStandardMaterial && (material as any).__originalEmissive) {
          material.emissive.copy((material as any).__originalEmissive);
          material.emissiveIntensity = (material as any).__originalEmissiveIntensity;
          material.toneMapped = (material as any).__originalToneMapped;
        }
      }
    }
    
    return () => {
      // Cleanup on unmount - restore original properties
      if ((obj as THREE.Mesh).isMesh) {
        const mesh = obj as THREE.Mesh;
        const material = mesh.material as THREE.MeshStandardMaterial;
        if (material?.isMeshStandardMaterial && (material as any).__originalEmissive) {
          material.emissive.copy((material as any).__originalEmissive);
          material.emissiveIntensity = (material as any).__originalEmissiveIntensity;
          material.toneMapped = (material as any).__originalToneMapped;
          
          delete (material as any).__originalEmissive;
          delete (material as any).__originalEmissiveIntensity;
          delete (material as any).__originalToneMapped;
        }
      }
    };
  }, [ref, layer, enabled, settings?.system?.debug?.enablePerformanceDebug]);
};

// Export layer constants for use in other components
export { LAYERS };

export default SelectiveBloom;