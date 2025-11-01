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



interface SelectiveBloomProps {
  enabled?: boolean;
}


const LAYERS = {
  BASE: 0,
  GRAPH_BLOOM: 1,
  ENVIRONMENT_GLOW: 2,
} as const;

export const SelectiveBloom: React.FC<SelectiveBloomProps> = ({ enabled = true }) => {
  const { scene, camera } = useThree();
  const settings = useSettingsStore(state => state.settings);
  
  
  const bloomSettings = settings?.visualisation?.bloom;
  const glowSettings = settings?.visualisation?.glow;
  
  
  const hasEffects = enabled && (bloomSettings?.enabled || glowSettings?.enabled);
  
  
  const bloomParams = useMemo(() => {
    
    const activeSettings = !bloomSettings?.enabled && glowSettings?.enabled ? glowSettings : bloomSettings;
    
    if (!activeSettings?.enabled) {
      console.warn('SelectiveBloom: No active settings, bloom disabled');
      return null;
    }
    
    const params = {
      intensity: activeSettings.strength ?? activeSettings.intensity ?? 1.5,
      luminanceThreshold: activeSettings.threshold ?? 0.1, 
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
      
      
      if ((obj as THREE.Mesh).isMesh) {
        const mesh = obj as THREE.Mesh;
        const material = mesh.material as THREE.MeshStandardMaterial;
        if (material?.isMeshStandardMaterial) {
          
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