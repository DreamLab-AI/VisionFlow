import * as THREE from 'three';
import { createLogger } from '../../utils/loggerConfig';
import { useSettingsStore } from '../../store/settingsStore';

const logger = createLogger('BloomStandardMaterial');


export class BloomStandardMaterial extends THREE.MeshStandardMaterial {
  constructor(parameters: {
    
    color?: THREE.Color | string;
    
    emissive?: THREE.Color | string;
    
    emissiveIntensity?: number;
    
    opacity?: number;
    
    wireframe?: boolean;
    
    roughness?: number;
    
    metalness?: number;
  } = {}) {
    
    super({
      
      color: parameters.color || '#00ffff',
      emissive: parameters.emissive || parameters.color || '#00ffff',
      emissiveIntensity: parameters.emissiveIntensity || 2.0,
      opacity: parameters.opacity || 0.8,
      
      
      transparent: true,
      wireframe: parameters.wireframe !== false, 
      
      
      roughness: parameters.roughness ?? 0.3, 
      metalness: parameters.metalness ?? 0.8,  
      
      
      toneMapped: false,                  
      blending: THREE.NormalBlending,     
      
      
      depthWrite: false,                  
      depthTest: true,                    
      side: THREE.DoubleSide              
    });
    
    
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Material created', {
        color: this.color.getHexString(),
        emissive: this.emissive.getHexString(),
        emissiveIntensity: this.emissiveIntensity,
        wireframe: this.wireframe
      });
    }
  }
  
  
  updateColors(baseColor: string | THREE.Color, emissiveColor?: string | THREE.Color) {
    this.color = new THREE.Color(baseColor);
    this.emissive = new THREE.Color(emissiveColor || baseColor);
    this.needsUpdate = true;
    
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Colors updated', {
        color: this.color.getHexString(),
        emissive: this.emissive.getHexString()
      });
    }
  }
  
  
  updateBloomIntensity(intensity: number) {
    this.emissiveIntensity = intensity;
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Bloom intensity updated', { intensity });
    }
  }
  
  
  createVariant(overrides: {
    color?: THREE.Color | string;
    emissive?: THREE.Color | string;
    emissiveIntensity?: number;
    opacity?: number;
    wireframe?: boolean;
  }): BloomStandardMaterial {
    return new BloomStandardMaterial({
      color: overrides.color || this.color,
      emissive: overrides.emissive || this.emissive,
      emissiveIntensity: overrides.emissiveIntensity ?? this.emissiveIntensity,
      opacity: overrides.opacity ?? this.opacity,
      wireframe: overrides.wireframe ?? this.wireframe,
      roughness: this.roughness,
      metalness: this.metalness
    });
  }
}


export const createBloomStandardPresets = () => {
  
  const settings = useSettingsStore.getState().settings;
  const bloomSettings = settings?.visualisation?.bloom;
  const glowSettings = settings?.visualisation?.glow;

  
  const baseIntensity = bloomSettings?.intensity ?? 1.0;
  const glowStrength = glowSettings?.intensity ?? 3.2142856;

  return {
    
    GraphPrimary: new BloomStandardMaterial({
      color: '#00ffff',
      emissiveIntensity: baseIntensity * 3.0,
      opacity: 0.9,
      wireframe: true,
      roughness: 0.2,
      metalness: 0.9
    }),

    
    GraphSecondary: new BloomStandardMaterial({
      color: '#0099ff',
      emissiveIntensity: baseIntensity * 2.0,
      opacity: 0.7,
      wireframe: true,
      roughness: 0.4,
      metalness: 0.7
    }),

    
    EnvironmentGlow: new BloomStandardMaterial({
      color: '#00ffaa',
      emissiveIntensity: baseIntensity * 1.5,
      opacity: 0.6,
      wireframe: false,
      roughness: 0.8,
      metalness: 0.3
    }),

    
    HologramSubtle: new BloomStandardMaterial({
      color: '#0066ff',
      emissiveIntensity: baseIntensity * 1.0,
      opacity: 0.4,
      wireframe: true,
      roughness: 0.6,
      metalness: 0.5
    })
  };
};


type BloomStandardPresetsType = ReturnType<typeof createBloomStandardPresets>;
let _presets: BloomStandardPresetsType | null = null;

export function getBloomStandardPresets(): BloomStandardPresetsType {
  if (!_presets) {
    _presets = createBloomStandardPresets();
  }
  return _presets;
}

export function disposeBloomStandardPresets(): void {
  if (_presets) {
    _presets.GraphPrimary.dispose();
    _presets.GraphSecondary.dispose();
    _presets.EnvironmentGlow.dispose();
    _presets.HologramSubtle.dispose();
    _presets = null;
  }
}

export default BloomStandardMaterial;