import * as THREE from 'three';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('BloomStandardMaterial');

/**
 * BLOOM STANDARD MATERIAL
 * 
 * A specialized material designed to work seamlessly with the SelectiveBloom component.
 * This material is optimized for objects that should glow in the post-processing
 * bloom/glow effects without creating their own shader-based glow effects.
 * 
 * DESIGN PRINCIPLES:
 * - Uses standard PBR shading with high emissive values
 * - Lets the post-processing pipeline handle the actual bloom effect
 * - Tone mapping disabled to prevent bloom interference
 * - Optimized for performance with minimal shader overhead
 * 
 * USAGE:
 * - Assign objects using this material to Layer 1 (graph) or Layer 2 (environment)
 * - The SelectiveBloom will automatically apply the appropriate bloom effect
 * - Adjust emissiveIntensity to control bloom contribution
 */
export class BloomStandardMaterial extends THREE.MeshStandardMaterial {
  constructor(parameters: {
    /** Base color of the material */
    color?: THREE.Color | string;
    /** Emissive color for bloom effect - defaults to same as base color */
    emissive?: THREE.Color | string;
    /** Intensity of emissive glow - higher values create stronger bloom */
    emissiveIntensity?: number;
    /** Overall opacity of the material */
    opacity?: number;
    /** Whether to render as wireframe */
    wireframe?: boolean;
    /** Surface roughness (0 = mirror, 1 = rough) */
    roughness?: number;
    /** Metallic appearance (0 = dielectric, 1 = metallic) */
    metalness?: number;
  } = {}) {
    
    super({
      // Base material properties
      color: parameters.color || '#00ffff',
      emissive: parameters.emissive || parameters.color || '#00ffff',
      emissiveIntensity: parameters.emissiveIntensity || 2.0,
      opacity: parameters.opacity || 0.8,
      
      // Rendering properties
      transparent: true,
      wireframe: parameters.wireframe !== false, // Default to wireframe
      
      // PBR properties optimized for bloom
      roughness: parameters.roughness ?? 0.3, // Slight roughness for realistic reflection
      metalness: parameters.metalness ?? 0.8,  // High metalness for sci-fi appearance
      
      // Bloom optimization settings
      toneMapped: false,                  // Critical: Disable tone mapping for proper bloom
      blending: THREE.NormalBlending,     // Normal blending preserves depth sorting
      
      // Transparency settings
      depthWrite: false,                  // Don't write to depth buffer for better transparency
      depthTest: true,                    // But still test against depth for proper sorting
      side: THREE.DoubleSide              // Render both sides for wireframe visibility
    });
    
    // Debug logging gated by performance debug setting
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Material created', {
        color: this.color.getHexString(),
        emissive: this.emissive.getHexString(),
        emissiveIntensity: this.emissiveIntensity,
        wireframe: this.wireframe
      });
    }
  }
  
  /**
   * Updates the material colors dynamically
   */
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
  
  /**
   * Updates the bloom intensity (emissive intensity)
   */
  updateBloomIntensity(intensity: number) {
    this.emissiveIntensity = intensity;
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Bloom intensity updated', { intensity });
    }
  }
  
  /**
   * Creates a copy of this material with modified parameters
   */
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

/**
 * PRESET CONFIGURATIONS
 * 
 * Pre-configured materials for common use cases
 */
export const BloomStandardPresets = {
  /**
   * High-intensity material for primary graph elements (nodes, important edges)
   */
  GraphPrimary: new BloomStandardMaterial({
    color: '#00ffff',
    emissiveIntensity: 3.0,
    opacity: 0.9,
    wireframe: true,
    roughness: 0.2,
    metalness: 0.9
  }),
  
  /**
   * Medium-intensity material for secondary graph elements
   */
  GraphSecondary: new BloomStandardMaterial({
    color: '#0099ff',
    emissiveIntensity: 2.0,
    opacity: 0.7,
    wireframe: true,
    roughness: 0.4,
    metalness: 0.7
  }),
  
  /**
   * Soft glow material for environmental elements (particles, atmosphere)
   */
  EnvironmentGlow: new BloomStandardMaterial({
    color: '#00ffaa',
    emissiveIntensity: 1.5,
    opacity: 0.6,
    wireframe: false,
    roughness: 0.8,
    metalness: 0.3
  }),
  
  /**
   * Subtle material for background hologram elements
   */
  HologramSubtle: new BloomStandardMaterial({
    color: '#0066ff',
    emissiveIntensity: 1.0,
    opacity: 0.4,
    wireframe: true,
    roughness: 0.6,
    metalness: 0.5
  })
};

export default BloomStandardMaterial;