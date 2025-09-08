import * as THREE from 'three';
import { extend } from '@react-three/fiber';

/**
 * HOLOGRAM NODE MATERIAL
 * 
 * A specialized shader material for rendering data visualization nodes with
 * holographic effects including scanlines, rim lighting, and glitch effects.
 * 
 * This material is designed to work with the SelectiveBloom's Graph Pipeline
 * (Layer 1) and provides unique visual effects that are NOT replicated by the
 * post-processing bloom pass:
 * 
 * UNIQUE FEATURES (not in post-processing):
 * - Animated scanlines for holographic appearance
 * - Rim lighting for edge definition
 * - Glitch effects for sci-fi aesthetic
 * - Instance color support for per-node coloring
 * - Vertex displacement for organic animation
 * 
 * BLOOM INTEGRATION:
 * - glowStrength uniform controls bloom contribution
 * - Emissive color designed to work with post-processing
 * - toneMapped disabled for proper bloom interaction
 */

// Vertex shader with instancing support and vertex displacement
const hologramVertexShader = `
  uniform float time;
  uniform float pulseSpeed;
  uniform float pulseStrength;

  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying vec3 vInstanceColor;

  void main() {
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);

    // Get instance color from built-in instanceColor attribute
    #ifdef USE_INSTANCING_COLOR
      vInstanceColor = instanceColor;
    #else
      vInstanceColor = vec3(1.0);
    #endif

    // Apply instance transform (matrix already includes position and scale)
    vec3 transformed = position;

    // Add subtle vertex displacement for organic feel
    vec4 worldPosition = modelMatrix * instanceMatrix * vec4(transformed, 1.0);
    float displacement = sin(time * pulseSpeed + worldPosition.x * 0.1) * pulseStrength;
    worldPosition.xyz += normalize(normalMatrix * normal) * displacement * 0.1;

    vWorldPosition = worldPosition.xyz;

    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

// Fragment shader with holographic effects
const hologramFragmentShader = `
  uniform float time;
  uniform vec3 baseColor;
  uniform vec3 emissiveColor;
  uniform float opacity;
  uniform float scanlineSpeed;
  uniform float scanlineCount;
  uniform float glowStrength;
  uniform float rimPower;
  uniform bool enableHologram;
  uniform float hologramStrength;

  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying vec3 vInstanceColor;

  void main() {
    vec3 viewDirection = normalize(cameraPosition - vWorldPosition);

    // Base color with instance color modulation - favor instance color
    vec3 color = mix(baseColor, vInstanceColor, 0.9);

    // Fresnel rim lighting for edge definition
    float rim = 1.0 - max(dot(viewDirection, vNormal), 0.0);
    rim = pow(rim, rimPower);

    // Hologram scanlines - creates the classic holographic look
    float scanline = 0.0;
    if (enableHologram) {
      float scan = sin(vWorldPosition.y * scanlineCount + time * scanlineSpeed);
      scanline = smoothstep(0.0, 0.1, scan) * hologramStrength;
    }

    // Glitch effect for sci-fi aesthetic
    float glitch = 0.0;
    if (enableHologram) {
      float glitchTime = time * 10.0;
      glitch = step(0.99, sin(glitchTime * 1.0 + vWorldPosition.y * 12.0)) * 0.1;
    }

    // Combine effects - glowStrength acts as bloom multiplier
    float totalGlow = rim + scanline + glitch;
    vec3 emission = emissiveColor * totalGlow * glowStrength;
    color += emission;

    // Alpha with rim fade
    float alpha = mix(opacity, 1.0, rim * 0.5);
    alpha *= (1.0 - glitch * 0.5); // Flicker during glitch

    // Distance fade for depth - adjusted for better visibility
    float distance = length(cameraPosition - vWorldPosition);
    float distanceFade = 1.0 - smoothstep(100.0, 500.0, distance); // Increased range
    alpha *= distanceFade;
    
    // Ensure minimum alpha to prevent complete transparency
    alpha = max(alpha, 0.1);

    gl_FragColor = vec4(color, alpha);
  }
`;

/**
 * HologramNodeMaterial Class
 * 
 * A high-performance shader material for rendering graph nodes with
 * holographic effects and bloom pipeline integration.
 */
export class HologramNodeMaterial extends THREE.ShaderMaterial {
  constructor(parameters?: {
    /** Base color of the node */
    baseColor?: THREE.Color | string;
    /** Emissive color for bloom effects */
    emissiveColor?: THREE.Color | string;
    /** Overall opacity */
    opacity?: number;
    /** Enable/disable holographic effects */
    enableHologram?: boolean;
    /** Speed of scanline animation */
    scanlineSpeed?: number;
    /** Density of scanlines */
    scanlineCount?: number;
    /** Intensity multiplier for bloom contribution */
    glowStrength?: number;
    /** Power of rim lighting effect */
    rimPower?: number;
    /** Speed of vertex pulse animation */
    pulseSpeed?: number;
    /** Strength of vertex displacement */
    pulseStrength?: number;
    /** Intensity of holographic effects */
    hologramStrength?: number;
  }) {
    
    const params = {
      baseColor: new THREE.Color(parameters?.baseColor || '#00ffff'),
      emissiveColor: new THREE.Color(parameters?.emissiveColor || '#00ffff'),
      opacity: parameters?.opacity || 0.8,
      enableHologram: parameters?.enableHologram !== undefined ? parameters.enableHologram : true,
      scanlineSpeed: parameters?.scanlineSpeed || 2.0,
      scanlineCount: parameters?.scanlineCount || 30.0,
      glowStrength: parameters?.glowStrength || 1.0,
      rimPower: parameters?.rimPower || 2.0,
      pulseSpeed: parameters?.pulseSpeed || 1.0,
      pulseStrength: parameters?.pulseStrength || 0.1,
      hologramStrength: parameters?.hologramStrength || 0.3,
    };

    super({
      uniforms: {
        time: { value: 0 },
        baseColor: { value: params.baseColor },
        emissiveColor: { value: params.emissiveColor },
        opacity: { value: params.opacity },
        enableHologram: { value: params.enableHologram },
        scanlineSpeed: { value: params.scanlineSpeed },
        scanlineCount: { value: params.scanlineCount },
        glowStrength: { value: params.glowStrength },
        rimPower: { value: params.rimPower },
        pulseSpeed: { value: params.pulseSpeed },
        pulseStrength: { value: params.pulseStrength },
        hologramStrength: { value: params.hologramStrength },
      },
      vertexShader: hologramVertexShader,
      fragmentShader: hologramFragmentShader,
      
      // Optimized transparency settings
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false, // Don't write to depth buffer for transparent objects
      depthTest: true,   // But still test against depth for proper sorting
      blending: THREE.NormalBlending, // Use normal blending to prevent scene vanishing
      
      // Bloom optimization
      toneMapped: false  // Critical: Disable tone mapping for proper bloom
    });
    
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      console.log('HologramNodeMaterial: Created with parameters', {
        baseColor: params.baseColor.getHexString(),
        emissiveColor: params.emissiveColor.getHexString(),
        enableHologram: params.enableHologram,
        glowStrength: params.glowStrength
      });
    }
  }

  /**
   * Update time uniform for animations
   * Should be called each frame for animated effects
   */
  updateTime(time: number) {
    this.uniforms.time.value = time;
  }

  /**
   * Update colors from settings
   * Useful for dynamic theming or state changes
   */
  updateColors(baseColor: string | THREE.Color, emissiveColor?: string | THREE.Color) {
    this.uniforms.baseColor.value = new THREE.Color(baseColor);
    this.uniforms.emissiveColor.value = new THREE.Color(emissiveColor || baseColor);
    
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      console.log('HologramNodeMaterial: Updated colors', {
        baseColor: this.uniforms.baseColor.value.getHexString(),
        emissiveColor: this.uniforms.emissiveColor.value.getHexString()
      });
    }
  }

  /**
   * Toggle hologram effect on/off
   * Useful for performance optimization or style changes
   */
  setHologramEnabled(enabled: boolean) {
    this.uniforms.enableHologram.value = enabled;
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      console.log('HologramNodeMaterial: Hologram effects', enabled ? 'enabled' : 'disabled');
    }
  }

  /**
   * Update hologram parameters for fine-tuning effects
   */
  updateHologramParams(params: {
    scanlineSpeed?: number;
    scanlineCount?: number;
    glowStrength?: number;
    rimPower?: number;
    hologramStrength?: number;
  }) {
    if (params.scanlineSpeed !== undefined) {
      this.uniforms.scanlineSpeed.value = params.scanlineSpeed;
    }
    if (params.scanlineCount !== undefined) {
      this.uniforms.scanlineCount.value = params.scanlineCount;
    }
    if (params.glowStrength !== undefined) {
      this.uniforms.glowStrength.value = params.glowStrength;
    }
    if (params.rimPower !== undefined) {
      this.uniforms.rimPower.value = params.rimPower;
    }
    if (params.hologramStrength !== undefined) {
      this.uniforms.hologramStrength.value = params.hologramStrength;
    }
    
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      console.log('HologramNodeMaterial: Updated hologram parameters', params);
    }
  }
  
  /**
   * Update glow strength specifically for bloom pipeline integration
   */
  updateBloomContribution(glowStrength: number) {
    this.uniforms.glowStrength.value = glowStrength;
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      console.log('HologramNodeMaterial: Updated bloom contribution to', glowStrength);
    }
  }
  
  /**
   * Create a copy of this material with modified parameters
   */
  clone(): this {
    const cloned = new HologramNodeMaterial({
      baseColor: this.uniforms.baseColor.value,
      emissiveColor: this.uniforms.emissiveColor.value,
      opacity: this.uniforms.opacity.value,
      enableHologram: this.uniforms.enableHologram.value,
      scanlineSpeed: this.uniforms.scanlineSpeed.value,
      scanlineCount: this.uniforms.scanlineCount.value,
      glowStrength: this.uniforms.glowStrength.value,
      rimPower: this.uniforms.rimPower.value,
      pulseSpeed: this.uniforms.pulseSpeed.value,
      pulseStrength: this.uniforms.pulseStrength.value,
      hologramStrength: this.uniforms.hologramStrength.value,
    });
    
    return cloned as this;
  }
}

// Extend for use in React Three Fiber
extend({ HologramNodeMaterial });

/**
 * PRESET CONFIGURATIONS
 * 
 * Pre-configured materials for different node types
 */
export const HologramNodePresets = {
  /**
   * Standard data node with full holographic effects
   */
  Standard: new HologramNodeMaterial({
    baseColor: '#00ffff',
    emissiveColor: '#00ffff',
    glowStrength: 1.0,
    enableHologram: true,
    hologramStrength: 0.3
  }),
  
  /**
   * High-priority node with enhanced effects
   */
  HighPriority: new HologramNodeMaterial({
    baseColor: '#ff0080',
    emissiveColor: '#ff0080',
    glowStrength: 1.5,
    enableHologram: true,
    hologramStrength: 0.5,
    scanlineSpeed: 3.0
  }),
  
  /**
   * Subtle node with minimal effects
   */
  Subtle: new HologramNodeMaterial({
    baseColor: '#0066ff',
    emissiveColor: '#0066ff',
    opacity: 0.6,
    glowStrength: 0.5,
    enableHologram: true,
    hologramStrength: 0.1
  }),
  
  /**
   * Performance-optimized node with effects disabled
   */
  Performance: new HologramNodeMaterial({
    baseColor: '#00ff88',
    emissiveColor: '#00ff88',
    enableHologram: false,
    glowStrength: 0.8
  })
};

export default HologramNodeMaterial;