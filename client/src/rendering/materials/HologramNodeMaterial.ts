import * as THREE from 'three';
import { extend } from '@react-three/fiber';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('HologramNodeMaterial');



// Vertex shader with instancing support and vertex displacement
const hologramVertexShader = `
  uniform float time;
  uniform float pulseSpeed;
  uniform float pulseStrength;
  uniform int shaderMode;

  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying vec3 vInstanceColor;
  varying float vViewAngle;

  void main() {
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);


    #ifdef USE_INSTANCING_COLOR
      vInstanceColor = instanceColor;
    #else
      vInstanceColor = vec3(1.0);
    #endif


    vec3 transformed = position;


    vec4 worldPosition = modelMatrix * instanceMatrix * vec4(transformed, 1.0);
    float displacement = sin(time * pulseSpeed + worldPosition.x * 0.1) * pulseStrength;
    worldPosition.xyz += normalize(normalMatrix * normal) * displacement * 0.1;

    vWorldPosition = worldPosition.xyz;

    // Compute view angle for crystal dispersion (used in fragment)
    vec3 viewDir = normalize(cameraPosition - worldPosition.xyz);
    vViewAngle = dot(viewDir, vNormal);

    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

// Fragment shader with holographic effects and graph-mode visual branches
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
  uniform float pulseSpeed;
  uniform float pulseStrength;
  uniform int shaderMode;
  uniform float metalness;
  uniform float roughness;

  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying vec3 vInstanceColor;
  varying float vViewAngle;

  void main() {
    vec3 viewDirection = normalize(cameraPosition - vWorldPosition);


    vec3 color = mix(baseColor, vInstanceColor, 0.9);


    float rim = 1.0 - max(dot(viewDirection, vNormal), 0.0);
    rim = pow(rim, rimPower);


    float scanline = 0.0;
    if (enableHologram) {
      float scan = sin(vWorldPosition.y * scanlineCount + time * scanlineSpeed);
      scanline = smoothstep(0.0, 0.1, scan) * hologramStrength;
    }


    float glitch = 0.0;
    if (enableHologram) {
      float glitchTime = time * 10.0;
      glitch = step(0.99, sin(glitchTime * 1.0 + vWorldPosition.y * 12.0)) * 0.1;
    }


    float totalGlow = rim + scanline + glitch;
    vec3 emission = emissiveColor * totalGlow * glowStrength;
    color += emission;


    float alpha = mix(opacity, 1.0, rim * 0.5);
    alpha *= (1.0 - glitch * 0.5);


    // ---------------------------------------------------------------
    // Mode-specific visual effects
    // ---------------------------------------------------------------

    if (shaderMode == 0) {
      // -- MODE 0: Crystal (Knowledge Graph) --

      // Crystal refraction / facet noise
      float facetNoise = sin(vWorldPosition.x * 8.0 + vWorldPosition.y * 6.0 + vWorldPosition.z * 7.0);
      float crystalFacet = smoothstep(0.3, 0.7, facetNoise) * 0.15;

      // Inner glow (knowledge density)
      float innerGlow = pow(max(0.0, 1.0 - length(vPosition) * 1.5), 2.0);

      // Spectral dispersion (rainbow flash on rotation)
      float viewAngle = vViewAngle;
      vec3 dispersion = vec3(
        sin(viewAngle * 3.0 + 0.0) * 0.1,
        sin(viewAngle * 3.0 + 2.1) * 0.1,
        sin(viewAngle * 3.0 + 4.2) * 0.1
      );

      // Metallic specular highlight
      float specular = pow(max(0.0, vViewAngle), mix(8.0, 64.0, 1.0 - roughness)) * metalness;

      color += emissiveColor * innerGlow * glowStrength;
      color += dispersion * rim;
      color += crystalFacet;
      color += vec3(specular) * 0.3;

    } else if (shaderMode == 1) {
      // -- MODE 1: Constellation (Ontology) --

      // Orbital ring effect
      float ring = abs(sin(vWorldPosition.y * scanlineCount * 0.5 + time * scanlineSpeed));
      float orbitalRing = smoothstep(0.95, 1.0, ring) * hologramStrength;

      // Nebula glow (soft, atmospheric)
      float nebulaGlow = pow(rim, 1.5) * glowStrength * 0.5;

      // Star twinkle (position-based noise modulated by time)
      float twinkle = sin(vWorldPosition.x * 12.0 + time * 2.0)
                    * sin(vWorldPosition.z * 10.0 + time * 1.7) * 0.08;

      // Depth-based ethereal fade
      float ethereal = mix(1.0, 0.6, smoothstep(0.0, 1.0, vWorldPosition.y * 0.01));

      color += emissiveColor * nebulaGlow;
      color += vec3(orbitalRing) * emissiveColor * 0.8;
      color += emissiveColor * max(0.0, twinkle);
      alpha *= ethereal;

    } else if (shaderMode == 2) {
      // -- MODE 2: Organic (Agents) --

      // Membrane effect (subsurface scattering approximation)
      float membrane = pow(rim, 2.0) * 0.8;
      float subsurface = pow(max(0.0, dot(viewDirection, -vNormal)), 3.0) * 0.3;

      // Cytoplasm flow (internal churning)
      float flow = sin(vPosition.x * 4.0 + time * pulseSpeed * 2.0)
                 * cos(vPosition.y * 3.0 + time * pulseSpeed * 1.5)
                 * sin(vPosition.z * 5.0 + time * pulseSpeed) * 0.15;

      // Heartbeat pulse
      float heartbeat = pow(sin(time * pulseSpeed * 3.14159) * 0.5 + 0.5, 4.0) * pulseStrength;

      // Nucleus glow (center brighter)
      float nucleus = pow(max(0.0, 1.0 - length(vPosition) * 2.0), 3.0) * 0.2;

      color += emissiveColor * (membrane + subsurface);
      color += emissiveColor * flow;
      color += emissiveColor * heartbeat * 0.3;
      color += emissiveColor * nucleus;
    }


    // Distance-based fade (shared across all modes)
    float distance = length(cameraPosition - vWorldPosition);
    float distanceFade = 1.0 - smoothstep(100.0, 500.0, distance);
    alpha *= distanceFade;


    alpha = max(alpha, 0.1);

    gl_FragColor = vec4(color, alpha);
  }
`;


export interface HologramNodeMaterialParams {
  baseColor?: THREE.Color | string;
  emissiveColor?: THREE.Color | string;
  opacity?: number;
  enableHologram?: boolean;
  scanlineSpeed?: number;
  scanlineCount?: number;
  glowStrength?: number;
  rimPower?: number;
  pulseSpeed?: number;
  pulseStrength?: number;
  hologramStrength?: number;
  /** 0 = crystal (knowledge graph), 1 = constellation (ontology), 2 = organic (agent) */
  shaderMode?: number;
  metalness?: number;
  roughness?: number;
}

export class HologramNodeMaterial extends THREE.ShaderMaterial {
  constructor(parameters?: HologramNodeMaterialParams) {
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
      shaderMode: parameters?.shaderMode ?? 0,
      metalness: parameters?.metalness ?? 0.0,
      roughness: parameters?.roughness ?? 0.5,
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
        shaderMode: { value: params.shaderMode },
        metalness: { value: params.metalness },
        roughness: { value: params.roughness },
      },
      vertexShader: hologramVertexShader,
      fragmentShader: hologramFragmentShader,
      
      
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false, 
      depthTest: true,   
      blending: THREE.NormalBlending, 
      
      
      toneMapped: false  
    });
    
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Material created', {
        baseColor: params.baseColor.getHexString(),
        emissiveColor: params.emissiveColor.getHexString(),
        enableHologram: params.enableHologram,
        glowStrength: params.glowStrength
      });
    }
  }

  
  updateTime(time: number) {
    this.uniforms.time.value = time;
  }

  
  updateColors(baseColor: string | THREE.Color, emissiveColor?: string | THREE.Color) {
    (this.uniforms.baseColor.value as THREE.Color).set(baseColor);
    (this.uniforms.emissiveColor.value as THREE.Color).set(emissiveColor || baseColor);
    
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Colors updated', {
        baseColor: this.uniforms.baseColor.value.getHexString(),
        emissiveColor: this.uniforms.emissiveColor.value.getHexString()
      });
    }
  }

  
  setHologramEnabled(enabled: boolean) {
    this.uniforms.enableHologram.value = enabled;
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Hologram effects ' + (enabled ? 'enabled' : 'disabled'));
    }
  }

  
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
      logger.debug('Hologram parameters updated', params);
    }
  }
  

  updateBloomContribution(glowStrength: number) {
    this.uniforms.glowStrength.value = glowStrength;
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Bloom contribution updated', { glowStrength });
    }
  }

  /**
   * Sets the shader visual mode.
   * 0 = crystal (knowledge graph), 1 = constellation (ontology), 2 = organic (agent)
   */
  setShaderMode(mode: number): void {
    this.uniforms.shaderMode.value = mode;
    if ((globalThis as any).__SETTINGS__?.system?.debug?.enablePerformanceDebug) {
      logger.debug('Shader mode set', { mode });
    }
  }

  /**
   * Updates metalness and roughness uniforms used by the crystal shader mode.
   */
  updateSurfaceParams(params: { metalness?: number; roughness?: number }): void {
    if (params.metalness !== undefined) {
      this.uniforms.metalness.value = params.metalness;
    }
    if (params.roughness !== undefined) {
      this.uniforms.roughness.value = params.roughness;
    }
  }


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
      shaderMode: this.uniforms.shaderMode.value,
      metalness: this.uniforms.metalness.value,
      roughness: this.uniforms.roughness.value,
    });

    return cloned as this;
  }
}

// Extend for use in React Three Fiber
extend({ HologramNodeMaterial });


interface HologramNodePresetsType {
  Standard: HologramNodeMaterial;
  HighPriority: HologramNodeMaterial;
  Subtle: HologramNodeMaterial;
  Performance: HologramNodeMaterial;
  CrystalKnowledge: HologramNodeMaterial;
  CosmicOntology: HologramNodeMaterial;
  OrganicAgent: HologramNodeMaterial;
}

let _hologramPresets: HologramNodePresetsType | null = null;

export function getHologramNodePresets(): HologramNodePresetsType {
  if (!_hologramPresets) {
    _hologramPresets = {
      Standard: new HologramNodeMaterial({
        baseColor: '#00ffff',
        emissiveColor: '#00ffff',
        glowStrength: 1.0,
        enableHologram: true,
        hologramStrength: 0.3,
      }),
      HighPriority: new HologramNodeMaterial({
        baseColor: '#ff0080',
        emissiveColor: '#ff0080',
        glowStrength: 1.5,
        enableHologram: true,
        hologramStrength: 0.5,
        scanlineSpeed: 3.0,
      }),
      Subtle: new HologramNodeMaterial({
        baseColor: '#0066ff',
        emissiveColor: '#0066ff',
        opacity: 0.6,
        glowStrength: 0.5,
        enableHologram: true,
        hologramStrength: 0.1,
      }),
      Performance: new HologramNodeMaterial({
        baseColor: '#00ff88',
        emissiveColor: '#00ff88',
        enableHologram: false,
        glowStrength: 0.8,
      }),

      // -- Graph-mode presets --

      /** Data Crystals: sharp faceted gems with inner glow and spectral dispersion */
      CrystalKnowledge: new HologramNodeMaterial({
        baseColor: '#E0E0E0',
        emissiveColor: '#E0E0E0',
        shaderMode: 0,
        rimPower: 3.0,
        scanlineCount: 0,
        scanlineSpeed: 0,
        glowStrength: 2.5,
        hologramStrength: 0.3,
        enableHologram: false,
        metalness: 0.6,
        roughness: 0.15,
        pulseSpeed: 0.8,
        pulseStrength: 0.1,
        opacity: 0.85,
      }),

      /** Cosmic Taxonomy: nebula glow with orbital rings and ethereal fade */
      CosmicOntology: new HologramNodeMaterial({
        baseColor: '#FFD93D',
        emissiveColor: '#FFD93D',
        shaderMode: 1,
        rimPower: 1.5,
        scanlineCount: 8,
        scanlineSpeed: 0.5,
        glowStrength: 1.8,
        hologramStrength: 0.7,
        enableHologram: true,
        metalness: 0.1,
        roughness: 0.4,
        pulseSpeed: 0.3,
        pulseStrength: 0.08,
        opacity: 0.8,
      }),

      /** Living Organisms: bioluminescent membrane with heartbeat and cytoplasm flow */
      OrganicAgent: new HologramNodeMaterial({
        baseColor: '#2ECC71',
        emissiveColor: '#2ECC71',
        shaderMode: 2,
        rimPower: 2.0,
        scanlineCount: 0,
        scanlineSpeed: 0,
        glowStrength: 2.0,
        hologramStrength: 0.1,
        enableHologram: false,
        metalness: 0.0,
        roughness: 0.7,
        pulseSpeed: 1.5,
        pulseStrength: 0.4,
        opacity: 0.75,
      }),
    };
  }
  return _hologramPresets;
}

export function disposeHologramNodePresets(): void {
  if (_hologramPresets) {
    _hologramPresets.Standard.dispose();
    _hologramPresets.HighPriority.dispose();
    _hologramPresets.Subtle.dispose();
    _hologramPresets.Performance.dispose();
    _hologramPresets.CrystalKnowledge.dispose();
    _hologramPresets.CosmicOntology.dispose();
    _hologramPresets.OrganicAgent.dispose();
    _hologramPresets = null;
  }
}

export default HologramNodeMaterial;