import * as THREE from 'three';
import { extend } from '@react-three/fiber';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('HologramNodeMaterial');



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

    
    float distance = length(cameraPosition - vWorldPosition);
    float distanceFade = 1.0 - smoothstep(100.0, 500.0, distance); 
    alpha *= distanceFade;
    
    
    alpha = max(alpha, 0.1);

    gl_FragColor = vec4(color, alpha);
  }
`;


export class HologramNodeMaterial extends THREE.ShaderMaterial {
  constructor(parameters?: {
    
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
    this.uniforms.baseColor.value = new THREE.Color(baseColor);
    this.uniforms.emissiveColor.value = new THREE.Color(emissiveColor || baseColor);
    
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


export const HologramNodePresets = {
  
  Standard: new HologramNodeMaterial({
    baseColor: '#00ffff',
    emissiveColor: '#00ffff',
    glowStrength: 1.0,
    enableHologram: true,
    hologramStrength: 0.3
  }),
  
  
  HighPriority: new HologramNodeMaterial({
    baseColor: '#ff0080',
    emissiveColor: '#ff0080',
    glowStrength: 1.5,
    enableHologram: true,
    hologramStrength: 0.5,
    scanlineSpeed: 3.0
  }),
  
  
  Subtle: new HologramNodeMaterial({
    baseColor: '#0066ff',
    emissiveColor: '#0066ff',
    opacity: 0.6,
    glowStrength: 0.5,
    enableHologram: true,
    hologramStrength: 0.1
  }),
  
  
  Performance: new HologramNodeMaterial({
    baseColor: '#00ff88',
    emissiveColor: '#00ff88',
    enableHologram: false,
    glowStrength: 0.8
  })
};

export default HologramNodeMaterial;