import * as THREE from 'three';
import { createLogger } from '@/utils/logger';

const logger = createLogger('DiffuseWireframeMaterial');

// Custom shader material for diffuse wireframe rendering
export class DiffuseWireframeMaterial extends THREE.ShaderMaterial {
  constructor(parameters: {
    color?: THREE.ColorRepresentation;
    opacity?: number;
    glowIntensity?: number;
    diffuseRadius?: number;
    wireframeThickness?: number;
    distanceFieldScale?: number;
    animated?: boolean;
    time?: number;
  } = {}) {
    const {
      color = 0x00ffff,
      opacity = 0.7,
      glowIntensity = 0.8,
      diffuseRadius = 2.0,
      wireframeThickness = 0.002,
      distanceFieldScale = 1.0,
      animated = true,
      time = 0
    } = parameters;

    super({
      uniforms: {
        time: { value: time },
        color: { value: new THREE.Color(color) },
        opacity: { value: opacity },
        glowIntensity: { value: glowIntensity },
        diffuseRadius: { value: diffuseRadius },
        wireframeThickness: { value: wireframeThickness },
        distanceFieldScale: { value: distanceFieldScale },
        animated: { value: animated },
        resolution: { value: new THREE.Vector2(1024, 1024) }
      },
      
      vertexShader: `
        uniform float time;
        uniform bool animated;
        
        varying vec3 vPosition;
        varying vec3 vNormal;
        varying vec2 vUv;
        varying float vDistanceFromCenter;
        
        void main() {
          vUv = uv;
          vNormal = normalize(normalMatrix * normal);
          vPosition = position;
          
          // Calculate distance from center for effects
          vDistanceFromCenter = length(position);
          
          vec3 pos = position;
          
          // Apply subtle animation to vertices if enabled
          if (animated) {
            float pulse = sin(time * 2.0 + vDistanceFromCenter * 0.5) * 0.02;
            pos += normal * pulse;
          }
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      
      fragmentShader: `
        uniform float time;
        uniform vec3 color;
        uniform float opacity;
        uniform float glowIntensity;
        uniform float diffuseRadius;
        uniform float wireframeThickness;
        uniform float distanceFieldScale;
        uniform bool animated;
        uniform vec2 resolution;
        
        varying vec3 vPosition;
        varying vec3 vNormal;
        varying vec2 vUv;
        varying float vDistanceFromCenter;
        
        // Distance field function for smooth wireframe
        float wireframeDF(vec3 pos, vec3 normal) {
          // Create distance field based on wireframe structure
          vec3 derivative = fwidth(pos);
          vec3 grid = abs(fract(pos) - 0.5) / derivative;
          float line = min(min(grid.x, grid.y), grid.z);
          
          return smoothstep(0.0, wireframeThickness * distanceFieldScale, line);
        }
        
        // Diffuse glow calculation
        float calculateDiffuseGlow(float distanceField, float radius) {
          float glow = 1.0 - smoothstep(0.0, radius, distanceField);
          return glow * glow; // Quadratic falloff for softer edges
        }
        
        // Fresnel effect for enhanced wireframe appearance
        float fresnel(vec3 normal, vec3 viewDir, float power) {
          return pow(1.0 - max(0.0, dot(normal, viewDir)), power);
        }
        
        void main() {
          vec3 viewDir = normalize(cameraPosition - vPosition);
          
          // Calculate distance field for wireframe
          float df = wireframeDF(vPosition * 2.0, vNormal);
          
          // Base wireframe intensity
          float wireframe = 1.0 - df;
          
          // Apply diffuse glow effect
          float diffuseGlow = calculateDiffuseGlow(df, diffuseRadius);
          
          // Fresnel rim lighting for depth
          float fresnelEffect = fresnel(vNormal, viewDir, 2.0);
          
          // Combine effects
          float totalIntensity = wireframe + diffuseGlow * glowIntensity + fresnelEffect * 0.3;
          
          // Apply time-based animation if enabled
          float animationFactor = 1.0;
          if (animated) {
            animationFactor = sin(time * 1.5 + vDistanceFromCenter * 0.3) * 0.2 + 0.8;
          }
          
          // Calculate final color
          vec3 finalColor = color * totalIntensity * animationFactor;
          
          // Add subtle sparkle effect for enhanced visual appeal
          float sparkle = sin(time * 10.0 + vDistanceFromCenter * 50.0) * 0.1 + 0.9;
          finalColor *= sparkle;
          
          // Distance-based opacity fade
          float distanceFade = 1.0 - smoothstep(50.0, 100.0, vDistanceFromCenter);
          float finalOpacity = opacity * distanceFade * totalIntensity;
          
          gl_FragColor = vec4(finalColor, finalOpacity);
        }
      `,
      
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    logger.debug('DiffuseWireframeMaterial created', parameters);
  }

  // Update time for animation
  public updateTime(time: number): void {
    this.uniforms.time.value = time;
  }

  // Update material properties
  public updateProperties(properties: {
    color?: THREE.ColorRepresentation;
    opacity?: number;
    glowIntensity?: number;
    diffuseRadius?: number;
    wireframeThickness?: number;
    distanceFieldScale?: number;
    animated?: boolean;
  }): void {
    if (properties.color !== undefined) {
      this.uniforms.color.value.set(properties.color);
    }
    if (properties.opacity !== undefined) {
      this.uniforms.opacity.value = properties.opacity;
    }
    if (properties.glowIntensity !== undefined) {
      this.uniforms.glowIntensity.value = properties.glowIntensity;
    }
    if (properties.diffuseRadius !== undefined) {
      this.uniforms.diffuseRadius.value = properties.diffuseRadius;
    }
    if (properties.wireframeThickness !== undefined) {
      this.uniforms.wireframeThickness.value = properties.wireframeThickness;
    }
    if (properties.distanceFieldScale !== undefined) {
      this.uniforms.distanceFieldScale.value = properties.distanceFieldScale;
    }
    if (properties.animated !== undefined) {
      this.uniforms.animated.value = properties.animated;
    }
  }

  // Set resolution for proper scaling
  public setResolution(width: number, height: number): void {
    this.uniforms.resolution.value.set(width, height);
  }
}

// Enhanced hologram ring material with diffuse effects
export class DiffuseHologramRingMaterial extends DiffuseWireframeMaterial {
  constructor(parameters: {
    color?: THREE.ColorRepresentation;
    opacity?: number;
    innerRadius?: number;
    outerRadius?: number;
    rotationSpeed?: number;
    glowIntensity?: number;
  } = {}) {
    const {
      color = 0x00ffff,
      opacity = 0.7,
      innerRadius = 0.8,
      outerRadius = 1.0,
      rotationSpeed = 0.5,
      glowIntensity = 0.9
    } = parameters;

    super({
      color,
      opacity,
      glowIntensity,
      animated: true
    });

    // Add ring-specific uniforms
    this.uniforms.innerRadius = { value: innerRadius };
    this.uniforms.outerRadius = { value: outerRadius };
    this.uniforms.rotationSpeed = { value: rotationSpeed };

    // Override fragment shader for ring-specific effects
    this.fragmentShader = `
      uniform float time;
      uniform vec3 color;
      uniform float opacity;
      uniform float glowIntensity;
      uniform float innerRadius;
      uniform float outerRadius;
      uniform float rotationSpeed;
      
      varying vec3 vPosition;
      varying vec3 vNormal;
      varying vec2 vUv;
      varying float vDistanceFromCenter;
      
      void main() {
        vec2 center = vec2(0.5, 0.5);
        float dist = distance(vUv, center);
        
        // Ring mask
        float ringMask = smoothstep(innerRadius - 0.02, innerRadius, dist) * 
                        smoothstep(outerRadius, outerRadius - 0.02, dist);
        
        // Rotating pattern
        float angle = atan(vUv.y - center.y, vUv.x - center.x);
        float rotation = time * rotationSpeed;
        float pattern = sin(angle * 8.0 + rotation) * 0.5 + 0.5;
        
        // Diffuse glow effect
        float glow = 1.0 - smoothstep(0.0, 0.3, abs(dist - (innerRadius + outerRadius) * 0.5));
        glow *= glowIntensity;
        
        // Combine effects
        float intensity = ringMask * (pattern * 0.7 + 0.3) + glow * 0.5;
        
        // Apply pulsing animation
        float pulse = sin(time * 2.0) * 0.2 + 0.8;
        intensity *= pulse;
        
        vec3 finalColor = color * intensity;
        gl_FragColor = vec4(finalColor, intensity * opacity);
      }
    `;

    this.needsUpdate = true;
  }
}

// Mote particle material with diffuse effects
export class DiffuseMoteMaterial extends THREE.ShaderMaterial {
  constructor(parameters: {
    color?: THREE.ColorRepresentation;
    size?: number;
    opacity?: number;
    density?: number;
    speed?: number;
  } = {}) {
    const {
      color = 0x00ffff,
      size = 1.0,
      opacity = 0.8,
      density = 0.5,
      speed = 0.3
    } = parameters;

    super({
      uniforms: {
        time: { value: 0 },
        color: { value: new THREE.Color(color) },
        size: { value: size },
        opacity: { value: opacity },
        density: { value: density },
        speed: { value: speed }
      },
      
      vertexShader: `
        uniform float time;
        uniform float size;
        
        varying vec2 vUv;
        varying float vRandom;
        
        // Noise function for mote positioning
        float random(vec2 st) {
          return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
        }
        
        void main() {
          vUv = uv;
          vRandom = random(position.xy);
          
          // Apply random movement for mote particles
          vec3 pos = position;
          pos.x += sin(time * speed + vRandom * 10.0) * 0.1;
          pos.y += cos(time * speed * 0.7 + vRandom * 8.0) * 0.05;
          pos.z += sin(time * speed * 1.3 + vRandom * 6.0) * 0.08;
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
          gl_PointSize = size * (1.0 + sin(time * 2.0 + vRandom * 5.0) * 0.3);
        }
      `,
      
      fragmentShader: `
        uniform float time;
        uniform vec3 color;
        uniform float opacity;
        uniform float density;
        
        varying vec2 vUv;
        varying float vRandom;
        
        void main() {
          // Create circular motes
          vec2 center = gl_PointCoord - vec2(0.5);
          float dist = length(center);
          
          if (dist > 0.5) discard;
          
          // Soft edge falloff
          float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
          
          // Flickering effect
          float flicker = sin(time * 3.0 + vRandom * 20.0) * 0.3 + 0.7;
          
          // Apply density-based visibility
          if (vRandom > density) discard;
          
          gl_FragColor = vec4(color, alpha * opacity * flicker);
        }
      `,
      
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    logger.debug('DiffuseMoteMaterial created', parameters);
  }

  public updateTime(time: number): void {
    this.uniforms.time.value = time;
  }
}

export default {
  DiffuseWireframeMaterial,
  DiffuseHologramRingMaterial,
  DiffuseMoteMaterial
};