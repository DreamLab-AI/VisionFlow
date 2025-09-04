import * as THREE from 'three';

// Bloom-like hologram shader with heavy blur and alpha blending
export class BloomHologramMaterial extends THREE.ShaderMaterial {
  constructor(parameters: {
    color?: THREE.Color | string;
    opacity?: number;
    glowRadius?: number;
    glowIntensity?: number;
    blurAmount?: number;
  } = {}) {
    const params = {
      color: new THREE.Color(parameters.color || '#00ffff'),
      opacity: parameters.opacity || 0.6,
      glowRadius: parameters.glowRadius || 2.0,
      glowIntensity: parameters.glowIntensity || 2.0,
      blurAmount: parameters.blurAmount || 3.0
    };

    super({
      uniforms: {
        time: { value: 0 },
        color: { value: params.color },
        opacity: { value: params.opacity },
        glowRadius: { value: params.glowRadius },
        glowIntensity: { value: params.glowIntensity },
        blurAmount: { value: params.blurAmount }
      },

      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        varying vec2 vUv;
        
        void main() {
          vUv = uv;
          vNormal = normalize(normalMatrix * normal);
          vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,

      fragmentShader: `
        uniform vec3 color;
        uniform float opacity;
        uniform float time;
        uniform float glowRadius;
        uniform float glowIntensity;
        uniform float blurAmount;
        
        varying vec3 vNormal;
        varying vec3 vPosition;
        varying vec2 vUv;
        
        // Creates a soft glow effect
        float getGlow(float dist, float radius, float intensity) {
          return pow(radius / dist, intensity);
        }
        
        // Multi-sample blur for bloom effect
        vec3 getBlurredColor(vec2 uv) {
          vec3 blurred = vec3(0.0);
          float total = 0.0;
          
          // 9-sample blur pattern for performance
          for(float x = -1.0; x <= 1.0; x += 1.0) {
            for(float y = -1.0; y <= 1.0; y += 1.0) {
              float weight = 1.0 / (1.0 + length(vec2(x, y)));
              vec2 offset = vec2(x, y) * blurAmount * 0.01;
              
              // Sample with offset for blur
              float dist = length(vUv + offset - vec2(0.5, 0.5));
              float glow = getGlow(dist, glowRadius * 0.3, glowIntensity);
              
              blurred += color * glow * weight;
              total += weight;
            }
          }
          
          return blurred / total;
        }
        
        void main() {
          // Get view direction for rim lighting
          vec3 viewDir = normalize(-vPosition);
          
          // Calculate rim lighting for edge glow
          float rim = 1.0 - max(0.0, dot(viewDir, vNormal));
          rim = pow(rim, 2.0);
          
          // Get distance from center for radial glow
          float dist = length(vUv - vec2(0.5, 0.5));
          
          // Core glow
          float coreGlow = getGlow(dist, glowRadius * 0.5, glowIntensity * 1.5);
          
          // Blurred outer glow for bloom effect
          vec3 bloomColor = getBlurredColor(vUv);
          
          // Combine core and bloom
          vec3 finalColor = color * coreGlow + bloomColor * 0.5;
          
          // Add rim lighting
          finalColor += color * rim * glowIntensity;
          
          // Animated pulse
          float pulse = sin(time * 2.0) * 0.1 + 0.9;
          finalColor *= pulse;
          
          // Calculate alpha with soft edges
          float alpha = opacity;
          alpha *= (coreGlow + rim) * 0.5;
          alpha *= smoothstep(1.0, 0.0, dist * 2.0); // Soft edge falloff
          
          // Ensure minimum visibility
          alpha = max(alpha, opacity * 0.1);
          
          gl_FragColor = vec4(finalColor, alpha);
        }
      `,

      transparent: true,
      blending: THREE.AdditiveBlending, // Additive for glow effect
      side: THREE.DoubleSide,
      depthWrite: false, // Don't write to depth buffer for transparency
      depthTest: true
    });
  }

  updateTime(time: number) {
    this.uniforms.time.value = time;
  }

  updateGlow(radius: number, intensity: number, blur: number) {
    this.uniforms.glowRadius.value = radius;
    this.uniforms.glowIntensity.value = intensity;
    this.uniforms.blurAmount.value = blur;
  }
}

// Standard material with bloom-friendly settings
export class BloomStandardMaterial extends THREE.MeshStandardMaterial {
  constructor(parameters: {
    color?: THREE.Color | string;
    emissive?: THREE.Color | string;
    emissiveIntensity?: number;
    opacity?: number;
    wireframe?: boolean;
  } = {}) {
    super({
      color: parameters.color || '#00ffff',
      emissive: parameters.emissive || parameters.color || '#00ffff',
      emissiveIntensity: parameters.emissiveIntensity || 2.0,
      opacity: parameters.opacity || 0.8,
      transparent: true,
      wireframe: parameters.wireframe !== false,
      roughness: 0.3,
      metalness: 0.8,
      toneMapped: false, // Disable tone mapping for bloom
      blending: THREE.NormalBlending
    });
  }
}