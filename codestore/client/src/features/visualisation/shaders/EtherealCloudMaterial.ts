import * as THREE from 'three';

// Ethereal cloud shader with HEAVY gaussian blur effect
export class EtherealCloudMaterial extends THREE.ShaderMaterial {
  constructor(parameters: {
    color?: THREE.Color | string;
    opacity?: number;
    cloudScale?: number;  // How large the cloud extends beyond geometry
    blurSpread?: number;  // How diffused the cloud is
    cloudDensity?: number; // How opaque the cloud center is
  } = {}) {
    const params = {
      color: new THREE.Color(parameters.color || '#00ffff'),
      opacity: parameters.opacity || 0.3,
      cloudScale: parameters.cloudScale || 5.0,  // 5x the size of geometry
      blurSpread: parameters.blurSpread || 10.0, // Very heavy blur
      cloudDensity: parameters.cloudDensity || 0.5
    };

    super({
      uniforms: {
        time: { value: 0 },
        color: { value: params.color },
        opacity: { value: params.opacity },
        cloudScale: { value: params.cloudScale },
        blurSpread: { value: params.blurSpread },
        cloudDensity: { value: params.cloudDensity },
        resolution: { value: new THREE.Vector2(1024, 1024) }
      },

      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec3 vViewPosition;
        varying vec2 vUv;
        
        uniform float cloudScale;
        uniform float time;
        
        void main() {
          vUv = uv;
          vNormal = normalize(normalMatrix * normal);
          
          // Expand vertices outward to create larger cloud area
          vec3 expandedPosition = position + normal * cloudScale;
          
          // Add some animation to the cloud
          float wave = sin(time * 0.5 + position.x * 0.1) * 0.5;
          expandedPosition += normal * wave;
          
          vec4 worldPosition = modelMatrix * vec4(expandedPosition, 1.0);
          vWorldPosition = worldPosition.xyz;
          
          vec4 viewPosition = viewMatrix * worldPosition;
          vViewPosition = viewPosition.xyz;
          
          gl_Position = projectionMatrix * viewPosition;
        }
      `,

      fragmentShader: `
        uniform vec3 color;
        uniform float opacity;
        uniform float time;
        uniform float cloudScale;
        uniform float blurSpread;
        uniform float cloudDensity;
        uniform vec2 resolution;
        
        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec3 vViewPosition;
        varying vec2 vUv;
        
        // Smooth noise for cloud texture
        float hash(vec2 p) {
          p = fract(p * vec2(123.34, 456.21));
          p += dot(p, p + 45.32);
          return fract(p.x * p.y);
        }
        
        float noise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          
          float a = hash(i);
          float b = hash(i + vec2(1.0, 0.0));
          float c = hash(i + vec2(0.0, 1.0));
          float d = hash(i + vec2(1.0, 1.0));
          
          return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
        
        // Multi-octave noise for cloud detail
        float cloudNoise(vec2 uv) {
          float n = 0.0;
          float amplitude = 1.0;
          float frequency = 1.0;
          
          for(int i = 0; i < 4; i++) {
            n += noise(uv * frequency + time * 0.1) * amplitude;
            amplitude *= 0.5;
            frequency *= 2.0;
          }
          
          return n;
        }
        
        // Heavy gaussian blur approximation
        float gaussianBlur(vec2 center, vec2 point, float spread) {
          float dist = distance(center, point);
          return exp(-(dist * dist) / (2.0 * spread * spread));
        }
        
        void main() {
          // Calculate view-dependent effects
          vec3 viewDir = normalize(-vViewPosition);
          float fresnel = pow(1.0 - abs(dot(viewDir, vNormal)), 1.5);
          
          // Create cloud density based on distance from center
          vec2 center = vec2(0.5, 0.5);
          float cloudBase = gaussianBlur(center, vUv, blurSpread * 0.1);
          
          // Add cloud noise for ethereal effect
          vec2 noiseUv = vUv * 3.0;
          float cloudTexture = cloudNoise(noiseUv);
          
          // Combine for final cloud density
          float density = cloudBase * cloudDensity;
          density *= (0.5 + cloudTexture * 0.5);
          
          // Add rotating ethereal wisps
          float angle = atan(vUv.y - 0.5, vUv.x - 0.5);
          float spiral = sin(angle * 3.0 - time * 2.0) * 0.5 + 0.5;
          density *= (0.7 + spiral * 0.3);
          
          // Edge fade for soft boundaries
          float edgeFade = 1.0 - smoothstep(0.0, 1.0, length(vUv - center) * 2.0);
          density *= edgeFade;
          
          // Add fresnel for depth
          density += fresnel * 0.2;
          
          // Animate with pulsing
          float pulse = sin(time * 1.5) * 0.1 + 0.9;
          density *= pulse;
          
          // Final color with glow
          vec3 glowColor = color * (1.0 + fresnel * 2.0);
          
          // Alpha calculation - very transparent with soft edges
          float alpha = density * opacity;
          alpha = clamp(alpha, 0.0, 1.0);
          
          // Ensure some minimum visibility
          alpha = max(alpha, opacity * 0.05);
          
          gl_FragColor = vec4(glowColor, alpha);
        }
      `,

      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: true,
      vertexColors: false,
      wireframe: false  // We want solid geometry to blur
    });
  }

  updateTime(time: number) {
    this.uniforms.time.value = time;
  }

  updateCloud(scale: number, spread: number, density: number, opacity: number) {
    this.uniforms.cloudScale.value = scale;
    this.uniforms.blurSpread.value = spread;
    this.uniforms.cloudDensity.value = density;
    this.uniforms.opacity.value = opacity;
  }
}

// Multi-layer cloud material for even more diffusion
export class MultiLayerCloudMaterial {
  layers: EtherealCloudMaterial[];
  
  constructor(parameters: {
    color?: THREE.Color | string;
    opacity?: number;
    layers?: number;
  } = {}) {
    const layerCount = parameters.layers || 3;
    this.layers = [];
    
    for (let i = 0; i < layerCount; i++) {
      const scale = 3.0 + i * 2.0;  // Each layer gets progressively larger
      const opacity = (parameters.opacity || 0.3) / layerCount;  // Distribute opacity
      
      this.layers.push(new EtherealCloudMaterial({
        color: parameters.color,
        opacity: opacity,
        cloudScale: scale,
        blurSpread: 10.0 + i * 5.0,  // Each layer more blurred
        cloudDensity: 0.5 - i * 0.1   // Outer layers less dense
      }));
    }
  }
  
  updateTime(time: number) {
    this.layers.forEach((layer, i) => {
      layer.updateTime(time + i * 0.5);  // Offset time for each layer
    });
  }
}