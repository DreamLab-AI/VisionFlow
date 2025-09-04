import * as THREE from 'three';

// Combines the best of DiffuseWireframeMaterial with EtherealCloudMaterial
// Creates an ultra-diffuse, heavily blurred cloud effect
export class EtherealDiffuseCloudMaterial extends THREE.ShaderMaterial {
  constructor(parameters: {
    color?: THREE.Color | string;
    opacity?: number;
    cloudScale?: number;
    blurSpread?: number;
    cloudDensity?: number;
    glowIntensity?: number;
    diffuseRadius?: number;
  } = {}) {
    const params = {
      color: new THREE.Color(parameters.color || '#00ffff'),
      opacity: parameters.opacity || 0.3,
      cloudScale: parameters.cloudScale || 5.0,
      blurSpread: parameters.blurSpread || 15.0, // Even heavier blur
      cloudDensity: parameters.cloudDensity || 0.5,
      glowIntensity: parameters.glowIntensity || 2.0,
      diffuseRadius: parameters.diffuseRadius || 3.0
    };

    super({
      uniforms: {
        time: { value: 0 },
        color: { value: params.color },
        opacity: { value: params.opacity },
        cloudScale: { value: params.cloudScale },
        blurSpread: { value: params.blurSpread },
        cloudDensity: { value: params.cloudDensity },
        glowIntensity: { value: params.glowIntensity },
        diffuseRadius: { value: params.diffuseRadius },
        resolution: { value: new THREE.Vector2(1024, 1024) }
      },

      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec3 vViewPosition;
        varying vec2 vUv;
        varying float vDistanceFromCenter;
        
        uniform float cloudScale;
        uniform float time;
        
        // Simplex noise for organic expansion
        vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }
        
        float snoise(vec2 v) {
          const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
          vec2 i = floor(v + dot(v, C.yy));
          vec2 x0 = v - i + dot(i, C.xx);
          vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
          vec4 x12 = x0.xyxy + C.xxzz;
          x12.xy -= i1;
          i = mod289(i);
          vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
          vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
          m = m*m;
          m = m*m;
          vec3 x = 2.0 * fract(p * C.www) - 1.0;
          vec3 h = abs(x) - 0.5;
          vec3 ox = floor(x + 0.5);
          vec3 a0 = x - ox;
          m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
          vec3 g;
          g.x = a0.x * x0.x + h.x * x0.y;
          g.yz = a0.yz * x12.xz + h.yz * x12.yw;
          return 130.0 * dot(m, g);
        }
        
        void main() {
          vUv = uv;
          vNormal = normalize(normalMatrix * normal);
          vDistanceFromCenter = length(position);
          
          // Expand vertices with organic noise
          float noiseScale = snoise(position.xy * 0.5 + time * 0.1) * 0.5 + 
                             snoise(position.yz * 0.7 - time * 0.15) * 0.3 +
                             snoise(position.xz * 0.3 + time * 0.08) * 0.2;
          
          vec3 expandedPosition = position + normal * cloudScale * (1.0 + noiseScale * 0.3);
          
          // Add turbulent wave motion
          float wave = sin(time * 0.5 + position.x * 0.1) * 0.5;
          float wave2 = cos(time * 0.3 + position.y * 0.15) * 0.3;
          expandedPosition += normal * (wave + wave2);
          
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
        uniform float glowIntensity;
        uniform float diffuseRadius;
        uniform vec2 resolution;
        
        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec3 vViewPosition;
        varying vec2 vUv;
        varying float vDistanceFromCenter;
        
        // Multi-octave noise for cloud texture
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
        
        float fbm(vec2 uv) {
          float n = 0.0;
          float amplitude = 1.0;
          float frequency = 1.0;
          
          for(int i = 0; i < 6; i++) {
            n += noise(uv * frequency + time * 0.05 * float(i + 1)) * amplitude;
            amplitude *= 0.5;
            frequency *= 2.2;
          }
          
          return n;
        }
        
        // Heavy gaussian blur with multiple samples
        float gaussianBlur(vec2 center, vec2 point, float spread) {
          float dist = distance(center, point);
          return exp(-(dist * dist) / (2.0 * spread * spread));
        }
        
        // Diffuse glow calculation from DiffuseWireframeMaterial
        float calculateDiffuseGlow(float distanceField, float radius) {
          float glow = 1.0 - smoothstep(0.0, radius, distanceField);
          return glow * glow * glow; // Cubic falloff for even softer edges
        }
        
        // Fresnel effect for enhanced depth
        float fresnel(vec3 normal, vec3 viewDir, float power) {
          return pow(1.0 - max(0.0, dot(normal, viewDir)), power);
        }
        
        void main() {
          vec3 viewDir = normalize(-vViewPosition);
          
          // Enhanced fresnel with multiple power levels
          float fresnel1 = fresnel(vNormal, viewDir, 1.0);
          float fresnel2 = fresnel(vNormal, viewDir, 2.0);
          float fresnel3 = fresnel(vNormal, viewDir, 3.0);
          float combinedFresnel = (fresnel1 * 0.3 + fresnel2 * 0.5 + fresnel3 * 0.2);
          
          // Create multi-layer cloud density
          vec2 center = vec2(0.5, 0.5);
          
          // Base cloud layer - very wide spread
          float cloudBase = gaussianBlur(center, vUv, blurSpread * 0.15);
          
          // Middle cloud layer - medium spread  
          float cloudMid = gaussianBlur(center, vUv, blurSpread * 0.08);
          
          // Core cloud layer - tighter spread
          float cloudCore = gaussianBlur(center, vUv, blurSpread * 0.04);
          
          // Combine layers with different weights
          float layeredDensity = cloudBase * 0.3 + cloudMid * 0.4 + cloudCore * 0.3;
          
          // Add complex noise for ethereal effect
          vec2 noiseUv = vUv * 2.0;
          float cloudTexture = fbm(noiseUv);
          float cloudTexture2 = fbm(noiseUv * 1.5 + vec2(100.0, 100.0));
          
          // Combine noise layers
          float combinedNoise = (cloudTexture * 0.6 + cloudTexture2 * 0.4);
          
          // Calculate distance field for diffuse glow
          float distFromCenter = length(vUv - center);
          float diffuseGlow = calculateDiffuseGlow(distFromCenter, diffuseRadius);
          
          // Combine all density components
          float density = layeredDensity * cloudDensity;
          density *= (0.3 + combinedNoise * 0.7);
          density += diffuseGlow * glowIntensity * 0.3;
          
          // Add rotating ethereal wisps with more complexity
          float angle = atan(vUv.y - 0.5, vUv.x - 0.5);
          float spiral1 = sin(angle * 3.0 - time * 1.5) * 0.5 + 0.5;
          float spiral2 = cos(angle * 5.0 + time * 2.0) * 0.3 + 0.7;
          float spiral3 = sin(angle * 7.0 - time * 1.0) * 0.2 + 0.8;
          density *= (spiral1 * 0.4 + spiral2 * 0.3 + spiral3 * 0.3);
          
          // Soft edge fade with multiple falloff curves
          float edgeFade1 = 1.0 - smoothstep(0.0, 1.0, distFromCenter * 1.5);
          float edgeFade2 = 1.0 - smoothstep(0.2, 0.8, distFromCenter);
          float edgeFade = edgeFade1 * 0.6 + edgeFade2 * 0.4;
          density *= edgeFade;
          
          // Add fresnel for depth and glow
          density += combinedFresnel * glowIntensity * 0.4;
          
          // Complex pulsing animation
          float pulse1 = sin(time * 1.5) * 0.1 + 0.9;
          float pulse2 = cos(time * 2.3) * 0.05 + 0.95;
          float pulse = pulse1 * pulse2;
          density *= pulse;
          
          // Distance-based intensity falloff
          float distanceFade = 1.0 / (1.0 + vDistanceFromCenter * 0.01);
          density *= distanceFade;
          
          // Final color with enhanced glow
          vec3 glowColor = color * (1.0 + combinedFresnel * 3.0);
          
          // Add color variations based on density
          vec3 finalColor = mix(glowColor, color * 2.0, density);
          
          // Alpha calculation with very soft edges
          float alpha = density * opacity;
          alpha = smoothstep(0.0, 1.0, alpha);
          
          // Ensure minimum visibility for outer glow
          alpha = max(alpha, opacity * 0.02 * (combinedFresnel + diffuseGlow));
          
          gl_FragColor = vec4(finalColor, alpha);
        }
      `,

      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: true,
      vertexColors: false,
      wireframe: true  // Enable wireframe rendering
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

  updateGlow(intensity: number, radius: number) {
    this.uniforms.glowIntensity.value = intensity;
    this.uniforms.diffuseRadius.value = radius;
  }
}