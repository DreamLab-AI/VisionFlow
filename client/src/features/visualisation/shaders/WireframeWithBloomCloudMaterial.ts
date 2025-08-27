import * as THREE from 'three';

// Wireframe geometry with extended ethereal bloom cloud
export class WireframeWithBloomCloudMaterial extends THREE.ShaderMaterial {
  constructor(parameters: {
    color?: THREE.Color | string;
    opacity?: number;
    wireframeColor?: THREE.Color | string;
    wireframeOpacity?: number;
    cloudExtension?: number;  // How far beyond geometry the cloud extends
    blurRadius?: number;      // Radius of the blur effect
    glowIntensity?: number;
  } = {}) {
    const params = {
      color: new THREE.Color(parameters.color || '#00ffff'),
      wireframeColor: new THREE.Color(parameters.wireframeColor || parameters.color || '#00ffff'),
      opacity: parameters.opacity || 0.3,
      wireframeOpacity: parameters.wireframeOpacity || 0.8,
      cloudExtension: parameters.cloudExtension || 10.0,  // Extend 10 units beyond geometry
      blurRadius: parameters.blurRadius || 15.0,
      glowIntensity: parameters.glowIntensity || 2.0
    };

    super({
      uniforms: {
        time: { value: 0 },
        color: { value: params.color },
        wireframeColor: { value: params.wireframeColor },
        opacity: { value: params.opacity },
        wireframeOpacity: { value: params.wireframeOpacity },
        cloudExtension: { value: params.cloudExtension },
        blurRadius: { value: params.blurRadius },
        glowIntensity: { value: params.glowIntensity },
        resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
      },

      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec3 vViewPosition;
        varying vec2 vUv;
        varying float vDistanceFromCamera;
        
        uniform float time;
        uniform float cloudExtension;
        
        void main() {
          vUv = uv;
          vNormal = normalize(normalMatrix * normal);
          
          // Don't expand vertices here - we'll do it in screen space
          vec4 worldPosition = modelMatrix * vec4(position, 1.0);
          vWorldPosition = worldPosition.xyz;
          
          vec4 viewPosition = viewMatrix * worldPosition;
          vViewPosition = viewPosition.xyz;
          vDistanceFromCamera = length(viewPosition.xyz);
          
          gl_Position = projectionMatrix * viewPosition;
        }
      `,

      fragmentShader: `
        uniform vec3 color;
        uniform vec3 wireframeColor;
        uniform float opacity;
        uniform float wireframeOpacity;
        uniform float time;
        uniform float cloudExtension;
        uniform float blurRadius;
        uniform float glowIntensity;
        uniform vec2 resolution;
        
        varying vec3 vNormal;
        varying vec3 vWorldPosition;
        varying vec3 vViewPosition;
        varying vec2 vUv;
        varying float vDistanceFromCamera;
        
        // Extended gaussian blur that goes beyond geometry
        float extendedGaussian(vec2 coord, float radius) {
          // Sample multiple points in screen space around this fragment
          float blur = 0.0;
          float totalWeight = 0.0;
          
          const int samples = 9;
          float step = radius / float(samples);
          
          for(int x = -samples; x <= samples; x++) {
            for(int y = -samples; y <= samples; y++) {
              vec2 offset = vec2(float(x), float(y)) * step / resolution;
              float dist = length(offset) * resolution.x;
              float weight = exp(-dist * dist / (2.0 * radius * radius));
              blur += weight;
              totalWeight += weight;
            }
          }
          
          return blur / totalWeight;
        }
        
        // Noise for organic cloud effect
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
        
        void main() {
          // Simple edge detection based on screen derivatives
          float edge = length(fwidth(vWorldPosition)) * 10.0;
          edge = 1.0 - smoothstep(0.0, 1.0, edge);
          
          // Get view direction for fresnel
          vec3 viewDir = normalize(-vViewPosition);
          float fresnel = pow(1.0 - abs(dot(viewDir, vNormal)), 2.0);
          
          // Screen space position for extended blur
          vec2 screenPos = gl_FragCoord.xy / resolution;
          
          // Calculate extended blur that goes beyond geometry
          float extendedBlur = extendedGaussian(screenPos, blurRadius + cloudExtension);
          
          // Add noise for organic cloud
          float cloudNoise = noise(screenPos * 10.0 + time * 0.1);
          cloudNoise += noise(screenPos * 20.0 - time * 0.15) * 0.5;
          cloudNoise += noise(screenPos * 40.0 + time * 0.2) * 0.25;
          
          // Distance-based falloff for the extended cloud
          float distanceFalloff = 1.0 / (1.0 + vDistanceFromCamera * 0.01);
          
          // Combine wireframe with extended cloud
          vec3 finalColor = vec3(0.0);
          float finalAlpha = 0.0;
          
          // Wireframe pass
          if (edge > 0.01) {
            finalColor += wireframeColor * edge;
            finalAlpha += wireframeOpacity * edge;
          }
          
          // Extended cloud pass - this extends beyond the geometry
          float cloudIntensity = extendedBlur * glowIntensity;
          cloudIntensity *= (0.5 + cloudNoise * 0.5);
          cloudIntensity *= distanceFalloff;
          cloudIntensity += fresnel * 0.3;
          
          // Add pulsing
          float pulse = sin(time * 2.0) * 0.1 + 0.9;
          cloudIntensity *= pulse;
          
          // Blend cloud with wireframe
          finalColor += color * cloudIntensity;
          finalAlpha += opacity * cloudIntensity;
          
          // Ensure we have some minimum glow around wireframe
          if (edge > 0.01) {
            finalColor += color * glowIntensity * 0.5;
            finalAlpha = max(finalAlpha, opacity * 0.1);
          }
          
          gl_FragColor = vec4(finalColor, min(finalAlpha, 1.0));
        }
      `,

      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: true,
      wireframe: true,  // Enable wireframe mode
      extensions: {
        derivatives: true  // Enable for fwidth
      }
    });
  }

  updateTime(time: number) {
    this.uniforms.time.value = time;
  }

  updateCloud(extension: number, radius: number, intensity: number) {
    this.uniforms.cloudExtension.value = extension;
    this.uniforms.blurRadius.value = radius;
    this.uniforms.glowIntensity.value = intensity;
  }
}

// Helper function to add barycentric coordinates to geometry
export function addBarycentricCoordinates(geometry: THREE.BufferGeometry) {
  const positionAttribute = geometry.attributes.position;
  const vertices = positionAttribute.count;

  const barycentrics = new Float32Array(vertices * 3);

  // For triangulated geometry, set barycentric coordinates
  for (let i = 0; i < vertices; i += 3) {
    barycentrics[i * 3] = 1;
    barycentrics[i * 3 + 1] = 0;
    barycentrics[i * 3 + 2] = 0;

    barycentrics[(i + 1) * 3] = 0;
    barycentrics[(i + 1) * 3 + 1] = 1;
    barycentrics[(i + 1) * 3 + 2] = 0;

    barycentrics[(i + 2) * 3] = 0;
    barycentrics[(i + 2) * 3 + 1] = 0;
    barycentrics[(i + 2) * 3 + 2] = 1;
  }

  geometry.setAttribute('barycentric', new THREE.BufferAttribute(barycentrics, 3));
  return geometry;
}