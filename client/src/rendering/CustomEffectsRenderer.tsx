import React, { useRef, useMemo, useCallback, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass';
import { createLogger } from '@/utils/logger';

const logger = createLogger('CustomEffectsRenderer');

// Distance Field Glow Shader for diffuse wireframe rendering
const DiffuseWireframeShader = {
  uniforms: {
    tDiffuse: { value: null },
    glowIntensity: { value: 0.8 },
    glowColor: { value: new THREE.Color(0x00ffff) },
    diffuseRadius: { value: 2.0 },
    opacity: { value: 0.7 },
    time: { value: 0.0 },
    resolution: { value: new THREE.Vector2(1024, 1024) },
    wireframeThickness: { value: 0.002 },
    distanceFieldScale: { value: 1.0 }
  },
  
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float glowIntensity;
    uniform vec3 glowColor;
    uniform float diffuseRadius;
    uniform float opacity;
    uniform float time;
    uniform vec2 resolution;
    uniform float wireframeThickness;
    uniform float distanceFieldScale;
    
    varying vec2 vUv;
    
    // Distance field functions for smooth wireframe rendering
    float sdSegment(vec2 p, vec2 a, vec2 b) {
      vec2 pa = p - a;
      vec2 ba = b - a;
      float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
      return length(pa - ba * h);
    }
    
    float sdBox(vec2 p, vec2 b) {
      vec2 d = abs(p) - b;
      return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    }
    
    // Multi-sample blur for diffuse glow effect
    vec4 diffuseBlur(sampler2D tex, vec2 uv, float radius) {
      vec4 color = vec4(0.0);
      float total = 0.0;
      float samples = 16.0;
      
      // Circular sampling pattern
      for (float i = 0.0; i < samples; i++) {
        float angle = (i / samples) * 2.0 * 3.14159;
        vec2 offset = vec2(cos(angle), sin(angle)) * radius / resolution;
        
        // Multiple radius samples for smoother falloff
        for (float j = 0.2; j <= 1.0; j += 0.2) {
          vec2 sampleUv = uv + offset * j;
          vec4 sample = texture2D(tex, sampleUv);
          float weight = 1.0 - j * 0.8; // Falloff weight
          color += sample * weight;
          total += weight;
        }
      }
      
      return color / total;
    }
    
    // Wireframe edge detection
    float getWireframeIntensity(vec2 uv) {
      vec4 texel = texture2D(tDiffuse, uv);
      
      // Sample neighboring pixels for edge detection
      vec2 texelSize = 1.0 / resolution;
      float edge = 0.0;
      
      // Sobel edge detection for wireframe elements
      vec3 tl = texture2D(tDiffuse, uv + vec2(-texelSize.x, -texelSize.y)).rgb;
      vec3 tr = texture2D(tDiffuse, uv + vec2(texelSize.x, -texelSize.y)).rgb;
      vec3 bl = texture2D(tDiffuse, uv + vec2(-texelSize.x, texelSize.y)).rgb;
      vec3 br = texture2D(tDiffuse, uv + vec2(texelSize.x, texelSize.y)).rgb;
      
      vec3 horizontal = (tl + 2.0 * texture2D(tDiffuse, uv + vec2(-texelSize.x, 0.0)).rgb + bl) -
                       (tr + 2.0 * texture2D(tDiffuse, uv + vec2(texelSize.x, 0.0)).rgb + br);
      
      vec3 vertical = (tl + 2.0 * texture2D(tDiffuse, uv + vec2(0.0, -texelSize.y)).rgb + tr) -
                     (bl + 2.0 * texture2D(tDiffuse, uv + vec2(0.0, texelSize.y)).rgb + br);
      
      float edgeStrength = length(horizontal) + length(vertical);
      
      // Enhanced wireframe detection for thin lines
      float wireframe = step(0.01, edgeStrength) * texel.a;
      
      return wireframe;
    }
    
    void main() {
      vec2 uv = vUv;
      vec4 originalColor = texture2D(tDiffuse, uv);
      
      // Get wireframe intensity using edge detection
      float wireframeIntensity = getWireframeIntensity(uv);
      
      // Apply diffuse blur only to wireframe elements
      vec4 diffuseGlow = vec4(0.0);
      if (wireframeIntensity > 0.01) {
        diffuseGlow = diffuseBlur(tDiffuse, uv, diffuseRadius * distanceFieldScale);
        
        // Apply distance field smoothing
        float distanceField = smoothstep(wireframeThickness * 0.5, wireframeThickness * 2.0, wireframeIntensity);
        diffuseGlow *= distanceField;
      }
      
      // Combine original color with diffuse glow
      vec3 glowedColor = originalColor.rgb + (diffuseGlow.rgb * glowColor * glowIntensity);
      
      // Apply time-based animation for dynamic effect
      float pulse = sin(time * 2.0) * 0.1 + 0.9;
      glowedColor *= pulse;
      
      // Output with controlled opacity
      gl_FragColor = vec4(glowedColor, originalColor.a * opacity);
    }
  `
};

// Mote particle effects shader
const MoteParticleShader = {
  uniforms: {
    tDiffuse: { value: null },
    moteIntensity: { value: 0.6 },
    moteColor: { value: new THREE.Color(0x00ffff) },
    moteSize: { value: 1.0 },
    time: { value: 0.0 },
    opacity: { value: 0.8 }
  },
  
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float moteIntensity;
    uniform vec3 moteColor;
    uniform float moteSize;
    uniform float time;
    uniform float opacity;
    
    varying vec2 vUv;
    
    // Noise function for mote generation
    float random(vec2 st) {
      return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
    }
    
    float noise(vec2 st) {
      vec2 i = floor(st);
      vec2 f = fract(st);
      
      float a = random(i);
      float b = random(i + vec2(1.0, 0.0));
      float c = random(i + vec2(0.0, 1.0));
      float d = random(i + vec2(1.0, 1.0));
      
      vec2 u = f * f * (3.0 - 2.0 * f);
      
      return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
    }
    
    void main() {
      vec4 originalColor = texture2D(tDiffuse, vUv);
      
      // Generate animated motes
      vec2 moteCoord = vUv * 20.0 + time * 0.5;
      float moteNoise = noise(moteCoord);
      
      // Create mote particles based on noise
      float mote = smoothstep(0.8, 1.0, moteNoise) * moteIntensity;
      mote *= sin(time * 3.0 + moteNoise * 10.0) * 0.5 + 0.5;
      
      // Apply mote effect
      vec3 finalColor = originalColor.rgb + moteColor * mote * moteSize;
      
      gl_FragColor = vec4(finalColor, originalColor.a * opacity);
    }
  `
};

// Ring enhancement shader for hologram rings
const RingEnhancementShader = {
  uniforms: {
    tDiffuse: { value: null },
    ringGlow: { value: 0.9 },
    ringColor: { value: new THREE.Color(0x00ffff) },
    ringWidth: { value: 0.05 },
    time: { value: 0.0 },
    opacity: { value: 0.7 }
  },
  
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float ringGlow;
    uniform vec3 ringColor;
    uniform float ringWidth;
    uniform float time;
    uniform float opacity;
    
    varying vec2 vUv;
    
    void main() {
      vec4 originalColor = texture2D(tDiffuse, vUv);
      
      // Distance from center for ring detection
      float dist = distance(vUv, vec2(0.5));
      
      // Ring enhancement based on distance
      float ring = smoothstep(0.3 - ringWidth, 0.3, dist) * 
                   smoothstep(0.7, 0.7 - ringWidth, dist);
      
      // Animated ring glow
      ring *= sin(time * 2.0) * 0.3 + 0.7;
      
      // Apply ring glow effect
      vec3 finalColor = originalColor.rgb + ringColor * ring * ringGlow;
      
      gl_FragColor = vec4(finalColor, originalColor.a * opacity);
    }
  `
};

export interface CustomEffectsConfig {
  diffuse: {
    enabled: boolean;
    intensity: number;
    radius: number;
    color: THREE.Color;
    opacity: number;
    distanceFieldScale: number;
    wireframeThickness: number;
  };
  motes: {
    enabled: boolean;
    intensity: number;
    size: number;
    color: THREE.Color;
    opacity: number;
  };
  rings: {
    enabled: boolean;
    glow: number;
    width: number;
    color: THREE.Color;
    opacity: number;
  };
}

export const defaultCustomEffectsConfig: CustomEffectsConfig = {
  diffuse: {
    enabled: true,
    intensity: 0.8,
    radius: 2.0,
    color: new THREE.Color(0x00ffff),
    opacity: 0.7,
    distanceFieldScale: 1.0,
    wireframeThickness: 0.002
  },
  motes: {
    enabled: true,
    intensity: 0.6,
    size: 1.0,
    color: new THREE.Color(0x00ffff),
    opacity: 0.8
  },
  rings: {
    enabled: true,
    glow: 0.9,
    width: 0.05,
    color: new THREE.Color(0x00ffff),
    opacity: 0.7
  }
};

export interface CustomEffectsRendererProps {
  config?: Partial<CustomEffectsConfig>;
  backgroundElementsOnly?: boolean;
  children?: React.ReactNode;
}

export const CustomEffectsRenderer: React.FC<CustomEffectsRendererProps> = ({
  config = {},
  backgroundElementsOnly = true,
  children
}) => {
  const { gl, scene, camera, size } = useThree();
  const composerRef = useRef<EffectComposer>();
  const timeRef = useRef(0);
  
  // Merge config with defaults
  const effectsConfig = useMemo(() => {
    const merged = { ...defaultCustomEffectsConfig };
    
    if (config.diffuse) {
      merged.diffuse = { ...merged.diffuse, ...config.diffuse };
    }
    if (config.motes) {
      merged.motes = { ...merged.motes, ...config.motes };
    }
    if (config.rings) {
      merged.rings = { ...merged.rings, ...config.rings };
    }
    
    return merged;
  }, [config]);
  
  // Create effect composer with custom passes
  const composer = useMemo(() => {
    const composer = new EffectComposer(gl);
    composer.setSize(size.width, size.height);
    
    // Base render pass
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);
    
    // Diffuse wireframe pass
    if (effectsConfig.diffuse.enabled) {
      const diffusePass = new ShaderPass(DiffuseWireframeShader);
      diffusePass.uniforms.glowIntensity.value = effectsConfig.diffuse.intensity;
      diffusePass.uniforms.glowColor.value = effectsConfig.diffuse.color;
      diffusePass.uniforms.diffuseRadius.value = effectsConfig.diffuse.radius;
      diffusePass.uniforms.opacity.value = effectsConfig.diffuse.opacity;
      diffusePass.uniforms.resolution.value.set(size.width, size.height);
      diffusePass.uniforms.wireframeThickness.value = effectsConfig.diffuse.wireframeThickness;
      diffusePass.uniforms.distanceFieldScale.value = effectsConfig.diffuse.distanceFieldScale;
      composer.addPass(diffusePass);
    }
    
    // Mote particles pass
    if (effectsConfig.motes.enabled) {
      const motePass = new ShaderPass(MoteParticleShader);
      motePass.uniforms.moteIntensity.value = effectsConfig.motes.intensity;
      motePass.uniforms.moteColor.value = effectsConfig.motes.color;
      motePass.uniforms.moteSize.value = effectsConfig.motes.size;
      motePass.uniforms.opacity.value = effectsConfig.motes.opacity;
      composer.addPass(motePass);
    }
    
    // Ring enhancement pass
    if (effectsConfig.rings.enabled) {
      const ringPass = new ShaderPass(RingEnhancementShader);
      ringPass.uniforms.ringGlow.value = effectsConfig.rings.glow;
      ringPass.uniforms.ringColor.value = effectsConfig.rings.color;
      ringPass.uniforms.ringWidth.value = effectsConfig.rings.width;
      ringPass.uniforms.opacity.value = effectsConfig.rings.opacity;
      composer.addPass(ringPass);
    }
    
    return composer;
  }, [gl, scene, camera, size, effectsConfig]);
  
  // Update composer on resize
  useEffect(() => {
    if (composer) {
      composer.setSize(size.width, size.height);
      
      // Update resolution uniforms
      composer.passes.forEach(pass => {
        if (pass instanceof ShaderPass && pass.uniforms.resolution) {
          pass.uniforms.resolution.value.set(size.width, size.height);
        }
      });
    }
  }, [composer, size]);
  
  // Animation loop
  useFrame((state, delta) => {
    timeRef.current += delta;
    
    if (composer) {
      // Update time uniforms for all shader passes
      composer.passes.forEach(pass => {
        if (pass instanceof ShaderPass && pass.uniforms.time) {
          pass.uniforms.time.value = timeRef.current;
        }
      });
      
      // Render the effect composer
      composer.render();
    }
  }, 1);
  
  // Store composer reference
  useEffect(() => {
    composerRef.current = composer;
    
    return () => {
      if (composer) {
        composer.dispose();
      }
    };
  }, [composer]);
  
  // Update config callback
  const updateConfig = useCallback((newConfig: Partial<CustomEffectsConfig>) => {
    if (!composer) return;
    
    composer.passes.forEach(pass => {
      if (pass instanceof ShaderPass) {
        // Update diffuse pass
        if (pass.uniforms.glowIntensity && newConfig.diffuse) {
          if (newConfig.diffuse.intensity !== undefined) {
            pass.uniforms.glowIntensity.value = newConfig.diffuse.intensity;
          }
          if (newConfig.diffuse.color !== undefined) {
            pass.uniforms.glowColor.value = newConfig.diffuse.color;
          }
          if (newConfig.diffuse.radius !== undefined) {
            pass.uniforms.diffuseRadius.value = newConfig.diffuse.radius;
          }
          if (newConfig.diffuse.opacity !== undefined) {
            pass.uniforms.opacity.value = newConfig.diffuse.opacity;
          }
          if (newConfig.diffuse.distanceFieldScale !== undefined) {
            pass.uniforms.distanceFieldScale.value = newConfig.diffuse.distanceFieldScale;
          }
          if (newConfig.diffuse.wireframeThickness !== undefined) {
            pass.uniforms.wireframeThickness.value = newConfig.diffuse.wireframeThickness;
          }
        }
        
        // Update mote pass
        if (pass.uniforms.moteIntensity && newConfig.motes) {
          if (newConfig.motes.intensity !== undefined) {
            pass.uniforms.moteIntensity.value = newConfig.motes.intensity;
          }
          if (newConfig.motes.color !== undefined) {
            pass.uniforms.moteColor.value = newConfig.motes.color;
          }
          if (newConfig.motes.size !== undefined) {
            pass.uniforms.moteSize.value = newConfig.motes.size;
          }
          if (newConfig.motes.opacity !== undefined) {
            pass.uniforms.opacity.value = newConfig.motes.opacity;
          }
        }
        
        // Update ring pass
        if (pass.uniforms.ringGlow && newConfig.rings) {
          if (newConfig.rings.glow !== undefined) {
            pass.uniforms.ringGlow.value = newConfig.rings.glow;
          }
          if (newConfig.rings.color !== undefined) {
            pass.uniforms.ringColor.value = newConfig.rings.color;
          }
          if (newConfig.rings.width !== undefined) {
            pass.uniforms.ringWidth.value = newConfig.rings.width;
          }
          if (newConfig.rings.opacity !== undefined) {
            pass.uniforms.opacity.value = newConfig.rings.opacity;
          }
        }
      }
    });
  }, [composer]);
  
  // Expose update function via ref
  React.useImperativeHandle(composerRef, () => ({
    updateConfig,
    getComposer: () => composer,
    dispose: () => composer?.dispose()
  }), [updateConfig, composer]);
  
  logger.debug('CustomEffectsRenderer initialized', {
    config: effectsConfig,
    backgroundElementsOnly,
    passes: composer?.passes.length
  });
  
  return <>{children}</>;
};

export default CustomEffectsRenderer;