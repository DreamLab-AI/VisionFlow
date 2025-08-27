import React, { useRef, useMemo, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass';
import * as THREE from 'three';
import { useSettingsStore } from '../../../store/settingsStore';

// Shader for combining multiple bloom layers
const LayerCombineShader = {
  uniforms: {
    baseTexture: { value: null },
    bloomTexture1: { value: null },
    bloomTexture2: { value: null }
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D baseTexture;
    uniform sampler2D bloomTexture1;
    uniform sampler2D bloomTexture2;
    varying vec2 vUv;
    
    void main() {
      vec4 base = texture2D(baseTexture, vUv);
      vec4 bloom1 = texture2D(bloomTexture1, vUv);
      vec4 bloom2 = texture2D(bloomTexture2, vUv);
      
      // Combine layers with additive blending
      gl_FragColor = base + bloom1 + bloom2;
      gl_FragColor.a = base.a;
    }
  `
};

export const MultiLayerPostProcessing: React.FC = () => {
  const { gl, scene, camera, size } = useThree();
  const composerRef = useRef<EffectComposer>();
  const settings = useSettingsStore(state => state.settings?.visualisation);
  
  // Get bloom and glow settings
  const bloomSettings = settings?.bloom;
  const glowSettings = settings?.glow;
  
  // Create composers for different layers
  const composers = useMemo(() => {
    // Main composer for final output
    const mainComposer = new EffectComposer(gl);
    mainComposer.setSize(size.width, size.height);
    
    // Create separate render targets for each layer
    const renderTargetParams = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType
    };
    
    const layer1Target = new THREE.WebGLRenderTarget(size.width, size.height, renderTargetParams);
    const layer2Target = new THREE.WebGLRenderTarget(size.width, size.height, renderTargetParams);
    
    // Layer 1: Nodes and Edges (Legacy Bloom)
    const layer1Composer = new EffectComposer(gl, layer1Target);
    layer1Composer.setSize(size.width, size.height);
    layer1Composer.renderToScreen = false;
    
    // Configure camera to see base layer (0) and layer 1
    const layer1Camera = camera.clone() as THREE.PerspectiveCamera;
    layer1Camera.layers.set(0); // Start with base layer
    layer1Camera.layers.enable(1); // Also see layer 1
    
    const layer1RenderPass = new RenderPass(scene, layer1Camera);
    layer1Composer.addPass(layer1RenderPass);
    
    // Apply bloom settings to layer 1
    if (bloomSettings?.enabled !== false) {
      const layer1BloomPass = new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        bloomSettings?.strength || 1.5,
        bloomSettings?.radius || 0.4,
        bloomSettings?.threshold || 0.85
      );
      layer1Composer.addPass(layer1BloomPass);
    }
    
    // Layer 2: Hologram/Environment (Glow Settings)
    const layer2Composer = new EffectComposer(gl, layer2Target);
    layer2Composer.setSize(size.width, size.height);
    layer2Composer.renderToScreen = false;
    
    // Configure camera to see base layer (0) and layer 2
    const layer2Camera = camera.clone() as THREE.PerspectiveCamera;
    layer2Camera.layers.set(0); // Start with base layer
    layer2Camera.layers.enable(2); // Also see layer 2
    
    const layer2RenderPass = new RenderPass(scene, layer2Camera);
    layer2Composer.addPass(layer2RenderPass);
    
    // Apply glow settings to layer 2
    if (glowSettings?.enabled !== false) {
      const layer2BloomPass = new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        glowSettings?.intensity || 2.0,
        glowSettings?.radius || 0.6,
        glowSettings?.threshold || 0.5
      );
      layer2Composer.addPass(layer2BloomPass);
    }
    
    // Main composer combines all layers
    const baseRenderPass = new RenderPass(scene, camera);
    baseRenderPass.clear = false; // Don't clear, we'll combine layers
    mainComposer.addPass(baseRenderPass);
    
    // Combine layers pass
    const combinePass = new ShaderPass(LayerCombineShader);
    combinePass.uniforms.bloomTexture1.value = layer1Target.texture;
    combinePass.uniforms.bloomTexture2.value = layer2Target.texture;
    mainComposer.addPass(combinePass);
    
    return {
      main: mainComposer,
      layer1: layer1Composer,
      layer2: layer2Composer,
      layer1Target,
      layer2Target
    };
  }, [gl, scene, camera, size, bloomSettings, glowSettings]);
  
  // Update composers on resize
  useEffect(() => {
    composers.main.setSize(size.width, size.height);
    composers.layer1.setSize(size.width, size.height);
    composers.layer2.setSize(size.width, size.height);
    composers.layer1Target.setSize(size.width, size.height);
    composers.layer2Target.setSize(size.width, size.height);
  }, [composers, size]);
  
  // Render with multi-layer compositing
  useFrame(() => {
    // Render each layer separately
    composers.layer1.render();
    composers.layer2.render();
    
    // Combine and render final output
    composers.main.render();
  }, 1);
  
  // Store ref and cleanup
  useEffect(() => {
    composerRef.current = composers.main;
    return () => {
      composers.main.dispose();
      composers.layer1.dispose();
      composers.layer2.dispose();
      composers.layer1Target.dispose();
      composers.layer2Target.dispose();
    };
  }, [composers]);
  
  return null;
};