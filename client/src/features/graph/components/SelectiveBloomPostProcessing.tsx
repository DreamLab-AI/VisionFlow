import React, { useRef, useMemo, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass';
import * as THREE from 'three';
import { useSettingsStore } from '../../../store/settingsStore';

// Shader for selective bloom mixing
const SelectiveBloomShader = {
  uniforms: {
    baseTexture: { value: null },
    bloomTexture: { value: null }
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
    uniform sampler2D bloomTexture;
    varying vec2 vUv;
    
    void main() {
      vec4 base = texture2D(baseTexture, vUv);
      vec4 bloom = texture2D(bloomTexture, vUv);
      
      // Additive blend
      gl_FragColor = base + bloom;
      gl_FragColor.a = base.a;
    }
  `
};

export const SelectiveBloomPostProcessing: React.FC = () => {
  const { gl, scene, camera, size } = useThree();
  const settings = useSettingsStore(state => state.settings?.visualisation);
  
  // Get bloom and glow settings
  const bloomSettings = settings?.bloom;
  const glowSettings = settings?.glow;
  
  // Create composers
  const composers = useMemo(() => {
    // Render target for bloom layers
    const renderTargetParams = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat
    };
    
    const bloomRenderTarget = new THREE.WebGLRenderTarget(
      size.width, 
      size.height, 
      renderTargetParams
    );
    
    // Main scene composer (no bloom)
    const sceneComposer = new EffectComposer(gl);
    sceneComposer.renderToScreen = false;
    const scenePass = new RenderPass(scene, camera);
    scenePass.clear = true;
    sceneComposer.addPass(scenePass);
    
    // Bloom composer for specific layers
    const bloomComposer = new EffectComposer(gl, bloomRenderTarget);
    bloomComposer.renderToScreen = false;
    
    // Create bloom layer camera
    const bloomCamera = camera.clone() as THREE.PerspectiveCamera;
    const bloomScene = scene.clone();
    
    // Function to set layer visibility
    const setLayerVisibility = (layer: number, visible: boolean) => {
      scene.traverse((obj: any) => {
        if (obj.isMesh || obj.isPoints || obj.isLine) {
          if (obj.layers.test({ mask: 1 << layer } as any)) {
            obj.visible = visible;
          }
        }
      });
    };
    
    // Bloom render pass
    const bloomRenderPass = new RenderPass(scene, camera);
    bloomRenderPass.clear = true;
    bloomComposer.addPass(bloomRenderPass);
    
    // Determine which bloom pass to use based on settings
    let bloomPass: UnrealBloomPass | null = null;
    
    if (bloomSettings?.enabled && glowSettings?.enabled) {
      // Both enabled - use average settings
      bloomPass = new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        (bloomSettings.strength + glowSettings.intensity) / 2,
        (bloomSettings.radius + glowSettings.radius) / 2,
        (bloomSettings.threshold + glowSettings.threshold) / 2
      );
    } else if (bloomSettings?.enabled) {
      // Only nodes/edges bloom
      bloomPass = new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        bloomSettings.strength || 1.5,
        bloomSettings.radius || 0.4,
        bloomSettings.threshold || 0.85
      );
    } else if (glowSettings?.enabled) {
      // Only hologram glow
      bloomPass = new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        glowSettings.intensity || 2.0,
        glowSettings.radius || 0.6,
        glowSettings.threshold || 0.5
      );
    }
    
    if (bloomPass) {
      bloomComposer.addPass(bloomPass);
    }
    
    // Final composer to mix everything
    const finalComposer = new EffectComposer(gl);
    finalComposer.addPass(new RenderPass(scene, camera));
    
    if (bloomPass) {
      const mixPass = new ShaderPass(SelectiveBloomShader);
      mixPass.uniforms.baseTexture.value = sceneComposer.renderTarget2.texture;
      mixPass.uniforms.bloomTexture.value = bloomRenderTarget.texture;
      finalComposer.addPass(mixPass);
    }
    
    return {
      scene: sceneComposer,
      bloom: bloomComposer,
      final: finalComposer,
      bloomRenderTarget,
      setLayerVisibility,
      bloomEnabled: bloomSettings?.enabled || glowSettings?.enabled
    };
  }, [gl, scene, camera, size, bloomSettings, glowSettings]);
  
  // Update on resize
  useEffect(() => {
    composers.scene.setSize(size.width, size.height);
    composers.bloom.setSize(size.width, size.height);
    composers.final.setSize(size.width, size.height);
    composers.bloomRenderTarget.setSize(size.width, size.height);
  }, [composers, size]);
  
  // Render with selective bloom
  useFrame(() => {
    if (!composers.bloomEnabled) {
      // No bloom - just render scene normally
      composers.scene.render();
      gl.setRenderTarget(null);
      gl.clear();
      gl.render(scene, camera);
      return;
    }
    
    const bloomSettings = settings?.bloom;
    const glowSettings = settings?.glow;
    
    // Store original visibility
    const visibilityCache = new Map();
    
    scene.traverse((obj: any) => {
      if (obj.isMesh || obj.isPoints || obj.isLine) {
        visibilityCache.set(obj, obj.visible);
      }
    });
    
    // Render bloom layers only
    scene.traverse((obj: any) => {
      if (obj.isMesh || obj.isPoints || obj.isLine) {
        const onLayer1 = obj.layers.test({ mask: 1 << 1 } as any);
        const onLayer2 = obj.layers.test({ mask: 1 << 2 } as any);
        
        // Show objects based on which bloom is enabled
        if (bloomSettings?.enabled && glowSettings?.enabled) {
          // Both enabled - show if on either layer
          obj.visible = onLayer1 || onLayer2;
        } else if (bloomSettings?.enabled) {
          // Only show layer 1 (nodes/edges)
          obj.visible = onLayer1;
        } else if (glowSettings?.enabled) {
          // Only show layer 2 (hologram)
          obj.visible = onLayer2;
        } else {
          obj.visible = false;
        }
      }
    });
    
    // Render bloom pass
    composers.bloom.render();
    
    // Restore visibility for main render
    visibilityCache.forEach((visible, obj) => {
      obj.visible = visible;
    });
    
    // Final composite render
    composers.final.render();
  }, 1);
  
  // Cleanup
  useEffect(() => {
    return () => {
      composers.scene.dispose();
      composers.bloom.dispose();
      composers.final.dispose();
      composers.bloomRenderTarget.dispose();
    };
  }, [composers]);
  
  return null;
};