import React, { useRef, useMemo, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass';
import * as THREE from 'three';
import { useSettingsStore } from '../../../store/settingsStore';

// Copy shader - just passes through the texture
const CopyShader = {
  uniforms: {
    tDiffuse: { value: null }
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
    varying vec2 vUv;
    void main() {
      gl_FragColor = texture2D(tDiffuse, vUv);
    }
  `
};

export const PostProcessingEffects: React.FC<{
  graphElementsOnly?: boolean;
}> = ({ 
  graphElementsOnly = false 
}) => {
  const { gl, scene, camera, size } = useThree();
  const settings = useSettingsStore(state => state.settings?.visualisation);
  const bloomSettings = settings?.bloom;
  const glowSettings = settings?.glow;
  
  // Create effect composer and passes
  const [composer, bloomPass, glowPass] = useMemo(() => {
    const composer = new EffectComposer(gl);
    composer.setSize(size.width, size.height);
    
    // Standard render pass - renders everything normally
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);
    
    // Bloom pass for nodes/edges (layer 1)
    let bloomPass: UnrealBloomPass | null = null;
    if (bloomSettings?.enabled) {
      bloomPass = new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        bloomSettings.strength || 1.5,
        bloomSettings.radius || 0.4,
        bloomSettings.threshold || 0.85
      );
      
      // Custom selective bloom - only affect layer 1
      const originalRender = bloomPass.render.bind(bloomPass);
      bloomPass.render = function(renderer: any, writeBuffer: any, readBuffer: any, deltaTime: any, maskActive: any) {
        // Save visibility state
        const visibilityCache = new Map();
        scene.traverse((obj: any) => {
          if (obj.isMesh || obj.isPoints || obj.isLine) {
            visibilityCache.set(obj, obj.visible);
            // Only show objects on layer 1 for bloom
            const onLayer1 = obj.layers.test({ mask: 1 << 1 } as any);
            obj.visible = obj.visible && onLayer1;
          }
        });
        
        // Render bloom
        originalRender(renderer, writeBuffer, readBuffer, deltaTime, maskActive);
        
        // Restore visibility
        visibilityCache.forEach((visible, obj) => {
          obj.visible = visible;
        });
      };
      
      composer.addPass(bloomPass);
    }
    
    // Glow pass for hologram (layer 2)
    let glowPass: UnrealBloomPass | null = null;
    if (glowSettings?.enabled) {
      glowPass = new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        glowSettings.intensity || 2.0,
        glowSettings.radius || 0.6,
        glowSettings.threshold || 0.5
      );
      
      // Custom selective glow - only affect layer 2
      const originalRender = glowPass.render.bind(glowPass);
      glowPass.render = function(renderer: any, writeBuffer: any, readBuffer: any, deltaTime: any, maskActive: any) {
        // Save visibility state
        const visibilityCache = new Map();
        scene.traverse((obj: any) => {
          if (obj.isMesh || obj.isPoints || obj.isLine) {
            visibilityCache.set(obj, obj.visible);
            // Only show objects on layer 2 for glow
            const onLayer2 = obj.layers.test({ mask: 1 << 2 } as any);
            obj.visible = obj.visible && onLayer2;
          }
        });
        
        // Render glow
        originalRender(renderer, writeBuffer, readBuffer, deltaTime, maskActive);
        
        // Restore visibility
        visibilityCache.forEach((visible, obj) => {
          obj.visible = visible;
        });
      };
      
      composer.addPass(glowPass);
    }
    
    // If no effects enabled, add a copy pass to ensure rendering
    if (!bloomSettings?.enabled && !glowSettings?.enabled) {
      const copyPass = new ShaderPass(CopyShader);
      copyPass.renderToScreen = true;
      composer.addPass(copyPass);
    }
    
    return [composer, bloomPass, glowPass];
  }, [gl, scene, camera, size, bloomSettings?.enabled, glowSettings?.enabled]);
  
  // Update composer on resize
  React.useEffect(() => {
    composer.setSize(size.width, size.height);
  }, [composer, size]);
  
  // Update bloom settings dynamically
  React.useEffect(() => {
    if (bloomPass && bloomSettings?.enabled) {
      bloomPass.strength = bloomSettings.strength || 1.5;
      bloomPass.radius = bloomSettings.radius || 0.4;
      bloomPass.threshold = bloomSettings.threshold || 0.85;
    }
    
    if (glowPass && glowSettings?.enabled) {
      glowPass.strength = glowSettings.intensity || 2.0;
      glowPass.radius = glowSettings.radius || 0.6;
      glowPass.threshold = glowSettings.threshold || 0.5;
    }
  }, [bloomPass, glowPass, bloomSettings, glowSettings]);
  
  // Render with composer
  useFrame(() => {
    composer.render();
  }, 1);
  
  // Cleanup
  React.useEffect(() => {
    return () => {
      composer.dispose();
    };
  }, [composer]);
  
  return null;
};