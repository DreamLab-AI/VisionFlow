// Node shader toggle handler - switches between basic and hologram shaders
import { useEffect } from 'react';
import { useSettingsStore } from '@/store/settingsStore';
import { HologramNodeMaterial } from '../../../rendering/materials/HologramNodeMaterial';
import * as THREE from 'three';

export interface NodeShaderToggleProps {
  materialRef: React.MutableRefObject<HologramNodeMaterial | null>;
}

// Simple shader for when animations are disabled
const createBasicNodeShader = () => {
  return new THREE.ShaderMaterial({
    uniforms: {
      baseColor: { value: new THREE.Color('#00ffff') },
      opacity: { value: 0.9 },
      metalness: { value: 0.8 },
      roughness: { value: 0.2 },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      void main() {
        vNormal = normalize(normalMatrix * normal);
        vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 baseColor;
      uniform float opacity;
      uniform float metalness;
      uniform float roughness;
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      void main() {
        // Simple lighting calculation
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        float diffuse = max(dot(vNormal, lightDir), 0.0);
        
        // Rim lighting for edge glow
        vec3 viewDir = normalize(-vPosition);
        float rim = 1.0 - max(dot(viewDir, vNormal), 0.0);
        rim = pow(rim, 2.0);
        
        // Final color
        vec3 color = baseColor * (0.3 + diffuse * 0.7);
        color += baseColor * rim * 0.5;
        
        gl_FragColor = vec4(color, opacity);
      }
    `,
    transparent: true,
    side: THREE.DoubleSide,
  });
};

export const NodeShaderToggle: React.FC<NodeShaderToggleProps> = ({ materialRef }) => {
  const settings = useSettingsStore(state => state.settings);
  
  // Check if node animations are enabled (controls animation intensity)
  const enableNodeAnimations = settings?.visualisation?.animations?.enableNodeAnimations || false;
  // Get node settings from logseq graph or fallback to global
  const nodeSettings = settings?.visualisation?.graphs?.logseq?.nodes || settings?.visualisation?.nodes;
  // Check if hologram effect is enabled (controls hologram shader features) - check both paths
  const enableHologram = nodeSettings?.enableHologram !== false;
  const nodeBloom = settings?.visualisation?.bloom?.nodeBloomStrength ?? 1;
  
  useEffect(() => {
    if (!materialRef.current) return;
    
    // Hologram is enabled if user has enabled it specifically
    const shouldEnableHologram = enableHologram;
    
    if (shouldEnableHologram) {
      // Enable hologram shader features
      materialRef.current.setHologramEnabled(true);
      
      // If animations are also enabled, use full animated effects
      if (enableNodeAnimations) {
        materialRef.current.updateHologramParams({
          scanlineSpeed: 2.0,
          scanlineCount: 30.0,
          hologramStrength: 1.0,  // Increased from 0.8
          glowStrength: 3.0 * nodeBloom,  // Increased from 2.0
          rimPower: 2.0,  // Reduced for stronger rim
        });
        
        // Enable animation
        if (materialRef.current.uniforms) {
          materialRef.current.uniforms.pulseSpeed = { value: 1.0 };
          materialRef.current.uniforms.pulseStrength = { value: 0.3 };
        }
      } else {
        // Hologram enabled but animations disabled - static hologram
        materialRef.current.updateHologramParams({
          scanlineSpeed: 0.5,  // Slow scanlines
          scanlineCount: 15.0,  // Fewer scanlines
          hologramStrength: 0.5,  // Reduced hologram strength
          glowStrength: 1.5 * nodeBloom,
          rimPower: 2.5,
        });
        
        // Disable pulsing animation
        if (materialRef.current.uniforms) {
          materialRef.current.uniforms.pulseSpeed = { value: 0 };
          materialRef.current.uniforms.pulseStrength = { value: 0 };
        }
      }
    } else {
      // Hologram disabled - use simple material
      materialRef.current.setHologramEnabled(false);
      materialRef.current.updateHologramParams({
        scanlineSpeed: 0,
        scanlineCount: 0,
        hologramStrength: 0,
        glowStrength: 1.0 * nodeBloom, // Basic glow only
        rimPower: 2.0, // Keep rim lighting
      });
      
      // Disable all animations
      if (materialRef.current.uniforms) {
        materialRef.current.uniforms.pulseSpeed = { value: 0 };
        materialRef.current.uniforms.pulseStrength = { value: 0 };
      }
    }
    
    // Apply color settings
    if (nodeSettings?.baseColor) {
      materialRef.current.updateColors(
        nodeSettings.baseColor,
        nodeSettings.baseColor
      );
    }
    
    // Apply material properties
    materialRef.current.uniforms.opacity.value = nodeSettings?.opacity ?? 0.9;
    materialRef.current.uniforms.metalness = { value: nodeSettings?.metalness ?? 0.8 };
    materialRef.current.uniforms.roughness = { value: nodeSettings?.roughness ?? 0.2 };
    
  }, [enableNodeAnimations, enableHologram, nodeSettings, nodeBloom, materialRef]);
  
  return null;
};

export default NodeShaderToggle;