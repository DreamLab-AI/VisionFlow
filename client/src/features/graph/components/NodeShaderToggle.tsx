// Node shader toggle handler - switches between basic and hologram shaders
import { useEffect } from 'react';
import { useSettingsStore } from '@/store/settingsStore';
import { HologramNodeMaterial } from '../shaders/HologramNodeMaterial';
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
  
  // Check if node animations are enabled (controls shader type)
  const enableNodeAnimations = settings?.visualisation?.animations?.enableNodeAnimations || false;
  const nodeSettings = settings?.visualisation?.graphs?.logseq?.nodes || settings?.visualisation?.nodes;
  
  useEffect(() => {
    if (!materialRef.current) return;
    
    if (enableNodeAnimations) {
      // Use animated hologram shader when node animations are ON
      // This includes scanlines, glitch effects, and pulsing
      materialRef.current.setHologramEnabled(true);
      materialRef.current.updateHologramParams({
        scanlineSpeed: 2.0,
        scanlineCount: 30.0,
        hologramStrength: 0.8,
        glowStrength: 2.0,
        rimPower: 3.0,
      });
      
      // Update animation properties
      if (materialRef.current.uniforms) {
        materialRef.current.uniforms.pulseSpeed = { value: 1.0 };
        materialRef.current.uniforms.pulseStrength = { value: 0.3 };
      }
    } else {
      // Disable animated effects when node animations are OFF
      // Keep the material but turn off the animated components
      materialRef.current.setHologramEnabled(false);
      materialRef.current.updateHologramParams({
        scanlineSpeed: 0,
        scanlineCount: 0,
        hologramStrength: 0,
        glowStrength: 1.0, // Keep some glow for visibility
        rimPower: 2.0, // Keep rim lighting
      });
      
      // Disable pulsing
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
    
  }, [enableNodeAnimations, nodeSettings, materialRef]);
  
  return null;
};

export default NodeShaderToggle;