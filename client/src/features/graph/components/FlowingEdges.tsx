import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { EdgeSettings } from '../../settings/config/settings';
import { useSettingsStore } from '../../../store/settingsStore';
import { registerEdgeObject, unregisterEdgeObject } from '../../visualisation/hooks/bloomRegistry';
// import { useBloomStrength } from '../contexts/BloomContext'; // Removed - bloom managed via settings

interface FlowingEdgesProps {
  points: number[];
  settings: EdgeSettings;
  edgeData?: Array<{
    source: string;
    target: string;
    weight?: number;
    active?: boolean;
  }>;
}

// Custom shader for flowing edges
const flowVertexShader = `
  attribute float lineDistance;
  attribute vec3 instanceColorStart;
  attribute vec3 instanceColorEnd;
  
  varying float vLineDistance;
  varying vec3 vColor;
  
  void main() {
    vLineDistance = lineDistance;
    vColor = mix(instanceColorStart, instanceColorEnd, lineDistance);
    
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const flowFragmentShader = `
  uniform float time;
  uniform float flowSpeed;
  uniform float flowIntensity;
  uniform float opacity;
  uniform vec3 baseColor;
  uniform bool enableFlowEffect;
  uniform bool useGradient;
  uniform float glowStrength;
  uniform float distanceIntensity;
  
  varying float vLineDistance;
  varying vec3 vColor;
  
  void main() {
    vec3 color = useGradient ? vColor : baseColor;
    
    // Flow effect
    float flow = 0.0;
    if (enableFlowEffect) {
      float offset = time * flowSpeed;
      flow = sin(vLineDistance * 10.0 - offset) * 0.5 + 0.5;
      flow = pow(flow, 3.0) * flowIntensity;
    }
    
    // Distance-based intensity
    float distanceFade = 1.0 - vLineDistance * distanceIntensity;
    
    // Glow effect
    float glow = pow(1.0 - abs(vLineDistance - 0.5) * 2.0, 2.0) * glowStrength;
    
    // Combine effects
    color += vec3(flow + glow) * 0.5;
    float alpha = opacity * distanceFade * (1.0 + flow * 0.5);
    
    gl_FragColor = vec4(color, alpha);
  }
`;

export const FlowingEdges: React.FC<FlowingEdgesProps> = ({ points, settings: propSettings, edgeData }) => {
  const globalSettings = useSettingsStore((state) => state.settings);
  // Handle both snake_case and camelCase field names
  const edgeBloomStrength = (globalSettings?.visualisation?.bloom as any)?.edge_bloom_strength ?? 
                           globalSettings?.visualisation?.bloom?.edgeBloomStrength ?? 0.5;
  const lineRef = useRef<THREE.LineSegments>(null);
  const materialRef = useRef<THREE.LineBasicMaterial>(null);
  // Remove redundant edgeBloom - we're getting it from context now
  
  // Create single geometry for all edges
  const geometry = useMemo(() => {
    if (points.length < 6) return null; // Need at least 2 points (6 values)
    
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(points);
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    return geo;
  }, [points]);
  
  // Create simple line material
  const material = useMemo(() => {
    const color = new THREE.Color(propSettings.color || '#56b6c2');
    
    // Apply bloom strength to brightness
    const bloomAdjustedColor = color.clone().multiplyScalar(edgeBloomStrength);
    
    const mat = new THREE.LineBasicMaterial({
      color: bloomAdjustedColor,
      transparent: true,
      opacity: Math.min(1.0, (propSettings.opacity || 0.6)), // Keep opacity stable
      linewidth: propSettings.baseWidth || 2, // Increased line width
      depthWrite: false, // Disable depth write for transparent edges
      depthTest: true,
      alphaTest: 0.01, // Prevent z-fighting on nearly transparent areas
      toneMapped: false, // Keep lines bright for bloom threshold
    });
    
    return mat;
  }, [propSettings.color, propSettings.opacity, propSettings.baseWidth, edgeBloomStrength]);
  
  // Update material reference
  useEffect(() => {
    materialRef.current = material;
  }, [material]);

  // Render this on the "edge bloom" layer for selective bloom pass and register
  useEffect(() => {
    const obj = lineRef.current as any;
    if (obj) {
      // Initialize layers if not present
      if (!obj.layers) {
        obj.layers = new THREE.Layers();
      }
      obj.layers.set(0); // Base layer for rendering
      obj.layers.enable(1); // Layer 1 for graph bloom
      obj.layers.disable(2); // Explicitly disable Layer 2 (environment glow) to prevent cross-contamination
      registerEdgeObject(obj);
    }
    return () => {
      if (obj) unregisterEdgeObject(obj);
    };
  }, []);
  
  // Animate flow effect by modifying opacity
  useFrame((state) => {
    if (materialRef.current && (propSettings as any).enableFlowEffect) {
      const flowIntensity = Math.sin(state.clock.elapsedTime * ((propSettings as any).flowSpeed || 1.0)) * 0.3 + 0.7;
      materialRef.current.opacity = (propSettings.opacity || 0.25) * flowIntensity;
    }
  });
  
  if (!geometry) return null;
  
  return (
    <lineSegments ref={lineRef} geometry={geometry} material={material} renderOrder={5} />
  );
};