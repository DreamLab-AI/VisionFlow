import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { EdgeSettings } from '../../settings/config/settings';
import { useSelectiveSetting } from '../../../hooks/useSelectiveSettingsStore';
import { registerEdgeObject, unregisterEdgeObject } from '../../visualisation/hooks/bloomRegistry';
import { memoizeComponent, settingsEqual } from '../../../utils/performanceUtils';
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

const FlowingEdgesComponent: React.FC<FlowingEdgesProps> = ({ points, settings: propSettings, edgeData }) => {
  // Get edge bloom strength using selective hook
  const edgeBloomStrength = useSelectiveSetting<number>('visualisation.bloom.edgeBloomStrength') ?? 
                           useSelectiveSetting<number>('visualisation.bloom.edge_bloom_strength') ?? 0.5;
  const lineRef = useRef<THREE.LineSegments>(null);
  const materialRef = useRef<THREE.LineBasicMaterial>(null);
  // Remove redundant edgeBloom - we're getting it from context now
  
  // Create optimized geometry for all edges with proper line width handling
  const geometry = useMemo(() => {
    if (points.length < 6) return null; // Need at least 2 points (6 values)
    
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(points);
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    // Add line width attribute for custom shaders if needed
    const lineWidth = propSettings.baseWidth || 1;
    const widths = new Float32Array(points.length / 3).fill(Math.max(1, Math.min(lineWidth, 10))); // Clamp width
    geo.setAttribute('lineWidth', new THREE.BufferAttribute(widths, 1));
    
    return geo;
  }, [points, propSettings.baseWidth]);
  
  // Create optimized line material with WebGL compatibility
  const material = useMemo(() => {
    const color = new THREE.Color(propSettings.color || '#56b6c2');
    
    // Apply bloom strength to brightness
    const bloomAdjustedColor = color.clone().multiplyScalar(Math.max(0.1, edgeBloomStrength));
    
    const mat = new THREE.LineBasicMaterial({
      color: bloomAdjustedColor,
      transparent: true,
      opacity: Math.min(1.0, Math.max(0.1, propSettings.opacity || 0.6)), // Clamp opacity
      // Remove linewidth as it's not supported in WebGL and causes errors
      // Use geometry-based width instead
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
      obj.layers.enable(1); // Use layer 1 for all bloom objects to avoid SelectiveBloom issues
      registerEdgeObject(obj);
    }
    return () => {
      if (obj) unregisterEdgeObject(obj);
    };
  }, []);
  
  // Optimize animation frame updates - throttle to prevent excessive updates
  const lastAnimationUpdate = useRef(0);
  const ANIMATION_UPDATE_INTERVAL = 1000 / 30; // 30 FPS max for animation
  
  useFrame((state) => {
    const now = performance.now();
    if (now - lastAnimationUpdate.current < ANIMATION_UPDATE_INTERVAL) {
      return; // Skip this frame
    }
    lastAnimationUpdate.current = now;
    
    if (materialRef.current && (propSettings as any).enableFlowEffect) {
      const flowIntensity = Math.sin(state.clock.elapsedTime * ((propSettings as any).flowSpeed || 1.0)) * 0.3 + 0.7;
      materialRef.current.opacity = Math.max(0.1, (propSettings.opacity || 0.25) * flowIntensity);
    }
  });
  
  if (!geometry) return null;
  
  return (
    <lineSegments ref={lineRef} geometry={geometry} material={material} renderOrder={5} />
  );
};

// Memoized version with custom comparison
export const FlowingEdges = React.memo(FlowingEdgesComponent, (prevProps, nextProps) => {
  // Compare points arrays efficiently
  if (prevProps.points.length !== nextProps.points.length) return false;
  for (let i = 0; i < prevProps.points.length; i++) {
    if (prevProps.points[i] !== nextProps.points[i]) return false;
  }
  
  // Compare settings
  if (!settingsEqual(prevProps.settings, nextProps.settings)) return false;
  
  // Compare edge data
  if (prevProps.edgeData?.length !== nextProps.edgeData?.length) return false;
  
  return true;
});