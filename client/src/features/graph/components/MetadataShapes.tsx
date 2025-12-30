// client/src/features/graph/components/MetadataShapes.tsx
import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { type Node as GraphNode } from '../managers/graphDataManager';
import { HologramNodeMaterial } from '../../../rendering/materials/HologramNodeMaterial';

// --- 1. Define Visual Metaphor Logic ---
// This helper function is the heart of our new system.
const getVisualsForNode = (node: GraphNode, settingsBaseColor?: string, ssspResult?: any) => {
  const visuals = {
    geometryType: 'sphere' as 'sphere' | 'box' | 'octahedron' | 'icosahedron',
    scale: 1.0,
    color: new THREE.Color(settingsBaseColor || '#00ffff'), 
    emissive: new THREE.Color(settingsBaseColor || '#00ffff'),
    pulseSpeed: 0.5,
  };

  
  if (ssspResult) {
    const distance = ssspResult.distances[node.id];
    
    
    if (node.id === ssspResult.sourceNodeId) {
      visuals.color = new THREE.Color('#00FFFF');
      visuals.emissive = new THREE.Color('#00FFFF');
      visuals.scale = 1.5;
      visuals.pulseSpeed = 2.0; 
      visuals.geometryType = 'icosahedron'; 
    }
    
    else if (!isFinite(distance)) {
      visuals.color = new THREE.Color('#666666');
      visuals.emissive = new THREE.Color('#333333');
      visuals.scale = 0.7;
      visuals.pulseSpeed = 0.1;
    }
    
    else {
      const normalizedDistances = ssspResult.normalizedDistances || {};
      const normalizedDistance = normalizedDistances[node.id] || 0;
      
      
      const red = Math.min(1, normalizedDistance * 1.2);
      const green = Math.min(1, (1 - normalizedDistance) * 1.2);
      const blue = 0.1; 
      
      visuals.color = new THREE.Color(red, green, blue);
      visuals.emissive = new THREE.Color(red * 0.5, green * 0.5, blue * 0.5);
      visuals.scale = 0.8 + (1 - normalizedDistance) * 0.4; 
    }
    
    
    return visuals;
  }

  const { metadata } = node;
  if (!metadata) return visuals;
  
  
  
  if (typeof window !== 'undefined' && (window as any).debugState?.isEnabled?.() && Math.random() < 0.05) { 
    console.log('MetadataShapes: Node metadata sample:', {
      id: node.id,
      label: node.label,
      metadata: metadata,
      hasLastModified: !!metadata.lastModified,
      lastModified: metadata.lastModified,
      fileSize: metadata.fileSize,
      hyperlinkCount: metadata.hyperlinkCount
    });
  }

  
  const hyperlinkCount = parseInt(metadata.hyperlinkCount || '0', 10);
  if (hyperlinkCount > 7) {
    visuals.geometryType = 'icosahedron'; 
  } else if (hyperlinkCount > 3) {
    visuals.geometryType = 'octahedron'; 
  } else if (hyperlinkCount > 0) {
    visuals.geometryType = 'box'; 
  } else {
    visuals.geometryType = 'sphere'; 
  }

  
  const fileSize = parseInt(metadata.fileSize || '0', 10);
  const sizeScale = 0.8 + Math.log10(Math.max(1, fileSize / 1024)) * 0.2; 
  const connectionScale = 1 + hyperlinkCount * 0.05;
  visuals.scale = THREE.MathUtils.clamp(sizeScale * connectionScale, 0.5, 3.0);

  
  const originalColor = new THREE.Color(visuals.color); 
  const lastModified = metadata.lastModified ? new Date(metadata.lastModified).getTime() : 0;
  
  if (lastModified > 0) {
    const ageInDays = (Date.now() - lastModified) / (1000 * 60 * 60 * 24);
    
    const heat = Math.max(0, 1 - ageInDays / 90);
    
    
    const hsl = { h: 0, s: 0, l: 0 };
    originalColor.getHSL(hsl);
    
    
    const hueShift = heat * 0.15; 
    const saturationBoost = heat * 0.3; 
    const lightnessBoost = heat * 0.25; 
    
    visuals.color.setHSL(
      (hsl.h + hueShift) % 1,
      Math.min(1, hsl.s + saturationBoost),
      Math.min(1, hsl.l + lightnessBoost)
    );
  } else if (metadata.type) {
    
    const typeColorShifts: Record<string, { hue: number, sat: number, light: number }> = {
      'folder': { hue: 0.1, sat: 0.2, light: 0.15 },     
      'file': { hue: 0.0, sat: 0.1, light: 0.05 },       
      'function': { hue: -0.1, sat: 0.2, light: 0.1 },   
      'class': { hue: 0.05, sat: 0.15, light: 0.1 },     
      'variable': { hue: 0.15, sat: 0.12, light: 0.08 }, 
      'import': { hue: -0.06, sat: 0.1, light: 0.05 },   
      'export': { hue: -0.15, sat: 0.15, light: 0.08 },  
      'default': { hue: 0.0, sat: 0.0, light: 0.0 }      
    };
    
    const shift = typeColorShifts[metadata.type] || typeColorShifts['default'];
    const hsl = { h: 0, s: 0, l: 0 };
    originalColor.getHSL(hsl);
    
    visuals.color.setHSL(
      (hsl.h + shift.hue) % 1,
      Math.min(1, hsl.s + shift.sat),
      Math.min(1, hsl.l + shift.light)
    );
  } else {
    
    const colorIntensity = Math.min(hyperlinkCount / 10, 1);
    const hsl = { h: 0, s: 0, l: 0 };
    originalColor.getHSL(hsl);
    
    
    const saturationBoost = colorIntensity * 0.25;
    const lightnessBoost = colorIntensity * 0.2;
    
    visuals.color.setHSL(
      hsl.h,
      Math.min(1, hsl.s + saturationBoost),
      Math.min(1, hsl.l + lightnessBoost)
    );
  }

  
  if (metadata.perplexityLink) {
    
    const goldTint = new THREE.Color('#FFD700');
    visuals.emissive.copy(originalColor).lerp(goldTint, 0.6); 
  } else {
    
    visuals.emissive.copy(visuals.color).multiplyScalar(0.3);
  }

  
  visuals.pulseSpeed = 0.5 + Math.log10(Math.max(1, fileSize / 1024)) * 0.5;

  return visuals;
};


// --- 2. Create Geometry and Material Resources ---
const useGeometries = () => useMemo(() => ({
  sphere: new THREE.SphereGeometry(0.5, 32, 16),
  box: new THREE.BoxGeometry(0.8, 0.8, 0.8),
  octahedron: new THREE.OctahedronGeometry(0.7, 0),
  icosahedron: new THREE.IcosahedronGeometry(0.6, 1),
}), []);

const useHologramMaterial = (settings: any) => useMemo(() => {
  const nodeSettings = settings?.visualisation?.graphs?.logseq?.nodes || settings?.visualisation?.nodes;
  const material = new HologramNodeMaterial({
    baseColor: nodeSettings?.baseColor || '#00ffff',
    emissiveColor: nodeSettings?.emissiveColor || '#00ffff',
    opacity: nodeSettings?.opacity ?? 0.8,
    glowStrength: 2.0,
    rimPower: 2.5,
  });
  material.defines = { ...material.defines, USE_INSTANCING_COLOR: '' };
  material.needsUpdate = true;
  return material;
}, [settings]);


// --- 3. The React Component ---
interface MetadataShapesProps {
  nodes: GraphNode[];
  nodePositions: Float32Array | null;
  onNodeClick?: (nodeId: string, event: any) => void;
  settings: any;
  ssspResult?: any;
}

export const MetadataShapes: React.FC<MetadataShapesProps> = ({ nodes, nodePositions, onNodeClick, settings, ssspResult }) => {
  const geometries = useGeometries();
  const material = useHologramMaterial(settings);
  const meshRefs = useRef<Map<string, THREE.InstancedMesh>>(new Map());

  
  const nodeGroups = useMemo(() => {
    const groups = new Map<string, { nodes: GraphNode[], originalIndices: number[] }>();
    const nodeSettings = settings?.visualisation?.graphs?.logseq?.nodes || settings?.visualisation?.nodes;
    const baseColor = nodeSettings?.baseColor || '#00ffff';
    
    nodes.forEach((node, index) => {
      const { geometryType } = getVisualsForNode(node, baseColor, ssspResult);
      if (!groups.has(geometryType)) {
        groups.set(geometryType, { nodes: [], originalIndices: [] });
      }
      groups.get(geometryType)!.nodes.push(node);
      groups.get(geometryType)!.originalIndices.push(index);
    });
    return groups;
  }, [nodes, settings, ssspResult]);

  
  useFrame((state) => {
    if (!nodePositions) return;

    material.updateTime(state.clock.elapsedTime);
    const tempMatrix = new THREE.Matrix4();
    const tempColor = new THREE.Color();

    nodeGroups.forEach((group, geometryType) => {
      const mesh = meshRefs.current.get(geometryType);
      if (!mesh) return;

      group.nodes.forEach((node, localIndex) => {
        const originalIndex = group.originalIndices[localIndex];
        const i3 = originalIndex * 3;

        const nodeSettings = settings?.visualisation?.graphs?.logseq?.nodes || settings?.visualisation?.nodes;
        const baseColorForNode = nodeSettings?.baseColor || '#00ffff';
        const visuals = getVisualsForNode(node, baseColorForNode, ssspResult);
        material.uniforms.pulseSpeed.value = visuals.pulseSpeed;

        
        tempMatrix.makeScale(visuals.scale, visuals.scale, visuals.scale);
        tempMatrix.setPosition(nodePositions[i3], nodePositions[i3 + 1], nodePositions[i3 + 2]);
        mesh.setMatrixAt(localIndex, tempMatrix);

        
        tempColor.copy(visuals.color);
        mesh.setColorAt(localIndex, tempColor);
      });

      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) {
        mesh.instanceColor.needsUpdate = true;
      }
    });
  });

  return (
    <>
      {Array.from(nodeGroups.entries()).map(([geometryType, group]) => (
        <instancedMesh
          key={geometryType}
          ref={(ref) => { 
            if (ref) {
              meshRefs.current.set(geometryType, ref);
              
              if (!ref.layers) {
                ref.layers = new THREE.Layers();
              }
              
              ref.layers.set(0); 
              ref.layers.enable(1); 
              ref.layers.disable(2); 
            }
          }}
          args={[(geometries as Record<string, THREE.BufferGeometry>)[geometryType], material, group.nodes.length]}
          frustumCulled={false}
          onClick={(e) => {
            if (e.instanceId !== undefined && onNodeClick) {
              onNodeClick(group.nodes[e.instanceId].id, e);
            }
          }}
        />
      ))}
    </>
  );
};