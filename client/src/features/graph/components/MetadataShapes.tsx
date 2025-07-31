// client/src/features/graph/components/MetadataShapes.tsx
import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { type Node as GraphNode } from '../managers/graphDataManager';
import { HologramNodeMaterial } from '../shaders/HologramNodeMaterial';

// --- 1. Define Visual Metaphor Logic ---
// This helper function is the heart of our new system.
const getVisualsForNode = (node: GraphNode, settingsBaseColor?: string) => {
  const visuals = {
    geometryType: 'sphere' as 'sphere' | 'box' | 'octahedron' | 'icosahedron',
    scale: 1.0,
    color: new THREE.Color(settingsBaseColor || '#00ffff'), // Use base color from settings
    emissive: new THREE.Color(settingsBaseColor || '#00ffff'),
    pulseSpeed: 0.5,
  };

  const { metadata } = node;
  if (!metadata) return visuals;
  
  // Debug: Log first few nodes to see what metadata we have
  // Only log when debug is enabled via debugState
  if (typeof window !== 'undefined' && window.debugState?.isEnabled?.() && Math.random() < 0.05) { // Log 5% of nodes to see variation
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

  // METAPHOR 1: Geometry from Connectivity (hyperlinkCount)
  const hyperlinkCount = parseInt(metadata.hyperlinkCount || '0', 10);
  if (hyperlinkCount > 7) {
    visuals.geometryType = 'icosahedron'; // Highly connected: Complex shape
  } else if (hyperlinkCount > 3) {
    visuals.geometryType = 'octahedron'; // Well-connected: Interesting shape
  } else if (hyperlinkCount > 0) {
    visuals.geometryType = 'box'; // Some connections: Simple, distinct shape
  } else {
    visuals.geometryType = 'sphere'; // No connections: Base shape
  }

  // METAPHOR 2: Scale from Connectivity & File Size
  const fileSize = parseInt(metadata.fileSize || '0', 10);
  const sizeScale = 0.8 + Math.log10(Math.max(1, fileSize / 1024)) * 0.2; // Log scale for size
  const connectionScale = 1 + hyperlinkCount * 0.05;
  visuals.scale = THREE.MathUtils.clamp(sizeScale * connectionScale, 0.5, 3.0);

  // METAPHOR 3: Color modulation from Recency (lastModified) or Type - ADDITIVE to base color
  const originalColor = new THREE.Color(visuals.color); // Store original base color
  const lastModified = metadata.lastModified ? new Date(metadata.lastModified).getTime() : 0;
  
  if (lastModified > 0) {
    const ageInDays = (Date.now() - lastModified) / (1000 * 60 * 60 * 24);
    // Create age-based color modulation (heat effect)
    const heat = Math.max(0, 1 - ageInDays / 90);
    
    // Convert base color to HSL for modulation
    const hsl = { h: 0, s: 0, l: 0 };
    originalColor.getHSL(hsl);
    
    // Modulate the base color: shift hue slightly toward warm colors when recent
    const hueShift = heat * 0.15; // More noticeable hue shift toward yellow/orange
    const saturationBoost = heat * 0.3; // Increase saturation for recent files
    const lightnessBoost = heat * 0.25; // Brighten recent files
    
    visuals.color.setHSL(
      (hsl.h + hueShift) % 1,
      Math.min(1, hsl.s + saturationBoost),
      Math.min(1, hsl.l + lightnessBoost)
    );
  } else if (metadata.type) {
    // Apply more noticeable type-based color tinting to base color
    const typeColorShifts: Record<string, { hue: number, sat: number, light: number }> = {
      'folder': { hue: 0.1, sat: 0.2, light: 0.15 },     // Yellow shift
      'file': { hue: 0.0, sat: 0.1, light: 0.05 },       // Slight saturation boost
      'function': { hue: -0.1, sat: 0.2, light: 0.1 },   // Red shift
      'class': { hue: 0.05, sat: 0.15, light: 0.1 },     // Green shift
      'variable': { hue: 0.15, sat: 0.12, light: 0.08 }, // Green shift
      'import': { hue: -0.06, sat: 0.1, light: 0.05 },   // Red shift
      'export': { hue: -0.15, sat: 0.15, light: 0.08 },  // Purple shift
      'default': { hue: 0.0, sat: 0.0, light: 0.0 }      // No change
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
    // Apply connectivity-based subtle modulation to base color
    const colorIntensity = Math.min(hyperlinkCount / 10, 1);
    const hsl = { h: 0, s: 0, l: 0 };
    originalColor.getHSL(hsl);
    
    // More noticeable modulation based on connectivity
    const saturationBoost = colorIntensity * 0.25;
    const lightnessBoost = colorIntensity * 0.2;
    
    visuals.color.setHSL(
      hsl.h,
      Math.min(1, hsl.s + saturationBoost),
      Math.min(1, hsl.l + lightnessBoost)
    );
  }

  // METAPHOR 4: Emissive Glow from AI Processing (perplexityLink) - ADDITIVE to base color
  if (metadata.perplexityLink) {
    // Blend gold glow with base color for AI-processed nodes
    const goldTint = new THREE.Color('#FFD700');
    visuals.emissive.copy(originalColor).lerp(goldTint, 0.6); // 60% gold, 40% base color
  } else {
    // Use modulated color for emissive at reduced intensity
    visuals.emissive.copy(visuals.color).multiplyScalar(0.3);
  }

  // METAPHOR 5: Pulse Speed from File Size
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
}

export const MetadataShapes: React.FC<MetadataShapesProps> = ({ nodes, nodePositions, onNodeClick, settings }) => {
  const geometries = useGeometries();
  const material = useHologramMaterial(settings);
  const meshRefs = useRef<Map<string, THREE.InstancedMesh>>(new Map());

  // Group nodes by their new geometry type for instanced rendering
  const nodeGroups = useMemo(() => {
    const groups = new Map<string, { nodes: GraphNode[], originalIndices: number[] }>();
    const nodeSettings = settings?.visualisation?.graphs?.logseq?.nodes || settings?.visualisation?.nodes;
    const baseColor = nodeSettings?.baseColor || '#00ffff';
    
    nodes.forEach((node, index) => {
      const { geometryType } = getVisualsForNode(node, baseColor);
      if (!groups.has(geometryType)) {
        groups.set(geometryType, { nodes: [], originalIndices: [] });
      }
      groups.get(geometryType)!.nodes.push(node);
      groups.get(geometryType)!.originalIndices.push(index);
    });
    return groups;
  }, [nodes, settings]);

  // Frame loop to update instances
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
        const visuals = getVisualsForNode(node, baseColorForNode);
        material.uniforms.pulseSpeed.value = visuals.pulseSpeed;

        // Position & Scale
        tempMatrix.makeScale(visuals.scale, visuals.scale, visuals.scale);
        tempMatrix.setPosition(nodePositions[i3], nodePositions[i3 + 1], nodePositions[i3 + 2]);
        mesh.setMatrixAt(localIndex, tempMatrix);

        // Color
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
          ref={(ref) => { if (ref) meshRefs.current.set(geometryType, ref); }}
          args={[geometries[geometryType], material, group.nodes.length]}
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