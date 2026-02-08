// client/src/features/graph/components/MetadataShapes.tsx
import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { type Node as GraphNode } from '../managers/graphDataManager';
import { HologramNodeMaterial } from '../../../rendering/materials/HologramNodeMaterial';
import type { GraphVisualMode } from './GraphManager';

// Extended geometry type set for all three modes
type MetadataGeometryType = 'sphere' | 'box' | 'octahedron' | 'icosahedron' | 'dodecahedron' | 'tetrahedron' | 'torus';

// === Ontology depth colors (pre-computed) ===
const ONTOLOGY_DEPTH_COLORS = [
  new THREE.Color('#FF6B6B'), // depth 0: red giant
  new THREE.Color('#FFD93D'), // depth 1: yellow star
  new THREE.Color('#4ECDC4'), // depth 2: cyan nebula
  new THREE.Color('#AA96DA'), // depth 3: purple distant
  new THREE.Color('#95E1D3'), // depth 4+: pale ethereal
];

// === Agent status colors (pre-computed) ===
const AGENT_STATUS_COLORS_MD: Record<string, THREE.Color> = {
  'active': new THREE.Color('#2ECC71'),
  'busy': new THREE.Color('#F39C12'),
  'idle': new THREE.Color('#95A5A6'),
  'error': new THREE.Color('#E74C3C'),
  'default': new THREE.Color('#2ECC71'),
};

// --- 1. Define Visual Metaphor Logic (mode-aware) ---
const getVisualsForNode = (
  node: GraphNode,
  settingsBaseColor?: string,
  ssspResult?: any,
  graphMode: GraphVisualMode = 'knowledge_graph',
  hierarchyMap?: Map<string, any>
) => {
  const visuals = {
    geometryType: 'sphere' as MetadataGeometryType,
    scale: 1.0,
    color: new THREE.Color(settingsBaseColor || '#00ffff'),
    emissive: new THREE.Color(settingsBaseColor || '#00ffff'),
    pulseSpeed: 0.5,
  };

  // SSSP visualization overrides all modes
  if (ssspResult) {
    const distance = ssspResult.distances[node.id];

    if (node.id === ssspResult.sourceNodeId) {
      visuals.color = new THREE.Color('#00FFFF');
      visuals.emissive = new THREE.Color('#00FFFF');
      visuals.scale = 1.5;
      visuals.pulseSpeed = 2.0;
      visuals.geometryType = 'icosahedron';
    } else if (!isFinite(distance)) {
      visuals.color = new THREE.Color('#666666');
      visuals.emissive = new THREE.Color('#333333');
      visuals.scale = 0.7;
      visuals.pulseSpeed = 0.1;
    } else {
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

  // === ONTOLOGY MODE ===
  if (graphMode === 'ontology') {
    const metadata = node.metadata || {};
    const nodeType = (metadata.type || metadata.role || '').toLowerCase();
    const hierarchyNode = hierarchyMap?.get(node.id);
    const depth = hierarchyNode?.depth ?? (metadata.depth ?? 0);
    const instanceCount = parseInt(metadata.instanceCount || '0', 10);

    // Geometry by hierarchy role
    if (nodeType === 'property' || nodeType === 'datatype_property' || nodeType === 'object_property') {
      visuals.geometryType = 'torus';
      visuals.color = new THREE.Color('#F38181');
      visuals.emissive = new THREE.Color('#F38181').multiplyScalar(0.4);
      visuals.scale = 0.7;
    } else if (depth === 0) {
      // Root: large, smooth, stellar
      visuals.geometryType = 'sphere';
      visuals.scale = 1.4;
      const depthColor = ONTOLOGY_DEPTH_COLORS[0];
      visuals.color = depthColor.clone();
      visuals.emissive = depthColor.clone().multiplyScalar(0.5);
    } else if (depth <= 2) {
      // Mid-depth: faceted
      visuals.geometryType = 'icosahedron';
      visuals.scale = Math.max(0.6, 1.2 - depth * 0.2);
      const depthColor = ONTOLOGY_DEPTH_COLORS[Math.min(depth, ONTOLOGY_DEPTH_COLORS.length - 1)];
      visuals.color = depthColor.clone();
      visuals.emissive = depthColor.clone().multiplyScalar(0.4);
    } else {
      // Leaf: small octahedron
      visuals.geometryType = 'octahedron';
      visuals.scale = Math.max(0.4, 1.0 - depth * 0.15);
      const depthColor = ONTOLOGY_DEPTH_COLORS[Math.min(depth, ONTOLOGY_DEPTH_COLORS.length - 1)];
      visuals.color = depthColor.clone();
      visuals.emissive = depthColor.clone().multiplyScalar(0.3);
    }

    // Instance count scales the node
    if (instanceCount > 0) {
      visuals.scale *= (1 + Math.log(instanceCount + 1) * 0.1);
      // Emissive glow proportional to instanceCount
      const glowFactor = Math.min(instanceCount / 50, 0.4);
      visuals.emissive.offsetHSL(0, glowFactor * 0.2, glowFactor * 0.15);
    }

    visuals.pulseSpeed = 0.3 + depth * 0.1;
    return visuals;
  }

  // === AGENT MODE ===
  if (graphMode === 'agent') {
    const metadata = node.metadata || {};
    const agentType = (metadata.agentType || '').toLowerCase();
    const agentStatus = (metadata.status || 'active').toLowerCase();
    const workload = metadata.workload ?? 0;
    const tokenRate = metadata.tokenRate ?? 0;

    // Geometry by agent type
    if (agentType === 'queen') {
      visuals.geometryType = 'icosahedron';
      visuals.scale = 1.5;
      visuals.color = new THREE.Color('#FFD700');
      visuals.emissive = new THREE.Color('#FFD700').multiplyScalar(0.6);
    } else if (agentType === 'coordinator') {
      visuals.geometryType = 'dodecahedron';
      visuals.scale = 1.2;
      visuals.color = new THREE.Color('#E67E22');
      visuals.emissive = new THREE.Color('#E67E22').multiplyScalar(0.5);
    } else if (agentStatus === 'error') {
      visuals.geometryType = 'tetrahedron';
      visuals.scale = 0.9;
      visuals.color = new THREE.Color('#E74C3C');
      visuals.emissive = new THREE.Color('#E74C3C').multiplyScalar(0.7);
    } else {
      // Worker: smooth sphere
      visuals.geometryType = 'sphere';
      const statusColor = AGENT_STATUS_COLORS_MD[agentStatus] || AGENT_STATUS_COLORS_MD['default'];
      visuals.color = statusColor.clone();
      visuals.emissive = statusColor.clone().multiplyScalar(0.4);
    }

    // Workload-driven scale
    visuals.scale *= (1 + workload * 0.3 + Math.min(tokenRate / 100, 0.5));

    // Busy agents pulse faster
    visuals.pulseSpeed = agentStatus === 'busy' ? 2.0 : (agentStatus === 'idle' ? 0.3 : 1.0);
    return visuals;
  }

  // === KNOWLEDGE GRAPH MODE (default, enhanced) ===
  const { metadata } = node;
  if (!metadata) return visuals;


  // Geometry selection: authority and connections drive shape
  const authority = metadata.authority ?? metadata.authorityScore ?? 0;
  const hyperlinkCount = parseInt(metadata.hyperlinkCount || '0', 10);
  const nodeType = (metadata.type || '').toLowerCase();

  if (authority > 0.8) {
    visuals.geometryType = 'icosahedron';
  } else if (hyperlinkCount > 10) {
    visuals.geometryType = 'dodecahedron';
  } else if (nodeType === 'folder') {
    visuals.geometryType = 'octahedron';
  } else if (nodeType === 'function') {
    visuals.geometryType = 'tetrahedron';
  } else if (hyperlinkCount > 3) {
    visuals.geometryType = 'icosahedron';
  } else if (hyperlinkCount > 0) {
    visuals.geometryType = 'box';
  } else {
    visuals.geometryType = 'sphere';
  }

  // Scale: include authority and quality factors
  const fileSize = parseInt(metadata.fileSize || '0', 10);
  const qualityScore = parseFloat(metadata.quality_score || metadata.quality || '0');
  const sizeScale = 0.8 + Math.log10(Math.max(1, fileSize / 1024)) * 0.2;
  const connectionScale = 1 + hyperlinkCount * 0.05;
  const authorityBoost = 1 + authority * 0.4;
  const qualityBoost = 1 + (isNaN(qualityScore) ? 0 : qualityScore * 0.2);
  visuals.scale = THREE.MathUtils.clamp(sizeScale * connectionScale * authorityBoost * qualityBoost, 0.5, 3.0);

  // Color: domain-tinted with recency heat
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

  // Emissive: domain-colored inner glow
  if (metadata.perplexityLink) {
    const goldTint = new THREE.Color('#FFD700');
    visuals.emissive.copy(originalColor).lerp(goldTint, 0.6);
  } else {
    visuals.emissive.copy(visuals.color).multiplyScalar(0.3);
  }

  // Authority-based emissive intensity boost
  if (authority > 0) {
    visuals.emissive.offsetHSL(0, authority * 0.15, authority * 0.1);
  }

  visuals.pulseSpeed = 0.5 + Math.log10(Math.max(1, fileSize / 1024)) * 0.5;

  return visuals;
};


// --- 2. Create Geometry and Material Resources ---
// All geometries normalized to ~0.5 bounding sphere radius to match the base sphere.
// Shape differences are visual distinction only; size is controlled by nodeSize setting.
const BASE_SPHERE_RADIUS = 0.5;
const useGeometries = () => useMemo(() => ({
  sphere: new THREE.SphereGeometry(0.5, 32, 16),
  box: new THREE.BoxGeometry(0.58, 0.58, 0.58),         // bounding radius ≈ 0.5
  octahedron: new THREE.OctahedronGeometry(0.5, 0),
  icosahedron: new THREE.IcosahedronGeometry(0.5, 1),
  dodecahedron: new THREE.DodecahedronGeometry(0.5, 0),
  tetrahedron: new THREE.TetrahedronGeometry(0.5, 0),
  torus: new THREE.TorusGeometry(0.35, 0.12, 8, 16),    // bounding radius ≈ 0.47
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
  onNodeDoubleClick?: (nodeId: string, node: GraphNode, event: any) => void;
  settings: any;
  ssspResult?: any;
  graphMode?: GraphVisualMode;
  hierarchyMap?: Map<string, any>;
}

export const MetadataShapes: React.FC<MetadataShapesProps> = ({
  nodes,
  nodePositions,
  onNodeClick,
  onNodeDoubleClick,
  settings,
  ssspResult,
  graphMode = 'knowledge_graph',
  hierarchyMap
}) => {
  const geometries = useGeometries();
  const material = useHologramMaterial(settings);
  const meshRefs = useRef<Map<string, THREE.InstancedMesh>>(new Map());

  // Group nodes by geometry type (mode-aware)
  const nodeGroups = useMemo(() => {
    const groups = new Map<string, { nodes: GraphNode[], originalIndices: number[] }>();
    const nodeSettings = settings?.visualisation?.graphs?.logseq?.nodes || settings?.visualisation?.nodes;
    const baseColor = nodeSettings?.baseColor || '#00ffff';

    nodes.forEach((node, index) => {
      const { geometryType } = getVisualsForNode(node, baseColor, ssspResult, graphMode, hierarchyMap);
      if (!groups.has(geometryType)) {
        groups.set(geometryType, { nodes: [], originalIndices: [] });
      }
      groups.get(geometryType)!.nodes.push(node);
      groups.get(geometryType)!.originalIndices.push(index);
    });
    return groups;
  }, [nodes, settings, ssspResult, graphMode, hierarchyMap]);

  // Per-frame updates
  useFrame((state) => {
    if (!nodePositions) return;

    material.updateTime(state.clock.elapsedTime);
    const tempMatrix = new THREE.Matrix4();
    const tempColor = new THREE.Color();

    // Hoist settings lookups out of per-node loop
    const nodeSettings = settings?.visualisation?.graphs?.logseq?.nodes || settings?.visualisation?.nodes;
    const baseColorForNode = nodeSettings?.baseColor || '#00ffff';
    const nodeSize = nodeSettings?.nodeSize || 0.5;
    const sizeMultiplier = nodeSize / BASE_SPHERE_RADIUS;

    nodeGroups.forEach((group, geometryType) => {
      const mesh = meshRefs.current.get(geometryType);
      if (!mesh) return;

      group.nodes.forEach((node, localIndex) => {
        const originalIndex = group.originalIndices[localIndex];
        const i3 = originalIndex * 3;
        if (!nodePositions || i3 + 2 >= nodePositions.length) return;

        const visuals = getVisualsForNode(node, baseColorForNode, ssspResult, graphMode, hierarchyMap);
        material.uniforms.pulseSpeed.value = visuals.pulseSpeed;

        const finalScale = visuals.scale * sizeMultiplier;
        tempMatrix.makeScale(finalScale, finalScale, finalScale);
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
          onDoubleClick={(e) => {
            if (e.instanceId !== undefined && onNodeDoubleClick) {
              const node = group.nodes[e.instanceId];
              onNodeDoubleClick(node.id, node, e);
            }
          }}
        />
      ))}
    </>
  );
};
