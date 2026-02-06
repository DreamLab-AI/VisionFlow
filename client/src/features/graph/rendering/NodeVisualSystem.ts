/**
 * NodeVisualSystem.ts
 *
 * Central configuration mapping graph visual modes to full visual profiles.
 * Each mode defines geometry, shader parameters, scale logic, and color palettes
 * that feed into HologramNodeMaterial for rendering visually distinct node types
 * across knowledge graph, ontology, and agent views.
 */

import type { GraphNode } from '../types/graphTypes';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type GraphVisualMode = 'knowledge_graph' | 'ontology' | 'agent';

export type NodeGeometry =
  | 'sphere'
  | 'icosahedron'
  | 'dodecahedron'
  | 'octahedron'
  | 'torus'
  | 'tetrahedron';

export interface NodeVisualProfile {
  // Geometry
  defaultGeometry: NodeGeometry;
  geometryDetailHigh: number;
  geometryDetailMed: number;
  geometryDetailLow: number;

  // Material parameters (fed to HologramNodeMaterial)
  shaderMode: number; // 0=crystal, 1=constellation, 2=organic
  rimPower: number;
  scanlineCount: number;
  scanlineSpeed: number;
  glowStrength: number;
  hologramStrength: number;
  pulseSpeed: number;
  pulseStrength: number;
  metalness: number;
  roughness: number;

  // Scale computation
  baseScale: number;
  connectionScaleFactor: number;
  maxScale: number;
  minScale: number;

  // Color palette
  palette: Record<string, string>; // type/status -> hex color
  defaultColor: string;
  emissiveIntensity: number;
}

// ---------------------------------------------------------------------------
// Profile definitions
// ---------------------------------------------------------------------------

const KNOWLEDGE_GRAPH_PROFILE: NodeVisualProfile = {
  // Geometry
  defaultGeometry: 'icosahedron',
  geometryDetailHigh: 2,
  geometryDetailMed: 1,
  geometryDetailLow: 0,

  // Shader: Crystal
  shaderMode: 0,
  rimPower: 3.0,
  scanlineCount: 0,
  scanlineSpeed: 0,
  glowStrength: 2.5,
  hologramStrength: 0.3,
  metalness: 0.6,
  roughness: 0.15,
  pulseSpeed: 0.8,
  pulseStrength: 0.1,

  // Scale
  baseScale: 1.0,
  connectionScaleFactor: 0.4,
  maxScale: 3.5,
  minScale: 0.5,

  // Colors: Domain-based metallic crystals
  palette: {
    AI: '#4FC3F7', // metallic blue crystal
    BC: '#81C784', // emerald crystal
    RB: '#FFB74D', // amber crystal
    MV: '#CE93D8', // amethyst crystal
    TC: '#FFD54F', // topaz crystal
    DT: '#EF5350', // ruby crystal
  },
  defaultColor: '#E0E0E0', // diamond / clear crystal
  emissiveIntensity: 0.5,
};

const ONTOLOGY_PROFILE: NodeVisualProfile = {
  // Geometry
  defaultGeometry: 'sphere',
  geometryDetailHigh: 32,
  geometryDetailMed: 16,
  geometryDetailLow: 8,

  // Shader: Constellation
  shaderMode: 1,
  rimPower: 1.5,
  scanlineCount: 8,
  scanlineSpeed: 0.5,
  glowStrength: 1.8,
  hologramStrength: 0.7,
  metalness: 0.1,
  roughness: 0.4,
  pulseSpeed: 0.3,
  pulseStrength: 0.08,

  // Scale
  baseScale: 1.0,
  connectionScaleFactor: 0.02,
  maxScale: 4.0,
  minScale: 0.4,

  // Colors: Depth-gradient cosmic spectrum
  palette: {
    depth_0: '#FF6B6B', // red giant -- root
    depth_1: '#FFD93D', // yellow star -- primary class
    depth_2: '#4ECDC4', // cyan nebula -- intermediate
    depth_3: '#AA96DA', // purple -- deep hierarchy
    depth_4: '#95E1D3', // pale -- distant
    property: '#F38181', // warm pink link
  },
  defaultColor: '#95E1D3',
  emissiveIntensity: 0.7,
};

const AGENT_PROFILE: NodeVisualProfile = {
  // Geometry
  defaultGeometry: 'sphere',
  geometryDetailHigh: 32,
  geometryDetailMed: 16,
  geometryDetailLow: 8,

  // Shader: Organic
  shaderMode: 2,
  rimPower: 2.0,
  scanlineCount: 0,
  scanlineSpeed: 0,
  glowStrength: 2.0,
  hologramStrength: 0.1,
  metalness: 0.0,
  roughness: 0.7,
  pulseSpeed: 1.5,
  pulseStrength: 0.4,

  // Scale
  baseScale: 1.0,
  connectionScaleFactor: 0.3,
  maxScale: 3.0,
  minScale: 0.6,

  // Colors: Status-based bioluminescence
  palette: {
    active: '#2ECC71', // healthy green
    busy: '#F39C12',   // working amber
    idle: '#95A5A6',   // dormant gray
    error: '#E74C3C',  // distress red
    queen: '#FFD700',  // golden
  },
  defaultColor: '#2ECC71',
  emissiveIntensity: 0.6,
};

const PROFILES: Record<GraphVisualMode, NodeVisualProfile> = {
  knowledge_graph: KNOWLEDGE_GRAPH_PROFILE,
  ontology: ONTOLOGY_PROFILE,
  agent: AGENT_PROFILE,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Returns the full visual profile for a given graph mode.
 */
export function getVisualProfile(mode: GraphVisualMode): NodeVisualProfile {
  return PROFILES[mode];
}

/**
 * Determines the appropriate geometry type for a node based on graph mode
 * and node metadata.
 */
export function getNodeGeometryForMode(
  mode: GraphVisualMode,
  node: GraphNode,
): NodeGeometry {
  const meta = node.metadata ?? {};

  switch (mode) {
    // Knowledge Graph: geometry driven by authority / connections / type
    case 'knowledge_graph': {
      const authority = typeof meta.authority === 'number' ? meta.authority : 0;
      const connections = typeof meta.connections === 'number' ? meta.connections : 0;
      const nodeType = node.nodeType ?? meta.type ?? '';

      if (authority > 0.8) return 'icosahedron';
      if (connections > 10) return 'dodecahedron';
      if (nodeType === 'folder') return 'octahedron';
      if (nodeType === 'function') return 'tetrahedron';
      return 'icosahedron';
    }

    // Ontology: geometry driven by hierarchy depth and role
    case 'ontology': {
      const depth = typeof meta.depth === 'number' ? meta.depth : 1;
      const role = meta.role ?? meta.type ?? '';

      if (role === 'property') return 'torus';
      if (role === 'instance') return 'dodecahedron';
      if (depth === 0) return 'sphere';
      if (depth <= 2) return 'icosahedron';
      return 'octahedron';
    }

    // Agent: always sphere (biological cell metaphor)
    case 'agent':
      return 'sphere';

    default: {
      // Exhaustive check — all GraphVisualMode values handled above
      const _exhaustive: never = mode;
      void _exhaustive;
      return 'sphere';
    }
  }
}

/**
 * Computes the rendered scale of a node based on graph mode, node metadata,
 * and connection count.
 */
export function getNodeScaleForMode(
  mode: GraphVisualMode,
  node: GraphNode,
  connectionCount: number,
): number {
  const profile = PROFILES[mode];
  const meta = node.metadata ?? {};

  let scale: number;

  switch (mode) {
    case 'knowledge_graph': {
      const authority = typeof meta.authority === 'number' ? meta.authority : 0;
      const authorityBoost = 1 + authority * 0.5;
      scale =
        profile.baseScale *
        (1 + Math.log(connectionCount + 1) * profile.connectionScaleFactor) *
        authorityBoost;
      break;
    }

    case 'ontology': {
      const depth = typeof meta.depth === 'number' ? meta.depth : 1;
      const instanceCount = typeof meta.instanceCount === 'number' ? meta.instanceCount : 0;
      const depthFactor = Math.max(0.4, 1.0 - depth * 0.15);
      scale =
        profile.baseScale *
        (1 + instanceCount * profile.connectionScaleFactor) *
        depthFactor;
      break;
    }

    case 'agent': {
      const workload = typeof meta.workload === 'number' ? meta.workload : 0;
      const tokenRate = typeof meta.tokenRate === 'number' ? meta.tokenRate : 0;
      scale =
        profile.baseScale +
        workload * profile.connectionScaleFactor +
        (tokenRate / 100) * 0.5;
      break;
    }

    default:
      scale = profile.baseScale;
  }

  return Math.min(profile.maxScale, Math.max(profile.minScale, scale));
}

/**
 * Resolves base color, emissive color, and emissive intensity for a node
 * based on its graph mode and metadata.
 */
export function getNodeColorForMode(
  mode: GraphVisualMode,
  node: GraphNode,
): { base: string; emissive: string; emissiveIntensity: number } {
  const profile = PROFILES[mode];
  const meta = node.metadata ?? {};

  let base: string;

  switch (mode) {
    case 'knowledge_graph': {
      const domain =
        (meta.domain as string | undefined) ??
        (meta.category as string | undefined) ??
        '';
      base = profile.palette[domain] ?? profile.defaultColor;
      break;
    }

    case 'ontology': {
      const depth = typeof meta.depth === 'number' ? meta.depth : 1;
      const role = meta.role ?? meta.type ?? '';

      if (role === 'property') {
        base = profile.palette.property ?? profile.defaultColor;
      } else {
        const depthKey = `depth_${Math.min(depth, 4)}`;
        base = profile.palette[depthKey] ?? profile.defaultColor;
      }
      break;
    }

    case 'agent': {
      const status =
        (meta.status as string | undefined) ??
        (meta.role as string | undefined) ??
        'active';
      base = profile.palette[status] ?? profile.defaultColor;
      break;
    }

    default:
      base = profile.defaultColor;
  }

  return {
    base,
    emissive: base,
    emissiveIntensity: profile.emissiveIntensity,
  };
}
