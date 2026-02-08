import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
import { useThree, useFrame, ThreeEvent } from '@react-three/fiber'
import { Text, Billboard } from '@react-three/drei'
import * as THREE from 'three'
import { graphDataManager, type GraphData, type Node as GraphNode } from '../managers/graphDataManager'
import { graphWorkerProxy } from '../managers/graphWorkerProxy'
import { usePlatformStore } from '../../../services/platformManager'
import { createLogger, createErrorMetadata } from '../../../utils/loggerConfig'
import { debugState } from '../../../utils/clientDebugState'
import { useSettingsStore } from '../../../store/settingsStore'
import { BinaryNodeData, createBinaryNodeData, NodeType } from '../../../types/binaryProtocol'
import { useWebSocketStore } from '../../../store/websocketStore'
import { HologramNodeMaterial } from '../../../rendering/materials/HologramNodeMaterial'
import { FlowingEdges } from './FlowingEdges'
import { KnowledgeRings } from './KnowledgeRings'
import { ClusterHulls } from './ClusterHulls'
import { useGraphEventHandlers } from '../hooks/useGraphEventHandlers'
import { MetadataShapes } from './MetadataShapes'
import { NodeShaderToggle } from './NodeShaderToggle'
import { EdgeSettings } from '../../settings/config/settings'
import { registerNodeObject, unregisterNodeObject } from '../../visualisation/hooks/bloomRegistry'
import { useAnalyticsStore, useCurrentSSSPResult } from '../../analytics/store/analyticsStore'
import { detectHierarchy } from '../utils/hierarchyDetector'
import { useExpansionState } from '../hooks/useExpansionState'
// import { useBloomStrength } from '../contexts/BloomContext'

const logger = createLogger('GraphManager')

// === GRAPH VISUAL MODE ===
export type GraphVisualMode = 'knowledge_graph' | 'ontology' | 'agent';

// === PERFORMANCE OPTIMIZATION: Domain colors defined once outside component ===
const DOMAIN_COLORS: Record<string, string> = {
  'AI': '#4FC3F7',   // Light blue
  'BC': '#81C784',   // Green
  'RB': '#FFB74D',   // Orange
  'MV': '#CE93D8',   // Purple
  'TC': '#FFD54F',   // Yellow
  'DT': '#EF5350',   // Red
  'NGM': '#4DB6AC',  // Teal
};
const DEFAULT_DOMAIN_COLOR = '#90A4AE'; // Grey

// Pre-computed THREE.Color instances for domain colors (avoids GC pressure)
const DOMAIN_THREE_COLORS: Record<string, THREE.Color> = {};
Object.entries(DOMAIN_COLORS).forEach(([domain, hex]) => {
  DOMAIN_THREE_COLORS[domain] = new THREE.Color(hex);
});
DOMAIN_THREE_COLORS['default'] = new THREE.Color(DEFAULT_DOMAIN_COLOR);

// Muted domain colors (pre-computed at 0.7 intensity for metadata)
const DOMAIN_MUTED_COLORS: Record<string, string> = {};
Object.entries(DOMAIN_COLORS).forEach(([domain, hex]) => {
  DOMAIN_MUTED_COLORS[domain] = new THREE.Color(hex).multiplyScalar(0.7).getStyle();
});
DOMAIN_MUTED_COLORS['default'] = new THREE.Color(DEFAULT_DOMAIN_COLOR).multiplyScalar(0.7).getStyle();

// O(1) domain color lookup
const getDomainColor = (domain?: string): string => {
  return domain && DOMAIN_COLORS[domain] ? DOMAIN_COLORS[domain] : DEFAULT_DOMAIN_COLOR;
};

const getDomainMutedColor = (domain?: string): string => {
  return domain && DOMAIN_MUTED_COLORS[domain] ? DOMAIN_MUTED_COLORS[domain] : DOMAIN_MUTED_COLORS['default'];
};

// === ONTOLOGY MODE: Hierarchy depth color spectrum (cosmic) ===
const ONTOLOGY_DEPTH_COLORS: THREE.Color[] = [
  new THREE.Color('#FF6B6B'), // depth 0: red giant
  new THREE.Color('#FFD93D'), // depth 1: yellow star
  new THREE.Color('#4ECDC4'), // depth 2: cyan nebula
  new THREE.Color('#AA96DA'), // depth 3: purple distant
  new THREE.Color('#95E1D3'), // depth 4+: pale ethereal
];
const ONTOLOGY_PROPERTY_COLOR = new THREE.Color('#F38181');
const ONTOLOGY_INSTANCE_COLOR = new THREE.Color('#B8D4E3');

// === AGENT MODE: Status-based bioluminescence ===
const AGENT_STATUS_COLORS: Record<string, THREE.Color> = {
  'active': new THREE.Color('#2ECC71'),
  'busy': new THREE.Color('#F39C12'),
  'idle': new THREE.Color('#95A5A6'),
  'error': new THREE.Color('#E74C3C'),
  'default': new THREE.Color('#2ECC71'),
};
const AGENT_TYPE_COLORS: Record<string, THREE.Color> = {
  'queen': new THREE.Color('#FFD700'),
  'coordinator': new THREE.Color('#E67E22'),
};

// === MODE-AWARE MATERIAL PRESETS ===
interface MaterialModePreset {
  rimPower: number;
  glowStrength: number;
  hologramStrength: number;
  scanlineCount: number;
  pulseSpeed: number;
  pulseStrength: number;
}

const MATERIAL_MODE_PRESETS: Record<GraphVisualMode, MaterialModePreset> = {
  knowledge_graph: {
    rimPower: 3.0,
    glowStrength: 2.5,
    hologramStrength: 0.3,
    scanlineCount: 30.0,
    pulseSpeed: 1.0,
    pulseStrength: 0.1,
  },
  ontology: {
    rimPower: 1.5,
    glowStrength: 1.8,
    hologramStrength: 0.7,
    scanlineCount: 8.0,
    pulseSpeed: 0.8,
    pulseStrength: 0.05,
  },
  agent: {
    rimPower: 2.0,
    glowStrength: 2.0,
    hologramStrength: 0.4,
    scanlineCount: 15.0,
    pulseSpeed: 1.5,
    pulseStrength: 0.15,
  },
};

// === MODE-SPECIFIC METADATA OVERLAY HELPERS ===

// Detect the dominant graph visual mode from node population (sampled for perf)
const detectGraphMode = (nodes: GraphNode[]): GraphVisualMode => {
  if (nodes.length === 0) return 'knowledge_graph';
  const sample = nodes.length > 50 ? nodes.slice(0, 50) : nodes;
  let ontologySignals = 0;
  let agentSignals = 0;
  for (const n of sample) {
    if ((n as any).owlClassIri || n.metadata?.hierarchyDepth !== undefined || n.metadata?.depth !== undefined) {
      ontologySignals++;
    }
    if (n.metadata?.agentType || n.metadata?.status === 'active' || n.metadata?.status === 'idle'
        || n.metadata?.status === 'busy' || n.metadata?.status === 'error') {
      agentSignals++;
    }
  }
  const threshold = sample.length * 0.2;
  if (agentSignals > threshold && agentSignals >= ontologySignals) return 'agent';
  if (ontologySignals > threshold) return 'ontology';
  return 'knowledge_graph';
};

// Map binary protocol NodeType to GraphVisualMode for per-node rendering
const nodeTypeToVisualMode = (nodeType: NodeType): GraphVisualMode => {
  switch (nodeType) {
    case NodeType.Agent:
      return 'agent';
    case NodeType.OntologyClass:
    case NodeType.OntologyIndividual:
    case NodeType.OntologyProperty:
      return 'ontology';
    case NodeType.Knowledge:
      return 'knowledge_graph';
    default:
      return 'knowledge_graph';
  }
};

// Quality score -> star rating string (1-5 filled stars, unicode)
const getQualityStars = (quality?: number | string): string => {
  if (quality === undefined || quality === null) return '';
  const score = typeof quality === 'string' ? parseFloat(quality) : quality;
  if (isNaN(score)) return '';
  const normalized = score <= 1 ? score * 5 : Math.min(score, 5);
  const filled = Math.round(normalized);
  return '\u2605'.repeat(filled) + '\u2606'.repeat(5 - filled);
};

// Time-ago text from a date/timestamp
const getRecencyText = (lastModified?: string | number | Date): string => {
  if (!lastModified) return '';
  const modDate = lastModified instanceof Date ? lastModified : new Date(lastModified);
  if (isNaN(modDate.getTime())) return '';
  const diffMs = Date.now() - modDate.getTime();
  if (diffMs < 0) return 'Updated just now';
  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 1) return 'Updated just now';
  if (minutes < 60) return `Updated ${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `Updated ${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `Updated ${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `Updated ${months}mo ago`;
  return `Updated ${Math.floor(months / 12)}y ago`;
};

// Recency color: warm (recent) -> cool (old)
const getRecencyColor = (lastModified?: string | number | Date): string => {
  if (!lastModified) return '#666666';
  const modDate = lastModified instanceof Date ? lastModified : new Date(lastModified);
  if (isNaN(modDate.getTime())) return '#666666';
  const diffDays = (Date.now() - modDate.getTime()) / 86400000;
  if (diffDays < 1) return '#4FC3F7';
  if (diffDays < 7) return '#81C784';
  if (diffDays < 30) return '#FFD54F';
  if (diffDays < 90) return '#FFB74D';
  return '#90A4AE';
};

// Ontology depth hex colors (string version for Text components)
const ONTOLOGY_DEPTH_HEX = ['#FF6B6B', '#FFD93D', '#4ECDC4', '#AA96DA', '#95E1D3'];
const getOntologyDepthHex = (depth: number): string => {
  return ONTOLOGY_DEPTH_HEX[Math.min(depth, ONTOLOGY_DEPTH_HEX.length - 1)];
};

// Ontology node category detection
const getOntologyCategory = (node: GraphNode): 'class' | 'property' | 'instance' => {
  const meta = node.metadata ?? {};
  const role = meta.role ?? meta.type ?? '';
  if (role === 'property' || (node as any).nodeType === 'property') return 'property';
  if (role === 'instance' || (node as any).nodeType === 'instance') return 'instance';
  return 'class';
};

// Category indicator symbols for ontology nodes
const ONTOLOGY_CATEGORY_DISPLAY: Record<string, string> = {
  class: '\u25C9 Class',
  property: '\u25C7 Property',
  instance: '\u25CB Instance',
};

// Agent status hex colors (string version for Text components)
const AGENT_STATUS_HEX: Record<string, string> = {
  active: '#2ECC71',
  busy: '#F39C12',
  idle: '#95A5A6',
  error: '#E74C3C',
  queen: '#FFD700',
};
const getAgentStatusHex = (status?: string): string => {
  return AGENT_STATUS_HEX[status ?? 'idle'] ?? '#95A5A6';
};

// === END METADATA OVERLAY HELPERS ===

// Enhanced position calculation with better distribution
const getPositionForNode = (node: GraphNode, index: number, totalNodes: number): [number, number, number] => {
  if (!node.position || (node.position.x === 0 && node.position.y === 0 && node.position.z === 0)) {
    
    const goldenAngle = Math.PI * (3 - Math.sqrt(5))
    const theta = index * goldenAngle
    const y = 1 - (index / totalNodes) * 2
    const radius = Math.sqrt(1 - y * y)

    const scaleFactor = 15 + Math.random() * 5 
    const x = Math.cos(theta) * radius * scaleFactor
    const z = Math.sin(theta) * radius * scaleFactor
    const yScaled = y * scaleFactor

    if (node.position) {
      node.position.x = x
      node.position.y = yScaled
      node.position.z = z
    } else {
      node.position = { x, y: yScaled, z }
    }

    return [x, yScaled, z]
  }

  return [node.position.x, node.position.y, node.position.z]
}

// === MODE-AWARE LOD GEOMETRY SETS ===
type LODGeometrySet = {
  high: THREE.BufferGeometry;
  medium: THREE.BufferGeometry;
  low: THREE.BufferGeometry;
};

// Default LOD geometries - static fallback for knowledge_graph
const LOD_GEOMETRIES: LODGeometrySet = {
  high: new THREE.IcosahedronGeometry(0.5, 2),     // 80 faceted faces, gem-like
  medium: new THREE.IcosahedronGeometry(0.5, 1),   // 20 faces
  low: new THREE.OctahedronGeometry(0.5),           // 8 faces
};

const createLODGeometries = (mode: GraphVisualMode): LODGeometrySet => {
  switch (mode) {
    case 'ontology':
      return {
        high: new THREE.SphereGeometry(0.5, 32, 32),   // Smooth stellar
        medium: new THREE.SphereGeometry(0.5, 16, 16),
        low: new THREE.SphereGeometry(0.5, 8, 8),
      };
    case 'agent':
      return {
        high: new THREE.SphereGeometry(0.5, 24, 24),   // Smooth organic
        medium: new THREE.SphereGeometry(0.5, 12, 12),
        low: new THREE.SphereGeometry(0.5, 8, 8),
      };
    case 'knowledge_graph':
    default:
      return {
        high: new THREE.IcosahedronGeometry(0.5, 2),   // Faceted crystal
        medium: new THREE.IcosahedronGeometry(0.5, 1),
        low: new THREE.OctahedronGeometry(0.5),
      };
  }
};

// Get geometry for node type (kept for metadata shapes)
const getGeometryForNodeType = (type?: string): THREE.BufferGeometry => {
  switch (type?.toLowerCase()) {
    case 'folder':
      return new THREE.OctahedronGeometry(0.6, 0);
    case 'file':
      return new THREE.BoxGeometry(0.8, 0.8, 0.8);
    case 'concept':
      return new THREE.IcosahedronGeometry(0.5, 0);
    case 'todo':
      return new THREE.ConeGeometry(0.5, 1, 4);
    case 'reference':
      return new THREE.TorusGeometry(0.5, 0.2, 8, 16);
    default:
      return LOD_GEOMETRIES.high;
  }
};

// Reusable Color for getNodeColor to eliminate per-call allocation
const _nodeColor = new THREE.Color();

// Pre-computed type colors as THREE.Color instances (avoid re-parsing hex strings)
const TYPE_THREE_COLORS: Record<string, THREE.Color> = {
  'folder': new THREE.Color('#FFD700'),
  'file': new THREE.Color('#00CED1'),
  'function': new THREE.Color('#FF6B6B'),
  'class': new THREE.Color('#4ECDC4'),
  'variable': new THREE.Color('#95E1D3'),
  'import': new THREE.Color('#F38181'),
  'export': new THREE.Color('#AA96DA'),
  'default': new THREE.Color('#00ffff'),
};

// === MODE-AWARE NODE COLOR ===
// Returns the shared _nodeColor instance -- caller must use values before next call
const getNodeColor = (
  node: GraphNode,
  ssspResult?: any,
  graphMode: GraphVisualMode = 'knowledge_graph',
  hierarchyMap?: Map<string, any>,
  connectionCountMap?: Map<string, number>
): THREE.Color => {

  // SSSP visualization overrides all modes
  if (ssspResult) {
    const distance = ssspResult.distances[node.id]

    if (node.id === ssspResult.sourceNodeId) {
      return _nodeColor.set('#00FFFF')
    }

    if (!isFinite(distance)) {
      return _nodeColor.set('#666666')
    }

    const normalizedDistances = ssspResult.normalizedDistances || {}
    const normalizedDistance = normalizedDistances[node.id] || 0

    const red = Math.min(1, normalizedDistance * 1.2)
    const green = Math.min(1, (1 - normalizedDistance) * 1.2)
    const blue = 0.1

    return _nodeColor.setRGB(red, green, blue)
  }

  // --- ONTOLOGY MODE: cosmic hierarchy spectrum ---
  if (graphMode === 'ontology') {
    const nodeType = node.metadata?.type?.toLowerCase() || '';

    // Properties get warm pink
    if (nodeType === 'property' || nodeType === 'datatype_property' || nodeType === 'object_property') {
      return _nodeColor.copy(ONTOLOGY_PROPERTY_COLOR);
    }
    // Instances get white-blue
    if (nodeType === 'instance' || nodeType === 'individual') {
      return _nodeColor.copy(ONTOLOGY_INSTANCE_COLOR);
    }

    // Class nodes: color by hierarchy depth
    const hierarchyNode = hierarchyMap?.get(node.id);
    const depth = hierarchyNode?.depth ?? (node.metadata?.depth ?? 0);
    const depthIndex = Math.min(depth, ONTOLOGY_DEPTH_COLORS.length - 1);
    _nodeColor.copy(ONTOLOGY_DEPTH_COLORS[depthIndex]);

    // Emissive glow proportional to instanceCount
    const instanceCount = parseInt(node.metadata?.instanceCount || '0', 10);
    if (instanceCount > 0) {
      const glowFactor = Math.min(instanceCount / 50, 0.4);
      _nodeColor.offsetHSL(0, glowFactor * 0.2, glowFactor * 0.15);
    }

    return _nodeColor;
  }

  // --- AGENT MODE: status-based bioluminescence ---
  if (graphMode === 'agent') {
    const agentType = node.metadata?.agentType?.toLowerCase() || '';
    const agentStatus = node.metadata?.status?.toLowerCase() || 'active';

    // Queen and coordinator types override status color
    if (AGENT_TYPE_COLORS[agentType]) {
      return _nodeColor.copy(AGENT_TYPE_COLORS[agentType]);
    }

    // Status-based color
    const statusColor = AGENT_STATUS_COLORS[agentStatus] || AGENT_STATUS_COLORS['default'];
    return _nodeColor.copy(statusColor);
  }

  // --- KNOWLEDGE GRAPH MODE (default): enhanced with authority brightness ---
  const nodeType = node.metadata?.type || 'default'
  const precomputed = TYPE_THREE_COLORS[nodeType];
  if (precomputed) {
    _nodeColor.copy(precomputed);
  } else {
    _nodeColor.copy(TYPE_THREE_COLORS['default']);
  }

  // Authority-based brightness boost: higher authority = brighter, more saturated
  const authority = node.metadata?.authority ?? node.metadata?.authorityScore ?? 0;
  if (authority > 0) {
    const brightnessFactor = authority * 0.3;
    _nodeColor.offsetHSL(0, brightnessFactor * 0.2, brightnessFactor);
  }

  // Metallic tinting for crystal aesthetic on highly-connected nodes
  const connections = connectionCountMap?.get(node.id) || 0;
  if (connections > 5) {
    const metallicShift = Math.min(connections / 30, 0.15);
    _nodeColor.offsetHSL(-0.02 * metallicShift, 0.1 * metallicShift, 0.05 * metallicShift);
  }

  return _nodeColor;
}

// === MODE-AWARE NODE SCALE ===
const getNodeScale = (
  node: GraphNode,
  edges: any[],
  connectionCountMap?: Map<string, number>,
  graphMode: GraphVisualMode = 'knowledge_graph',
  hierarchyMap?: Map<string, any>
): number => {
  const baseSize = node.metadata?.size || 1.0;
  let connectionCount: number;
  const nodeIdStr = String(node.id);
  if (connectionCountMap) {
    connectionCount = connectionCountMap.get(nodeIdStr) || 0;
  } else {
    connectionCount = edges.filter(e =>
      String(e.source) === nodeIdStr || String(e.target) === nodeIdStr
    ).length;
  }

  // --- ONTOLOGY MODE: hierarchy-driven sizing ---
  if (graphMode === 'ontology') {
    const hierarchyNode = hierarchyMap?.get(node.id);
    const depth = hierarchyNode?.depth ?? (node.metadata?.depth ?? 0);
    const instanceCount = parseInt(node.metadata?.instanceCount || '0', 10);
    const depthScale = Math.max(0.4, 1.0 - depth * 0.15);
    const instanceScale = 1 + Math.log(instanceCount + 1) * 0.1;
    return baseSize * depthScale * instanceScale;
  }

  // --- AGENT MODE: workload-driven sizing ---
  if (graphMode === 'agent') {
    const workload = node.metadata?.workload ?? 0;
    const tokenRate = node.metadata?.tokenRate ?? 0;
    const workloadScale = 1 + workload * 0.3 + Math.min(tokenRate / 100, 0.5);
    return baseSize * workloadScale;
  }

  // --- KNOWLEDGE GRAPH MODE (default): larger nodes with authority-driven sizing ---
  const authority = node.metadata?.authority ?? node.metadata?.authorityScore ?? 0;
  const connectionScale = 1 + Math.log(connectionCount + 1) * 0.4;
  const authorityScale = 1 + authority * 0.5;
  const typeScale = getTypeImportance(node.metadata?.type);

  // KG nodes are 2.5x larger than base to give them visual prominence
  return baseSize * connectionScale * authorityScale * typeScale * 2.5;
}

// Get importance multiplier based on node type
const getTypeImportance = (nodeType?: string): number => {
  const importanceMap: Record<string, number> = {
    'folder': 1.5,
    'function': 1.3,
    'class': 1.4,
    'file': 1.0,
    'variable': 0.8,
    'import': 0.7,
    'export': 0.9,
  }

  return importanceMap[nodeType || 'default'] || 1.0
}

interface GraphManagerProps {
  onDragStateChange?: (isDragging: boolean) => void;
}

const GraphManager: React.FC<GraphManagerProps> = ({ onDragStateChange }) => {
  
  // Performance: Removed mount-time logging
  const settings = useSettingsStore((state) => state.settings);
  
  const nodeBloomStrength = settings?.visualisation?.glow?.nodeGlowStrength ?? 0.5;
  const edgeBloomStrength = settings?.visualisation?.glow?.edgeGlowStrength ?? 0.5;
  
  
  const ssspResult = useCurrentSSSPResult();
  const normalizeDistances = useAnalyticsStore(state => state.normalizeDistances);
  const [normalizedSSSPResult, setNormalizedSSSPResult] = useState<any>(null);
  const isXRMode = usePlatformStore((state) => state.isXRMode);
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const materialRef = useRef<HologramNodeMaterial | null>(null)

  
  // Pre-allocated reusable objects to eliminate GC churn
  const tempMatrix = useMemo(() => new THREE.Matrix4(), [])
  const tempPosition = useMemo(() => new THREE.Vector3(), [])
  const tempScale = useMemo(() => new THREE.Vector3(), [])
  const tempQuaternion = useMemo(() => new THREE.Quaternion(), [])
  const tempColor = useMemo(() => new THREE.Color(), [])
  const tempVec3 = useMemo(() => new THREE.Vector3(), [])
  const tempDirection = useMemo(() => new THREE.Vector3(), [])
  const tempSourceOffset = useMemo(() => new THREE.Vector3(), [])
  const tempTargetOffset = useMemo(() => new THREE.Vector3(), [])

  // Frustum for label culling
  const frustum = useMemo(() => new THREE.Frustum(), [])
  const cameraViewProjectionMatrix = useMemo(() => new THREE.Matrix4(), [])

  // LOD state - track current geometry level
  const [currentLODLevel, setCurrentLODLevel] = useState<'high' | 'medium' | 'low'>('high')
  const lodCheckIntervalRef = useRef(0)

  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })

  // O(n) lookup maps to replace O(n^2) findIndex calls (must be after graphData declaration)
  // String() coercion ensures lookups work even if server returns numeric IDs
  const nodeIdToIndexMap = useMemo(() =>
    new Map(graphData.nodes.map((n, i) => [String(n.id), i])),
    [graphData.nodes]
  )

  // Pre-built connection count map: O(E) build once, O(1) lookup per node
  // Replaces O(n*E) filtering in getNodeScale
  const connectionCountMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const edge of graphData.edges) {
      const src = String(edge.source);
      const tgt = String(edge.target);
      map.set(src, (map.get(src) || 0) + 1);
      map.set(tgt, (map.get(tgt) || 0) + 1);
    }
    return map;
  }, [graphData.edges])
  const nodePositionsRef = useRef<Float32Array | null>(null)
  const [edgePoints, setEdgePoints] = useState<number[]>([])
  const prevEdgePointsRef = useRef<number[]>([])
  const prevLabelPositionsLengthRef = useRef<number>(0)
  const labelPositionsRef = useRef<Array<{x: number, y: number, z: number}>>([])
  const edgeUpdatePendingRef = useRef<number[] | null>(null)
  const labelUpdatePendingRef = useRef<Array<{x: number, y: number, z: number}> | null>(null)
  const [nodesAreAtOrigin, setNodesAreAtOrigin] = useState(false)

  const [forceUpdate, setForceUpdate] = useState(0)
  const [labelUpdateTick, setLabelUpdateTick] = useState(0)
  const labelTickRef = useRef(0)

  // CLIENT-SIDE HIERARCHICAL LOD: Detect hierarchy from node IDs
  const hierarchyMap = useMemo(() => {
    if (graphData.nodes.length === 0) return new Map();
    const hierarchy = detectHierarchy(graphData.nodes);
    logger.info(`Detected hierarchy: ${hierarchy.size} nodes, max depth: ${
      Math.max(...Array.from(hierarchy.values()).map(n => n.depth))
    }`);
    return hierarchy;
  }, [graphData.nodes]);

  // CLIENT-SIDE HIERARCHICAL LOD: Expansion state (per-client, no server persistence)
  const expansionState = useExpansionState(true); // Default: all expanded

  // === GRAPH VISUAL MODE DETECTION ===
  // Priority: 1) settings store, 2) auto-detect from node data, 3) default
  const settingsGraphMode = (settings?.visualisation as any)?.graphs?.mode as GraphVisualMode | undefined;
  const graphMode: GraphVisualMode = useMemo(() => {
    if (settingsGraphMode && (settingsGraphMode === 'knowledge_graph' || settingsGraphMode === 'ontology' || settingsGraphMode === 'agent')) {
      return settingsGraphMode;
    }
    return detectGraphMode(graphData.nodes);
  }, [settingsGraphMode, graphData.nodes]);

  // === MODE-AWARE LOD GEOMETRIES (rebuilt only on mode change) ===
  const modeLODGeometries = useMemo(() => {
    logger.info(`Creating LOD geometries for mode: ${graphMode}`);
    return createLODGeometries(graphMode);
  }, [graphMode]);

  // === PER-NODE VISUAL MODE MAP ===
  // Binary protocol flags are ground truth; metadata heuristics are fallback.
  // The store's nodeTypeMap is populated from binary protocol position updates.
  const binaryNodeTypeMap = useWebSocketStore(state => state.nodeTypeMap);

  const perNodeVisualModeMap = useMemo(() => {
    const map = new Map<string, GraphVisualMode>();
    for (const node of graphData.nodes) {
      const nodeIdNum = parseInt(String(node.id), 10);

      // Priority 1: Binary protocol type flags (ground truth)
      if (!isNaN(nodeIdNum) && binaryNodeTypeMap.size > 0) {
        const binaryType = binaryNodeTypeMap.get(nodeIdNum);
        if (binaryType && binaryType !== NodeType.Unknown) {
          map.set(String(node.id), nodeTypeToVisualMode(binaryType));
          continue;
        }
      }

      // Priority 2: Metadata heuristics (fallback)
      const nt = node.metadata?.nodeType || (node as any).nodeType || '';
      const owlIri = (node as any).owlClassIri;
      if (node.metadata?.agentType || node.metadata?.status === 'active' || node.metadata?.status === 'busy') {
        map.set(String(node.id), 'agent');
      } else if (owlIri || nt === 'owl_class' || node.metadata?.hierarchyDepth !== undefined) {
        map.set(String(node.id), 'ontology');
      }
      // If no signals found, don't set -- will fall through to global graphMode
    }
    return map;
  }, [graphData.nodes, binaryNodeTypeMap]);

  // === APPLY MATERIAL MODE PRESET when mode changes ===
  const prevGraphModeRef = useRef<GraphVisualMode>(graphMode);
  useEffect(() => {
    if (materialRef.current && prevGraphModeRef.current !== graphMode) {
      prevGraphModeRef.current = graphMode;
      const preset = MATERIAL_MODE_PRESETS[graphMode];
      logger.info(`Applying material preset for mode: ${graphMode}`, preset);

      materialRef.current.updateHologramParams({
        rimPower: preset.rimPower,
        glowStrength: preset.glowStrength,
        hologramStrength: preset.hologramStrength,
        scanlineCount: preset.scanlineCount,
      });
      materialRef.current.uniforms.pulseSpeed.value = preset.pulseSpeed;
      materialRef.current.uniforms.pulseStrength.value = preset.pulseStrength;
      materialRef.current.needsUpdate = true;
    }
  }, [graphMode]);

  // Get nodeFilter settings from store - extract individual values for stable deps
  const storeNodeFilter = settings?.nodeFilter;
  const filterEnabled = storeNodeFilter?.enabled ?? false;
  const qualityThreshold = storeNodeFilter?.qualityThreshold ?? 0.7;
  const authorityThreshold = storeNodeFilter?.authorityThreshold ?? 0.5;
  const filterByQuality = storeNodeFilter?.filterByQuality ?? true;
  const filterByAuthority = storeNodeFilter?.filterByAuthority ?? false;
  const filterMode = storeNodeFilter?.filterMode ?? 'or';

  // Log filter settings changes for debugging
  useEffect(() => {
    logger.info('[NodeFilter] Settings updated:', {
      enabled: filterEnabled,
      qualityThreshold,
      authorityThreshold,
      filterByQuality,
      filterByAuthority,
      filterMode,
      hasStoreFilter: !!storeNodeFilter
    });
  }, [filterEnabled, qualityThreshold, authorityThreshold, filterByQuality, filterByAuthority, filterMode, storeNodeFilter]);

  // CLIENT-SIDE HIERARCHICAL LOD + QUALITY/AUTHORITY FILTERING
  // Physics still uses ALL graphData.nodes!
  const visibleNodes = useMemo(() => {
    if (graphData.nodes.length === 0) return [];

    logger.debug(`[NodeFilter] Computing visible nodes: filterEnabled=${filterEnabled}, qualityThreshold=${qualityThreshold}, authorityThreshold=${authorityThreshold}`);

    const visible = graphData.nodes.filter(node => {
      // First apply hierarchy/expansion filtering
      const hierarchyNode = hierarchyMap.get(node.id);
      if (hierarchyNode) {
        // Root nodes always pass hierarchy check
        if (!hierarchyNode.isRoot) {
          // Child nodes visible only if parent is expanded
          if (!expansionState.isVisible(node.id, hierarchyNode.parentId)) {
            return false;
          }
        }
      }

      // Then apply quality/authority filtering if enabled
      if (filterEnabled) {
        // Get quality score - use metadata if available, otherwise compute from connections
        let quality = node.metadata?.quality ?? node.metadata?.qualityScore;
        if (quality === undefined || quality === null) {
          // Compute quality from node connections (normalized 0-1) using pre-built map
          const connectionCount = connectionCountMap.get(node.id) || 0;
          // Map connections to 0-1 range: 0 connections = 0, 10+ connections = 1
          quality = Math.min(1.0, connectionCount / 10);
        }

        // Get authority score - use metadata if available, otherwise compute from hierarchy
        let authority = node.metadata?.authority ?? node.metadata?.authorityScore;
        if (authority === undefined || authority === null) {
          // Compute authority from hierarchy depth and connections
          const hierarchyNode = hierarchyMap.get(node.id);
          const depth = hierarchyNode?.depth ?? 0;
          // Root nodes (depth 0) have high authority, deeper nodes have less
          authority = Math.max(0, 1.0 - (depth * 0.2));
        }

        const passesQuality = !filterByQuality || quality >= qualityThreshold;
        const passesAuthority = !filterByAuthority || authority >= authorityThreshold;

        // Apply filter mode (AND requires both, OR requires at least one)
        if (filterMode === 'and') {
          if (!passesQuality || !passesAuthority) {
            return false;
          }
        } else {
          // OR mode - but only if at least one filter is active
          if (filterByQuality || filterByAuthority) {
            if (!passesQuality && !passesAuthority) {
              return false;
            }
          }
        }
      }

      return true;
    });

    // Always log when filtering is active
    if (filterEnabled) {
      logger.info(`[NodeFilter] Result: ${visible.length}/${graphData.nodes.length} nodes visible (quality>=${qualityThreshold}, authority>=${authorityThreshold}, mode=${filterMode})`);
    }

    return visible;
  }, [graphData.nodes, connectionCountMap, hierarchyMap, expansionState, filterEnabled, qualityThreshold, authorityThreshold, filterByQuality, filterByAuthority, filterMode])

  const animationStateRef = useRef({
    time: 0,
    selectedNode: null as string | null,
    hoveredNode: null as string | null,
    pulsePhase: 0,
  })

  
  const [dragState, setDragState] = useState<{
    nodeId: string | null;
    instanceId: number | null;
  }>({ nodeId: null, instanceId: null })

  const dragDataRef = useRef({
    isDragging: false,
    pointerDown: false,
    nodeId: null as string | null,
    instanceId: null as number | null,
    startPointerPos: new THREE.Vector2(),
    startTime: 0,
    startNodePos3D: new THREE.Vector3(),
    currentNodePos3D: new THREE.Vector3(),
    lastUpdateTime: 0,
    pendingUpdate: null as BinaryNodeData | null,
  })

  const { camera, size } = useThree()

  
  const logseqSettings = settings?.visualisation?.graphs?.logseq
  const nodeSettings = logseqSettings?.nodes || settings?.visualisation?.nodes
  const enableMetadataShape = nodeSettings?.enableMetadataShape ?? false

  
  useEffect(() => {
    if (!materialRef.current) {
      materialRef.current = new HologramNodeMaterial({
        baseColor: '#0066ff', 
        emissiveColor: '#00ffff', 
        opacity: settings?.visualisation?.graphs?.logseq?.nodes?.opacity ?? settings?.visualisation?.nodes?.opacity ?? 0.8,
        enableHologram: true, 
        glowStrength: (nodeBloomStrength || 1) * (settings?.visualisation?.glow?.nodeGlowStrength ?? 0.7), 
        pulseSpeed: 1.0,
        hologramStrength: 0.8, 
        rimPower: 2.0, 
      })

      
      ;(materialRef.current as any).toneMapped = false

      
      materialRef.current.defines = { ...materialRef.current.defines, USE_INSTANCING_COLOR: '' }
      materialRef.current.needsUpdate = true
    }
  }, [])
  
  
  useEffect(() => {
    const obj = meshRef.current as any;
    if (obj) {
      
      obj.layers.set(0); 
      obj.layers.enable(1); 
      registerNodeObject(obj);
    }
    return () => {
      if (obj) unregisterNodeObject(obj);
    };
  }, [graphData.nodes.length]) 
  
  
  useEffect(() => {
    if (materialRef.current) {
      
      const strength = nodeBloomStrength || 1;
      materialRef.current.updateHologramParams({
        glowStrength: strength * (settings?.visualisation?.glow?.nodeGlowStrength ?? 0.7) 
      });
      
      materialRef.current.uniforms.glowStrength.value = strength * (settings?.visualisation?.glow?.nodeGlowStrength ?? 0.7);
      materialRef.current.needsUpdate = true;
    }
  }, [nodeBloomStrength]);
  
  
  useEffect(() => {
    if (materialRef.current && settings?.visualisation) {
      
      const logseqSettings = settings.visualisation.graphs?.logseq;
      const nodeSettings = logseqSettings?.nodes || settings.visualisation.nodes;

      materialRef.current.updateColors(
        nodeSettings?.baseColor || '#00ffff',
        nodeSettings?.baseColor || '#00ffff'
      )
      materialRef.current.uniforms.opacity.value = nodeSettings?.opacity ?? 0.8;
      materialRef.current.setHologramEnabled(
        nodeSettings?.enableHologram !== false
      )
      materialRef.current.updateHologramParams({
        glowStrength: settings.visualisation.animations?.pulseStrength || 1.0,
      })
    }
  }, [settings?.visualisation])

  
  useEffect(() => {
    if (ssspResult) {
      const normalized = normalizeDistances(ssspResult);
      setNormalizedSSSPResult({
        ...ssspResult,
        normalizedDistances: normalized
      });
    } else {
      setNormalizedSSSPResult(null);
    }
  }, [ssspResult, normalizeDistances]);

  // Pre-allocate color array and attribute for zero-allocation updates
  const colorArrayRef = useRef<Float32Array | null>(null)
  const colorAttributeRef = useRef<THREE.InstancedBufferAttribute | null>(null)

  // Initialize color array when node count changes
  useEffect(() => {
    if (graphData.nodes.length > 0) {
      const nodeCount = graphData.nodes.length
      colorArrayRef.current = new Float32Array(nodeCount * 3)
      colorAttributeRef.current = new THREE.InstancedBufferAttribute(colorArrayRef.current, 3)
    }
  }, [graphData.nodes.length])


  const updateNodeColors = useCallback(() => {
    if (!meshRef.current || graphData.nodes.length === 0) return;
    if (!colorArrayRef.current || !colorAttributeRef.current) return;

    const mesh = meshRef.current;
    const colors = colorArrayRef.current;

    // Direct Float32Array writes - no allocation, reuse tempColor
    // Per-node visual mode: use binary protocol flags when available, fallback to global graphMode
    graphData.nodes.forEach((node, i) => {
      const nodeVisualMode = perNodeVisualModeMap.get(String(node.id)) || graphMode;
      const color = getNodeColor(node, normalizedSSSPResult, nodeVisualMode, hierarchyMap, connectionCountMap);
      const idx = i * 3;
      colors[idx] = color.r;
      colors[idx + 1] = color.g;
      colors[idx + 2] = color.b;
    });

    // Reuse existing attribute, just mark dirty
    mesh.geometry.setAttribute('instanceColor', colorAttributeRef.current);
    colorAttributeRef.current.needsUpdate = true;
  }, [graphData.nodes, normalizedSSSPResult, graphMode, perNodeVisualModeMap, hierarchyMap, connectionCountMap]);

  
  useEffect(() => {
    updateNodeColors();
  }, [updateNodeColors]);

  
  useEffect(() => {
    if (meshRef.current && graphData.nodes.length > 0) {
      const mesh = meshRef.current
      mesh.count = graphData.nodes.length
      
      
      const debugSettings = settings?.system?.debug;
      if (debugSettings?.enableNodeDebug) {
        logger.debug('Node mesh initialized', {
          nodeCount: graphData.nodes.length,
          meshCount: mesh.count,
          hasPositions: !!nodePositionsRef.current,
          meshRef: meshRef.current,
          hasSSSPResult: !!normalizedSSSPResult
        });
      }

      updateNodeColors();

      
      const tempMatrix = new THREE.Matrix4();
      const nodeSize = settings?.visualisation?.graphs?.logseq?.nodes?.nodeSize || 0.5;
      const BASE_SPHERE_RADIUS = 0.5;
      const baseScale = nodeSize / BASE_SPHERE_RADIUS;

      graphData.nodes.forEach((node, i) => {
        const nodeVisualMode = perNodeVisualModeMap.get(String(node.id)) || graphMode;
        const nodeScale = getNodeScale(node, graphData.edges, connectionCountMap, nodeVisualMode, hierarchyMap) * baseScale;
        tempMatrix.makeScale(nodeScale, nodeScale, nodeScale);


        const angle = (i / graphData.nodes.length) * Math.PI * 2;
        const radius = 10;
        tempMatrix.setPosition(
          Math.cos(angle) * radius,
          (Math.random() - 0.5) * 5,
          Math.sin(angle) * radius
        );

        mesh.setMatrixAt(i, tempMatrix);
      })
      
      
      mesh.instanceMatrix.needsUpdate = true;
      
      
      mesh.computeBoundingSphere();

      
      if (materialRef.current) {
        materialRef.current.needsUpdate = true
      }

    }
  }, [graphData, normalizedSSSPResult])

  
  useEffect(() => {
    graphWorkerProxy.updateSettings(settings);
  }, [settings]);


  useFrame(async (state, delta) => {
    animationStateRef.current.time = state.clock.elapsedTime

    // LOD System: Check every 15 frames (~250ms at 60fps)
    lodCheckIntervalRef.current += 1;
    if (lodCheckIntervalRef.current >= 15 && meshRef.current && nodePositionsRef.current) {
      lodCheckIntervalRef.current = 0;

      // Calculate average distance of visible nodes from camera
      let totalDistance = 0;
      let nodeCount = 0;

      for (let i = 0; i < Math.min(visibleNodes.length, 100); i++) { // Sample up to 100 nodes
        const i3 = i * 3;
        if (i3 + 2 < nodePositionsRef.current.length) {
          tempVec3.set(
            nodePositionsRef.current[i3],
            nodePositionsRef.current[i3 + 1],
            nodePositionsRef.current[i3 + 2]
          );
          totalDistance += tempVec3.distanceTo(camera.position);
          nodeCount++;
        }
      }

      const avgDistance = nodeCount > 0 ? totalDistance / nodeCount : 0;

      // Determine LOD level based on average distance
      let newLODLevel: 'high' | 'medium' | 'low' = 'high';
      if (avgDistance > 40) {
        newLODLevel = 'low';
      } else if (avgDistance > 20) {
        newLODLevel = 'medium';
      }

      // Update geometry if LOD level changed (uses mode-aware geometries)
      if (newLODLevel !== currentLODLevel) {
        setCurrentLODLevel(newLODLevel);
        meshRef.current.geometry = modeLODGeometries[newLODLevel];
      }
    }


    const debugSettings = settings?.system?.debug;
    if (debugSettings?.enablePhysicsDebug && debugState.isEnabled()) {
      const frameCount = Math.floor(state.clock.elapsedTime * 60);
      if (frameCount === 1 || frameCount % 300 === 0) {
        logger.debug('Physics frame update', {
          time: state.clock.elapsedTime,
          delta,
          nodeCount: graphData.nodes.length,
          hasPositions: !!nodePositionsRef.current,
          lodLevel: currentLODLevel
        });
      }
    }


    if (materialRef.current) {
      materialRef.current.updateTime(animationStateRef.current.time)
    }

    // Periodic label frustum refresh (~6 updates/sec at 60fps)
    labelTickRef.current++;
    if (labelTickRef.current >= 10) {
      labelTickRef.current = 0;
      setLabelUpdateTick(prev => prev + 1);
    }


    if ((meshRef.current || enableMetadataShape) && graphData.nodes.length > 0) {
      const positions = await graphWorkerProxy.tick(delta);
      nodePositionsRef.current = positions;

      if (positions && positions.length > 0) {
        // Validate positions array has enough data for all nodes
        const expectedLength = graphData.nodes.length * 3;
        if (positions.length < expectedLength) {
          // Positions array is stale/incomplete - skip this frame to avoid crash
          return;
        }

        const logseqSettings = settings?.visualisation?.graphs?.logseq;
        const nodeSettings = logseqSettings?.nodes || settings?.visualisation?.nodes;
        const nodeSize = nodeSettings?.nodeSize || 0.5;
        const BASE_SPHERE_RADIUS = 0.5;
        const baseScale = nodeSize / BASE_SPHERE_RADIUS;


        if (meshRef.current && !enableMetadataShape) {
          for (let i = 0; i < graphData.nodes.length; i++) {
            const i3 = i * 3;
            // Bounds check for safety
            if (i3 + 2 >= positions.length) break;

            const node = graphData.nodes[i];
            // Per-node visual mode from binary protocol flags (fallback to detected global mode)
            const nodeVisualMode = perNodeVisualModeMap.get(String(node.id)) || graphMode;
            let nodeScale = getNodeScale(node, graphData.edges, connectionCountMap, nodeVisualMode, hierarchyMap) * baseScale;


            if (normalizedSSSPResult && node.id === normalizedSSSPResult.sourceNodeId) {
              const pulseScale = 1 + Math.sin(animationStateRef.current.time * 2) * 0.3;
              nodeScale *= pulseScale;
            }

            tempMatrix.makeScale(nodeScale, nodeScale, nodeScale);
            tempMatrix.setPosition(positions[i3], positions[i3 + 1], positions[i3 + 2]);
            meshRef.current.setMatrixAt(i, tempMatrix);
          }
          meshRef.current.instanceMatrix.needsUpdate = true;

          meshRef.current.computeBoundingSphere();
        }

        // OPTIMIZED: O(n) edge rendering using nodeIdToIndexMap
        const edgeCount = graphData.edges.length;
        const newEdgePoints = new Array<number>(edgeCount * 6);
        let edgePointIdx = 0;

        graphData.edges.forEach(edge => {
          // O(1) lookup instead of O(n) findIndex — String() for type-safe matching
          const sourceNodeIndex = nodeIdToIndexMap.get(String(edge.source));
          const targetNodeIndex = nodeIdToIndexMap.get(String(edge.target));

          if (sourceNodeIndex !== undefined && targetNodeIndex !== undefined) {
            const i3s = sourceNodeIndex * 3;
            const i3t = targetNodeIndex * 3;

            // Bounds check for edge positions
            if (i3s + 2 >= positions.length || i3t + 2 >= positions.length) return;

            // Reuse temp vectors - zero allocation
            tempVec3.set(positions[i3s], positions[i3s + 1], positions[i3s + 2]);
            const sourcePos = tempVec3;
            tempPosition.set(positions[i3t], positions[i3t + 1], positions[i3t + 2]);
            const targetPos = tempPosition;

            // Reuse direction vector
            tempDirection.subVectors(targetPos, sourcePos);
            const edgeLength = tempDirection.length();

            if (edgeLength > 0) {
              tempDirection.normalize();

              const sourceNode = graphData.nodes[sourceNodeIndex];
              const targetNode = graphData.nodes[targetNodeIndex];
              const sourceVisualMode = perNodeVisualModeMap.get(String(sourceNode.id)) || graphMode;
              const targetVisualMode = perNodeVisualModeMap.get(String(targetNode.id)) || graphMode;
              const sourceRadius = getNodeScale(sourceNode, graphData.edges, connectionCountMap, sourceVisualMode, hierarchyMap) * (nodeSettings?.nodeSize || 0.5);
              const targetRadius = getNodeScale(targetNode, graphData.edges, connectionCountMap, targetVisualMode, hierarchyMap) * (nodeSettings?.nodeSize || 0.5);

              // Reuse offset vectors - zero allocation
              tempSourceOffset.copy(sourcePos).addScaledVector(tempDirection, sourceRadius + 0.1);
              tempTargetOffset.copy(targetPos).addScaledVector(tempDirection, -(targetRadius + 0.1));

              if (tempSourceOffset.distanceTo(tempTargetOffset) > 0.2) {
                newEdgePoints[edgePointIdx++] = tempSourceOffset.x;
                newEdgePoints[edgePointIdx++] = tempSourceOffset.y;
                newEdgePoints[edgePointIdx++] = tempSourceOffset.z;
                newEdgePoints[edgePointIdx++] = tempTargetOffset.x;
                newEdgePoints[edgePointIdx++] = tempTargetOffset.y;
                newEdgePoints[edgePointIdx++] = tempTargetOffset.z;
              }
            }
          }
        });
        newEdgePoints.length = edgePointIdx;
        // Compare edge points content, not just length, to detect position changes
        // Use sampling for performance: check length + first/last 6 values (2 edge endpoints)
        const prev = prevEdgePointsRef.current;
        const edgesChanged =
          newEdgePoints.length !== prev.length ||
          (newEdgePoints.length > 0 && (
            newEdgePoints[0] !== prev[0] ||
            newEdgePoints[1] !== prev[1] ||
            newEdgePoints[2] !== prev[2] ||
            newEdgePoints[newEdgePoints.length - 1] !== prev[prev.length - 1] ||
            newEdgePoints[newEdgePoints.length - 2] !== prev[prev.length - 2] ||
            newEdgePoints[newEdgePoints.length - 3] !== prev[prev.length - 3]
          ));

        if (edgesChanged) {
          prevEdgePointsRef.current = newEdgePoints.slice();
          edgeUpdatePendingRef.current = newEdgePoints;
        }


        // Update label positions ref every frame (fast, no re-render)
        // Reuse or grow label positions array to avoid per-frame allocation
        const labelCount = graphData.nodes.length;
        let currentLabelArr = labelPositionsRef.current;
        if (currentLabelArr.length !== labelCount) {
          currentLabelArr = new Array(labelCount);
          for (let i = 0; i < labelCount; i++) {
            currentLabelArr[i] = { x: 0, y: 0, z: 0 };
          }
        }
        for (let i = 0; i < labelCount; i++) {
          const i3 = i * 3;
          currentLabelArr[i].x = positions[i3];
          currentLabelArr[i].y = positions[i3 + 1];
          currentLabelArr[i].z = positions[i3 + 2];
        }
        labelPositionsRef.current = currentLabelArr;

        // Always queue label update -- labelUpdateTick controls re-render frequency
        prevLabelPositionsLengthRef.current = labelCount;
        labelUpdatePendingRef.current = currentLabelArr;
      }
    }

    
    
    
    
    
    
    
    
    
    
    

    
    if (meshRef.current && animationStateRef.current.selectedNode !== null) {
      const mesh = meshRef.current;
      const selectedIndex = nodeIdToIndexMap.get(String(animationStateRef.current.selectedNode));
      if (selectedIndex !== undefined) {
        mesh.getMatrixAt(selectedIndex, tempMatrix);
        tempMatrix.decompose(tempPosition, tempQuaternion, tempScale);
        const pulseFactor = 1 + Math.sin(animationStateRef.current.time * 3) * 0.1;
        tempScale.multiplyScalar(pulseFactor);
        tempMatrix.compose(tempPosition, tempQuaternion, tempScale);
        mesh.setMatrixAt(selectedIndex, tempMatrix);
        mesh.instanceMatrix.needsUpdate = true;
      }
    }

    // Process pending state updates -- after await, React 18 batches automatically
    if (edgeUpdatePendingRef.current) {
      const pendingEdges = edgeUpdatePendingRef.current;
      edgeUpdatePendingRef.current = null;
      setEdgePoints(pendingEdges);
    }
    if (labelUpdatePendingRef.current) {
      const pendingLabels = labelUpdatePendingRef.current;
      labelUpdatePendingRef.current = null;
      setLabelPositions(pendingLabels);
    }
  })


  useEffect(() => {

    const handleGraphUpdate = (data: GraphData): GraphData | undefined => {

      const debugSettings = settings?.system?.debug;
      if (debugSettings?.enableNodeDebug) {
        logger.debug('Graph data updated', {
          nodeCount: data.nodes.length,
          edgeCount: data.edges.length,
          firstNode: data.nodes.length > 0 ? data.nodes[0] : null,
          hasValidData: data && Array.isArray(data.nodes) && Array.isArray(data.edges)
        });
      }

      if (debugState.isEnabled()) {
        logger.info('Graph data updated', {
          nodeCount: data.nodes.length,
          edgeCount: data.edges.length,
          firstNode: data.nodes.length > 0 ? data.nodes[0] : null
        })
      }


      if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.edges)) {
        return undefined;
      }

      
      const dataWithPositions = {
        ...data,
        nodes: data.nodes.map((node, i) => {
          // Normalize node ID to string (server may return numeric IDs)
          const normalizedNode = typeof node.id !== 'string' ? { ...node, id: String(node.id) } : node;
          if (!normalizedNode.position || (normalizedNode.position.x === 0 && normalizedNode.position.y === 0 && normalizedNode.position.z === 0)) {
            const position = getPositionForNode(normalizedNode, i, data.nodes.length)
            return {
              ...normalizedNode,
              position: { x: position[0], y: position[1], z: position[2] }
            }
          }
          return normalizedNode
        }),
        edges: data.edges.map(edge => ({
          ...edge,
          source: String(edge.source),
          target: String(edge.target)
        }))
      }

      const allAtOrigin = dataWithPositions.nodes.every(node =>
        !node.position || (node.position.x === 0 && node.position.y === 0 && node.position.z === 0)
      )
      setNodesAreAtOrigin(allAtOrigin)

      setGraphData(dataWithPositions)

      
      // Use dataWithPositions.nodes (which have generated positions) for initial edge computation
      // String() coercion ensures matching even when server returns numeric node IDs
      const posNodeMap = new Map(dataWithPositions.nodes.map(n => [String(n.id), n]))
      const newEdgePoints: number[] = []
      dataWithPositions.edges.forEach((edge) => {
        const sourceNode = posNodeMap.get(String(edge.source))
        const targetNode = posNodeMap.get(String(edge.target))

        if (sourceNode?.position && targetNode?.position) {
          newEdgePoints.push(
            sourceNode.position.x, sourceNode.position.y, sourceNode.position.z,
            targetNode.position.x, targetNode.position.y, targetNode.position.z
          )
        }
      })

      setEdgePoints(newEdgePoints)

      return dataWithPositions
    }

    const unsubscribe = graphDataManager.onGraphDataChange((data) => {
      // Process data locally only — do NOT send back to graphWorkerProxy.setGraphData()
      // as that triggers notifyGraphDataListeners → this callback → infinite loop.
      // The worker already has the data from graphDataManager.fetchInitialData().
      handleGraphUpdate(data)
    })


    graphDataManager.getGraphData().then((data) => {

      const debugSettings = settings?.system?.debug;
      if (debugSettings?.enableNodeDebug) {
        logger.debug('Initial graph data loaded', {
          nodeCount: data.nodes.length,
          edgeCount: data.edges.length
        });
      }
      handleGraphUpdate(data)
    }).then(() => {
    }).catch((error) => {

      const fallbackData = {
        nodes: [
          { id: 'fallback1', label: 'Test Node 1', position: { x: -5, y: 0, z: 0 } },
          { id: 'fallback2', label: 'Test Node 2', position: { x: 5, y: 0, z: 0 } },
          { id: 'fallback3', label: 'Test Node 3', position: { x: 0, y: 5, z: 0 } }
        ],
        edges: [
          { id: 'fallback_edge1', source: 'fallback1', target: 'fallback2' },
          { id: 'fallback_edge2', source: 'fallback2', target: 'fallback3' }
        ]
      };
      handleGraphUpdate(fallbackData);
    })

    return () => {
      unsubscribe()
    }
  }, [])

  
  const { handlePointerDown, handlePointerMove, handlePointerUp } = useGraphEventHandlers(
    meshRef as any,
    dragDataRef,
    setDragState,
    graphData,
    camera,
    size,
    settings,
    setGraphData,
    onDragStateChange
  )

  // Particle geometry removed - was always null (dead code)

  
  const [labelPositions, setLabelPositions] = useState<Array<{x: number, y: number, z: number}>>([])

  
  useEffect(() => {
    if (nodePositionsRef.current && graphData.nodes.length > 0) {
      const positions = nodePositionsRef.current
      const newPositions = graphData.nodes.map((_node, i) => {
        const i3 = i * 3
        return {
          x: positions[i3],
          y: positions[i3 + 1],
          z: positions[i3 + 2]
        }
      })
      setLabelPositions(newPositions)
    }
  }, [graphData.nodes.length])

  
  // Default edge settings - opacity increased to 0.6 for bloom visibility
  // Bloom threshold is typically 0.15, so edges need opacity > 0.3 to remain visible
  const defaultEdgeSettings: EdgeSettings = {
    arrowSize: 0.5,
    baseWidth: 0.2,
    color: '#FF5722',
    enableArrows: true,
    opacity: 0.6, // Increased from 0.2 to ensure visibility above bloom threshold
    widthRange: [0.1, 0.3],
    quality: 'medium',
    enableFlowEffect: false,
    flowSpeed: 1,
    flowIntensity: 1,
    glowStrength: 1,
    distanceIntensity: 0.5,
    useGradient: false,
    gradientColors: ['#ff0000', '#0000ff'],
  };

  // PERFORMANCE: Pre-compute node ID to index map for O(1) lookups (vs O(n) findIndex)
  const nodeIdToIndex = useMemo(() => {
    const map = new Map<string, number>();
    graphData.nodes.forEach((node, index) => {
      map.set(node.id, index);
    });
    return map;
  }, [graphData.nodes]);

  const NodeLabels = useMemo(() => {
    const logseqSettings = settings?.visualisation?.graphs?.logseq;
    const labelSettings = logseqSettings?.labels ?? settings?.visualisation?.labels;
    if (!labelSettings?.enableLabels || visibleNodes.length === 0) return null;

    cameraViewProjectionMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(cameraViewProjectionMatrix);

    const LABEL_DISTANCE_THRESHOLD = labelSettings?.labelDistanceThreshold ?? 500;
    const METADATA_DISTANCE_THRESHOLD = LABEL_DISTANCE_THRESHOLD * 0.6;
    const vrMode = isXRMode;

    const currentLabelPositions = labelPositionsRef.current;
    return visibleNodes.map((node) => {
      const originalIndex = nodeIdToIndex.get(node.id) ?? -1;
      const physicsPos = originalIndex !== -1 ? currentLabelPositions[originalIndex] : undefined;
      const position = physicsPos || node.position || { x: 0, y: 0, z: 0 };

      tempVec3.set(position.x, position.y, position.z);
      if (!frustum.containsPoint(tempVec3)) return null;

      const distanceToCamera = tempVec3.distanceTo(camera.position);
      if (distanceToCamera > LABEL_DISTANCE_THRESHOLD) return null;

      const showMetadataLines = distanceToCamera <= METADATA_DISTANCE_THRESHOLD;
      const nodeLabelVisualMode = perNodeVisualModeMap.get(String(node.id)) || graphMode;
      const scale = getNodeScale(node, graphData.edges, connectionCountMap, nodeLabelVisualMode, hierarchyMap);
      const textPadding = labelSettings.textPadding ?? 0.6;
      const labelOffsetY = scale * 1.5 + textPadding;

      let distanceInfo: string | null = null;
      if (normalizedSSSPResult && normalizedSSSPResult.distances) {
        const dist = normalizedSSSPResult.distances[node.id];
        if (node.id === normalizedSSSPResult.sourceNodeId) distanceInfo = "Source (0)";
        else if (dist === undefined || !isFinite(dist)) distanceInfo = "Unreachable";
        else distanceInfo = `Distance: ${dist.toFixed(2)}`;
      }

      const maxWidth = labelSettings.maxLabelWidth ?? 8;
      const fontSize = labelSettings.desktopFontSize ?? 0.4;
      const metaFontSize = fontSize * 0.8;
      const outlineW = 0.012;
      const lineSpacing = metaFontSize * 1.3;
      const lineY = -(textPadding * 0.25);
      const labelText = node.label && node.label.length > 40
        ? node.label.substring(0, 37) + '...' : (node.label || node.id);

      // === SSSP OVERRIDE ===
      if (distanceInfo) {
        return (
          <Billboard key={`label-${node.id}`}
            position={[position.x, position.y + labelOffsetY, position.z]}
            follow={true} lockX={false} lockY={false} lockZ={false}>
            <Text fontSize={fontSize} color={labelSettings.textColor || '#ffffff'}
              anchorX="center" anchorY="bottom"
              outlineWidth={labelSettings.textOutlineWidth || 0.02}
              outlineColor={labelSettings.textOutlineColor || '#000000'}
              maxWidth={maxWidth} textAlign="center">{labelText}</Text>
            <Text position={[0, lineY, 0]} fontSize={fontSize * 0.7}
              color={node.id === normalizedSSSPResult?.sourceNodeId ? '#00FFFF' :
                (!isFinite(normalizedSSSPResult?.distances[node.id] || 0) ? '#666666' : '#FFFF00')}
              anchorX="center" anchorY="top" maxWidth={maxWidth * 0.8} textAlign="center"
              outlineWidth={0.002} outlineColor="#000000">{distanceInfo}</Text>
          </Billboard>
        );
      }

      // === KNOWLEDGE GRAPH MODE ===
      if (nodeLabelVisualMode === 'knowledge_graph') {
        const sourceDomain = node.metadata?.source_domain ?? '';
        const domainColor = getDomainColor(sourceDomain);
        const qualityStars = getQualityStars(node.metadata?.quality ?? node.metadata?.quality_score);
        const connectionCount = connectionCountMap.get(node.id) ?? 0;
        const recencyField = node.metadata?.lastModified ?? node.metadata?.last_modified ?? node.metadata?.updated_at;
        const recencyText = getRecencyText(recencyField);
        const recencyColor = getRecencyColor(recencyField);

        const line2Parts: string[] = [];
        if (sourceDomain) line2Parts.push(`\u25CF ${sourceDomain}`);
        if (qualityStars) line2Parts.push(qualityStars);
        const line2 = line2Parts.join('  ');
        const line3 = `\u27E8${connectionCount} link${connectionCount !== 1 ? 's' : ''}\u27E9`;

        return (
          <Billboard key={`label-${node.id}`}
            position={[position.x, position.y + labelOffsetY, position.z]}
            follow={true} lockX={false} lockY={false} lockZ={false}>
            <Text fontSize={fontSize}
              color={sourceDomain ? domainColor : (labelSettings.textColor || '#ffffff')}
              anchorX="center" anchorY="bottom"
              outlineWidth={labelSettings.textOutlineWidth || 0.02}
              outlineColor={labelSettings.textOutlineColor || '#000000'}
              maxWidth={maxWidth} textAlign="center">{labelText}</Text>
            {showMetadataLines && line2 && (
              <Text position={[0, lineY, 0]} fontSize={metaFontSize}
                color={sourceDomain ? domainColor : '#B0BEC5'}
                anchorX="center" anchorY="top" maxWidth={maxWidth} textAlign="center"
                outlineWidth={outlineW} outlineColor="#000000">{line2}</Text>
            )}
            {showMetadataLines && !vrMode && (
              <Text position={[0, lineY - lineSpacing, 0]} fontSize={metaFontSize * 0.9}
                color="#B0BEC5" anchorX="center" anchorY="top" maxWidth={maxWidth} textAlign="center"
                outlineWidth={outlineW} outlineColor="#000000">{line3}</Text>
            )}
            {showMetadataLines && !vrMode && recencyText && (
              <Text position={[0, lineY - lineSpacing * 2, 0]} fontSize={metaFontSize * 0.85}
                color={recencyColor} anchorX="center" anchorY="top" maxWidth={maxWidth} textAlign="center"
                outlineWidth={outlineW} outlineColor="#000000">{recencyText}</Text>
            )}
          </Billboard>
        );
      }

      // === ONTOLOGY MODE ===
      if (nodeLabelVisualMode === 'ontology') {
        const depth = node.metadata?.hierarchyDepth ?? node.metadata?.depth ?? 0;
        const instanceCount = node.metadata?.instanceCount ?? 0;
        const category = getOntologyCategory(node);
        const categoryDisplay = ONTOLOGY_CATEGORY_DISPLAY[category];
        const depthColor = getOntologyDepthHex(depth);
        const violations = node.metadata?.violations ?? 0;
        const line2 = `\u21B3 Depth ${depth} \u00B7 ${instanceCount} instance${instanceCount !== 1 ? 's' : ''}`;
        const constraintLine = violations > 0
          ? `\u26A0 ${violations} violation${violations !== 1 ? 's' : ''}`
          : (node.metadata?.constraintValid !== undefined ? '\u2713 Valid' : '');
        const constraintColor = violations > 0 ? '#F39C12' : '#2ECC71';

        return (
          <Billboard key={`label-${node.id}`}
            position={[position.x, position.y + labelOffsetY, position.z]}
            follow={true} lockX={false} lockY={false} lockZ={false}>
            <Text fontSize={fontSize} color={depthColor}
              anchorX="center" anchorY="bottom"
              outlineWidth={labelSettings.textOutlineWidth || 0.02}
              outlineColor={labelSettings.textOutlineColor || '#000000'}
              maxWidth={maxWidth} textAlign="center">{labelText}</Text>
            {showMetadataLines && (
              <Text position={[0, lineY, 0]} fontSize={metaFontSize} color={depthColor}
                anchorX="center" anchorY="top" maxWidth={maxWidth} textAlign="center"
                outlineWidth={outlineW} outlineColor="#000000">{line2}</Text>
            )}
            {showMetadataLines && !vrMode && (
              <Text position={[0, lineY - lineSpacing, 0]} fontSize={metaFontSize * 0.9}
                color="#B0BEC5" anchorX="center" anchorY="top" maxWidth={maxWidth} textAlign="center"
                outlineWidth={outlineW} outlineColor="#000000">{categoryDisplay}</Text>
            )}
            {showMetadataLines && !vrMode && constraintLine && (
              <Text position={[0, lineY - lineSpacing * 2, 0]} fontSize={metaFontSize * 0.85}
                color={constraintColor} anchorX="center" anchorY="top" maxWidth={maxWidth} textAlign="center"
                outlineWidth={outlineW} outlineColor="#000000">{constraintLine}</Text>
            )}
          </Billboard>
        );
      }

      // === AGENT MODE ===
      if (nodeLabelVisualMode === 'agent') {
        const agentType = (node.metadata?.agentType ?? node.metadata?.type ?? 'unknown').toUpperCase();
        const status = node.metadata?.status ?? 'idle';
        const statusColor = getAgentStatusHex(status);
        const health = node.metadata?.health ?? 100;
        const tokenRate = node.metadata?.tokenRate ?? 0;
        const activeTasks = node.metadata?.tasksActive ?? node.metadata?.tasks ?? 0;
        const statusLabel = status.charAt(0).toUpperCase() + status.slice(1);
        const line2 = `\u25CF ${statusLabel}  \u2665 ${health}%`;
        const line3 = `\u26A1 ${tokenRate} tok/min \u00B7 ${activeTasks} task${activeTasks !== 1 ? 's' : ''}`;

        return (
          <Billboard key={`label-${node.id}`}
            position={[position.x, position.y + labelOffsetY, position.z]}
            follow={true} lockX={false} lockY={false} lockZ={false}>
            <Text fontSize={fontSize} color={statusColor}
              anchorX="center" anchorY="bottom"
              outlineWidth={labelSettings.textOutlineWidth || 0.02}
              outlineColor={labelSettings.textOutlineColor || '#000000'}
              maxWidth={maxWidth} textAlign="center">{agentType}</Text>
            {showMetadataLines && (
              <Text position={[0, lineY, 0]} fontSize={metaFontSize} color={statusColor}
                anchorX="center" anchorY="top" maxWidth={maxWidth} textAlign="center"
                outlineWidth={outlineW} outlineColor="#000000">{line2}</Text>
            )}
            {showMetadataLines && !vrMode && (
              <Text position={[0, lineY - lineSpacing, 0]} fontSize={metaFontSize * 0.9}
                color="#B0BEC5" anchorX="center" anchorY="top" maxWidth={maxWidth} textAlign="center"
                outlineWidth={outlineW} outlineColor="#000000">{line3}</Text>
            )}
          </Billboard>
        );
      }

      // Fallback
      return (
        <Billboard key={`label-${node.id}`}
          position={[position.x, position.y + labelOffsetY, position.z]}
          follow={true} lockX={false} lockY={false} lockZ={false}>
          <Text fontSize={fontSize} color={labelSettings.textColor || '#ffffff'}
            anchorX="center" anchorY="bottom"
            outlineWidth={labelSettings.textOutlineWidth || 0.02}
            outlineColor={labelSettings.textOutlineColor || '#000000'}
            maxWidth={maxWidth} textAlign="center">{labelText}</Text>
        </Billboard>
      );
    }).filter(Boolean)
  }, [visibleNodes, graphData.edges, connectionCountMap, labelUpdateTick, nodeIdToIndex, settings?.visualisation?.graphs?.logseq?.labels, settings?.visualisation?.labels, normalizedSSSPResult, graphMode, perNodeVisualModeMap, hierarchyMap, isXRMode])

  
  useEffect(() => {
    
    const debugSettings = settings?.system?.debug;
    if (debugSettings?.enableNodeDebug) {
      logger.debug('Component mounted', {
        nodeCount: graphData.nodes.length,
        edgeCount: graphData.edges.length,
        edgePointsLength: edgePoints.length,
        enableMetadataShape,
        meshRefCurrent: !!meshRef.current,
        materialRefCurrent: !!materialRef.current
      });
    }

    return () => {
      if (debugSettings?.enableNodeDebug) {
        logger.debug('Component unmounting');
      }
    };
  }, []); 

  return (
    <>
      {}
      <NodeShaderToggle materialRef={materialRef} />
      
      
      {}
      {enableMetadataShape ? (
  <MetadataShapes
    nodes={visibleNodes}
    nodePositions={nodePositionsRef.current}
    onNodeClick={(nodeId, event) => {
      const nodeIndex = visibleNodes.findIndex(n => n.id === nodeId);
      if (nodeIndex !== -1) {
        handlePointerDown({ ...event, instanceId: nodeIndex } as any);
      }
    }}
    onNodeDoubleClick={(nodeId, node, _event) => {
      // Priority 1: Check for explicit page URL in metadata
      const pageUrl = node.metadata?.page_url || node.metadata?.pageUrl || node.metadata?.url;
      if (pageUrl) {
        logger.info(`Navigating to page URL for "${node.label}": ${pageUrl}`);
        window.open(pageUrl, '_blank', 'noopener,noreferrer');
        return;
      }

      // Priority 2: Check for file path in metadata (construct local URL)
      const filePath = node.metadata?.file_path || node.metadata?.filePath || node.metadata?.path;
      if (filePath) {
        const encodedPath = encodeURIComponent(filePath);
        const url = `/#/page/${encodedPath}`;
        logger.info(`Navigating to file path for "${node.label}": ${url}`);
        window.location.href = url;
        return;
      }

      // Priority 3: Construct URL from node label (fallback)
      if (node.label) {
        const encodedLabel = encodeURIComponent(node.label);
        const url = `/#/page/${encodedLabel}`;
        logger.info(`Navigating to page for "${node.label}": ${url}`);
        window.location.href = url;
        return;
      }

      // Fallback: Toggle hierarchical expansion if node has children
      const hierarchyNode = hierarchyMap.get(node.id);
      if (hierarchyNode && hierarchyNode.childIds.length > 0) {
        expansionState.toggleExpansion(node.id);
        logger.info(`Toggled expansion for "${node.label}" (${node.id}): ${
          expansionState.isExpanded(node.id) ? 'expanded' : 'collapsed'
        }, ${hierarchyNode.childIds.length} children`);
      } else {
        logger.info(`Node "${node.label}" has no children to expand and no URL to navigate to`);
      }
    }}
    settings={settings}
    ssspResult={normalizedSSSPResult}
    graphMode={graphMode}
    hierarchyMap={hierarchyMap}
  />
) : (
        <instancedMesh
          ref={meshRef}
          args={[undefined, undefined, visibleNodes.length]}
          frustumCulled={false}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={(event) => handlePointerUp(event)}
          onPointerLeave={(event) => {
            // If dragging and pointer leaves the mesh, keep the drag active
            // The pointer capture will ensure we still get events
            // Only end if we lost the capture
          }}
          onPointerCancel={(event) => {
            // Pointer was cancelled (e.g., touch interrupted)
            if (dragDataRef.current.pointerDown) {
              handlePointerUp(event);
            }
          }}
          onPointerMissed={() => {
            // Clicked outside the mesh (on canvas background)
            if (dragDataRef.current.pointerDown) {
              handlePointerUp();
            }
          }}
          onDoubleClick={(event: ThreeEvent<MouseEvent>) => {
            if (event.instanceId !== undefined && event.instanceId < visibleNodes.length) {
              const node = visibleNodes[event.instanceId];
              if (node) {
                // Priority 1: Check for explicit page URL in metadata
                const pageUrl = node.metadata?.page_url || node.metadata?.pageUrl || node.metadata?.url;
                if (pageUrl) {
                  logger.info(`Navigating to page URL for "${node.label}": ${pageUrl}`);
                  window.open(pageUrl, '_blank', 'noopener,noreferrer');
                  return;
                }

                // Priority 2: Check for file path in metadata (construct local URL)
                const filePath = node.metadata?.file_path || node.metadata?.filePath || node.metadata?.path;
                if (filePath) {
                  // For file paths, open in the page viewer
                  const encodedPath = encodeURIComponent(filePath);
                  const url = `/#/page/${encodedPath}`;
                  logger.info(`Navigating to file path for "${node.label}": ${url}`);
                  window.location.href = url;
                  return;
                }

                // Priority 3: Construct URL from node label (fallback)
                if (node.label) {
                  const encodedLabel = encodeURIComponent(node.label);
                  const url = `/#/page/${encodedLabel}`;
                  logger.info(`Navigating to page for "${node.label}": ${url}`);
                  window.location.href = url;
                  return;
                }

                // Fallback: Toggle hierarchical expansion if node has children
                const hierarchyNode = hierarchyMap.get(node.id);
                if (hierarchyNode && hierarchyNode.childIds.length > 0) {
                  expansionState.toggleExpansion(node.id);
                  logger.info(`Toggled expansion for "${node.label}" (${node.id}): ${
                    expansionState.isExpanded(node.id) ? 'expanded' : 'collapsed'
                  }, ${hierarchyNode.childIds.length} children`);
                } else {
                  logger.info(`Node "${node.label}" has no children to expand and no URL to navigate to`);
                }
              }
            }
          }}
        >
          {/* LOD-aware geometry: mode-specific, dynamically switches on distance */}
          <primitive object={modeLODGeometries[currentLODLevel]} attach="geometry" />
          {materialRef.current ? (
            <primitive object={materialRef.current} attach="material" />
          ) : (
            <meshBasicMaterial color="#00ffff" />
          )}
        </instancedMesh>
      )}

      {}
      {edgePoints.length > 0 && (
        <FlowingEdges
          points={edgePoints}
          settings={(settings?.visualisation?.graphs?.logseq?.edges || settings?.visualisation?.edges || defaultEdgeSettings) as any}
          colorOverride={
            graphMode === 'knowledge_graph'
              ? (settings?.visualisation?.graphTypeVisuals?.knowledgeGraph as any)?.edgeColor
              : graphMode === 'ontology'
              ? (settings?.visualisation?.graphTypeVisuals?.ontology as any)?.edgeColor
              : undefined
          }
          edgeData={graphData.edges}
        />
      )}

      {/* Knowledge graph rotating rings */}
      <KnowledgeRings
        nodes={graphData.nodes}
        graphMode={graphMode}
        perNodeVisualModeMap={perNodeVisualModeMap}
        nodePositionsRef={nodePositionsRef}
        nodeIdToIndexMap={nodeIdToIndexMap}
        connectionCountMap={connectionCountMap}
        edges={graphData.edges}
        hierarchyMap={hierarchyMap}
        settings={settings}
      />

      {/* Cluster hull visualization */}
      <ClusterHulls
        nodes={graphData.nodes}
        nodePositionsRef={nodePositionsRef}
        nodeIdToIndexMap={nodeIdToIndexMap}
        settings={settings}
      />

      {}
      {NodeLabels}
    </>
  )
}

export default GraphManager