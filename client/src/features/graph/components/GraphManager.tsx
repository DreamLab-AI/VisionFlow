import React, { useRef, useEffect, useState, useMemo } from 'react'
import { useThree, useFrame, ThreeEvent } from '@react-three/fiber'
import { Text, Billboard } from '@react-three/drei'
import * as THREE from 'three'
import { graphDataManager, type GraphData, type Node as GraphNode } from '../managers/graphDataManager'
import { graphWorkerProxy } from '../managers/graphWorkerProxy'
import { usePlatformStore } from '../../../services/platformManager'
import { createLogger } from '../../../utils/loggerConfig'
import { debugState } from '../../../utils/clientDebugState'
import { useSettingsStore } from '../../../store/settingsStore'
import { BinaryNodeData, NodeType } from '../../../types/binaryProtocol'
import { useWebSocketStore } from '../../../store/websocketStore'
import { GemNodes, GemNodesHandle } from './GemNodes'
import { GlassEdges, GlassEdgesHandle } from './GlassEdges'
import { KnowledgeRings } from './KnowledgeRings'
import { ClusterHulls } from './ClusterHulls'
import { useGraphEventHandlers } from '../hooks/useGraphEventHandlers'
import { EdgeSettings } from '../../settings/config/settings'
import { useAnalyticsStore, useCurrentSSSPResult } from '../../analytics/store/analyticsStore'
import { detectHierarchy } from '../utils/hierarchyDetector'
import { useExpansionState } from '../hooks/useExpansionState'
import { AgentNodesLayer, useAgentNodes } from '../../visualisation/components/AgentNodesLayer'
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

// O(1) domain color lookup
const getDomainColor = (domain?: string): string => {
  return domain && DOMAIN_COLORS[domain] ? DOMAIN_COLORS[domain] : DEFAULT_DOMAIN_COLOR;
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

// (Material mode presets removed -- GemNodes handles mode switching internally)

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

// (LOD geometry sets removed -- GemNodes manages its own geometry)

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
  
  
  
  const ssspResult = useCurrentSSSPResult();
  const normalizeDistances = useAnalyticsStore(state => state.normalizeDistances);
  const [normalizedSSSPResult, setNormalizedSSSPResult] = useState<any>(null);
  const isXRMode = usePlatformStore((state) => state.isXRMode);
  const gemNodesRef = useRef<GemNodesHandle>(null)

  
  // Pre-allocated reusable objects to eliminate GC churn
  const tempPosition = useMemo(() => new THREE.Vector3(), [])
  const tempVec3 = useMemo(() => new THREE.Vector3(), [])
  const tempDirection = useMemo(() => new THREE.Vector3(), [])
  const tempSourceOffset = useMemo(() => new THREE.Vector3(), [])
  const tempTargetOffset = useMemo(() => new THREE.Vector3(), [])

  // Frustum for label culling
  const frustum = useMemo(() => new THREE.Frustum(), [])
  const cameraViewProjectionMatrix = useMemo(() => new THREE.Matrix4(), [])

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
  const [highlightEdgePoints, setHighlightEdgePoints] = useState<number[]>([]);
  const edgeFlowRef = useRef<GlassEdgesHandle>(null);
  const highlightEdgeFlowRef = useRef<GlassEdgesHandle>(null);
  const prevLabelPositionsLengthRef = useRef<number>(0)
  const labelPositionsRef = useRef<Array<{x: number, y: number, z: number}>>([])
  const edgeUpdatePendingRef = useRef<number[] | null>(null)
  const highlightEdgeUpdatePendingRef = useRef<number[] | null>(null);
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

  // Agent nodes overlay: polls /api/bots/agents for live agent telemetry
  const { agents: agentLayerNodes, connections: agentLayerConnections } = useAgentNodes();

  // === GRAPH VISUAL MODE DETECTION ===
  // Priority: 1) settings store, 2) auto-detect from node data, 3) default
  const settingsGraphMode = (settings?.visualisation as any)?.graphs?.mode as GraphVisualMode | undefined;
  const graphMode: GraphVisualMode = useMemo(() => {
    if (settingsGraphMode && (settingsGraphMode === 'knowledge_graph' || settingsGraphMode === 'ontology' || settingsGraphMode === 'agent')) {
      return settingsGraphMode;
    }
    return detectGraphMode(graphData.nodes);
  }, [settingsGraphMode, graphData.nodes]);

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

  // (Color arrays, updateNodeColors, and mesh init removed -- GemNodes handles all node rendering)

  
  // Only forward settings to the worker when physics parameters actually change.
  // Non-physics settings (edge opacity, glow, hologram, etc.) are irrelevant to the worker
  // and sending them would cause unnecessary physics parameter resets that disrupt layout.
  const physicsFingerprint = useMemo(() => JSON.stringify({
    vf: settings?.visualisation?.graphs?.visionflow?.physics,
    lq: settings?.visualisation?.graphs?.logseq?.physics,
  }), [settings?.visualisation?.graphs?.visionflow?.physics, settings?.visualisation?.graphs?.logseq?.physics]);

  useEffect(() => {
    graphWorkerProxy.updateSettings(settings);
  }, [physicsFingerprint]);


  useFrame((state, delta) => {
    animationStateRef.current.time = state.clock.elapsedTime

    // Periodic label frustum refresh (~6 updates/sec at 60fps)
    labelTickRef.current++;
    if (labelTickRef.current >= 10) {
      labelTickRef.current = 0;
      cameraViewProjectionMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      frustum.setFromProjectionMatrix(cameraViewProjectionMatrix);
      setLabelUpdateTick(prev => prev + 1);
    }

    // Position reading from SharedArrayBuffer (GemNodes reads from nodePositionsRef)
    if (graphData.nodes.length > 0) {
      graphWorkerProxy.requestTick(delta);
      const positions = graphWorkerProxy.getPositionsSync();
      if (!positions) return;
      nodePositionsRef.current = positions;

      const positionsValid = positions && positions.length > 0 && positions.length >= graphData.nodes.length * 3;
      if (positions && positions.length > 0 && !positionsValid) {
        logger.warn(`Positions array too short: ${positions.length} < ${graphData.nodes.length * 3} (${graphData.nodes.length} nodes). Skipping position-dependent rendering this frame.`);
      }

      if (positionsValid) {
        // Edge point computation (GlassEdges needs edgePoints)
        const edgeCount = graphData.edges.length;
        const newEdgePoints = new Array<number>(edgeCount * 6);
        let edgePointIdx = 0;

        graphData.edges.forEach(edge => {
          const sourceNodeIndex = nodeIdToIndexMap.get(String(edge.source));
          const targetNodeIndex = nodeIdToIndexMap.get(String(edge.target));

          if (sourceNodeIndex !== undefined && targetNodeIndex !== undefined) {
            const i3s = sourceNodeIndex * 3;
            const i3t = targetNodeIndex * 3;

            if (i3s + 2 >= positions.length || i3t + 2 >= positions.length) return;

            tempVec3.set(positions[i3s], positions[i3s + 1], positions[i3s + 2]);
            const sourcePos = tempVec3;
            tempPosition.set(positions[i3t], positions[i3t + 1], positions[i3t + 2]);
            const targetPos = tempPosition;

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

        // Compute highlighted edges for the selected node
        if (selectedNodeId) {
          const highlightPoints: number[] = [];
          graphData.edges.forEach((edge: any) => {
            const sourceStr = String(edge.source);
            const targetStr = String(edge.target);
            if (sourceStr !== selectedNodeId && targetStr !== selectedNodeId) return;

            const sourceIdx = nodeIdToIndexMap.get(sourceStr);
            const targetIdx = nodeIdToIndexMap.get(targetStr);
            if (sourceIdx === undefined || targetIdx === undefined) return;

            const si3 = sourceIdx * 3;
            const ti3 = targetIdx * 3;
            if (si3 + 2 >= positions.length || ti3 + 2 >= positions.length) return;

            tempVec3.set(positions[si3], positions[si3 + 1], positions[si3 + 2]);
            tempPosition.set(positions[ti3], positions[ti3 + 1], positions[ti3 + 2]);
            tempDirection.subVectors(tempPosition, tempVec3);
            const len = tempDirection.length();
            if (len > 0) {
              tempDirection.normalize();
              const srcNode = graphData.nodes[sourceIdx];
              const tgtNode = graphData.nodes[targetIdx];
              const srcMode = perNodeVisualModeMap.get(String(srcNode.id)) || graphMode;
              const tgtMode = perNodeVisualModeMap.get(String(tgtNode.id)) || graphMode;
              const srcR = getNodeScale(srcNode, graphData.edges, connectionCountMap, srcMode, hierarchyMap) * (nodeSettings?.nodeSize || 0.5);
              const tgtR = getNodeScale(tgtNode, graphData.edges, connectionCountMap, tgtMode, hierarchyMap) * (nodeSettings?.nodeSize || 0.5);
              tempSourceOffset.copy(tempVec3).addScaledVector(tempDirection, srcR + 0.1);
              tempTargetOffset.copy(tempPosition).addScaledVector(tempDirection, -(tgtR + 0.1));
              if (tempSourceOffset.distanceTo(tempTargetOffset) > 0.2) {
                highlightPoints.push(
                  tempSourceOffset.x, tempSourceOffset.y, tempSourceOffset.z,
                  tempTargetOffset.x, tempTargetOffset.y, tempTargetOffset.z
                );
              }
            }
          });
          if (highlightEdgeFlowRef.current) {
            highlightEdgeFlowRef.current.updatePoints(highlightPoints);
          } else {
            highlightEdgeUpdatePendingRef.current = highlightPoints;
          }
        } else if (highlightEdgePoints.length > 0) {
          if (highlightEdgeFlowRef.current) {
            highlightEdgeFlowRef.current.updatePoints([]);
          } else {
            highlightEdgeUpdatePendingRef.current = [];
          }
        }

        // Imperative edge update: push directly to GlassEdges geometry buffer
        if (edgeFlowRef.current) {
          edgeFlowRef.current.updatePoints(newEdgePoints);
        } else {
          edgeUpdatePendingRef.current = newEdgePoints;
        }

        // Update label positions ref every frame (fast, no re-render)
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

        prevLabelPositionsLengthRef.current = labelCount;
        labelUpdatePendingRef.current = currentLabelArr;
      }
    }

    // Process pending state updates -- only for initial mount before imperative handles are available
    if (edgeUpdatePendingRef.current && !edgeFlowRef.current) {
      const pendingEdges = edgeUpdatePendingRef.current;
      edgeUpdatePendingRef.current = null;
      setEdgePoints(pendingEdges);
    }
    if (labelUpdatePendingRef.current) {
      const pendingLabels = labelUpdatePendingRef.current;
      labelUpdatePendingRef.current = null;
      setLabelPositions(pendingLabels);
    }
    if (highlightEdgeUpdatePendingRef.current !== null && !highlightEdgeFlowRef.current) {
      const pendingHighlight = highlightEdgeUpdatePendingRef.current;
      highlightEdgeUpdatePendingRef.current = null;
      setHighlightEdgePoints(pendingHighlight);
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

  
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Proxy ref: useGraphEventHandlers expects RefObject<InstancedMesh>.
  // GemNodes manages its own mesh internally, so we bridge via a getter-backed ref.
  const meshProxyRef = useMemo(() => ({
    get current() { return gemNodesRef.current?.getMesh() ?? null; },
    set current(_v) { /* no-op: GemNodes owns the mesh */ },
  }), []) as React.RefObject<THREE.InstancedMesh>;

  const { handlePointerDown, handlePointerMove, handlePointerUp } = useGraphEventHandlers(
    meshProxyRef,
    dragDataRef,
    setDragState,
    graphData,
    camera,
    size,
    settings,
    setGraphData,
    onDragStateChange,
    setSelectedNodeId
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

  // nodeIdToIndex removed -- use nodeIdToIndexMap (line ~517) which computes the same Map

  const NodeLabels = useMemo(() => {
    const logseqSettings = settings?.visualisation?.graphs?.logseq;
    const labelSettings = logseqSettings?.labels ?? settings?.visualisation?.labels;
    if (!labelSettings?.enableLabels || visibleNodes.length === 0) return null;

    // Frustum is updated in useFrame (labelTickRef gate) -- read the already-computed state here

    const LABEL_DISTANCE_THRESHOLD = labelSettings?.labelDistanceThreshold ?? 500;
    const METADATA_DISTANCE_THRESHOLD = LABEL_DISTANCE_THRESHOLD * 0.6;
    const vrMode = isXRMode;

    const currentLabelPositions = labelPositionsRef.current;
    return visibleNodes.map((node) => {
      const originalIndex = nodeIdToIndexMap.get(node.id) ?? -1;
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
  }, [visibleNodes, graphData.edges, connectionCountMap, labelUpdateTick, nodeIdToIndexMap, settings?.visualisation?.graphs?.logseq?.labels, settings?.visualisation?.labels, normalizedSSSPResult, graphMode, perNodeVisualModeMap, hierarchyMap, isXRMode])

  
  useEffect(() => {
    const debugSettings = settings?.system?.debug;
    if (debugSettings?.enableNodeDebug) {
      logger.debug('Component mounted', {
        nodeCount: graphData.nodes.length,
        edgeCount: graphData.edges.length,
        edgePointsLength: edgePoints.length,
        gemNodesRef: !!gemNodesRef.current,
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
      {/* Gem node rendering (replaces old instancedMesh + MetadataShapes) */}
      <GemNodes
        ref={gemNodesRef}
        nodes={visibleNodes}
        edges={graphData.edges}
        graphMode={graphMode}
        perNodeVisualModeMap={perNodeVisualModeMap}
        nodePositionsRef={nodePositionsRef}
        connectionCountMap={connectionCountMap}
        hierarchyMap={hierarchyMap}
        nodeIdToIndexMap={nodeIdToIndexMap}
        settings={settings}
        ssspResult={normalizedSSSPResult}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={(event: any) => handlePointerUp(event)}
        onPointerMissed={() => {
          if (dragDataRef.current.pointerDown) {
            handlePointerUp();
          }
          setSelectedNodeId(null);
        }}
        onDoubleClick={(event: ThreeEvent<MouseEvent>) => {
          if (event.instanceId !== undefined && event.instanceId < visibleNodes.length) {
            const node = visibleNodes[event.instanceId];
            if (node) {
              const pageUrl = node.metadata?.page_url || node.metadata?.pageUrl || node.metadata?.url;
              if (pageUrl) {
                window.open(pageUrl, '_blank', 'noopener,noreferrer');
                return;
              }
              const filePath = node.metadata?.file_path || node.metadata?.filePath || node.metadata?.path;
              if (filePath) {
                window.open(`https://narrativegoldmine.com/#/page/${encodeURIComponent(filePath)}`, '_blank', 'noopener,noreferrer');
                return;
              }
              if (node.label) {
                window.open(`https://narrativegoldmine.com/#/page/${encodeURIComponent(node.label)}`, '_blank', 'noopener,noreferrer');
                return;
              }
              const hierarchyNode = hierarchyMap.get(node.id);
              if (hierarchyNode && hierarchyNode.childIds.length > 0) {
                expansionState.toggleExpansion(node.id);
              }
            }
          }
        }}
        selectedNodeId={selectedNodeId}
      />

      {/* Glass edge rendering */}
      {edgePoints.length > 0 && (
        <GlassEdges
          ref={edgeFlowRef}
          points={edgePoints}
          settings={settings?.visualisation?.graphs?.logseq?.edges || settings?.visualisation?.edges || defaultEdgeSettings}
          colorOverride={
            graphMode === 'knowledge_graph'
              ? (settings?.visualisation?.graphTypeVisuals?.knowledgeGraph as any)?.edgeColor
              : graphMode === 'ontology'
              ? (settings?.visualisation?.graphTypeVisuals?.ontology as any)?.edgeColor
              : undefined
          }
        />
      )}

      {/* Highlighted edges for selected node */}
      {highlightEdgePoints.length > 0 && (
        <GlassEdges
          ref={highlightEdgeFlowRef}
          points={highlightEdgePoints}
          settings={settings?.visualisation?.graphs?.logseq?.edges || settings?.visualisation?.edges || defaultEdgeSettings}
          colorOverride={settings?.visualisation?.interaction?.selectionHighlightColor || '#00FFFF'}
        />
      )}

      {/* Knowledge graph rotating rings */}
      <KnowledgeRings
        nodes={graphData.nodes}
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

      {/* Agent nodes overlay: bioluminescent agent visualization from Management API */}
      {agentLayerNodes.length > 0 && (
        <AgentNodesLayer agents={agentLayerNodes} connections={agentLayerConnections} />
      )}

      {/* Node labels */}
      {NodeLabels}
    </>
  )
}

export default GraphManager