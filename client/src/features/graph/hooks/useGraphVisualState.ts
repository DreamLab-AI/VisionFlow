/**
 * useGraphVisualState
 *
 * Extracts visual-mode classification, hierarchy detection, and
 * connection-count computation from GraphManager into a dedicated hook.
 *
 * Provides:
 *  - perNodeVisualModeMap: per-node visual mode assignment (binary protocol first, metadata fallback)
 *  - hierarchyMap: ontology depth hierarchy derived from node IDs
 *  - connectionCountMap: O(1) edge-connection lookup per node
 *  - dominantMode: the single GraphVisualMode governing the current graph population
 */

import { useMemo } from 'react';
import { useSettingsStore } from '../../../store/settingsStore';
import { useWebSocketStore } from '../../../store/websocketStore';
import { NodeType } from '../../../types/binaryProtocol';
import { detectHierarchy, type HierarchyNode } from '../utils/hierarchyDetector';
import { createLogger } from '../../../utils/loggerConfig';
import type { GraphData, Node as GraphNode } from '../managers/graphDataManager';

const logger = createLogger('useGraphVisualState');

// === GRAPH VISUAL MODE ===
export type GraphVisualMode = 'knowledge_graph' | 'ontology' | 'agent';

// ---------------------------------------------------------------------------
// Pure helper functions (no React hooks — deterministic, testable)
// ---------------------------------------------------------------------------

/** Detect the dominant graph visual mode from node population (sampled for perf) */
const detectGraphMode = (nodes: GraphNode[]): GraphVisualMode => {
  if (nodes.length === 0) return 'knowledge_graph';
  const sample = nodes.length > 50 ? nodes.slice(0, 50) : nodes;
  let ontologySignals = 0;
  let agentSignals = 0;
  for (const n of sample) {
    if ((n as unknown as { owlClassIri?: string }).owlClassIri || n.metadata?.hierarchyDepth !== undefined || n.metadata?.depth !== undefined) {
      ontologySignals++;
    }
    // Require agentType as the definitive agent signal.
    // `status` alone is too generic — knowledge graph nodes commonly have status fields.
    if (n.metadata?.agentType || n.metadata?.tokenRate !== undefined) {
      agentSignals++;
    }
  }
  const threshold = sample.length * 0.2;
  if (agentSignals > threshold && agentSignals >= ontologySignals) return 'agent';
  if (ontologySignals > threshold) return 'ontology';
  return 'knowledge_graph';
};

/** Map binary protocol NodeType to GraphVisualMode for per-node rendering */
export const nodeTypeToVisualMode = (nodeType: NodeType): GraphVisualMode => {
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

// ---------------------------------------------------------------------------
// Hook return type
// ---------------------------------------------------------------------------

export interface GraphVisualStateResult {
  /** Per-node visual mode, keyed by string node ID */
  perNodeVisualModeMap: Map<string, GraphVisualMode>;
  /** Ontology hierarchy map, keyed by string node ID */
  hierarchyMap: Map<string, HierarchyNode>;
  /** Connection count per node, keyed by string node ID */
  connectionCountMap: Map<string, number>;
  /** The single dominant visual mode for the graph */
  dominantMode: GraphVisualMode;
}

// ---------------------------------------------------------------------------
// Hook implementation
// ---------------------------------------------------------------------------

export function useGraphVisualState(graphData: GraphData): GraphVisualStateResult {
  // Narrow selector: only the graph mode override matters for visual state.
  // Avoids re-renders when unrelated settings (glow, edges, physics) change.
  const settingsGraphMode = useSettingsStore(
    s => (s.settings?.visualisation?.graphs as unknown as { mode?: GraphVisualMode } | undefined)?.mode
  );

  // Binary protocol node-type map from websocket store
  const binaryNodeTypeMap = useWebSocketStore(state => state.nodeTypeMap);

  // --- hierarchyMap ---
  const hierarchyMap = useMemo(() => {
    if (graphData.nodes.length === 0) return new Map<string, HierarchyNode>();
    const hierarchy = detectHierarchy(graphData.nodes);
    logger.info(`Detected hierarchy: ${hierarchy.size} nodes, max depth: ${
      Math.max(...Array.from(hierarchy.values()).map(n => n.depth))
    }`);
    return hierarchy;
  }, [graphData.nodes]);

  // --- connectionCountMap ---
  const connectionCountMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const edge of graphData.edges) {
      const src = String(edge.source);
      const tgt = String(edge.target);
      map.set(src, (map.get(src) || 0) + 1);
      map.set(tgt, (map.get(tgt) || 0) + 1);
    }
    return map;
  }, [graphData.edges]);

  // --- dominantMode ---
  const dominantMode: GraphVisualMode = useMemo(() => {
    if (settingsGraphMode && (settingsGraphMode === 'knowledge_graph' || settingsGraphMode === 'ontology' || settingsGraphMode === 'agent')) {
      return settingsGraphMode;
    }
    return detectGraphMode(graphData.nodes);
  }, [settingsGraphMode, graphData.nodes]);

  // --- perNodeVisualModeMap ---
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
      const nt = node.metadata?.nodeType || (node as unknown as { nodeType?: string }).nodeType || '';
      const owlIri = (node as unknown as { owlClassIri?: string }).owlClassIri;
      if (node.metadata?.agentType || node.metadata?.tokenRate !== undefined) {
        map.set(String(node.id), 'agent');
      } else if (owlIri || nt === 'owl_class' || node.metadata?.hierarchyDepth !== undefined) {
        map.set(String(node.id), 'ontology');
      }
      // If no signals found, don't set -- will fall through to global dominantMode
    }
    return map;
  }, [graphData.nodes, binaryNodeTypeMap]);

  return {
    perNodeVisualModeMap,
    hierarchyMap,
    connectionCountMap,
    dominantMode,
  };
}
