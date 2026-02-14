/**
 * useGraphFiltering
 *
 * Extracts node filtering logic (quality/authority thresholds, hierarchy-based
 * visibility, expansion state) and derived lookup maps from GraphManager.
 *
 * Provides:
 *  - visibleNodes: the filtered subset of graphData.nodes that should be rendered
 *  - nodeIdToIndexMap: O(1) lookup from string node ID to its index in graphData.nodes
 *  - filteredEdges: edges whose both endpoints survive filtering (same as graphData.edges
 *    today since edge filtering is position-based, but provided for future use)
 */

import { useMemo, useEffect } from 'react';
import { useSettingsStore } from '../../../store/settingsStore';
import { useExpansionState, type ExpansionState } from '../hooks/useExpansionState';
import { createLogger } from '../../../utils/loggerConfig';
import type { GraphData, Node as GraphNode, Edge } from '../managers/graphDataManager';
import type { HierarchyNode } from '../utils/hierarchyDetector';

const logger = createLogger('useGraphFiltering');

// ---------------------------------------------------------------------------
// Hook return type
// ---------------------------------------------------------------------------

export interface GraphFilteringResult {
  /** Nodes that pass hierarchy-expansion and quality/authority filters */
  visibleNodes: GraphNode[];
  /** O(1) lookup: string node-id -> index in graphData.nodes (full set, not filtered) */
  nodeIdToIndexMap: Map<string, number>;
  /** Edges from graphData (pass-through today; hook owns the contract for future filtering) */
  filteredEdges: Edge[];
  /** Expansion state controller (for toggling node expansion in hierarchy view) */
  expansionState: ExpansionState;
}

// ---------------------------------------------------------------------------
// Hook implementation
// ---------------------------------------------------------------------------

export function useGraphFiltering(
  graphData: GraphData,
  hierarchyMap: Map<string, HierarchyNode>,
  connectionCountMap: Map<string, number>,
): GraphFilteringResult {
  const settings = useSettingsStore((state) => state.settings);

  // --- nodeIdToIndexMap ---
  const nodeIdToIndexMap = useMemo(() =>
    new Map(graphData.nodes.map((n, i) => [String(n.id), i])),
    [graphData.nodes]
  );

  // --- Expansion state (per-client, no server persistence) ---
  const expansionState = useExpansionState(true); // Default: all expanded

  // --- Extract filter settings for stable deps ---
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
      hasStoreFilter: !!storeNodeFilter,
    });
  }, [filterEnabled, qualityThreshold, authorityThreshold, filterByQuality, filterByAuthority, filterMode, storeNodeFilter]);

  // --- visibleNodes ---
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
  }, [graphData.nodes, connectionCountMap, hierarchyMap, expansionState, filterEnabled, qualityThreshold, authorityThreshold, filterByQuality, filterByAuthority, filterMode]);

  // --- filteredEdges (pass-through today) ---
  const filteredEdges = graphData.edges;

  return {
    visibleNodes,
    nodeIdToIndexMap,
    filteredEdges,
    expansionState,
  };
}
