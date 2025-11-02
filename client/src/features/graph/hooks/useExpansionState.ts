/**
 * Client-Side Expansion State Hook
 *
 * Manages per-client node expansion/collapse state.
 * NO server-side persistence - state is local to each client.
 */

import { useState, useCallback, useMemo } from 'react';

export interface ExpansionState {
  /** Set of collapsed node IDs */
  collapsedNodes: Set<string>;

  /** Toggle expansion state for a node */
  toggleExpansion: (nodeId: string) => void;

  /** Check if a node is expanded (inverse of collapsed) */
  isExpanded: (nodeId: string) => boolean;

  /** Check if a node should be visible based on parent expansion */
  isVisible: (nodeId: string, parentId?: string) => boolean;

  /** Expand all nodes */
  expandAll: () => void;

  /** Collapse all nodes */
  collapseAll: () => void;

  /** Expand a node and all its ancestors */
  expandWithAncestors: (nodeId: string, ancestorIds: string[]) => void;
}

/**
 * Hook for managing client-side node expansion state
 * @param defaultExpanded Whether nodes should be expanded by default (recommended: true)
 * @returns ExpansionState object with expansion controls
 */
export function useExpansionState(defaultExpanded: boolean = true): ExpansionState {
  // Store collapsed nodes (if defaultExpanded=true) or expanded nodes (if defaultExpanded=false)
  const [collapsedNodes, setCollapsedNodes] = useState<Set<string>>(new Set());

  const toggleExpansion = useCallback((nodeId: string) => {
    setCollapsedNodes(prev => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, []);

  const isExpanded = useCallback((nodeId: string) => {
    // If defaultExpanded=true, node is expanded unless in collapsedNodes
    // If defaultExpanded=false, node is collapsed unless in collapsedNodes (which would be expandedNodes)
    return defaultExpanded ? !collapsedNodes.has(nodeId) : collapsedNodes.has(nodeId);
  }, [collapsedNodes, defaultExpanded]);

  const isVisible = useCallback((nodeId: string, parentId?: string) => {
    // Root nodes (no parent) are always visible
    if (!parentId) return true;

    // Child nodes are visible only if parent is expanded
    return isExpanded(parentId);
  }, [isExpanded]);

  const expandAll = useCallback(() => {
    setCollapsedNodes(new Set());
  }, []);

  const collapseAll = useCallback(() => {
    // Implementation would need list of all node IDs
    // For now, just clear (which returns to default state)
    setCollapsedNodes(new Set());
  }, []);

  const expandWithAncestors = useCallback((nodeId: string, ancestorIds: string[]) => {
    setCollapsedNodes(prev => {
      const next = new Set(prev);
      // Remove node and all ancestors from collapsed set
      next.delete(nodeId);
      ancestorIds.forEach(ancestorId => next.delete(ancestorId));
      return next;
    });
  }, []);

  return useMemo(() => ({
    collapsedNodes,
    toggleExpansion,
    isExpanded,
    isVisible,
    expandAll,
    collapseAll,
    expandWithAncestors
  }), [collapsedNodes, toggleExpansion, isExpanded, isVisible, expandAll, collapseAll, expandWithAncestors]);
}
