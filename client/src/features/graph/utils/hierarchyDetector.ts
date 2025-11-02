/**
 * Client-Side Hierarchy Detection
 *
 * Detects parent-child relationships from node IDs based on path structure.
 * Example: "pages/foo/bar.md" has parent "pages/foo/"
 *
 * This is PURE CLIENT-SIDE logic - no server-side changes needed.
 */

import type { Node } from '../managers/graphWorkerProxy';

export interface HierarchyNode extends Node {
  parentId?: string;
  depth: number;
  childIds: string[];
  isRoot: boolean;
}

/**
 * Detect hierarchy from node ID path structure
 * @param nodes All nodes from the graph
 * @returns Map of node ID to HierarchyNode with parent-child relationships
 */
export function detectHierarchy(nodes: Node[]): Map<string, HierarchyNode> {
  const hierarchyMap = new Map<string, HierarchyNode>();

  // First pass: Create hierarchy nodes with parent detection
  nodes.forEach(node => {
    const pathParts = node.id.split('/').filter(p => p.length > 0);
    const depth = pathParts.length - 1;

    // Parent is the path without the last component
    const parentPath = pathParts.slice(0, -1).join('/');
    const parentId = parentPath || undefined;

    hierarchyMap.set(node.id, {
      ...node,
      parentId,
      depth,
      childIds: [],
      isRoot: !parentId
    });
  });

  // Second pass: Build child arrays
  hierarchyMap.forEach((node, id) => {
    if (node.parentId) {
      const parent = hierarchyMap.get(node.parentId);
      if (parent) {
        parent.childIds.push(id);
      }
    }
  });

  return hierarchyMap;
}

/**
 * Get all descendant IDs (recursive)
 * @param nodeId Parent node ID
 * @param hierarchyMap Hierarchy map
 * @returns Array of all descendant node IDs
 */
export function getDescendants(
  nodeId: string,
  hierarchyMap: Map<string, HierarchyNode>
): string[] {
  const node = hierarchyMap.get(nodeId);
  if (!node) return [];

  const descendants: string[] = [];
  const queue = [...node.childIds];

  while (queue.length > 0) {
    const childId = queue.shift()!;
    descendants.push(childId);

    const child = hierarchyMap.get(childId);
    if (child) {
      queue.push(...child.childIds);
    }
  }

  return descendants;
}

/**
 * Get all ancestor IDs (recursive)
 * @param nodeId Child node ID
 * @param hierarchyMap Hierarchy map
 * @returns Array of all ancestor node IDs (closest first)
 */
export function getAncestors(
  nodeId: string,
  hierarchyMap: Map<string, HierarchyNode>
): string[] {
  const ancestors: string[] = [];
  let currentId: string | undefined = nodeId;

  while (currentId) {
    const node = hierarchyMap.get(currentId);
    if (!node || !node.parentId) break;

    ancestors.push(node.parentId);
    currentId = node.parentId;
  }

  return ancestors;
}

/**
 * Get root nodes (nodes with no parent)
 * @param hierarchyMap Hierarchy map
 * @returns Array of root node IDs
 */
export function getRootNodes(hierarchyMap: Map<string, HierarchyNode>): string[] {
  const roots: string[] = [];
  hierarchyMap.forEach((node, id) => {
    if (node.isRoot) {
      roots.push(id);
    }
  });
  return roots;
}

/**
 * Get maximum depth in the hierarchy
 * @param hierarchyMap Hierarchy map
 * @returns Maximum depth (0 for flat graph)
 */
export function getMaxDepth(hierarchyMap: Map<string, HierarchyNode>): number {
  let maxDepth = 0;
  hierarchyMap.forEach(node => {
    if (node.depth > maxDepth) {
      maxDepth = node.depth;
    }
  });
  return maxDepth;
}
