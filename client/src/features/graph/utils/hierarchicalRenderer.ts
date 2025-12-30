import * as THREE from 'three';
import { GraphNode } from '../managers/graphDataManager';
import { ClassNode } from '../../ontology/store/useOntologyStore';

/**
 * Hierarchical rendering utilities for class-based visualization
 */

export interface ClassGroupNode {
  classIri: string;
  label: string;
  instanceCount: number;
  position: THREE.Vector3;
  scale: number;
  color: THREE.Color;
  isCollapsed: boolean;
  childIris: string[];
  depth: number;
}

/**
 * Group nodes by their class hierarchy
 */
export const groupNodesByClass = (
  nodes: GraphNode[],
  classHierarchy: Map<string, ClassNode>,
  expandedClasses: Set<string>
): ClassGroupNode[] => {
  const groups: ClassGroupNode[] = [];
  const classNodeMap = new Map<string, GraphNode[]>();

  // Group nodes by their class IRI
  nodes.forEach(node => {
    const classIri = node.metadata?.classIri;
    if (!classIri) return;

    if (!classNodeMap.has(classIri)) {
      classNodeMap.set(classIri, []);
    }
    classNodeMap.get(classIri)!.push(node);
  });

  // Create group nodes for collapsed classes
  classNodeMap.forEach((nodeGroup, classIri) => {
    const classNode = classHierarchy.get(classIri);
    if (!classNode) return;

    const isCollapsed = !expandedClasses.has(classIri);

    if (isCollapsed) {
      // Calculate average position for the group
      const avgPosition = new THREE.Vector3();
      nodeGroup.forEach(node => {
        if (node.position) {
          avgPosition.add(
            new THREE.Vector3(node.position.x, node.position.y, node.position.z)
          );
        }
      });
      avgPosition.divideScalar(nodeGroup.length);

      // Scale based on instance count
      const scale = Math.min(5, 1 + Math.log(nodeGroup.length + 1));

      // Color based on depth in hierarchy
      const color = getColorForDepth(classNode.depth);

      groups.push({
        classIri,
        label: classNode.label,
        instanceCount: nodeGroup.length,
        position: avgPosition,
        scale,
        color,
        isCollapsed: true,
        childIris: classNode.childIris ?? [],
        depth: classNode.depth,
      });
    }
  });

  return groups;
};

/**
 * Get color based on hierarchy depth
 */
export const getColorForDepth = (depth: number): THREE.Color => {
  const colors = [
    new THREE.Color('#FF6B6B'), // Depth 0: Red
    new THREE.Color('#4ECDC4'), // Depth 1: Cyan
    new THREE.Color('#FFD93D'), // Depth 2: Yellow
    new THREE.Color('#95E1D3'), // Depth 3: Light cyan
    new THREE.Color('#AA96DA'), // Depth 4: Purple
    new THREE.Color('#F38181'), // Depth 5+: Pink
  ];

  return colors[Math.min(depth, colors.length - 1)];
};

/**
 * Calculate smooth transition between expanded and collapsed states
 */
export const calculateTransitionState = (
  fromPosition: THREE.Vector3,
  toPosition: THREE.Vector3,
  fromScale: number,
  toScale: number,
  progress: number // 0-1
): { position: THREE.Vector3; scale: number } => {
  const easedProgress = easeInOutCubic(progress);

  const position = new THREE.Vector3().lerpVectors(
    fromPosition,
    toPosition,
    easedProgress
  );

  const scale = THREE.MathUtils.lerp(fromScale, toScale, easedProgress);

  return { position, scale };
};

/**
 * Easing function for smooth animations
 */
const easeInOutCubic = (t: number): number => {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2;
};

/**
 * Render collapsed class as a large sphere with label
 */
export const renderCollapsedClass = (
  group: ClassGroupNode,
  geometry: THREE.SphereGeometry,
  material: THREE.MeshBasicMaterial
): THREE.Mesh => {
  const mesh = new THREE.Mesh(geometry, material.clone());

  mesh.position.copy(group.position);
  mesh.scale.setScalar(group.scale);
  (mesh.material as THREE.MeshBasicMaterial).color = group.color;
  (mesh.material as THREE.MeshBasicMaterial).opacity = 0.7;
  (mesh.material as THREE.MeshBasicMaterial).transparent = true;

  mesh.userData = {
    type: 'classGroup',
    classIri: group.classIri,
    label: group.label,
    instanceCount: group.instanceCount,
    isCollapsed: true,
  };

  return mesh;
};

/**
 * Filter visible nodes based on semantic zoom level
 */
export const filterNodesByZoomLevel = (
  nodes: GraphNode[],
  semanticZoomLevel: number,
  classHierarchy: Map<string, ClassNode>
): GraphNode[] => {
  if (semanticZoomLevel === 0) return nodes; // Show all

  return nodes.filter(node => {
    const classIri = node.metadata?.classIri;
    if (!classIri) return true; // Show unclassified nodes

    const classNode = classHierarchy.get(classIri);
    if (!classNode) return true;

    // Filter based on depth threshold
    const maxDepth = Math.max(...Array.from(classHierarchy.values()).map(c => c.depth));
    const visibleDepth = Math.max(0, maxDepth - semanticZoomLevel);

    return classNode.depth <= visibleDepth;
  });
};

/**
 * Calculate bounding box for a group of nodes
 */
export const calculateGroupBoundingBox = (nodes: GraphNode[]): THREE.Box3 => {
  const box = new THREE.Box3();

  nodes.forEach(node => {
    if (node.position) {
      box.expandByPoint(
        new THREE.Vector3(node.position.x, node.position.y, node.position.z)
      );
    }
  });

  return box;
};

/**
 * Highlight nodes of the same class
 */
export const highlightSameClass = (
  targetNode: GraphNode,
  allNodes: GraphNode[]
): string[] => {
  const classIri = targetNode.metadata?.classIri;
  if (!classIri) return [targetNode.id];

  return allNodes
    .filter(node => node.metadata?.classIri === classIri)
    .map(node => node.id);
};
