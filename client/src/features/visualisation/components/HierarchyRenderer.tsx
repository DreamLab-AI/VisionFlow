/**
 * Hierarchical Visualization Renderer with THREE.js
 *
 * Renders ontology class hierarchy with:
 * - Visual nesting using THREE.Group
 * - Bounding boxes for class groups
 * - Collapse/expand controls
 * - Semantic zoom (6 levels)
 * - Color-coded depth shading
 * - Level-of-detail (LOD) optimization
 */

import React, { useEffect, useRef, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { useHierarchyData, ClassNode } from '../../ontology/hooks/useHierarchyData';
import { useExpansionState } from '../../graph/hooks/useExpansionState';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('HierarchyRenderer');

// Visual constants
const COLORS = {
  depth0: 0x00ffff, // Cyan - root classes
  depth1: 0x00ccff,
  depth2: 0x0099ff,
  depth3: 0x0066ff,
  depth4: 0x0033ff,
  depth5: 0x0000ff, // Blue - deepest classes
  border: 0xffffff,
  selected: 0xffff00,
  collapsed: 0x888888,
};

const BOX_PADDING = 2;
const NODE_SIZE = 1;
const VERTICAL_SPACING = 3;
const HORIZONTAL_SPACING = 4;

export interface HierarchyRendererProps {
  scene: THREE.Scene;
  camera: THREE.Camera;
  semanticZoomLevel: number;
  ontologyId?: string;
  onNodeClick?: (nodeIri: string) => void;
  onNodeHover?: (nodeIri: string | null) => void;
}

interface NodePosition {
  iri: string;
  position: THREE.Vector3;
  depth: number;
  group: THREE.Group;
  mesh: THREE.Mesh;
  boundingBox?: THREE.Box3Helper;
  label?: THREE.Sprite;
}

/**
 * Hierarchical visualization renderer component
 */
export const HierarchyRenderer: React.FC<HierarchyRendererProps> = ({
  scene,
  camera,
  semanticZoomLevel,
  ontologyId = 'default',
  onNodeClick,
  onNodeHover,
}) => {
  const {
    hierarchy,
    loading,
    error,
    maxDepth,
    totalClasses,
    getChildren,
    getDescendants,
    getRootClasses,
  } = useHierarchyData({ ontologyId, autoRefresh: false });

  const expansionState = useExpansionState(true);
  const nodePositionsRef = useRef<Map<string, NodePosition>>(new Map());
  const rootGroupRef = useRef<THREE.Group | null>(null);
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2());

  /**
   * Get color for a node based on its depth
   */
  const getDepthColor = useCallback((depth: number): number => {
    const colors = [
      COLORS.depth0,
      COLORS.depth1,
      COLORS.depth2,
      COLORS.depth3,
      COLORS.depth4,
      COLORS.depth5,
    ];
    return colors[Math.min(depth, colors.length - 1)];
  }, []);

  /**
   * Create a text sprite for labels
   */
  const createLabelSprite = useCallback((text: string, color: number = 0xffffff): THREE.Sprite => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    canvas.width = 256;
    canvas.height = 64;

    context.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
    context.font = 'Bold 24px Arial';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(4, 1, 1);

    return sprite;
  }, []);

  /**
   * Create a bounding box for a class group
   */
  const createBoundingBox = useCallback(
    (min: THREE.Vector3, max: THREE.Vector3, color: number): THREE.Box3Helper => {
      const box = new THREE.Box3(min, max);
      const helper = new THREE.Box3Helper(box, new THREE.Color(color));
      (helper as any).type = 'BoundingBox'; // Tag for identification
      return helper;
    },
    []
  );

  /**
   * Calculate layout positions for hierarchy tree
   */
  const calculateLayout = useCallback(
    (node: ClassNode, x: number, y: number, depth: number): { width: number; height: number } => {
      const children = getChildren(node.iri);
      const isExpanded = expansionState.isExpanded(node.iri);

      let totalWidth = 0;
      let maxHeight = 0;

      if (isExpanded && children.length > 0) {
        let currentX = x;

        children.forEach((child) => {
          const childLayout = calculateLayout(child, currentX, y - VERTICAL_SPACING, depth + 1);
          currentX += childLayout.width + HORIZONTAL_SPACING;
          totalWidth += childLayout.width + HORIZONTAL_SPACING;
          maxHeight = Math.max(maxHeight, childLayout.height);
        });

        totalWidth = Math.max(totalWidth - HORIZONTAL_SPACING, 1);
      } else {
        totalWidth = 1;
      }

      // Store position
      const centerX = x + totalWidth / 2;
      const position = new THREE.Vector3(centerX, y, -depth * 2);

      if (!nodePositionsRef.current.has(node.iri)) {
        const group = new THREE.Group();
        group.userData.iri = node.iri;
        group.userData.depth = depth;

        nodePositionsRef.current.set(node.iri, {
          iri: node.iri,
          position,
          depth,
          group,
          mesh: new THREE.Mesh(), // Placeholder, will be created later
        });
      } else {
        const existing = nodePositionsRef.current.get(node.iri)!;
        existing.position.copy(position);
        existing.depth = depth;
      }

      return {
        width: totalWidth,
        height: maxHeight + VERTICAL_SPACING,
      };
    },
    [getChildren, expansionState]
  );

  /**
   * Build THREE.js scene from hierarchy
   */
  const buildScene = useCallback(() => {
    if (!hierarchy || loading) return;

    logger.info('Building hierarchical scene', {
      totalClasses,
      maxDepth,
      semanticZoomLevel,
    });

    // Clear previous scene
    if (rootGroupRef.current) {
      scene.remove(rootGroupRef.current);
      rootGroupRef.current.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          if (obj.material instanceof THREE.Material) {
            obj.material.dispose();
          }
        }
      });
    }

    const rootGroup = new THREE.Group();
    rootGroupRef.current = rootGroup;

    // Calculate layouts for all root classes
    const rootClasses = getRootClasses();
    let startX = 0;

    rootClasses.forEach((rootNode) => {
      const layout = calculateLayout(rootNode, startX, 0, 0);
      startX += layout.width + HORIZONTAL_SPACING * 2;
    });

    // Create visual elements for each node
    nodePositionsRef.current.forEach((nodePos, iri) => {
      const node = hierarchy.hierarchy[iri];
      if (!node) return;

      // Check semantic zoom visibility
      const shouldRender = node.depth <= (5 - semanticZoomLevel);
      if (!shouldRender) return;

      // Create sphere for node
      const geometry = new THREE.SphereGeometry(NODE_SIZE, 16, 16);
      const material = new THREE.MeshPhongMaterial({
        color: getDepthColor(node.depth),
        emissive: getDepthColor(node.depth),
        emissiveIntensity: 0.2,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(nodePos.position);
      mesh.userData.iri = iri;
      mesh.userData.type = 'ClassNode';

      nodePos.mesh = mesh;
      nodePos.group.add(mesh);

      // Create label
      const label = createLabelSprite(node.label, COLORS.border);
      label.position.set(0, NODE_SIZE + 1, 0);
      mesh.add(label);
      nodePos.label = label;

      // Create bounding box for expanded groups with children
      const children = getChildren(iri);
      const isExpanded = expansionState.isExpanded(iri);

      if (isExpanded && children.length > 0) {
        const descendants = getDescendants(iri);
        const positions = descendants
          .map((d) => nodePositionsRef.current.get(d.iri)?.position)
          .filter(Boolean) as THREE.Vector3[];

        if (positions.length > 0) {
          const min = new THREE.Vector3(
            Math.min(...positions.map((p) => p.x)) - BOX_PADDING,
            Math.min(...positions.map((p) => p.y)) - BOX_PADDING,
            Math.min(...positions.map((p) => p.z)) - BOX_PADDING
          );
          const max = new THREE.Vector3(
            Math.max(...positions.map((p) => p.x)) + BOX_PADDING,
            Math.max(...positions.map((p) => p.y)) + BOX_PADDING,
            Math.max(...positions.map((p) => p.z)) + BOX_PADDING
          );

          const bbox = createBoundingBox(min, max, getDepthColor(node.depth));
          nodePos.boundingBox = bbox;
          nodePos.group.add(bbox);
        }
      }

      rootGroup.add(nodePos.group);
    });

    scene.add(rootGroup);
    logger.info('Hierarchical scene built', {
      visibleNodes: nodePositionsRef.current.size,
    });
  }, [
    hierarchy,
    loading,
    scene,
    totalClasses,
    maxDepth,
    semanticZoomLevel,
    getRootClasses,
    calculateLayout,
    getChildren,
    getDescendants,
    expansionState,
    getDepthColor,
    createLabelSprite,
    createBoundingBox,
  ]);

  // Rebuild scene when dependencies change
  useEffect(() => {
    buildScene();
  }, [buildScene]);

  // Handle mouse interactions
  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (!rootGroupRef.current) return;

      const rect = (event.target as HTMLElement).getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);
      const intersects = raycasterRef.current.intersectObjects(
        rootGroupRef.current.children,
        true
      );

      const hoveredNode = intersects.find((i) => i.object.userData.type === 'ClassNode');
      if (hoveredNode && onNodeHover) {
        onNodeHover(hoveredNode.object.userData.iri);
      } else if (onNodeHover) {
        onNodeHover(null);
      }
    };

    const handleClick = (event: MouseEvent) => {
      if (!rootGroupRef.current) return;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);
      const intersects = raycasterRef.current.intersectObjects(
        rootGroupRef.current.children,
        true
      );

      const clickedNode = intersects.find((i) => i.object.userData.type === 'ClassNode');
      if (clickedNode) {
        const iri = clickedNode.object.userData.iri;
        expansionState.toggleExpansion(iri);
        if (onNodeClick) {
          onNodeClick(iri);
        }
      }
    };

    const canvas = scene.userData.canvas as HTMLCanvasElement | undefined;
    if (canvas) {
      canvas.addEventListener('mousemove', handleMouseMove);
      canvas.addEventListener('click', handleClick);

      return () => {
        canvas.removeEventListener('mousemove', handleMouseMove);
        canvas.removeEventListener('click', handleClick);
      };
    }
  }, [scene, camera, expansionState, onNodeClick, onNodeHover]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (rootGroupRef.current) {
        scene.remove(rootGroupRef.current);
      }
      nodePositionsRef.current.clear();
    };
  }, [scene]);

  // Error/loading states
  if (error) {
    logger.error('Hierarchy rendering error', { error });
  }

  if (loading) {
    logger.debug('Loading hierarchy data...');
  }

  return null; // Rendered directly to THREE.js scene
};

export default HierarchyRenderer;
