import React, { useMemo, useRef } from 'react';
import { Billboard, Text } from '@react-three/drei';
import * as THREE from 'three';
import { GraphNode } from '../managers/graphDataManager';
import { useOntologyStore } from '../../ontology/store/useOntologyStore';
import {
  groupNodesByClass,
  renderCollapsedClass,
  filterNodesByZoomLevel,
  highlightSameClass,
  ClassGroupNode,
} from '../utils/hierarchicalRenderer';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('HierarchicalGraphRenderer');

interface HierarchicalGraphRendererProps {
  nodes: GraphNode[];
  edges: any[];
  nodePositions: Float32Array | null;
  onNodeClick?: (nodeId: string, event: any) => void;
  settings: any;
}

/**
 * Hierarchical graph renderer with class grouping and semantic zoom
 */
export const HierarchicalGraphRenderer: React.FC<HierarchicalGraphRendererProps> = ({
  nodes,
  edges,
  nodePositions,
  onNodeClick,
  settings,
}) => {
  const {
    hierarchy,
    semanticZoomLevel,
    expandedClasses,
    highlightedClass,
    toggleClass,
  } = useOntologyStore();

  const groupMeshesRef = useRef<THREE.Mesh[]>([]);

  // Filter nodes based on semantic zoom level
  const filteredNodes = useMemo(() => {
    if (!hierarchy || semanticZoomLevel === 0) return nodes;

    return filterNodesByZoomLevel(nodes, semanticZoomLevel, hierarchy.classes);
  }, [nodes, semanticZoomLevel, hierarchy]);

  // Group nodes by class for hierarchical rendering
  const classGroups = useMemo(() => {
    if (!hierarchy || semanticZoomLevel < 3) return [];

    return groupNodesByClass(filteredNodes, hierarchy.classes, expandedClasses);
  }, [filteredNodes, hierarchy, expandedClasses, semanticZoomLevel]);

  // Render mode: individual nodes or grouped classes
  const renderMode = semanticZoomLevel >= 3 ? 'grouped' : 'individual';

  // Render collapsed class groups as large spheres
  const CollapsedClassGroups = useMemo(() => {
    if (renderMode !== 'grouped' || classGroups.length === 0) return null;

    return classGroups.map((group: ClassGroupNode, idx: number) => {
      const isHighlighted = highlightedClass === group.classIri;
      const scale = group.scale * (isHighlighted ? 1.3 : 1);
      const opacity = isHighlighted ? 0.9 : 0.7;

      return (
        <group key={`class-group-${group.classIri}-${idx}`}>
          {/* Large sphere for collapsed class */}
          <mesh
            position={group.position}
            scale={scale}
            onClick={(e) => {
              e.stopPropagation();
              toggleClass(group.classIri);
              logger.info('Class group clicked', {
                classIri: group.classIri,
                label: group.label,
                instanceCount: group.instanceCount,
              });
            }}
          >
            <sphereGeometry args={[1, 32, 32]} />
            <meshStandardMaterial
              color={group.color}
              transparent
              opacity={opacity}
              emissive={group.color}
              emissiveIntensity={isHighlighted ? 0.5 : 0.2}
              metalness={0.3}
              roughness={0.4}
            />
          </mesh>

          {/* Label for class group */}
          <Billboard
            position={[
              group.position.x,
              group.position.y + scale * 1.5,
              group.position.z,
            ]}
            follow={true}
          >
            <Text
              fontSize={0.8}
              color="#ffffff"
              anchorX="center"
              anchorY="bottom"
              outlineWidth={0.02}
              outlineColor="#000000"
              maxWidth={8}
              textAlign="center"
            >
              {group.label}
            </Text>
            <Text
              position={[0, -0.5, 0]}
              fontSize={0.5}
              color="#00ffff"
              anchorX="center"
              anchorY="top"
              outlineWidth={0.01}
              outlineColor="#000000"
            >
              {group.instanceCount} instances
            </Text>
            <Text
              position={[0, -1.0, 0]}
              fontSize={0.4}
              color="#aaaaaa"
              anchorX="center"
              anchorY="top"
              outlineWidth={0.01}
              outlineColor="#000000"
            >
              Click to expand
            </Text>
          </Billboard>
        </group>
      );
    });
  }, [classGroups, renderMode, highlightedClass, toggleClass]);

  // Render individual nodes (when zoomed in)
  const IndividualNodes = useMemo(() => {
    if (renderMode !== 'individual' || !nodePositions) return null;

    return filteredNodes.map((node, idx) => {
      const i3 = idx * 3;
      const position = new THREE.Vector3(
        nodePositions[i3],
        nodePositions[i3 + 1],
        nodePositions[i3 + 2]
      );

      const classIri = node.metadata?.classIri;
      const isHighlighted =
        highlightedClass && classIri === highlightedClass;

      const nodeScale = isHighlighted ? 1.3 : 1.0;
      const nodeColor = isHighlighted
        ? new THREE.Color('#ffff00')
        : new THREE.Color('#00ffff');

      return (
        <mesh
          key={`node-${node.id}-${idx}`}
          position={position}
          scale={nodeScale}
          onClick={(e) => {
            e.stopPropagation();
            if (onNodeClick) {
              onNodeClick(node.id, e);
            }

            // Highlight same-class nodes on double-click
            if (e.detail === 2 && classIri) {
              const sameClassIds = highlightSameClass(node, filteredNodes);
              logger.info('Highlighting same class', {
                classIri,
                count: sameClassIds.length,
              });
            }
          }}
        >
          <sphereGeometry args={[0.5, 32, 32]} />
          <meshStandardMaterial
            color={nodeColor}
            transparent
            opacity={0.8}
            emissive={nodeColor}
            emissiveIntensity={0.3}
          />
        </mesh>
      );
    });
  }, [
    filteredNodes,
    renderMode,
    nodePositions,
    highlightedClass,
    onNodeClick,
  ]);

  // Stats logging
  React.useEffect(() => {
    if (renderMode === 'grouped') {
      logger.info('Hierarchical rendering mode', {
        classGroups: classGroups.length,
        zoomLevel: semanticZoomLevel,
      });
    } else {
      logger.debug('Individual rendering mode', {
        nodeCount: filteredNodes.length,
        zoomLevel: semanticZoomLevel,
      });
    }
  }, [renderMode, classGroups.length, filteredNodes.length, semanticZoomLevel]);

  return (
    <>
      {renderMode === 'grouped' ? CollapsedClassGroups : IndividualNodes}
    </>
  );
};

export default HierarchicalGraphRenderer;
