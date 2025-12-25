import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
import { useThree, useFrame, ThreeEvent } from '@react-three/fiber'
import { Text, Billboard } from '@react-three/drei'
import * as THREE from 'three'
import { graphDataManager, type GraphData, type Node as GraphNode } from '../managers/graphDataManager'
import { graphWorkerProxy } from '../managers/graphWorkerProxy'
import { createLogger, createErrorMetadata } from '../../../utils/loggerConfig'
import { debugState } from '../../../utils/clientDebugState'
import { useSettingsStore } from '../../../store/settingsStore'
import { BinaryNodeData, createBinaryNodeData } from '../../../types/binaryProtocol'
import { HologramNodeMaterial } from '../../../rendering/materials/HologramNodeMaterial'
import { FlowingEdges } from './FlowingEdges'
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

// Get geometry for node type
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
      return new THREE.SphereGeometry(0.5, 32, 32); 
  }
};

// Get node color based on type/metadata and SSSP visualization
const getNodeColor = (node: GraphNode, ssspResult?: any): THREE.Color => {
  
  if (ssspResult) {
    const distance = ssspResult.distances[node.id]
    
    
    if (node.id === ssspResult.sourceNodeId) {
      return new THREE.Color('#00FFFF') 
    }
    
    
    if (!isFinite(distance)) {
      return new THREE.Color('#666666') 
    }
    
    
    const normalizedDistances = ssspResult.normalizedDistances || {}
    const normalizedDistance = normalizedDistances[node.id] || 0
    
    
    const red = Math.min(1, normalizedDistance * 1.2)
    const green = Math.min(1, (1 - normalizedDistance) * 1.2)
    const blue = 0.1 
    
    return new THREE.Color(red, green, blue)
  }
  
  
  const typeColors: Record<string, string> = {
    'folder': '#FFD700',     
    'file': '#00CED1',       
    'function': '#FF6B6B',   
    'class': '#4ECDC4',      
    'variable': '#95E1D3',   
    'import': '#F38181',     
    'export': '#AA96DA',     
  }

  const nodeType = node.metadata?.type || 'default'
  const color = typeColors[nodeType] || '#00ffff'
  return new THREE.Color(color)
}

// Get node scale based on importance/connections
const getNodeScale = (node: GraphNode, edges: any[]): number => {
  const baseSize = node.metadata?.size || 1.0
  const connectionCount = edges.filter(e =>
    e.source === node.id || e.target === node.id
  ).length

  
  const connectionScale = 1 + Math.log(connectionCount + 1) * 0.3

  
  const typeScale = getTypeImportance(node.metadata?.type)

  return baseSize * connectionScale * typeScale
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
  
  useEffect(() => {
    if (onDragStateChange) {
      console.log('GraphManager: onDragStateChange callback available');
    }
  }, []);
  const settings = useSettingsStore((state) => state.settings);
  
  const nodeBloomStrength = settings?.visualisation?.bloom?.node_bloom_strength ?? settings?.visualisation?.bloom?.nodeBloomStrength ?? 0.5;
  const edgeBloomStrength = settings?.visualisation?.bloom?.edge_bloom_strength ?? settings?.visualisation?.bloom?.edgeBloomStrength ?? 0.5;
  
  
  const ssspResult = useCurrentSSSPResult();
  const normalizeDistances = useAnalyticsStore(state => state.normalizeDistances);
  const [normalizedSSSPResult, setNormalizedSSSPResult] = useState<any>(null);
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const materialRef = useRef<HologramNodeMaterial | null>(null)
  const particleSystemRef = useRef<THREE.Points>(null)

  
  const tempMatrix = useMemo(() => new THREE.Matrix4(), [])
  const tempPosition = useMemo(() => new THREE.Vector3(), [])
  const tempScale = useMemo(() => new THREE.Vector3(), [])
  const tempQuaternion = useMemo(() => new THREE.Quaternion(), [])
  const tempColor = useMemo(() => new THREE.Color(), [])

  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })
  const nodePositionsRef = useRef<Float32Array | null>(null)
  const [edgePoints, setEdgePoints] = useState<number[]>([])
  const [nodesAreAtOrigin, setNodesAreAtOrigin] = useState(false)

  const [forceUpdate, setForceUpdate] = useState(0)

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
          // Compute quality from node connections (normalized 0-1)
          const connectionCount = graphData.edges.filter(e =>
            e.source === node.id || e.target === node.id
          ).length;
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
  }, [graphData.nodes, graphData.edges, hierarchyMap, expansionState, filterEnabled, qualityThreshold, authorityThreshold, filterByQuality, filterByAuthority, filterMode])

  
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

  
  const updateNodeColors = useCallback(() => {
    if (!meshRef.current || graphData.nodes.length === 0) return;

    const mesh = meshRef.current;
    const colors = new Float32Array(graphData.nodes.length * 3);

    graphData.nodes.forEach((node, i) => {
      const color = getNodeColor(node, normalizedSSSPResult);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });

    mesh.geometry.setAttribute('instanceColor', new THREE.InstancedBufferAttribute(colors, 3));
    mesh.geometry.attributes.instanceColor.needsUpdate = true;
  }, [graphData.nodes, normalizedSSSPResult]);

  
  useEffect(() => {
    updateNodeColors();
  }, [updateNodeColors]);

  
  useEffect(() => {
    if (meshRef.current && graphData.nodes.length > 0) {
      const mesh = meshRef.current
      mesh.count = graphData.nodes.length
      
      
      const debugSettings = settings?.system?.debug;
      if (debugSettings?.enableNodeDebug) {
        console.log('GraphManager: Node mesh initialized', {
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
        
        const nodeScale = getNodeScale(node, graphData.edges) * baseScale;
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
    
    
    const debugSettings = settings?.system?.debug;
    if (debugSettings?.enablePhysicsDebug && debugState.isEnabled()) {
      const frameCount = Math.floor(state.clock.elapsedTime * 60);
      if (frameCount === 1 || frameCount % 300 === 0) { 
        logger.debug('Physics frame update', {
          time: state.clock.elapsedTime,
          delta,
          nodeCount: graphData.nodes.length,
          hasPositions: !!nodePositionsRef.current
        });
      }
    }

    
    if (materialRef.current) {
      materialRef.current.updateTime(animationStateRef.current.time)
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
            let nodeScale = getNodeScale(node, graphData.edges) * baseScale;


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

        
        const newEdgePoints: number[] = [];
        graphData.edges.forEach(edge => {
          const sourceNodeIndex = graphData.nodes.findIndex(n => n.id === edge.source);
          const targetNodeIndex = graphData.nodes.findIndex(n => n.id === edge.target);
          if (sourceNodeIndex !== -1 && targetNodeIndex !== -1) {
            const i3s = sourceNodeIndex * 3;
            const i3t = targetNodeIndex * 3;

            // Bounds check for edge positions
            if (i3s + 2 >= positions.length || i3t + 2 >= positions.length) return;

            const sourcePos = new THREE.Vector3(positions[i3s], positions[i3s + 1], positions[i3s + 2]);
            const targetPos = new THREE.Vector3(positions[i3t], positions[i3t + 1], positions[i3t + 2]);

            
            const direction = new THREE.Vector3().subVectors(targetPos, sourcePos);
            const edgeLength = direction.length();

            if (edgeLength > 0) {
              direction.normalize();

              
              const sourceNode = graphData.nodes[sourceNodeIndex];
              const targetNode = graphData.nodes[targetNodeIndex];
              const logseqSettings = settings?.visualisation?.graphs?.logseq;
              const nodeSettings = logseqSettings?.nodes || settings?.visualisation?.nodes;
              const sourceRadius = getNodeScale(sourceNode, graphData.edges) * (nodeSettings?.nodeSize || 0.5);
              const targetRadius = getNodeScale(targetNode, graphData.edges) * (nodeSettings?.nodeSize || 0.5);

              
              const offsetSource = new THREE.Vector3().addVectors(sourcePos, direction.clone().multiplyScalar(sourceRadius + 0.1));
              const offsetTarget = new THREE.Vector3().subVectors(targetPos, direction.clone().multiplyScalar(targetRadius + 0.1));

              
              if (offsetSource.distanceTo(offsetTarget) > 0.2) {
                newEdgePoints.push(offsetSource.x, offsetSource.y, offsetSource.z);
                newEdgePoints.push(offsetTarget.x, offsetTarget.y, offsetTarget.z);
              }
            }
          }
        });
        setEdgePoints(newEdgePoints);

        
        const newLabelPositions = graphData.nodes.map((node, i) => {
          const i3 = i * 3;
          return {
            x: positions[i3],
            y: positions[i3 + 1],
            z: positions[i3 + 2]
          };
        });
        setLabelPositions(newLabelPositions);
      }
    }

    
    
    
    
    
    
    
    
    
    
    

    
    if (meshRef.current && animationStateRef.current.selectedNode !== null) {
      const mesh = meshRef.current
      const nodes = graphData.nodes

      nodes.forEach((node, i) => {
        if (node.id === animationStateRef.current.selectedNode) {
          mesh.getMatrixAt(i, tempMatrix)
          tempMatrix.decompose(tempPosition, tempQuaternion, tempScale)

          const pulseFactor = 1 + Math.sin(animationStateRef.current.time * 3) * 0.1
          tempScale.multiplyScalar(pulseFactor)

          tempMatrix.compose(tempPosition, tempQuaternion, tempScale)
          mesh.setMatrixAt(i, tempMatrix)
          mesh.instanceMatrix.needsUpdate = true
        }
      })
    }
  })

  
  useEffect(() => {
    
    const handleGraphUpdate = (data: GraphData) => {
      
      const debugSettings = settings?.system?.debug;
      if (debugSettings?.enableNodeDebug) {
        console.log('GraphManager: Graph data updated', {
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
        return;
      }

      
      const dataWithPositions = {
        ...data,
        nodes: data.nodes.map((node, i) => {
          if (!node.position || (node.position.x === 0 && node.position.y === 0 && node.position.z === 0)) {
            const position = getPositionForNode(node, i, data.nodes.length)
            return {
              ...node,
              position: { x: position[0], y: position[1], z: position[2] }
            }
          }
          return node
        })
      }

      const allAtOrigin = dataWithPositions.nodes.every(node =>
        !node.position || (node.position.x === 0 && node.position.y === 0 && node.position.z === 0)
      )
      setNodesAreAtOrigin(allAtOrigin)

      setGraphData(dataWithPositions)

      
      const newEdgePoints: number[] = []
      data.edges.forEach((edge) => {
        const sourceNode = data.nodes.find(n => n.id === edge.source)
        const targetNode = data.nodes.find(n => n.id === edge.target)

        if (sourceNode?.position && targetNode?.position) {
          newEdgePoints.push(
            sourceNode.position.x, sourceNode.position.y, sourceNode.position.z,
            targetNode.position.x, targetNode.position.y, targetNode.position.z
          )
        }
      })

      setEdgePoints(newEdgePoints)
    }

    const unsubscribe = graphDataManager.onGraphDataChange(handleGraphUpdate)

    
    graphDataManager.getGraphData().then((data) => {
      
      const debugSettings = settings?.system?.debug;
      if (debugSettings?.enableNodeDebug) {
        console.log('GraphManager: Initial graph data loaded', {
          nodeCount: data.nodes.length,
          edgeCount: data.edges.length
        });
      }
      handleGraphUpdate(data)
      
      return graphWorkerProxy.setGraphData(data)
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
    meshRef,
    dragDataRef,
    setDragState,
    graphData,
    camera,
    size,
    settings,
    setGraphData,
    onDragStateChange
  )

  
  const particleGeometry = useMemo(() => {
    return null 
  }, [])

  
  const [labelPositions, setLabelPositions] = useState<Array<{x: number, y: number, z: number}>>([])

  
  useEffect(() => {
    if (nodePositionsRef.current && graphData.nodes.length > 0) {
      const newPositions = graphData.nodes.map((node, i) => {
        const i3 = i * 3
        return {
          x: nodePositionsRef.current![i3],
          y: nodePositionsRef.current![i3 + 1],
          z: nodePositionsRef.current![i3 + 2]
        }
      })
      setLabelPositions(newPositions)
    }
  }, [nodePositionsRef.current, graphData.nodes])

  
  const defaultEdgeSettings: EdgeSettings = {
    arrowSize: 0.5,
    baseWidth: 1,
    color: '#ffffff',
    enableArrows: true,
    opacity: 0.8,
    widthRange: [1, 5],
    quality: 'medium',
    enableFlowEffect: false,
    flowSpeed: 1,
    flowIntensity: 1,
    glowStrength: 1,
    distanceIntensity: 0.5,
    useGradient: false,
    gradientColors: ['#ff0000', '#0000ff'],
  };

  const NodeLabels = useMemo(() => {
      const logseqSettings = settings?.visualisation?.graphs?.logseq;
      const labelSettings = logseqSettings?.labels ?? settings?.visualisation?.labels;
      // CLIENT-SIDE HIERARCHICAL LOD: Only render labels for visible nodes
      if (!labelSettings?.enableLabels || visibleNodes.length === 0) return null;

    return visibleNodes.map((node) => {
      // CLIENT-SIDE HIERARCHICAL LOD: Find original index in graphData.nodes for position lookup
      const originalIndex = graphData.nodes.findIndex(n => n.id === node.id);
      const physicsPos = originalIndex !== -1 ? labelPositions[originalIndex] : undefined;
      const position = physicsPos || node.position || { x: 0, y: 0, z: 0 }
      const scale = getNodeScale(node, graphData.edges)
      const textPadding = labelSettings.textPadding ?? 0.6;
      const labelOffsetY = scale * 1.5 + textPadding; 

      
      let metadataToShow = null;
      let distanceInfo = null;
      
      
      if (normalizedSSSPResult) {
        const distance = normalizedSSSPResult.distances[node.id];
        if (node.id === normalizedSSSPResult.sourceNodeId) {
          distanceInfo = "Source (0)";
        } else if (!isFinite(distance)) {
          distanceInfo = "Unreachable";
        } else {
          distanceInfo = `Distance: ${distance.toFixed(2)}`;
        }
      }
      
      if (labelSettings.showMetadata && node.metadata) {
        if (node.metadata.description) {
          metadataToShow = node.metadata.description;
        } else if (node.metadata.type) {
          metadataToShow = node.metadata.type;
        } else if (node.metadata.fileSize) {
          const sizeInBytes = parseInt(node.metadata.fileSize);
          if (sizeInBytes > 1024 * 1024) {
            metadataToShow = `${(sizeInBytes / (1024 * 1024)).toFixed(1)} MB`;
          } else if (sizeInBytes > 1024) {
            metadataToShow = `${(sizeInBytes / 1024).toFixed(1)} KB`;
          } else {
            metadataToShow = `${sizeInBytes.toLocaleString()} bytes`;
          }
        }
      }

      
      const maxWidth = labelSettings.maxLabelWidth ?? 5;
      const fontSize = labelSettings.desktopFontSize ?? 0.5;

      return (
        <Billboard
          key={`label-${node.id}`}
          position={[position.x, position.y + labelOffsetY, position.z]}
          follow={true}
          lockX={false}
          lockY={false}
          lockZ={false}
        >
          <Text
            fontSize={fontSize}
            color={labelSettings.textColor || '#ffffff'}
            anchorX="center"
            anchorY="bottom"
            outlineWidth={labelSettings.textOutlineWidth || 0.005}
            outlineColor={labelSettings.textOutlineColor || '#000000'}
            maxWidth={maxWidth}
            textAlign="center"
          >
            {node.label || node.id}
          </Text>
          {distanceInfo && (
            <Text
              position={[0, -(textPadding * 0.25), 0]}
              fontSize={fontSize * 0.7}
              color={node.id === normalizedSSSPResult?.sourceNodeId ? '#00FFFF' : 
                     (!isFinite(normalizedSSSPResult?.distances[node.id] || 0) ? '#666666' : '#FFFF00')}
              anchorX="center"
              anchorY="top"
              maxWidth={maxWidth * 0.8}
              textAlign="center"
              outlineWidth={0.002}
              outlineColor="#000000"
            >
              {distanceInfo}
            </Text>
          )}
          {metadataToShow && !distanceInfo && (
            <Text
              position={[0, -(textPadding * 0.25), 0]}
              fontSize={fontSize * 0.6}
              color={new THREE.Color(labelSettings.textColor || '#ffffff').multiplyScalar(0.7).getStyle()}
              anchorX="center"
              anchorY="top"
              maxWidth={maxWidth * 0.8}
              textAlign="center"
            >
              {metadataToShow}
            </Text>
          )}
          {metadataToShow && distanceInfo && (
            <Text
              position={[0, -(textPadding * 0.5), 0]}
              fontSize={fontSize * 0.5}
              color={new THREE.Color(labelSettings.textColor || '#ffffff').multiplyScalar(0.5).getStyle()}
              anchorX="center"
              anchorY="top"
              maxWidth={maxWidth * 0.8}
              textAlign="center"
            >
              {metadataToShow}
            </Text>
          )}
        </Billboard>
      )
    })
    // CLIENT-SIDE HIERARCHICAL LOD: Updated dependencies to use visibleNodes
  }, [visibleNodes, graphData.nodes, graphData.edges, labelPositions, settings?.visualisation?.graphs?.logseq?.labels, settings?.visualisation?.labels, normalizedSSSPResult])

  
  useEffect(() => {
    
    const debugSettings = settings?.system?.debug;
    if (debugSettings?.enableNodeDebug) {
      console.log('GraphManager: Component mounted', {
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
        console.log('GraphManager: Component unmounting');
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
    settings={settings}
    ssspResult={normalizedSSSPResult}
  />
) : (
        <instancedMesh
          ref={meshRef}
          args={[undefined, undefined, visibleNodes.length]}
          frustumCulled={false}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerMissed={() => {
            if (dragDataRef.current.isDragging) {
              handlePointerUp()
            }
          }}
          onDoubleClick={(event: ThreeEvent<MouseEvent>) => {
            // CLIENT-SIDE HIERARCHICAL LOD: Toggle expansion on double-click
            if (event.instanceId !== undefined && event.instanceId < visibleNodes.length) {
              const node = visibleNodes[event.instanceId];
              if (node) {
                const hierarchyNode = hierarchyMap.get(node.id);
                if (hierarchyNode && hierarchyNode.childIds.length > 0) {
                  expansionState.toggleExpansion(node.id);
                  logger.info(`Toggled expansion for "${node.label}" (${node.id}): ${
                    expansionState.isExpanded(node.id) ? 'expanded' : 'collapsed'
                  }, ${hierarchyNode.childIds.length} children`);
                } else {
                  logger.info(`Node "${node.label}" has no children to expand`);
                }
              }
            }
          }}
        >
          <sphereGeometry args={[0.5, 32, 32]} />
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
          settings={settings?.visualisation?.graphs?.logseq?.edges || settings?.visualisation?.edges || defaultEdgeSettings}
          edgeData={graphData.edges}
        />
      )}

      {}
      {particleGeometry && (
        <points 
          ref={(points) => {
            particleSystemRef.current = points;
            
            if (points) {
              
              if (!points.layers) {
                points.layers = new THREE.Layers();
              }
              points.layers.set(0); 
              points.layers.enable(1); 
              points.layers.disable(2); 
            }
          }}
          geometry={particleGeometry}
        >
          <pointsMaterial
            size={0.1}
            color={settings?.visualisation?.graphs?.logseq?.nodes?.baseColor || settings?.visualisation?.nodes?.baseColor || '#00ffff'}
            transparent
            opacity={0.3}
            vertexColors
            blending={THREE.NormalBlending}
            depthWrite={false}
          />
        </points>
      )}

      {}
      {NodeLabels}
    </>
  )
}

export default GraphManager