import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
import { useThree, useFrame, ThreeEvent } from '@react-three/fiber'
import { Text, Billboard } from '@react-three/drei'
import * as THREE from 'three'
import { graphDataManager, type GraphData, type Node as GraphNode } from '../managers/graphDataManager'
import { graphWorkerProxy } from '../managers/graphWorkerProxy'
import { createLogger, createErrorMetadata } from '../../../utils/logger'
import { debugState } from '../../../utils/clientDebugState'
import { useSettingsStore } from '../../../store/settingsStore'
import { BinaryNodeData, createBinaryNodeData } from '../../../types/binaryProtocol'
import { HologramNodeMaterial } from '../../../rendering/materials/HologramNodeMaterial'
import { FlowingEdges } from './FlowingEdges'
import { createEventHandlers } from './GraphManager_EventHandlers'
import { MetadataShapes } from './MetadataShapes'
import { NodeShaderToggle } from './NodeShaderToggle'
import { EdgeSettings } from '../../settings/config/settings'
import { registerNodeObject, unregisterNodeObject } from '../../visualisation/hooks/bloomRegistry'
import { useAnalyticsStore, useCurrentSSSPResult } from '../../analytics/store/analyticsStore'
// import { useBloomStrength } from '../contexts/BloomContext' // Removed - bloom managed via settings

const logger = createLogger('GraphManager')

// Enhanced position calculation with better distribution
const getPositionForNode = (node: GraphNode, index: number, totalNodes: number): [number, number, number] => {
  if (!node.position || (node.position.x === 0 && node.position.y === 0 && node.position.z === 0)) {
    // Golden angle for better distribution
    const goldenAngle = Math.PI * (3 - Math.sqrt(5))
    const theta = index * goldenAngle
    const y = 1 - (index / totalNodes) * 2
    const radius = Math.sqrt(1 - y * y)

    const scaleFactor = 15 + Math.random() * 5 // Vary radius for organic feel
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
      return new THREE.OctahedronGeometry(0.6, 0); // Folder = octahedron
    case 'file':
      return new THREE.BoxGeometry(0.8, 0.8, 0.8); // File = cube
    case 'concept':
      return new THREE.IcosahedronGeometry(0.5, 0); // Concept = icosahedron
    case 'todo':
      return new THREE.ConeGeometry(0.5, 1, 4); // Todo = pyramid
    case 'reference':
      return new THREE.TorusGeometry(0.5, 0.2, 8, 16); // Reference = torus
    default:
      return new THREE.SphereGeometry(0.5, 32, 32); // Default = sphere
  }
};

// Get node color based on type/metadata and SSSP visualization
const getNodeColor = (node: GraphNode, ssspResult?: any): THREE.Color => {
  // SSSP visualization takes priority when available
  if (ssspResult) {
    const distance = ssspResult.distances[node.id]
    
    // Source node - special bright cyan color
    if (node.id === ssspResult.sourceNodeId) {
      return new THREE.Color('#00FFFF') // Bright cyan for source
    }
    
    // Unreachable nodes - gray
    if (!isFinite(distance)) {
      return new THREE.Color('#666666') // Dark gray
    }
    
    // Reachable nodes - gradient from green to red based on distance
    const normalizedDistances = ssspResult.normalizedDistances || {}
    const normalizedDistance = normalizedDistances[node.id] || 0
    
    // Create gradient from green (close) to red (far)
    const red = Math.min(1, normalizedDistance * 1.2)
    const green = Math.min(1, (1 - normalizedDistance) * 1.2)
    const blue = 0.1 // Slight blue tint for depth
    
    return new THREE.Color(red, green, blue)
  }
  
  // Default type-based coloring when no SSSP result
  const typeColors: Record<string, string> = {
    'folder': '#FFD700',     // Gold
    'file': '#00CED1',       // Dark turquoise
    'function': '#FF6B6B',   // Coral
    'class': '#4ECDC4',      // Turquoise
    'variable': '#95E1D3',   // Mint
    'import': '#F38181',     // Light coral
    'export': '#AA96DA',     // Lavender
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

  // Scale based on connections (more connections = larger node)
  const connectionScale = 1 + Math.log(connectionCount + 1) * 0.3

  // Also scale based on node type importance
  const typeScale = getTypeImportance(node.metadata?.type)

  return baseSize * connectionScale * typeScale
}

// Get importance multiplier based on node type
const getTypeImportance = (nodeType?: string): number => {
  const importanceMap: Record<string, number> = {
    'folder': 1.5,      // Folders are important containers
    'function': 1.3,    // Functions are important code elements
    'class': 1.4,       // Classes are structural elements
    'file': 1.0,        // Files are baseline
    'variable': 0.8,    // Variables are smaller
    'import': 0.7,      // Imports are small
    'export': 0.9,      // Exports are medium
  }

  return importanceMap[nodeType || 'default'] || 1.0
}

interface GraphManagerProps {
  onDragStateChange?: (isDragging: boolean) => void;
}

const GraphManager: React.FC<GraphManagerProps> = ({ onDragStateChange }) => {
  // Only log once on mount, not every frame
  useEffect(() => {
    if (onDragStateChange) {
      console.log('GraphManager: onDragStateChange callback available');
    }
  }, []);
  const settings = useSettingsStore((state) => state.settings);
  // Handle both camelCase and snake_case field names from REST API
  const nodeBloomStrength = settings?.visualisation?.bloom?.node_bloom_strength ?? settings?.visualisation?.bloom?.nodeBloomStrength ?? 0.5;
  const edgeBloomStrength = settings?.visualisation?.bloom?.edge_bloom_strength ?? settings?.visualisation?.bloom?.edgeBloomStrength ?? 0.5;
  
  // SSSP visualization state
  const ssspResult = useCurrentSSSPResult();
  const normalizeDistances = useAnalyticsStore(state => state.normalizeDistances);
  const [normalizedSSSPResult, setNormalizedSSSPResult] = useState<any>(null);
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const materialRef = useRef<HologramNodeMaterial | null>(null)
  const particleSystemRef = useRef<THREE.Points>(null)

  // Memoized objects for performance
  const tempMatrix = useMemo(() => new THREE.Matrix4(), [])
  const tempPosition = useMemo(() => new THREE.Vector3(), [])
  const tempScale = useMemo(() => new THREE.Vector3(), [])
  const tempQuaternion = useMemo(() => new THREE.Quaternion(), [])
  const tempColor = useMemo(() => new THREE.Color(), [])

  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })
  const nodePositionsRef = useRef<Float32Array | null>(null)
  const [edgePoints, setEdgePoints] = useState<number[]>([])
  const [nodesAreAtOrigin, setNodesAreAtOrigin] = useState(false)
  // Remove duplicate settings declaration - already declared above
  const [forceUpdate, setForceUpdate] = useState(0)

  // Animation state
  const animationStateRef = useRef({
    time: 0,
    selectedNode: null as string | null,
    hoveredNode: null as string | null,
    pulsePhase: 0,
  })

  // Drag state (same as original)
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

  // Check if metadata shapes are enabled
  const logseqSettings = settings?.visualisation?.graphs?.logseq
  const nodeSettings = logseqSettings?.nodes || settings?.visualisation?.nodes
  const enableMetadataShape = nodeSettings?.enableMetadataShape ?? false

  // Create custom hologram material
  useEffect(() => {
    if (!materialRef.current) {
      materialRef.current = new HologramNodeMaterial({
        baseColor: '#0066ff', // Bright blue base
        emissiveColor: '#00ffff', // Cyan emissive
        opacity: settings?.visualisation?.graphs?.logseq?.nodes?.opacity ?? settings?.visualisation?.nodes?.opacity ?? 0.8,
        enableHologram: true, // Force enable to test
        glowStrength: 2.0 * (nodeBloomStrength || 1), // Multiply by bloom strength with fallback
        pulseSpeed: 1.0,
        hologramStrength: 0.8, // Strong hologram effect
        rimPower: 2.0, // Moderate rim lighting
      })

      // Keep nodes bright for postprocessing bloom
      ;(materialRef.current as any).toneMapped = false

      // Enable instance color support
      materialRef.current.defines = { ...materialRef.current.defines, USE_INSTANCING_COLOR: '' }
      materialRef.current.needsUpdate = true
    }
  }, [])
  
  // Set up bloom for nodes
  useEffect(() => {
    const obj = meshRef.current as any;
    if (obj) {
      // Make sure mesh is on default layer for raycasting
      obj.layers.set(0); // Default layer for raycasting
      obj.layers.enable(1); // Also enable bloom layer
      registerNodeObject(obj);
    }
    return () => {
      if (obj) unregisterNodeObject(obj);
    };
  }, [graphData.nodes.length]) // Re-run when nodes change
  
  // Update material settings for bloom strength changes
  useEffect(() => {
    if (materialRef.current) {
      // Update glow strength based on bloom slider
      const strength = nodeBloomStrength || 1;
      materialRef.current.updateHologramParams({
        glowStrength: strength * 2.0 // Scale the glow based on slider
      });
      // Also update the emissive intensity directly
      materialRef.current.uniforms.glowStrength.value = strength * 2.0;
      materialRef.current.needsUpdate = true;
    }
  }, [nodeBloomStrength]);
  
  // Update material settings for Logseq graph
  useEffect(() => {
    if (materialRef.current && settings?.visualisation) {
      // Use Logseq-specific settings, fallback to legacy nodes settings
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

  // Update normalized SSSP result when ssspResult changes
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

  // Function to update node colors with smooth transitions
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

  // Update colors when SSSP result changes
  useEffect(() => {
    updateNodeColors();
  }, [updateNodeColors]);

  // Initialize instance attributes
  useEffect(() => {
    if (meshRef.current && graphData.nodes.length > 0) {
      const mesh = meshRef.current
      mesh.count = graphData.nodes.length
      
      // Debug logging based on settings
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

      // CRITICAL: Initialize instance matrices immediately
      const tempMatrix = new THREE.Matrix4();
      const nodeSize = settings?.visualisation?.graphs?.logseq?.nodes?.nodeSize || 0.5;
      const BASE_SPHERE_RADIUS = 0.5;
      const baseScale = nodeSize / BASE_SPHERE_RADIUS;

      graphData.nodes.forEach((node, i) => {
        // CRITICAL: Set initial instance matrix for each node
        const nodeScale = getNodeScale(node, graphData.edges) * baseScale;
        tempMatrix.makeScale(nodeScale, nodeScale, nodeScale);
        
        // Use node's initial position or spread them out
        const angle = (i / graphData.nodes.length) * Math.PI * 2;
        const radius = 10;
        tempMatrix.setPosition(
          Math.cos(angle) * radius,
          (Math.random() - 0.5) * 5,
          Math.sin(angle) * radius
        );
        
        mesh.setMatrixAt(i, tempMatrix);
      })
      
      // CRITICAL: Mark instance matrix as needing update
      mesh.instanceMatrix.needsUpdate = true;
      
      // IMPORTANT: Update bounding sphere for proper raycasting
      mesh.computeBoundingSphere();

      // Force material update
      if (materialRef.current) {
        materialRef.current.needsUpdate = true
      }

    }
  }, [graphData, normalizedSSSPResult])

  // Pass settings to worker whenever they change
  useEffect(() => {
    graphWorkerProxy.updateSettings(settings);
  }, [settings]);

  // Animation loop with physics updates
  useFrame(async (state, delta) => {
    animationStateRef.current.time = state.clock.elapsedTime
    
    // Debug: Log first frame and periodic updates
    const debugSettings = settings?.system?.debug;
    if (debugSettings?.enablePhysicsDebug && debugState.isEnabled()) {
      const frameCount = Math.floor(state.clock.elapsedTime * 60);
      if (frameCount === 1 || frameCount % 300 === 0) { // Log first frame and every 5 seconds
        logger.debug('Physics frame update', {
          time: state.clock.elapsedTime,
          delta,
          nodeCount: graphData.nodes.length,
          hasPositions: !!nodePositionsRef.current
        });
      }
    }

    // Update material time
    if (materialRef.current) {
      materialRef.current.updateTime(animationStateRef.current.time)
    }

    // Get smooth positions from physics worker
    if ((meshRef.current || enableMetadataShape) && graphData.nodes.length > 0) {
      const positions = await graphWorkerProxy.tick(delta);
      nodePositionsRef.current = positions;

      if (positions) {
        // Update node positions from physics
        const logseqSettings = settings?.visualisation?.graphs?.logseq;
        const nodeSettings = logseqSettings?.nodes || settings?.visualisation?.nodes;
        const nodeSize = nodeSettings?.nodeSize || 0.5;
        const BASE_SPHERE_RADIUS = 0.5;
        const baseScale = nodeSize / BASE_SPHERE_RADIUS;

        // Only update instance matrix if using standard mesh (not metadata shapes)
        if (meshRef.current && !enableMetadataShape) {
          for (let i = 0; i < graphData.nodes.length; i++) {
            const i3 = i * 3;
            const node = graphData.nodes[i];
            let nodeScale = getNodeScale(node, graphData.edges) * baseScale;
            
            // Add pulsing effect for source node in SSSP visualization
            if (normalizedSSSPResult && node.id === normalizedSSSPResult.sourceNodeId) {
              const pulseScale = 1 + Math.sin(animationStateRef.current.time * 2) * 0.3;
              nodeScale *= pulseScale;
            }
            
            tempMatrix.makeScale(nodeScale, nodeScale, nodeScale);
            tempMatrix.setPosition(positions[i3], positions[i3 + 1], positions[i3 + 2]);
            meshRef.current.setMatrixAt(i, tempMatrix);
          }
          meshRef.current.instanceMatrix.needsUpdate = true;
          // Update bounding sphere for raycasting to work properly
          meshRef.current.computeBoundingSphere();
        }

        // Update edge points based on new positions with endpoint offset
        const newEdgePoints: number[] = [];
        graphData.edges.forEach(edge => {
          const sourceNodeIndex = graphData.nodes.findIndex(n => n.id === edge.source);
          const targetNodeIndex = graphData.nodes.findIndex(n => n.id === edge.target);
          if (sourceNodeIndex !== -1 && targetNodeIndex !== -1) {
            const i3s = sourceNodeIndex * 3;
            const i3t = targetNodeIndex * 3;

            // Get node positions
            const sourcePos = new THREE.Vector3(positions[i3s], positions[i3s + 1], positions[i3s + 2]);
            const targetPos = new THREE.Vector3(positions[i3t], positions[i3t + 1], positions[i3t + 2]);

            // Calculate edge direction
            const direction = new THREE.Vector3().subVectors(targetPos, sourcePos);
            const edgeLength = direction.length();

            if (edgeLength > 0) {
              direction.normalize();

              // Calculate node radii based on scale
              const sourceNode = graphData.nodes[sourceNodeIndex];
              const targetNode = graphData.nodes[targetNodeIndex];
              const logseqSettings = settings?.visualisation?.graphs?.logseq;
              const nodeSettings = logseqSettings?.nodes || settings?.visualisation?.nodes;
              const sourceRadius = getNodeScale(sourceNode, graphData.edges) * (nodeSettings?.nodeSize || 0.5);
              const targetRadius = getNodeScale(targetNode, graphData.edges) * (nodeSettings?.nodeSize || 0.5);

              // Offset endpoints to stop before node surfaces (plus small gap)
              const offsetSource = new THREE.Vector3().addVectors(sourcePos, direction.clone().multiplyScalar(sourceRadius + 0.1));
              const offsetTarget = new THREE.Vector3().subVectors(targetPos, direction.clone().multiplyScalar(targetRadius + 0.1));

              // Only add edge if there's still visible length after offset
              if (offsetSource.distanceTo(offsetTarget) > 0.2) {
                newEdgePoints.push(offsetSource.x, offsetSource.y, offsetSource.z);
                newEdgePoints.push(offsetTarget.x, offsetTarget.y, offsetTarget.z);
              }
            }
          }
        });
        setEdgePoints(newEdgePoints);

        // Update label positions in real-time
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

    // Animate particles - disabled
    // if (particleSystemRef.current) {
    //   particleSystemRef.current.rotation.y = animationStateRef.current.time * 0.05
    //   const positions = particleSystemRef.current.geometry.attributes.position.array as Float32Array
    //
    //   for (let i = 0; i < positions.length; i += 3) {
    //     positions[i + 1] += Math.sin(animationStateRef.current.time + i) * 0.01
    //   }
    //
    //   particleSystemRef.current.geometry.attributes.position.needsUpdate = true
    // }

    // Pulse selected nodes
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

  // Graph data subscription with enhanced error handling and diagnostics
  useEffect(() => {
    
    const handleGraphUpdate = (data: GraphData) => {
      // Debug logging if enabled
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

      // Validate data before processing
      if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.edges)) {
        return;
      }

      // Ensure nodes have valid positions
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

      // Update edge points
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

    // Get initial data and update worker with enhanced error handling
    graphDataManager.getGraphData().then((data) => {
      // Debug logging if enabled
      const debugSettings = settings?.system?.debug;
      if (debugSettings?.enableNodeDebug) {
        console.log('GraphManager: Initial graph data loaded', {
          nodeCount: data.nodes.length,
          edgeCount: data.edges.length
        });
      }
      handleGraphUpdate(data)
      // Ensure worker has the data
      return graphWorkerProxy.setGraphData(data)
    }).then(() => {
    }).catch((error) => {
      // Fallback: create sample data
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

  // Event handlers
  const { handlePointerDown, handlePointerMove, handlePointerUp } = createEventHandlers(
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

  // Create ambient particles (disabled for now to avoid white blocks)
  const particleGeometry = useMemo(() => {
    return null // Disable particles that were causing white blocks
  }, [])

  // Node labels with dynamic positioning
  const [labelPositions, setLabelPositions] = useState<Array<{x: number, y: number, z: number}>>([])

  // Update label positions from physics
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

  // Node labels (enhanced version) - using physics positions
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
      if (!labelSettings?.enableLabels || graphData.nodes.length === 0) return null;

    return graphData.nodes.map((node, index) => {
      // Use physics position if available, otherwise fallback to node position
      const physicsPos = labelPositions[index]
      const position = physicsPos || node.position || { x: 0, y: 0, z: 0 }
      const scale = getNodeScale(node, graphData.edges)
      const textPadding = labelSettings.textPadding ?? 0.6;
      const labelOffsetY = scale * 1.5 + textPadding; // Use textPadding setting

      // Determine which piece of metadata to show
      let metadataToShow = null;
      let distanceInfo = null;
      
      // SSSP distance information takes priority
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

      // Get max width from settings
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
  }, [graphData.nodes, graphData.edges, labelPositions, settings?.visualisation?.graphs?.logseq?.labels, settings?.visualisation?.labels, normalizedSSSPResult])

  // Debug logging for render - only log once on mount/unmount
  useEffect(() => {
    // Debug logging if enabled
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
  }, []); // Empty deps = only on mount/unmount

  return (
    <>
      {/* Node shader toggle - controls animation effects */}
      <NodeShaderToggle materialRef={materialRef} />
      
      
      {/* Render nodes based on metadata shape setting */}
      {enableMetadataShape ? (
  <MetadataShapes
    nodes={graphData.nodes}
    nodePositions={nodePositionsRef.current}
    onNodeClick={(nodeId, event) => {
      const nodeIndex = graphData.nodes.findIndex(n => n.id === nodeId);
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
          args={[undefined, undefined, graphData.nodes.length]}
          frustumCulled={false}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerMissed={() => {
            if (dragDataRef.current.isDragging) {
              handlePointerUp()
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

      {/* Enhanced flowing edges */}
      {edgePoints.length > 0 && (
        <FlowingEdges
          points={edgePoints}
          settings={settings?.visualisation?.graphs?.logseq?.edges || settings?.visualisation?.edges || defaultEdgeSettings}
          edgeData={graphData.edges}
        />
      )}

      {/* Ambient particle system - disabled to prevent white blocks */}
      {particleGeometry && (
        <points 
          ref={(points) => {
            particleSystemRef.current = points;
            // Ensure particles are on Layer 1 (graph bloom) only
            if (points) {
              // Initialize layers if not present
              if (!points.layers) {
                points.layers = new THREE.Layers();
              }
              points.layers.set(0); // Base layer for rendering
              points.layers.enable(1); // Layer 1 for graph bloom
              points.layers.disable(2); // Explicitly disable Layer 2 (environment glow)
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

      {/* Enhanced node labels */}
      {NodeLabels}
    </>
  )
}

export default GraphManager