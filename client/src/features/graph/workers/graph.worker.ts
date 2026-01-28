

import { expose } from 'comlink';
import { BinaryNodeData, parseBinaryNodeData, createBinaryNodeData, Vec3 } from '../../../types/binaryProtocol';

export interface Node {
  id: string;
  label: string;
  position: {
    x: number;
    y: number;
    z: number;
  };
  metadata?: Record<string, any>;
}

export interface Edge {
  id: string;
  source: string;
  target: string;
  label?: string;
  weight?: number;
  metadata?: Record<string, any>;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
}


async function decompressZlib(compressedData: ArrayBuffer): Promise<ArrayBuffer> {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const cs = new DecompressionStream('deflate-raw');
      const writer = cs.writable.getWriter();
      writer.write(new Uint8Array(compressedData.slice(2))); 
      writer.close();

      const output = [];
      const reader = cs.readable.getReader();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        output.push(value);
      }

      const totalLength = output.reduce((acc, arr) => acc + arr.length, 0);
      const result = new Uint8Array(totalLength);
      let offset = 0;

      for (const arr of output) {
        result.set(arr, offset);
        offset += arr.length;
      }

      return result.buffer;
    } catch (error) {
      console.error('Worker decompression failed:', error);
      throw error;
    }
  }
  throw new Error('DecompressionStream not available');
}


function isZlibCompressed(data: ArrayBuffer): boolean {
  if (data.byteLength < 2) return false;
  const view = new Uint8Array(data);
  return view[0] === 0x78 && [0x01, 0x5E, 0x9C, 0xDA].includes(view[1]);
}


// Force-directed physics settings for client-side simulation
interface ForcePhysicsSettings {
  repulsionStrength: number;     // Coulomb-like repulsion between all nodes
  attractionStrength: number;    // Spring attraction along edges
  centerGravity: number;         // Gentle pull toward center to prevent drift
  damping: number;               // Velocity damping (0-1)
  maxVelocity: number;           // Speed limit
  idealEdgeLength: number;       // Target spring rest length
  theta: number;                 // Barnes-Hut approximation threshold (0.5-1.0)
  enabled: boolean;              // Whether physics is running
  alpha: number;                 // Simulation "temperature" (decays over time)
  alphaDecay: number;            // How fast alpha decays
  alphaMin: number;              // Stop when alpha reaches this
  // Semantic clustering
  clusterStrength: number;       // Force pulling nodes of same domain together
  enableClustering: boolean;     // Enable domain-based clustering
}

class GraphWorker {
  private graphData: GraphData = { nodes: [], edges: [] };
  private nodeIdMap: Map<string, number> = new Map();
  private reverseNodeIdMap: Map<number, string> = new Map();
  private graphType: 'logseq' | 'visionflow' = 'logseq';


  private nodeIndexMap: Map<string, number> = new Map();

  private currentPositions: Float32Array | null = null;
  private targetPositions: Float32Array | null = null;
  private velocities: Float32Array | null = null;
  private pinnedNodeIds: Set<number> = new Set();
  private physicsSettings = {
    springStrength: 0.001,
    damping: 0.98,
    maxVelocity: 0.5,
    updateThreshold: 0.05,
  };


  private useServerPhysics: boolean = true;
  private positionBuffer: SharedArrayBuffer | null = null;
  private positionView: Float32Array | null = null;


  private frameCount: number = 0;
  private binaryUpdateCount: number = 0;
  private lastBinaryUpdate: number = 0;

  // Force-directed physics for client-side simulation (VisionFlow)
  private forcePhysics: ForcePhysicsSettings = {
    repulsionStrength: 500,       // Node repulsion force
    attractionStrength: 0.05,     // Edge spring force
    centerGravity: 0.01,          // Gentle centering
    damping: 0.85,                // Velocity decay
    maxVelocity: 5.0,             // Max speed
    idealEdgeLength: 30,          // Target edge length
    theta: 0.8,                   // Barnes-Hut threshold
    enabled: true,                // Physics on by default
    alpha: 1.0,                   // Initial temperature
    alphaDecay: 0.0228,           // ~300 iterations to cool
    alphaMin: 0.001,              // Stop threshold
    clusterStrength: 0.3,         // Domain clustering force
    enableClustering: true,       // Enable semantic clustering
  };

  // Edge lookup for O(1) neighbor access
  private edgeSourceMap: Map<string, string[]> = new Map();
  private edgeTargetMap: Map<string, string[]> = new Map();

  // Domain clustering - maps domain to node indices
  private domainClusters: Map<string, number[]> = new Map();
  private domainCenters: Map<string, { x: number; y: number; z: number }> = new Map();

  
  async initialize(): Promise<void> {
    console.log('GraphWorker: Initialize method called');
    return Promise.resolve();
  }
  
  
  async setGraphType(type: 'logseq' | 'visionflow'): Promise<void> {
    this.graphType = type;
    // VisionFlow uses client-side physics since it doesn't receive server binary updates
    if (type === 'visionflow') {
      this.useServerPhysics = false;
      this.forcePhysics.enabled = true;
      this.forcePhysics.alpha = 1.0; // Reset simulation temperature
      console.log(`GraphWorker: Graph type set to ${type} - using CLIENT-SIDE force-directed physics`);
    } else {
      // Logseq graphs receive server physics via binary protocol
      this.useServerPhysics = true;
      this.forcePhysics.enabled = false;
      console.log(`GraphWorker: Graph type set to ${type} - using SERVER physics`);
    }
  }


  async setGraphData(data: GraphData): Promise<void> {
    this.graphData = {
      nodes: data.nodes.map(node => this.ensureNodeHasValidPosition(node)),
      edges: data.edges
    };


    this.nodeIdMap.clear();
    this.reverseNodeIdMap.clear();
    this.nodeIndexMap.clear();
    this.graphData.nodes.forEach((node, index) => {
        const numericId = parseInt(node.id, 10);
        if (!isNaN(numericId) && numericId >= 0 && numericId <= 0xFFFFFFFF) {
            this.nodeIdMap.set(node.id, numericId);
            this.reverseNodeIdMap.set(numericId, node.id);
        } else {
            const mappedId = index + 1;
            this.nodeIdMap.set(node.id, mappedId);
            this.reverseNodeIdMap.set(mappedId, node.id);
        }
        this.nodeIndexMap.set(node.id, index);
    });

    // Build edge adjacency maps for O(1) neighbor lookup
    this.edgeSourceMap.clear();
    this.edgeTargetMap.clear();
    for (const edge of data.edges) {
      // Source -> targets
      if (!this.edgeSourceMap.has(edge.source)) {
        this.edgeSourceMap.set(edge.source, []);
      }
      this.edgeSourceMap.get(edge.source)!.push(edge.target);

      // Target -> sources (for bidirectional edge springs)
      if (!this.edgeTargetMap.has(edge.target)) {
        this.edgeTargetMap.set(edge.target, []);
      }
      this.edgeTargetMap.get(edge.target)!.push(edge.source);
    }

    // Build domain clusters for semantic grouping
    this.domainClusters.clear();
    this.domainCenters.clear();
    this.graphData.nodes.forEach((node, index) => {
      const domain = node.metadata?.source_domain || node.metadata?.domain || 'default';
      if (!this.domainClusters.has(domain)) {
        this.domainClusters.set(domain, []);
      }
      this.domainClusters.get(domain)!.push(index);
    });


    const nodeCount = data.nodes.length;
    this.currentPositions = new Float32Array(nodeCount * 3);
    this.targetPositions = new Float32Array(nodeCount * 3);
    this.velocities = new Float32Array(nodeCount * 3).fill(0);

    data.nodes.forEach((node, index) => {
      const i3 = index * 3;
      const pos = node.position;
      this.currentPositions![i3] = pos.x;
      this.currentPositions![i3 + 1] = pos.y;
      this.currentPositions![i3 + 2] = pos.z;

      this.targetPositions![i3] = pos.x;
      this.targetPositions![i3 + 1] = pos.y;
      this.targetPositions![i3 + 2] = pos.z;
    });

    // Reset physics simulation when new data arrives
    if (this.graphType === 'visionflow') {
      this.forcePhysics.alpha = 1.0; // Reheat simulation
      console.log(`GraphWorker: VisionFlow graph initialized with ${nodeCount} nodes, ${data.edges.length} edges, ${this.domainClusters.size} domains`);
    } else {
      console.log(`GraphWorker: Initialized ${this.graphType} graph with ${this.graphData.nodes.length} nodes`);
    }
  }

  
  async setupSharedPositions(buffer: SharedArrayBuffer): Promise<void> {
    this.positionBuffer = buffer;
    this.positionView = new Float32Array(buffer);
    console.log(`GraphWorker: SharedArrayBuffer set up with ${buffer.byteLength} bytes`);
  }

  
  async updateSettings(settings: any): Promise<void> {
    
    const graphSettings = settings?.visualisation?.graphs?.[this.graphType]?.physics || 
                         settings?.visualisation?.physics;
    
    this.physicsSettings = {
      springStrength: graphSettings?.springStrength ?? 0.001,
      damping: graphSettings?.damping ?? 0.98,
      maxVelocity: graphSettings?.maxVelocity ?? 0.5,
      updateThreshold: graphSettings?.updateThreshold ?? 0.05
    };
  }

  
  async processBinaryData(data: ArrayBuffer): Promise<Float32Array> { 
    
    if (this.graphType !== 'logseq') {
      console.log(`GraphWorker: Skipping binary data processing for ${this.graphType} graph`);
      return new Float32Array(0);
    }

    
    if (!this.useServerPhysics) {
      this.useServerPhysics = true;
      console.log('GraphWorker: Auto-enabled server physics mode due to binary position updates');
    }
    
    
    this.binaryUpdateCount = (this.binaryUpdateCount || 0) + 1;
    this.lastBinaryUpdate = Date.now();

    
    if (isZlibCompressed(data)) {
      data = await decompressZlib(data);
    }
    const nodeUpdates = parseBinaryNodeData(data);

    
    const positionArray = new Float32Array(nodeUpdates.length * 4);

    nodeUpdates.forEach((update, index) => {
      const stringNodeId = this.reverseNodeIdMap.get(update.nodeId);
      if (stringNodeId) {
        const nodeIndex = this.nodeIndexMap.get(stringNodeId);
        if (nodeIndex !== undefined && !this.pinnedNodeIds.has(update.nodeId)) {

          const i3 = nodeIndex * 3;
          this.targetPositions![i3] = update.position.x;
          this.targetPositions![i3 + 1] = update.position.y;
          this.targetPositions![i3 + 2] = update.position.z;

        }
      }

      const arrayOffset = index * 4;
      positionArray[arrayOffset] = update.nodeId;
      positionArray[arrayOffset + 1] = update.position.x;
      positionArray[arrayOffset + 2] = update.position.y;
      positionArray[arrayOffset + 3] = update.position.z;
    });

    
    return positionArray;
  }

  
  async getGraphData(): Promise<GraphData> {
    return this.graphData;
  }


  async updateNode(node: Node): Promise<void> {
    const existingIndex = this.nodeIndexMap.get(node.id);

    if (existingIndex !== undefined) {
      this.graphData.nodes[existingIndex] = { ...this.graphData.nodes[existingIndex], ...node };
    } else {
      const newIndex = this.graphData.nodes.length;
      this.graphData.nodes.push(this.ensureNodeHasValidPosition(node));


      const numericId = parseInt(node.id, 10);
      if (!isNaN(numericId)) {
        this.nodeIdMap.set(node.id, numericId);
        this.reverseNodeIdMap.set(numericId, node.id);
      } else {
        const mappedId = newIndex + 1;
        this.nodeIdMap.set(node.id, mappedId);
        this.reverseNodeIdMap.set(mappedId, node.id);
      }
      this.nodeIndexMap.set(node.id, newIndex);
    }
  }


  async removeNode(nodeId: string): Promise<void> {
    const numericId = this.nodeIdMap.get(nodeId);

    this.graphData.nodes = this.graphData.nodes.filter(node => node.id !== nodeId);
    this.graphData.edges = this.graphData.edges.filter(
      edge => edge.source !== nodeId && edge.target !== nodeId
    );

    if (numericId !== undefined) {
      this.nodeIdMap.delete(nodeId);
      this.reverseNodeIdMap.delete(numericId);
    }
    this.nodeIndexMap.delete(nodeId);

    this.nodeIndexMap.clear();
    this.graphData.nodes.forEach((node, index) => {
      this.nodeIndexMap.set(node.id, index);
    });
  }

  
  async createBinaryData(nodes: BinaryNodeData[]): Promise<ArrayBuffer> {
    return createBinaryNodeData(nodes);
  }

  private ensureNodeHasValidPosition(node: Node): Node {
    if (!node.position) {
      return { ...node, position: { x: 0, y: 0, z: 0 } };
    }

    return {
      ...node,
      position: {
        x: typeof node.position.x === 'number' ? node.position.x : 0,
        y: typeof node.position.y === 'number' ? node.position.y : 0,
        z: typeof node.position.z === 'number' ? node.position.z : 0
      }
    };
  }

  
  async pinNode(nodeId: number): Promise<void> { this.pinnedNodeIds.add(nodeId); }
  async unpinNode(nodeId: number): Promise<void> { this.pinnedNodeIds.delete(nodeId); }

  
  async setUseServerPhysics(useServer: boolean): Promise<void> {
    this.useServerPhysics = useServer;
    console.log(`GraphWorker: Physics mode set to ${useServer ? 'server' : 'local'}`);
  }
  
  
  async getPhysicsMode(): Promise<boolean> {
    return this.useServerPhysics;
  }
  
  async updateUserDrivenNodePosition(nodeId: number, position: Vec3): Promise<void> {
    const stringNodeId = this.reverseNodeIdMap.get(nodeId);
    if (stringNodeId) {
      const nodeIndex = this.nodeIndexMap.get(stringNodeId);
      if (nodeIndex !== undefined) {
        const i3 = nodeIndex * 3;

        this.currentPositions![i3] = position.x;
        this.currentPositions![i3 + 1] = position.y;
        this.currentPositions![i3 + 2] = position.z;
        this.targetPositions![i3] = position.x;
        this.targetPositions![i3 + 1] = position.y;
        this.targetPositions![i3 + 2] = position.z;

        this.velocities!.fill(0, i3, i3 + 3);
      }
    }
  }

  
  async tick(deltaTime: number): Promise<Float32Array> {
    if (!this.currentPositions || !this.targetPositions || !this.velocities) {
      return new Float32Array(0);
    }

    
    const dt = Math.min(deltaTime, 0.016); 


    this.frameCount = (this.frameCount || 0) + 1;
    // Performance: Removed per-frame logging - use DEBUG_PHYSICS env var if needed
    
    
    if (this.useServerPhysics) {
      
      let hasAnyMovement = false;
      for (let i = 0; i < this.graphData.nodes.length && !hasAnyMovement; i++) {
        const i3 = i * 3;
        const dx = Math.abs(this.targetPositions[i3] - this.currentPositions[i3]);
        const dy = Math.abs(this.targetPositions[i3 + 1] - this.currentPositions[i3 + 1]);
        const dz = Math.abs(this.targetPositions[i3 + 2] - this.currentPositions[i3 + 2]);
        if (dx > 0.001 || dy > 0.001 || dz > 0.001) {
          hasAnyMovement = true;
        }
      }
      

      if (!hasAnyMovement) {
        // Performance: Removed per-frame logging
        return this.currentPositions;
      }
      



      const dtSeconds = deltaTime / 1000;
      const lerpFactor = 1 - Math.pow(0.001, dtSeconds); 
      
      
      let totalMovement = 0;
      
      // Performance: Removed interpolation logging - use DEBUG_PHYSICS if needed
      
      for (let i = 0; i < this.graphData.nodes.length; i++) {
        const i3 = i * 3;
        
        
        const nodeId = this.nodeIdMap.get(this.graphData.nodes[i].id);
        if (nodeId !== undefined && this.pinnedNodeIds.has(nodeId)) {
          
          continue;
        }
        
        
        const dx = this.targetPositions[i3] - this.currentPositions[i3];
        const dy = this.targetPositions[i3 + 1] - this.currentPositions[i3 + 1];
        const dz = this.targetPositions[i3 + 2] - this.currentPositions[i3 + 2];
        const distanceSq = dx * dx + dy * dy + dz * dz;
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        const snapThreshold = 5.0; 
        if (distanceSq < snapThreshold * snapThreshold) {
          
          const positionChanged = Math.abs(this.currentPositions[i3] - this.targetPositions[i3]) > 0.01 ||
                                 Math.abs(this.currentPositions[i3 + 1] - this.targetPositions[i3 + 1]) > 0.01 ||
                                 Math.abs(this.currentPositions[i3 + 2] - this.targetPositions[i3 + 2]) > 0.01;

          if (positionChanged) {
            totalMovement += Math.sqrt(distanceSq);
            this.currentPositions[i3] = this.targetPositions[i3];
            this.currentPositions[i3 + 1] = this.targetPositions[i3 + 1];
            this.currentPositions[i3 + 2] = this.targetPositions[i3 + 2];
          }
          
          if (this.velocities) {
            this.velocities[i3] = 0;
            this.velocities[i3 + 1] = 0;
            this.velocities[i3 + 2] = 0;
          }
        } else {
          
          const moveX = dx * lerpFactor;
          const moveY = dy * lerpFactor;
          const moveZ = dz * lerpFactor;

          totalMovement += Math.sqrt(moveX * moveX + moveY * moveY + moveZ * moveZ);

          this.currentPositions[i3] += moveX;
          this.currentPositions[i3 + 1] += moveY;
          this.currentPositions[i3 + 2] += moveZ;
        }
        
      }
      
      // Performance: Removed movement logging
      
      return this.currentPositions;
    }


    const { springStrength, damping, maxVelocity, updateThreshold } = this.physicsSettings;

    for (let i = 0; i < this.graphData.nodes.length; i++) {
      const numericId = this.nodeIdMap.get(this.graphData.nodes[i].id)!;
      if (this.pinnedNodeIds.has(numericId)) continue; 

      const i3 = i * 3;

      const dx = this.targetPositions[i3] - this.currentPositions[i3];
      const dy = this.targetPositions[i3 + 1] - this.currentPositions[i3 + 1];
      const dz = this.targetPositions[i3 + 2] - this.currentPositions[i3 + 2];

      const distSq = dx * dx + dy * dy + dz * dz;

      
      if (distSq < updateThreshold * updateThreshold) {
        this.velocities.fill(0, i3, i3 + 3); 
        continue;
      }

      
      let ax = dx * springStrength;
      let ay = dy * springStrength;
      let az = dz * springStrength;

      
      this.velocities[i3] += ax * dt;
      this.velocities[i3 + 1] += ay * dt;
      this.velocities[i3 + 2] += az * dt;

      
      this.velocities[i3] *= damping;
      this.velocities[i3 + 1] *= damping;
      this.velocities[i3 + 2] *= damping;

      
      const currentVelSq = this.velocities[i3] * this.velocities[i3] + 
                          this.velocities[i3 + 1] * this.velocities[i3 + 1] + 
                          this.velocities[i3 + 2] * this.velocities[i3 + 2];
      if (currentVelSq > maxVelocity * maxVelocity) {
        const scale = maxVelocity / Math.sqrt(currentVelSq);
        this.velocities[i3] *= scale;
        this.velocities[i3 + 1] *= scale;
        this.velocities[i3 + 2] *= scale;
      }

      
      this.currentPositions[i3] += this.velocities[i3] * dt;
      this.currentPositions[i3 + 1] += this.velocities[i3 + 1] * dt;
      this.currentPositions[i3 + 2] += this.velocities[i3 + 2] * dt;
    }

    
    return this.currentPositions;
  }
}

// Expose the worker API using Comlink
const worker = new GraphWorker();
expose(worker);

export type GraphWorkerType = GraphWorker;