

import { expose } from 'comlink';
import { BinaryNodeData, parseBinaryNodeData, createBinaryNodeData, Vec3 } from '../../../types/binaryProtocol';
import { stringToU32 } from '../../../types/idMapping';

// Worker-safe logger (createLogger depends on localStorage/window which are unavailable in Workers)
// Only warn/error by default; set self.__WORKER_DEBUG = true in devtools to enable info/debug
const workerLogger = {
  info: (...args: unknown[]) => { if ((self as any).__WORKER_DEBUG) console.log('[GraphWorker]', ...args); },
  warn: (...args: unknown[]) => console.warn('[GraphWorker]', ...args),
  error: (...args: unknown[]) => console.error('[GraphWorker]', ...args),
  debug: (...args: unknown[]) => { if ((self as any).__WORKER_DEBUG) console.debug('[GraphWorker]', ...args); },
};

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
      workerLogger.error('Decompression failed:', error);
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


// Force-directed physics settings — retained for API compatibility.
// Client-side force simulation is REMOVED: the server (Rust/CUDA GPU physics)
// is the single source of truth for all graph types. The client only performs
// optimistic interpolation/tweening toward server-provided target positions.
export interface ForcePhysicsSettings {
  repulsionStrength: number;
  attractionStrength: number;
  centerGravity: number;
  damping: number;
  maxVelocity: number;
  idealEdgeLength: number;
  theta: number;
  enabled: boolean;
  alpha: number;
  alphaDecay: number;
  alphaMin: number;
  clusterStrength: number;
  enableClustering: boolean;
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


  // Server physics is ALWAYS authoritative — all graph types use server positions.
  // This flag is kept for API compatibility but always returns true.
  private useServerPhysics: boolean = true;
  private positionBuffer: SharedArrayBuffer | null = null;
  private positionView: Float32Array | null = null;


  private frameCount: number = 0;
  private binaryUpdateCount: number = 0;
  private lastBinaryUpdate: number = 0;

  // Retained for API compatibility — client-side force simulation is removed.
  // Physics settings are now sent to the server via REST API.
  private forcePhysics: ForcePhysicsSettings = {
    repulsionStrength: 500,
    attractionStrength: 0.05,
    centerGravity: 0.01,
    damping: 0.85,
    maxVelocity: 5.0,
    idealEdgeLength: 30,
    theta: 0.8,
    enabled: true,
    alpha: 1.0,
    alphaDecay: 0.0228,
    alphaMin: 0.001,
    clusterStrength: 0.3,
    enableClustering: true,
  };

  // Idempotency guard: skip updateSettings when physics values haven't changed
  private _lastPhysicsKey: string = '';

  // Edge lookup for O(1) neighbor access (kept for graph structure queries)
  private edgeSourceMap: Map<string, string[]> = new Map();
  private edgeTargetMap: Map<string, string[]> = new Map();

  // Pre-allocated buffer for binary position output (reused every processBinaryData call)
  private binaryOutputBuffer: Float32Array | null = null;
  private binaryOutputBufferSize: number = 0;

  
  async initialize(): Promise<void> {
    workerLogger.info('Initialize method called');
    return Promise.resolve();
  }
  
  
  async setGraphType(type: 'logseq' | 'visionflow'): Promise<void> {
    this.graphType = type;
    // All graph types use server-authoritative physics.
    // The server (Rust/CUDA GPU) is the single source of truth for positions.
    // Client only performs optimistic tweening toward server targets.
    this.useServerPhysics = true;
    workerLogger.info(`Graph type set to ${type} - using SERVER-AUTHORITATIVE physics (single source of truth)`);
  }


  async setGraphData(data: GraphData): Promise<void> {
    this.graphData = {
      nodes: data.nodes.map(node => this.ensureNodeHasValidPosition(node)),
      edges: data.edges
    };

    // Capture old state BEFORE clearing maps — needed for position preservation.
    // The nodeIndexMap maps nodeId → array index in the OLD position buffers.
    const nodeCount = data.nodes.length;
    const oldCurrentPos = this.currentPositions;
    const oldTargetPos = this.targetPositions;
    const oldNodeIndexMap = new Map(this.nodeIndexMap);

    this.nodeIdMap.clear();
    this.reverseNodeIdMap.clear();
    this.nodeIndexMap.clear();
    this.graphData.nodes.forEach((node, index) => {
        // Normalize node ID to string for consistent Map lookups.
        // Edge source/target are always strings — keys must match.
        const nodeId = String(node.id);
        node.id = nodeId;
        const numericId = parseInt(nodeId, 10);
        if (!isNaN(numericId) && numericId >= 0 && numericId <= 0xFFFFFFFF) {
            this.nodeIdMap.set(nodeId, numericId);
            this.reverseNodeIdMap.set(numericId, nodeId);
        } else {
            let mappedId = stringToU32(nodeId);
            while (this.reverseNodeIdMap.has(mappedId) && this.reverseNodeIdMap.get(mappedId) !== nodeId) {
              mappedId = (mappedId + 1) >>> 0;
            }
            this.nodeIdMap.set(nodeId, mappedId);
            this.reverseNodeIdMap.set(mappedId, nodeId);
        }
        this.nodeIndexMap.set(nodeId, index);
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

    // Preserve positions for nodes that already exist (prevents reset on
    // initialGraphLoad / filter-update / reconnect). Only allocate fresh
    // positions for genuinely new nodes.

    const newCurrentPositions = new Float32Array(nodeCount * 3);
    const newTargetPositions = new Float32Array(nodeCount * 3);
    const newVelocities = new Float32Array(nodeCount * 3);

    let preservedCount = 0;
    this.graphData.nodes.forEach((node, index) => {
      const i3 = index * 3;
      const oldIndex = oldNodeIndexMap.get(String(node.id));

      if (oldIndex !== undefined && oldCurrentPos && oldCurrentPos.length > oldIndex * 3 + 2) {
        // Existing node — keep its interpolated position
        const oi3 = oldIndex * 3;
        newCurrentPositions[i3] = oldCurrentPos[oi3];
        newCurrentPositions[i3 + 1] = oldCurrentPos[oi3 + 1];
        newCurrentPositions[i3 + 2] = oldCurrentPos[oi3 + 2];

        if (oldTargetPos && oldTargetPos.length > oi3 + 2) {
          newTargetPositions[i3] = oldTargetPos[oi3];
          newTargetPositions[i3 + 1] = oldTargetPos[oi3 + 1];
          newTargetPositions[i3 + 2] = oldTargetPos[oi3 + 2];
        } else {
          newTargetPositions[i3] = newCurrentPositions[i3];
          newTargetPositions[i3 + 1] = newCurrentPositions[i3 + 1];
          newTargetPositions[i3 + 2] = newCurrentPositions[i3 + 2];
        }
        preservedCount++;
      } else {
        // New node — use data position
        const pos = node.position;
        newCurrentPositions[i3] = pos.x;
        newCurrentPositions[i3 + 1] = pos.y;
        newCurrentPositions[i3 + 2] = pos.z;
        newTargetPositions[i3] = pos.x;
        newTargetPositions[i3 + 1] = pos.y;
        newTargetPositions[i3 + 2] = pos.z;
      }
    });

    this.currentPositions = newCurrentPositions;
    this.targetPositions = newTargetPositions;
    this.velocities = newVelocities;

    // Write preserved positions back into graphData so that:
    // 1) Any consumer reading node.position gets current values, not stale DB positions
    // 2) Future setGraphData() calls have up-to-date fallback data
    for (let i = 0; i < nodeCount; i++) {
      const i3 = i * 3;
      this.graphData.nodes[i].position = {
        x: newCurrentPositions[i3],
        y: newCurrentPositions[i3 + 1],
        z: newCurrentPositions[i3 + 2],
      };
    }

    workerLogger.info(`Initialized ${this.graphType} graph with ${nodeCount} nodes, ${data.edges.length} edges (${preservedCount} positions preserved, server-authoritative physics)`);
  }

  
  async setupSharedPositions(buffer: SharedArrayBuffer): Promise<void> {
    this.positionBuffer = buffer;
    this.positionView = new Float32Array(buffer);
    workerLogger.info(`SharedArrayBuffer set up with ${buffer.byteLength} bytes`);
  }

  /** Copy currentPositions into the SharedArrayBuffer so the main thread can read synchronously. */
  private syncToSharedBuffer(): void {
    if (this.positionView && this.currentPositions) {
      const len = Math.min(this.currentPositions.length, this.positionView.length);
      this.positionView.set(this.currentPositions.subarray(0, len));
    }
  }

  
  async updateSettings(settings: any): Promise<void> {
    // Extract only physics-relevant settings
    const graphSettings = settings?.visualisation?.graphs?.[this.graphType]?.physics ||
                         settings?.visualisation?.physics;
    const vfPhysics = (this.graphType === 'visionflow')
      ? (settings?.visualisation?.graphs?.visionflow?.physics || {})
      : null;

    // Idempotency: bail if physics values haven't changed (prevents unnecessary
    // parameter resets that can disrupt a running force-directed simulation)
    const physicsKey = JSON.stringify({ gs: graphSettings, vf: vfPhysics });
    if (physicsKey === this._lastPhysicsKey) return;
    this._lastPhysicsKey = physicsKey;

    this.physicsSettings = {
      springStrength: graphSettings?.springStrength ?? 0.001,
      damping: graphSettings?.damping ?? 0.98,
      maxVelocity: graphSettings?.maxVelocity ?? 0.5,
      updateThreshold: graphSettings?.updateThreshold ?? 0.05
    };

    // Physics settings for visionflow are now routed to the server via REST API.
    // The client stores them for reference but does not run local force simulation.
  }

  
  async processBinaryData(data: ArrayBuffer): Promise<Float32Array> {
    // All graph types process binary position updates from the server.
    // Server is the single source of truth for positions.
    
    
    this.binaryUpdateCount = (this.binaryUpdateCount || 0) + 1;
    this.lastBinaryUpdate = Date.now();

    
    if (isZlibCompressed(data)) {
      data = await decompressZlib(data);
    }
    const nodeUpdates = parseBinaryNodeData(data);

    
    // Reuse binary output buffer, only reallocate if size changed
    const requiredBinarySize = nodeUpdates.length * 4;
    if (!this.binaryOutputBuffer || this.binaryOutputBufferSize !== requiredBinarySize) {
      this.binaryOutputBuffer = new Float32Array(requiredBinarySize);
      this.binaryOutputBufferSize = requiredBinarySize;
    }
    const positionArray = this.binaryOutputBuffer;

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
        let mappedId = stringToU32(node.id);
        while (this.reverseNodeIdMap.has(mappedId) && this.reverseNodeIdMap.get(mappedId) !== node.id) {
          mappedId = (mappedId + 1) >>> 0;
        }
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

  // Client-side force computation (computeForces, applyForces) REMOVED.
  // The server (Rust/CUDA GPU) handles all force-directed layout.
  // Client only performs optimistic interpolation toward server targets.

  
  async pinNode(nodeId: number): Promise<void> { this.pinnedNodeIds.add(nodeId); }
  async unpinNode(nodeId: number): Promise<void> { this.pinnedNodeIds.delete(nodeId); }

  
  async setUseServerPhysics(useServer: boolean): Promise<void> {
    // Server physics is always authoritative. This method is kept for API
    // compatibility but always enforces server mode.
    this.useServerPhysics = true;
    if (!useServer) {
      workerLogger.warn('Client-side physics requested but server is authoritative — ignoring');
    }
    workerLogger.info('Physics mode: server-authoritative (single source of truth)');
  }


  async getPhysicsMode(): Promise<boolean> {
    return this.useServerPhysics;
  }

  /**
   * Reheat the force simulation (restart physics from current positions).
   * Call this when user drags a node or wants to re-layout.
   */
  async reheatSimulation(alpha: number = 1.0): Promise<void> {
    this.forcePhysics.alpha = alpha;
    workerLogger.info(`Simulation reheated to alpha=${alpha}`);
  }

  /**
   * Update force-directed physics settings from UI.
   */
  async updateForcePhysicsSettings(settings: Partial<ForcePhysicsSettings>): Promise<void> {
    Object.assign(this.forcePhysics, settings);
    workerLogger.info('Force physics settings updated', this.forcePhysics);
  }

  /**
   * Get current force physics settings.
   */
  async getForcePhysicsSettings(): Promise<ForcePhysicsSettings> {
    return { ...this.forcePhysics };
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

        // User drag position is applied optimistically on the client.
        // The position should also be sent to the server via REST API
        // so the server can apply it as a constraint and rebroadcast.
      }
    }
  }

  
  async tick(deltaTime: number): Promise<Float32Array> {
    if (!this.currentPositions || !this.targetPositions || !this.velocities) {
      return new Float32Array(0);
    }

    // Clamp delta time for stability
    const dt = Math.min(deltaTime, 0.033); // Max 30fps equivalent

    this.frameCount = (this.frameCount || 0) + 1;

    // ====== SERVER-AUTHORITATIVE PHYSICS — Interpolate toward target positions ======
    // All graph types (visionflow, logseq) use server-computed positions as the
    // single source of truth. The client only performs optimistic tweening.
    {
      
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
        this.syncToSharedBuffer();
        return this.currentPositions;
      }
      



      // deltaTime is already in seconds (from Three.js useFrame delta)
      const lerpFactor = 1 - Math.pow(0.001, deltaTime); 
      
      
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
      
      // Keep graphData.nodes[i].position in sync with currentPositions.
      // This ensures any future setGraphData() / getGraphData() uses the
      // latest interpolated positions — not stale DB values.
      // Only sync every 30 frames (~0.5s at 60fps) to limit overhead.
      if (this.frameCount % 30 === 0) {
        for (let i = 0; i < this.graphData.nodes.length; i++) {
          const i3 = i * 3;
          const node = this.graphData.nodes[i];
          node.position = {
            x: this.currentPositions[i3],
            y: this.currentPositions[i3 + 1],
            z: this.currentPositions[i3 + 2],
          };
        }
      }

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

    this.syncToSharedBuffer();
    return this.currentPositions;
  }
}

// Expose the worker API using Comlink
const worker = new GraphWorker();
expose(worker);

export type GraphWorkerType = GraphWorker;