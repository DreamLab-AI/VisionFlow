

import { expose } from 'comlink';
import { BinaryNodeData, parseBinaryNodeData, createBinaryNodeData, Vec3 } from '../../../types/binaryProtocol';

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


// Force-directed physics settings for client-side simulation
export interface ForcePhysicsSettings {
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

  // Pre-allocated buffers for force computation (reused every tick)
  private forcesBuffer: Float32Array | null = null;
  private forcesBufferSize: number = 0;

  // Pre-allocated buffer for binary position output (reused every processBinaryData call)
  private binaryOutputBuffer: Float32Array | null = null;
  private binaryOutputBufferSize: number = 0;

  
  async initialize(): Promise<void> {
    workerLogger.info('Initialize method called');
    return Promise.resolve();
  }
  
  
  async setGraphType(type: 'logseq' | 'visionflow'): Promise<void> {
    this.graphType = type;
    // VisionFlow uses client-side physics since it doesn't receive server binary updates
    if (type === 'visionflow') {
      this.useServerPhysics = false;
      this.forcePhysics.enabled = true;
      this.forcePhysics.alpha = 1.0; // Reset simulation temperature
      workerLogger.info(`Graph type set to ${type} - using CLIENT-SIDE force-directed physics`);
    } else {
      // Logseq graphs receive server physics via binary protocol
      this.useServerPhysics = true;
      this.forcePhysics.enabled = false;
      workerLogger.info(`Graph type set to ${type} - using SERVER physics`);
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
      workerLogger.info(`VisionFlow graph initialized with ${nodeCount} nodes, ${data.edges.length} edges, ${this.domainClusters.size} domains`);
    } else {
      workerLogger.info(`Initialized ${this.graphType} graph with ${this.graphData.nodes.length} nodes`);
    }
  }

  
  async setupSharedPositions(buffer: SharedArrayBuffer): Promise<void> {
    this.positionBuffer = buffer;
    this.positionView = new Float32Array(buffer);
    workerLogger.info(`SharedArrayBuffer set up with ${buffer.byteLength} bytes`);
  }

  
  async updateSettings(settings: any): Promise<void> {
    // Server physics interpolation settings (for logseq)
    const graphSettings = settings?.visualisation?.graphs?.[this.graphType]?.physics ||
                         settings?.visualisation?.physics;

    this.physicsSettings = {
      springStrength: graphSettings?.springStrength ?? 0.001,
      damping: graphSettings?.damping ?? 0.98,
      maxVelocity: graphSettings?.maxVelocity ?? 0.5,
      updateThreshold: graphSettings?.updateThreshold ?? 0.05
    };

    // Force-directed physics settings (for visionflow)
    if (this.graphType === 'visionflow') {
      const vfPhysics = settings?.visualisation?.graphs?.visionflow?.physics || {};

      // Map UI settings to force physics parameters
      this.forcePhysics.enabled = vfPhysics.enabled ?? true;
      this.forcePhysics.repulsionStrength = (vfPhysics.repelK ?? 1.0) * 500; // Scale for good defaults
      this.forcePhysics.attractionStrength = vfPhysics.springK ?? 0.05;
      this.forcePhysics.damping = vfPhysics.damping ?? 0.85;
      this.forcePhysics.maxVelocity = vfPhysics.maxVelocity ?? 5.0;
      this.forcePhysics.idealEdgeLength = vfPhysics.restLength ?? 30;
      this.forcePhysics.centerGravity = vfPhysics.centerGravityK ?? 0.01;

      // If physics was just enabled, reheat simulation
      if (this.forcePhysics.enabled && this.forcePhysics.alpha < this.forcePhysics.alphaMin) {
        this.forcePhysics.alpha = 1.0;
        workerLogger.info('VisionFlow physics enabled, reheating simulation');
      }
    }
  }

  
  async processBinaryData(data: ArrayBuffer): Promise<Float32Array> { 
    
    if (this.graphType !== 'logseq') {
      workerLogger.debug(`Skipping binary data processing for ${this.graphType} graph`);
      return new Float32Array(0);
    }

    
    if (!this.useServerPhysics) {
      this.useServerPhysics = true;
      workerLogger.info('Auto-enabled server physics mode due to binary position updates');
    }
    
    
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

  /**
   * Compute force-directed layout forces for client-side physics.
   * Implements:
   * 1. Node-node repulsion (Coulomb's law) with spatial grid for ~O(n*k) complexity
   * 2. Edge-based attraction (spring force)
   * 3. Center gravity (prevents drift)
   * 4. Domain clustering (semantic grouping)
   */
  private computeForces(): Float32Array {
    const n = this.graphData.nodes.length;
    if (n === 0 || !this.currentPositions) {
      if (!this.forcesBuffer || this.forcesBufferSize !== 0) {
        this.forcesBuffer = new Float32Array(0);
        this.forcesBufferSize = 0;
      }
      return this.forcesBuffer;
    }

    // Reuse forces buffer, only reallocate if node count changed
    const requiredSize = n * 3;
    if (!this.forcesBuffer || this.forcesBufferSize !== requiredSize) {
      this.forcesBuffer = new Float32Array(requiredSize);
      this.forcesBufferSize = requiredSize;
    } else {
      this.forcesBuffer.fill(0);
    }

    const forces = this.forcesBuffer;
    const {
      repulsionStrength,
      attractionStrength,
      centerGravity,
      idealEdgeLength,
      clusterStrength,
      enableClustering,
      alpha
    } = this.forcePhysics;

    const pos = this.currentPositions;

    // 1. REPULSION: Spatial grid approximation for ~O(n*k) instead of O(n^2)
    // Nodes beyond cutoffRadius are skipped entirely. Nearby nodes use grid cells.
    const cutoffRadius = Math.max(idealEdgeLength * 4, 120); // Distance beyond which repulsion is negligible
    const cutoffRadiusSq = cutoffRadius * cutoffRadius;
    const cellSize = cutoffRadius; // One cell per cutoff radius

    // Build spatial grid: compute bounds first
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      const x = pos[i3], y = pos[i3 + 1], z = pos[i3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }

    // Add padding to prevent edge issues
    minX -= cellSize; minY -= cellSize; minZ -= cellSize;

    const gridW = Math.max(1, Math.ceil((maxX - minX) / cellSize) + 1);
    const gridH = Math.max(1, Math.ceil((maxY - minY) / cellSize) + 1);
    const gridD = Math.max(1, Math.ceil((maxZ - minZ) / cellSize) + 1);

    // Use flat Map for sparse grid (avoids allocating huge 3D array)
    const grid = new Map<number, number[]>();

    // Assign each node to a cell
    const nodeCellKeys = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      const cx = Math.floor((pos[i3] - minX) / cellSize);
      const cy = Math.floor((pos[i3 + 1] - minY) / cellSize);
      const cz = Math.floor((pos[i3 + 2] - minZ) / cellSize);
      const key = cx + cy * gridW + cz * gridW * gridH;
      nodeCellKeys[i] = key;
      let cell = grid.get(key);
      if (!cell) {
        cell = [];
        grid.set(key, cell);
      }
      cell.push(i);
    }

    // For each node, check repulsion only against nodes in neighboring cells (3x3x3 neighborhood)
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      const xi = pos[i3], yi = pos[i3 + 1], zi = pos[i3 + 2];

      const cx = Math.floor((xi - minX) / cellSize);
      const cy = Math.floor((yi - minY) / cellSize);
      const cz = Math.floor((zi - minZ) / cellSize);

      // Check 3x3x3 neighboring cells
      for (let dcx = -1; dcx <= 1; dcx++) {
        const ncx = cx + dcx;
        if (ncx < 0 || ncx >= gridW) continue;
        for (let dcy = -1; dcy <= 1; dcy++) {
          const ncy = cy + dcy;
          if (ncy < 0 || ncy >= gridH) continue;
          for (let dcz = -1; dcz <= 1; dcz++) {
            const ncz = cz + dcz;
            if (ncz < 0 || ncz >= gridD) continue;

            const neighborKey = ncx + ncy * gridW + ncz * gridW * gridH;
            const neighborCell = grid.get(neighborKey);
            if (!neighborCell) continue;

            for (let ni = 0; ni < neighborCell.length; ni++) {
              const j = neighborCell[ni];
              if (j <= i) continue; // Only process each pair once (j > i)

              const j3 = j * 3;
              let dx = pos[j3] - xi;
              let dy = pos[j3 + 1] - yi;
              let dz = pos[j3 + 2] - zi;

              // Add small jitter to prevent singularities when nodes overlap
              if (dx === 0 && dy === 0 && dz === 0) {
                dx = (Math.random() - 0.5) * 0.1;
                dy = (Math.random() - 0.5) * 0.1;
                dz = (Math.random() - 0.5) * 0.1;
              }

              const distSq = dx * dx + dy * dy + dz * dz;

              // Skip nodes beyond cutoff radius
              if (distSq > cutoffRadiusSq) continue;

              const dist = Math.sqrt(distSq);

              // Repulsion force magnitude: F = k / r^2 (clamped for stability)
              const minDist = 1.0;
              const effectiveDist = Math.max(dist, minDist);
              const repulseForce = (repulsionStrength * alpha) / (effectiveDist * effectiveDist);

              // Direction: from j to i (i is pushed away from j)
              const nx = dx / dist;
              const ny = dy / dist;
              const nz = dz / dist;

              // Apply equal and opposite forces
              forces[i3] -= repulseForce * nx;
              forces[i3 + 1] -= repulseForce * ny;
              forces[i3 + 2] -= repulseForce * nz;

              forces[j3] += repulseForce * nx;
              forces[j3 + 1] += repulseForce * ny;
              forces[j3 + 2] += repulseForce * nz;
            }
          }
        }
      }
    }

    // 2. ATTRACTION: Edges act as springs pulling connected nodes together
    // Using Hooke's law: F = k * (distance - restLength)
    for (const edge of this.graphData.edges) {
      const sourceIdx = this.nodeIndexMap.get(edge.source);
      const targetIdx = this.nodeIndexMap.get(edge.target);

      if (sourceIdx === undefined || targetIdx === undefined) continue;

      const s3 = sourceIdx * 3;
      const t3 = targetIdx * 3;

      const dx = pos[t3] - pos[s3];
      const dy = pos[t3 + 1] - pos[s3 + 1];
      const dz = pos[t3 + 2] - pos[s3 + 2];

      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 0.0001) continue; // Skip if nodes are at same position

      // Spring force: pull toward ideal length
      const displacement = dist - idealEdgeLength;
      const springForce = attractionStrength * displacement * alpha;

      const nx = dx / dist;
      const ny = dy / dist;
      const nz = dz / dist;

      // Pull source toward target
      forces[s3] += springForce * nx;
      forces[s3 + 1] += springForce * ny;
      forces[s3 + 2] += springForce * nz;

      // Pull target toward source
      forces[t3] -= springForce * nx;
      forces[t3 + 1] -= springForce * ny;
      forces[t3 + 2] -= springForce * nz;
    }

    // 3. CENTER GRAVITY: Gentle pull toward origin to prevent drift
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      forces[i3] -= pos[i3] * centerGravity * alpha;
      forces[i3 + 1] -= pos[i3 + 1] * centerGravity * alpha;
      forces[i3 + 2] -= pos[i3 + 2] * centerGravity * alpha;
    }

    // 4. DOMAIN CLUSTERING: Pull nodes of same domain toward cluster center
    if (enableClustering && this.domainClusters.size > 1) {
      // First, compute domain centers
      this.domainCenters.clear();
      this.domainClusters.forEach((nodeIndices, domain) => {
        if (nodeIndices.length === 0) return;

        let cx = 0, cy = 0, cz = 0;
        for (let i = 0; i < nodeIndices.length; i++) {
          const idx = nodeIndices[i];
          const i3 = idx * 3;
          cx += pos[i3];
          cy += pos[i3 + 1];
          cz += pos[i3 + 2];
        }
        cx /= nodeIndices.length;
        cy /= nodeIndices.length;
        cz /= nodeIndices.length;
        this.domainCenters.set(domain, { x: cx, y: cy, z: cz });
      });

      // Apply clustering force toward domain center
      this.domainClusters.forEach((nodeIndices, domain) => {
        const center = this.domainCenters.get(domain);
        if (!center) return;

        for (let i = 0; i < nodeIndices.length; i++) {
          const idx = nodeIndices[i];
          const i3 = idx * 3;
          const dx = center.x - pos[i3];
          const dy = center.y - pos[i3 + 1];
          const dz = center.z - pos[i3 + 2];

          forces[i3] += dx * clusterStrength * alpha;
          forces[i3 + 1] += dy * clusterStrength * alpha;
          forces[i3 + 2] += dz * clusterStrength * alpha;
        }
      });
    }

    return forces;
  }

  /**
   * Apply computed forces to update velocities and positions.
   * Implements velocity Verlet integration with damping and speed limits.
   */
  private applyForces(forces: Float32Array, dt: number): void {
    if (!this.currentPositions || !this.velocities) return;

    const n = this.graphData.nodes.length;
    const { damping, maxVelocity } = this.forcePhysics;
    const pos = this.currentPositions;
    const vel = this.velocities;

    for (let i = 0; i < n; i++) {
      // Skip pinned nodes
      const nodeId = this.nodeIdMap.get(this.graphData.nodes[i].id);
      if (nodeId !== undefined && this.pinnedNodeIds.has(nodeId)) continue;

      const i3 = i * 3;

      // Update velocities with forces (F = ma, assume m = 1)
      vel[i3] += forces[i3] * dt;
      vel[i3 + 1] += forces[i3 + 1] * dt;
      vel[i3 + 2] += forces[i3 + 2] * dt;

      // Apply damping
      vel[i3] *= damping;
      vel[i3 + 1] *= damping;
      vel[i3 + 2] *= damping;

      // Clamp velocity
      const speed = Math.sqrt(vel[i3] * vel[i3] + vel[i3 + 1] * vel[i3 + 1] + vel[i3 + 2] * vel[i3 + 2]);
      if (speed > maxVelocity) {
        const scale = maxVelocity / speed;
        vel[i3] *= scale;
        vel[i3 + 1] *= scale;
        vel[i3 + 2] *= scale;
      }

      // Update positions
      pos[i3] += vel[i3] * dt;
      pos[i3 + 1] += vel[i3 + 1] * dt;
      pos[i3 + 2] += vel[i3 + 2] * dt;
    }
  }

  
  async pinNode(nodeId: number): Promise<void> { this.pinnedNodeIds.add(nodeId); }
  async unpinNode(nodeId: number): Promise<void> { this.pinnedNodeIds.delete(nodeId); }

  
  async setUseServerPhysics(useServer: boolean): Promise<void> {
    this.useServerPhysics = useServer;
    // Also toggle force physics for visionflow
    if (this.graphType === 'visionflow') {
      this.forcePhysics.enabled = !useServer;
      if (!useServer) {
        this.forcePhysics.alpha = 1.0; // Reheat simulation
      }
    }
    workerLogger.info(`Physics mode set to ${useServer ? 'server' : 'local'}`);
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

        // Reheat simulation when user drags in VisionFlow mode
        if (this.graphType === 'visionflow' && this.forcePhysics.enabled) {
          // Partial reheat to allow re-equilibration without full restart
          this.forcePhysics.alpha = Math.max(this.forcePhysics.alpha, 0.3);
        }
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

    // ====== CLIENT-SIDE FORCE-DIRECTED PHYSICS (VisionFlow) ======
    if (this.graphType === 'visionflow' && this.forcePhysics.enabled) {
      // Check if simulation has cooled down
      if (this.forcePhysics.alpha < this.forcePhysics.alphaMin) {
        // Simulation has settled - return current positions without updates
        return this.currentPositions;
      }

      // Compute forces using repulsion/attraction model
      const forces = this.computeForces();

      // Apply forces to update positions
      this.applyForces(forces, dt);

      // Cool down simulation (alpha decay)
      this.forcePhysics.alpha *= (1 - this.forcePhysics.alphaDecay);

      // Log progress occasionally
      if (this.frameCount % 60 === 0 && this.forcePhysics.alpha > this.forcePhysics.alphaMin) {
        workerLogger.debug(`VisionFlow physics tick - alpha=${this.forcePhysics.alpha.toFixed(4)}, nodes=${this.graphData.nodes.length}`);
      }

      return this.currentPositions;
    }

    // ====== SERVER-SIDE PHYSICS (Logseq) - Interpolate toward target positions ======
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