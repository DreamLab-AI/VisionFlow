/**
 * Graph Worker Proxy
 * Main thread interface to the graph worker using Comlink
 */

import { wrap, Remote } from 'comlink';
import { GraphWorkerType } from '../workers/graph.worker';
import { createLogger } from '../../../utils/loggerConfig';
import { debugState } from '../../../utils/clientDebugState';

const logger = createLogger('GraphWorkerProxy');

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

// Add Vec3 to be used in updateUserDrivenNodePosition
export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

type GraphDataChangeListener = (data: GraphData) => void;
type PositionUpdateListener = (positions: Float32Array) => void;

/**
 * Proxy class that communicates with the graph worker
 */
class GraphWorkerProxy {
  private static instance: GraphWorkerProxy;
  private worker: Worker | null = null;
  private workerApi: Remote<GraphWorkerType> | null = null;
  private graphDataListeners: GraphDataChangeListener[] = [];
  private positionUpdateListeners: PositionUpdateListener[] = [];
  private sharedBuffer: SharedArrayBuffer | null = null;
  private isInitialized: boolean = false;
  private graphType: 'logseq' | 'visionflow' = 'logseq'; // Graph type identifier

  private constructor() {}

  public static getInstance(): GraphWorkerProxy {
    if (!GraphWorkerProxy.instance) {
      GraphWorkerProxy.instance = new GraphWorkerProxy();
    }
    return GraphWorkerProxy.instance;
  }

  public async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('[GraphWorkerProxy] Already initialized, skipping');
      return;
    }
    
    console.log('[GraphWorkerProxy] Starting worker initialization');
    try {
      // Create worker instance
      console.log('[GraphWorkerProxy] Creating worker');
      this.worker = new Worker(
        new URL('../workers/graph.worker.ts', import.meta.url),
        { type: 'module' }
      );

      // Add error handler for worker
      this.worker.onerror = (error) => {
        console.error('[GraphWorkerProxy] Worker error:', error);
        logger.error('Worker error:', error);
      };

      console.log('[GraphWorkerProxy] Wrapping worker with Comlink');
      // Wrap worker with Comlink
      this.workerApi = wrap<GraphWorkerType>(this.worker);

      // Test worker communication
      console.log('[GraphWorkerProxy] Testing worker communication');
      try {
        await this.workerApi.initialize();
        console.log('[GraphWorkerProxy] Worker communication test successful');
      } catch (commError) {
        console.error('[GraphWorkerProxy] Worker communication failed:', commError);
        throw new Error(`Worker communication failed: ${commError}`);
      }

      // Set up shared array buffer for position data (4 floats per node * max 10k nodes)
      const maxNodes = 10000;
      const bufferSize = maxNodes * 4 * 4; // 4 floats * 4 bytes per float

      if (typeof SharedArrayBuffer !== 'undefined') {
        console.log('[GraphWorkerProxy] Setting up SharedArrayBuffer');
        this.sharedBuffer = new SharedArrayBuffer(bufferSize);
        await this.workerApi.setupSharedPositions(this.sharedBuffer);
        console.log(`[GraphWorkerProxy] SharedArrayBuffer initialized: ${bufferSize} bytes`);
        if (debugState.isEnabled()) {
          logger.info(`Initialized SharedArrayBuffer: ${bufferSize} bytes for ${maxNodes} nodes`);
        }
      } else {
        console.warn('[GraphWorkerProxy] SharedArrayBuffer not available, using message passing');
        logger.warn('SharedArrayBuffer not available, falling back to regular message passing');
      }

      this.isInitialized = true;
      console.log('[GraphWorkerProxy] Initialization complete');
      if (debugState.isEnabled()) {
        logger.info('Graph worker initialized successfully');
      }

      // Set initial graph type
      console.log(`[GraphWorkerProxy] Setting initial graph type: ${this.graphType}`);
      await this.setGraphType(this.graphType);
    } catch (error) {
      console.error('[GraphWorkerProxy] Failed to initialize worker:', error);
      logger.error('Failed to initialize graph worker:', error);
      throw error;
    }
  }

  /**
   * Set the graph type
   */
  public async setGraphType(type: 'logseq' | 'visionflow'): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }

    this.graphType = type;
    await this.workerApi.setGraphType(type);

    if (debugState.isEnabled()) {
      logger.info(`Graph type set to: ${type}`);
    }
  }

  /**
   * Get the current graph type
   */
  public getGraphType(): 'logseq' | 'visionflow' {
    return this.graphType;
  }

  /**
   * Set graph data in the worker
   */
  public async setGraphData(data: GraphData): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }

    await this.workerApi.setGraphData(data);
    this.notifyGraphDataListeners(data);

    if (debugState.isEnabled()) {
      logger.info(`Set ${this.graphType} graph data: ${data.nodes.length} nodes, ${data.edges.length} edges`);
    }
  }

  /**
   * Process binary data through the worker (with decompression)
   */
  public async processBinaryData(data: ArrayBuffer): Promise<void> {
    // Only process binary data for Logseq graphs
    if (this.graphType !== 'logseq') {
      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Skipping binary data processing for ${this.graphType} graph`);
      }
      return;
    }
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }

    try {
      const positionArray = await this.workerApi.processBinaryData(data);
      this.notifyPositionUpdateListeners(positionArray);

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Processed binary data: ${positionArray.length / 4} position updates`);
      }
    } catch (error) {
      logger.error('Error processing binary data in worker:', error);
      throw error;
    }
  }

  /**
   * Get current graph data from worker
   */
  public async getGraphData(): Promise<GraphData> {
    if (!this.workerApi) {
      console.error('[GraphWorkerProxy] Worker not initialized for getGraphData');
      throw new Error('Worker not initialized');
    }
    console.log('[GraphWorkerProxy] Getting graph data from worker');
    try {
      const data = await this.workerApi.getGraphData();
      console.log(`[GraphWorkerProxy] Got ${data.nodes.length} nodes, ${data.edges.length} edges from worker`);
      return data;
    } catch (error) {
      console.error('[GraphWorkerProxy] Error getting graph data from worker:', error);
      throw error;
    }
  }

  /**
   * Add or update a node in the worker
   */
  public async updateNode(node: Node): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }

    await this.workerApi.updateNode(node);

    // Get updated data and notify listeners
    const graphData = await this.workerApi.getGraphData();
    this.notifyGraphDataListeners(graphData);
  }

  /**
   * Remove a node from the worker
   */
  public async removeNode(nodeId: string): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }

    await this.workerApi.removeNode(nodeId);

    // Get updated data and notify listeners
    const graphData = await this.workerApi.getGraphData();
    this.notifyGraphDataListeners(graphData);
  }

  public async updateSettings(settings: any): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }
    await this.workerApi.updateSettings(settings);
  }

  public async pinNode(nodeId: number): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }
    await this.workerApi.pinNode(nodeId);
  }

  public async unpinNode(nodeId: number): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }
    await this.workerApi.unpinNode(nodeId);
  }

  public async updateUserDrivenNodePosition(nodeId: number, position: Vec3): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }
    await this.workerApi.updateUserDrivenNodePosition(nodeId, position);
  }

  public async tick(deltaTime: number): Promise<Float32Array> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }
    return await this.workerApi.tick(deltaTime);
  }

  /**
   * Get shared array buffer view for position data
   */
  public getSharedPositionBuffer(): Float32Array | null {
    if (!this.sharedBuffer) {
      return null;
    }
    return new Float32Array(this.sharedBuffer);
  }

  /**
   * Add listener for graph data changes
   */
  public onGraphDataChange(listener: GraphDataChangeListener): () => void {
    this.graphDataListeners.push(listener);

    // Return unsubscribe function
    return () => {
      this.graphDataListeners = this.graphDataListeners.filter(l => l !== listener);
    };
  }

  /**
   * Add listener for position updates
   */
  public onPositionUpdate(listener: PositionUpdateListener): () => void {
    this.positionUpdateListeners.push(listener);

    // Return unsubscribe function
    return () => {
      this.positionUpdateListeners = this.positionUpdateListeners.filter(l => l !== listener);
    };
  }

  /**
   * Check if worker is ready
   */
  public isReady(): boolean {
    return this.isInitialized && this.workerApi !== null;
  }

  /**
   * Dispose of worker resources
   */
  public dispose(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }

    this.workerApi = null;
    this.graphDataListeners = [];
    this.positionUpdateListeners = [];
    this.sharedBuffer = null;
    this.isInitialized = false;

    if (debugState.isEnabled()) {
      logger.info('Graph worker disposed');
    }
  }

  private notifyGraphDataListeners(data: GraphData): void {
    this.graphDataListeners.forEach(listener => {
      try {
        listener(data);
      } catch (error) {
        logger.error('Error in graph data listener:', error);
      }
    });
  }

  private notifyPositionUpdateListeners(positions: Float32Array): void {
    this.positionUpdateListeners.forEach(listener => {
      try {
        listener(positions);
      } catch (error) {
        logger.error('Error in position update listener:', error);
      }
    });
  }
}

// Create singleton instance
export const graphWorkerProxy = GraphWorkerProxy.getInstance();