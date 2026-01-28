

import { wrap, Remote } from 'comlink';
import { GraphWorkerType, ForcePhysicsSettings } from '../workers/graph.worker';
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


class GraphWorkerProxy {
  private static instance: GraphWorkerProxy;
  private worker: Worker | null = null;
  private workerApi: Remote<GraphWorkerType> | null = null;
  private graphDataListeners: GraphDataChangeListener[] = [];
  private positionUpdateListeners: PositionUpdateListener[] = [];
  private sharedBuffer: SharedArrayBuffer | null = null;
  private isInitialized: boolean = false;
  private graphType: 'logseq' | 'visionflow' = 'logseq'; 

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
      
      console.log('[GraphWorkerProxy] Creating worker');
      this.worker = new Worker(
        new URL('../workers/graph.worker.ts', import.meta.url),
        { type: 'module' }
      );

      
      this.worker.onerror = (error) => {
        console.error('[GraphWorkerProxy] Worker error:', error);
        logger.error('Worker error:', error);
      };

      console.log('[GraphWorkerProxy] Wrapping worker with Comlink');
      
      this.workerApi = wrap<GraphWorkerType>(this.worker);

      
      console.log('[GraphWorkerProxy] Testing worker communication');
      try {
        await this.workerApi.initialize();
        console.log('[GraphWorkerProxy] Worker communication test successful');
      } catch (commError) {
        console.error('[GraphWorkerProxy] Worker communication failed:', commError);
        throw new Error(`Worker communication failed: ${commError}`);
      }

      
      const maxNodes = 10000;
      const bufferSize = maxNodes * 4 * 4; 

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

      
      console.log(`[GraphWorkerProxy] Setting initial graph type: ${this.graphType}`);
      await this.setGraphType(this.graphType);
    } catch (error) {
      console.error('[GraphWorkerProxy] Failed to initialize worker:', error);
      logger.error('Failed to initialize graph worker:', error);
      throw error;
    }
  }

  
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

  
  public getGraphType(): 'logseq' | 'visionflow' {
    return this.graphType;
  }

  
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

  
  public async processBinaryData(data: ArrayBuffer): Promise<void> {
    
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

  
  public async updateNode(node: Node): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }

    await this.workerApi.updateNode(node);

    
    const graphData = await this.workerApi.getGraphData();
    this.notifyGraphDataListeners(graphData);
  }

  
  public async removeNode(nodeId: string): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }

    await this.workerApi.removeNode(nodeId);

    
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
   * Reheat the force simulation (restart physics from current positions).
   * Use this when user wants to re-layout or after significant changes.
   */
  public async reheatSimulation(alpha: number = 1.0): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }
    await this.workerApi.reheatSimulation(alpha);

    if (debugState.isEnabled()) {
      logger.info(`Reheated simulation to alpha=${alpha}`);
    }
  }

  /**
   * Update force-directed physics settings.
   */
  public async updateForcePhysicsSettings(settings: Partial<ForcePhysicsSettings>): Promise<void> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }
    await this.workerApi.updateForcePhysicsSettings(settings);
  }

  /**
   * Get current force physics settings.
   */
  public async getForcePhysicsSettings(): Promise<ForcePhysicsSettings> {
    if (!this.workerApi) {
      throw new Error('Worker not initialized');
    }
    return await this.workerApi.getForcePhysicsSettings();
  }

  
  public getSharedPositionBuffer(): Float32Array | null {
    if (!this.sharedBuffer) {
      return null;
    }
    return new Float32Array(this.sharedBuffer);
  }

  
  public onGraphDataChange(listener: GraphDataChangeListener): () => void {
    this.graphDataListeners.push(listener);

    
    return () => {
      this.graphDataListeners = this.graphDataListeners.filter(l => l !== listener);
    };
  }

  
  public onPositionUpdate(listener: PositionUpdateListener): () => void {
    this.positionUpdateListeners.push(listener);

    
    return () => {
      this.positionUpdateListeners = this.positionUpdateListeners.filter(l => l !== listener);
    };
  }

  
  public isReady(): boolean {
    return this.isInitialized && this.workerApi !== null;
  }

  
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

// Re-export types for convenience
export type { ForcePhysicsSettings } from '../workers/graph.worker';