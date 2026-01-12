import { createLogger, createErrorMetadata } from '../../../utils/loggerConfig';
import { debugState, clientDebugState } from '../../../utils/clientDebugState';
import { unifiedApiClient } from '../../../services/api/UnifiedApiClient';
import { WebSocketAdapter } from '../../../services/WebSocketService';
import { useSettingsStore } from '../../../store/settingsStore';
import { BinaryNodeData, parseBinaryNodeData, createBinaryNodeData, Vec3, BINARY_NODE_SIZE } from '../../../types/binaryProtocol';
import { graphWorkerProxy } from './graphWorkerProxy';
import type { GraphData, Node, Edge } from './graphWorkerProxy';
import { startTransition } from 'react';
import { useWorkerErrorStore } from '../../../store/workerErrorStore';

const logger = createLogger('GraphDataManager');

// Re-export types from worker proxy for compatibility
export type { Node, Edge, GraphData } from './graphWorkerProxy';

// Alias for backward compatibility
export type GraphNode = Node;

type GraphDataChangeListener = (data: GraphData) => void;
type PositionUpdateListener = (positions: Float32Array) => void;

class GraphDataManager {
  private static instance: GraphDataManager;
  private binaryUpdatesEnabled: boolean = false;
  public webSocketService: WebSocketAdapter | null = null;
  private graphDataListeners: GraphDataChangeListener[] = [];
  private positionUpdateListeners: PositionUpdateListener[] = [];
  private lastBinaryUpdateTime: number = 0;
  private retryTimeout: number | null = null;
  public nodeIdMap: Map<string, number> = new Map();
  private reverseNodeIdMap: Map<number, string> = new Map();
  private workerInitialized: boolean = false;
  private graphType: 'logseq' | 'visionflow' = 'logseq'; 
  private isUserInteracting: boolean = false; 
  private interactionTimeoutRef: number | null = null;
  private updateCount: number = 0; 

  private constructor() {
    
    this.waitForWorker();
  }

  private async waitForWorker(): Promise<void> {
    try {
      console.log('[GraphDataManager] Waiting for worker to be ready...');
      let attempts = 0;
      const maxAttempts = 300; // Increased from 50 to 300 (3 seconds total)


      while (!graphWorkerProxy.isReady() && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 10));
        attempts++;
        // Log progress every 50 attempts
        if (attempts % 50 === 0) {
          console.log(`[GraphDataManager] Still waiting for worker... (${attempts}/${maxAttempts})`);
        }
      }

      if (!graphWorkerProxy.isReady()) {
        console.warn('[GraphDataManager] Worker not ready after timeout, continuing without worker');
        logger.warn('Graph worker proxy not ready after timeout, proceeding without worker');
        this.workerInitialized = false;
        // Show user-facing error modal via workerErrorStore
        useWorkerErrorStore.getState().setWorkerError(
          'The graph visualization worker failed to initialize.',
          'Worker initialization timed out after 3 seconds. The application will continue with reduced performance.'
        );
        return;
      }
      
      this.workerInitialized = true;
      console.log('[GraphDataManager] Worker is ready!');
      
      
      this.setupWorkerListeners();
      
      if (debugState.isEnabled()) {
        logger.info('Graph worker proxy is ready');
      }
    } catch (error) {
      console.error('[GraphDataManager] Failed to wait for worker:', error);
      logger.error('Failed to wait for graph worker proxy:', createErrorMetadata(error));
      this.workerInitialized = false;
    }
  }

  private setupWorkerListeners(): void {
    
    graphWorkerProxy.onGraphDataChange((data) => {
      this.graphDataListeners.forEach(listener => {
        try {
          startTransition(() => {
            listener(data);
          });
        } catch (error) {
          logger.error('Error in forwarded graph data listener:', createErrorMetadata(error));
        }
      });
    });

    
    graphWorkerProxy.onPositionUpdate((positions) => {
      this.positionUpdateListeners.forEach(listener => {
        try {
          listener(positions);
        } catch (error) {
          logger.error('Error in forwarded position update listener:', createErrorMetadata(error));
        }
      });
    });
  }

  public static getInstance(): GraphDataManager {
    if (!GraphDataManager.instance) {
      GraphDataManager.instance = new GraphDataManager();
    }
    return GraphDataManager.instance;
  }

  /**
   * Get the reverse node ID map (numeric ID -> string ID)
   * Used for resolving node positions from binary protocol IDs.
   */
  public get reverseNodeIds(): Map<number, string> {
    return this.reverseNodeIdMap;
  }

  /**
   * Get cached graph data synchronously (may be stale or null)
   * Used for fast position lookups during animation.
   * Note: Returns null since worker data requires async access.
   * Callers should use fallback positioning when null.
   */
  public getCachedGraphData(): GraphData | null {
    // Worker data is async-only; callers should use fallback positioning
    // For real-time visualization, ActionConnectionsLayer uses deterministic
    // position generation based on node IDs when positions aren't available.
    return null;
  }

  // Allow re-checking worker readiness after AppInitializer completes
  public async ensureWorkerReady(): Promise<boolean> {
    if (this.workerInitialized) {
      return true;
    }

    console.log('[GraphDataManager] ensureWorkerReady called, checking worker status...');

    if (graphWorkerProxy.isReady()) {
      this.workerInitialized = true;
      this.setupWorkerListeners();
      console.log('[GraphDataManager] Worker is now ready (late initialization)');
      return true;
    }

    // Wait a bit more for worker
    for (let i = 0; i < 100; i++) {
      await new Promise(resolve => setTimeout(resolve, 10));
      if (graphWorkerProxy.isReady()) {
        this.workerInitialized = true;
        this.setupWorkerListeners();
        console.log('[GraphDataManager] Worker became ready after additional wait');
        return true;
      }
    }

    console.warn('[GraphDataManager] Worker still not ready after ensureWorkerReady');
    return false;
  }

  
  public setWebSocketService(service: WebSocketAdapter): void {
    this.webSocketService = service;
    if (debugState.isDataDebugEnabled()) {
      logger.debug('WebSocket service set');
    }
  }

  
  public setGraphType(type: 'logseq' | 'visionflow'): void {
    this.graphType = type;
    if (debugState.isEnabled()) {
      logger.info(`Graph type set to: ${type}`);
    }
  }

  
  public getGraphType(): 'logseq' | 'visionflow' {
    return this.graphType;
  }

  
  
  public async fetchInitialData(): Promise<GraphData> {
    const maxRetries = 5;
    const initialDelay = 1000; 

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`[GraphDataManager] Fetching initial ${this.graphType} graph data with physics positions (Attempt ${attempt}/${maxRetries})`);
        if (debugState.isEnabled()) {
          logger.info(`Fetching initial ${this.graphType} graph data with physics positions (Attempt ${attempt}/${maxRetries})`);
        }

        const response = await unifiedApiClient.get('/graph/data');
        console.log(`[GraphDataManager] API response status: ${response.status}`);

        // Handle response structure: { success: true, data: { nodes: [], edges: [] } }
        const responseData = response.data.data || response.data;

        if (!responseData || typeof responseData !== 'object') {
          throw new Error('Invalid graph data format: data is not an object');
        }

        const nodes = Array.isArray(responseData.nodes) ? responseData.nodes : [];
        // Convert edge source/target from numbers to strings (API returns numeric IDs)
        const edges = Array.isArray(responseData.edges)
          ? responseData.edges.map((edge: any) => ({
              ...edge,
              source: String(edge.source),
              target: String(edge.target)
            }))
          : [];
        const metadata = responseData.metadata || {};
        const settlementState = responseData.settlementState || { isSettled: false, stableFrameCount: 0, kineticEnergy: 0 };

        console.log(`[GraphDataManager] Received settlement state: settled=${settlementState.isSettled}, frames=${settlementState.stableFrameCount}, KE=${settlementState.kineticEnergy}`);

        
        
        const enrichedNodes = nodes.map((node: Node) => {
          const nodeAny = node as any;
          const nodeMetadata = metadata[nodeAny.metadata_id || nodeAny.metadataId];
          if (nodeMetadata) {
            return { ...node, metadata: { ...node.metadata, ...nodeMetadata } };
          }
          return node;
        });

        const validatedData = { nodes: enrichedNodes, edges };

        if (debugState.isEnabled()) {
          logger.info(`Received initial graph data: ${validatedData.nodes.length} nodes, ${validatedData.edges.length} edges (physics settled: ${settlementState.isSettled})`);
        }

        console.log(`[GraphDataManager] Setting validated graph data with ${validatedData.nodes.length} nodes at physics-settled positions`);
        await this.setGraphData(validatedData);

        const currentData = await graphWorkerProxy.getGraphData();
        console.log(`[GraphDataManager] Worker returned data with ${currentData.nodes.length} nodes - no position "pop-in" expected!`);
        return currentData;

      } catch (error) {
        logger.error(`Attempt ${attempt} failed to fetch initial graph data:`, createErrorMetadata(error));
        if (attempt === maxRetries) {
          logger.error('All attempts to fetch initial graph data failed.');
          throw error; 
        }

        const delay = initialDelay * Math.pow(2, attempt - 1);
        console.log(`[GraphDataManager] Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    
    return { nodes: [], edges: [] };
  }

  
  public async setGraphData(data: GraphData): Promise<void> {
    if (debugState.isEnabled()) {
      logger.info(`Setting ${this.graphType} graph data: ${data.nodes.length} nodes, ${data.edges.length} edges`);
    }

    
    let validatedData = data;
    if (data && data.nodes) {
      const validatedNodes = data.nodes.map(node => this.ensureNodeHasValidPosition(node));
      validatedData = {
        ...data,
        nodes: validatedNodes
      };
      
      if (debugState.isEnabled()) {
        logger.info(`Validated ${validatedNodes.length} nodes with positions`);
      }
    } else {
      
      validatedData = { nodes: [], edges: data?.edges || [] };
      logger.warn('Initialized with empty graph data');
    }
    
    
    this.nodeIdMap.clear();
    this.reverseNodeIdMap.clear();
    
    
    validatedData.nodes.forEach((node, index) => {
      const numericId = parseInt(node.id, 10);
      if (!isNaN(numericId) && numericId >= 0 && numericId <= 0xFFFFFFFF) {
        
        this.nodeIdMap.set(node.id, numericId);
        this.reverseNodeIdMap.set(numericId, node.id);
      } else {
        
        
        const mappedId = index + 1;
        this.nodeIdMap.set(node.id, mappedId);
        this.reverseNodeIdMap.set(mappedId, node.id);
      }
    });
    
    
    await graphWorkerProxy.setGraphData(validatedData);
    
    if (debugState.isDataDebugEnabled()) {
      logger.debug(`Graph data updated: ${validatedData.nodes.length} nodes, ${validatedData.edges.length} edges`);
    }
  }

  
  private validateNodeMappings(nodes: Node[]): void {
    if (debugState.isDataDebugEnabled()) {
      logger.debug(`Validated ${nodes.length} nodes with ID mapping`);
    }
  }

  
  public enableBinaryUpdates(): void {
    if (!this.webSocketService) {
      logger.warn('Cannot enable binary updates: WebSocket service not set');
      return;
    }

    
    if (this.webSocketService.isReady()) {
      this.setBinaryUpdatesEnabled(true);
      return;
    }

    
    if (this.retryTimeout) {
      window.clearTimeout(this.retryTimeout);
    }

    this.retryTimeout = window.setTimeout(() => {
      if (this.webSocketService && this.webSocketService.isReady()) {
        this.setBinaryUpdatesEnabled(true);
        if (debugState.isEnabled()) {
          logger.info('WebSocket ready, binary updates enabled');
        }
      } else {
        if (debugState.isEnabled()) {
          logger.info('WebSocket not ready yet, retrying...');
        }
        this.enableBinaryUpdates();
      }
    }, 500);
  }

  public setBinaryUpdatesEnabled(enabled: boolean): void {
    this.binaryUpdatesEnabled = enabled;
    
    if (debugState.isEnabled()) {
      logger.info(`Binary updates ${enabled ? 'enabled' : 'disabled'}`);
    }
  }

  
  public async getGraphData(): Promise<GraphData> {
    // Check both local flag AND proxy ready state (handles race condition)
    if (!this.workerInitialized && !graphWorkerProxy.isReady()) {
      console.warn('[GraphDataManager] Worker not initialized, returning empty data');
      return { nodes: [], edges: [] };
    }

    // Update local flag if proxy is ready but we missed initialization
    if (!this.workerInitialized && graphWorkerProxy.isReady()) {
      console.log('[GraphDataManager] Proxy ready, updating workerInitialized flag');
      this.workerInitialized = true;
      this.setupWorkerListeners();
    }

    try {
      return await graphWorkerProxy.getGraphData();
    } catch (error) {
      console.error('[GraphDataManager] Error getting data from worker:', error);
      logger.error('Error getting graph data from worker:', createErrorMetadata(error));
      return { nodes: [], edges: [] };
    }
  }

  
  public async addNode(node: Node): Promise<void> {
    
    const numericId = parseInt(node.id, 10);
    if (!isNaN(numericId)) {
      this.nodeIdMap.set(node.id, numericId);
      this.reverseNodeIdMap.set(numericId, node.id);
    } else {
      
      const currentData = await graphWorkerProxy.getGraphData();
      const mappedId = currentData.nodes.length + 1;
      this.nodeIdMap.set(node.id, mappedId);
      this.reverseNodeIdMap.set(mappedId, node.id);
    }
    
    await graphWorkerProxy.updateNode(node);
  }

  
  public async addEdge(edge: Edge): Promise<void> {
    
    const currentData = await graphWorkerProxy.getGraphData();
    const existingIndex = currentData.edges.findIndex(e => e.id === edge.id);
    
    if (existingIndex >= 0) {
      currentData.edges[existingIndex] = {
        ...currentData.edges[existingIndex],
        ...edge
      };
    } else {
      currentData.edges.push(edge);
    }
    
    await graphWorkerProxy.setGraphData(currentData);
  }

  
  public async removeNode(nodeId: string): Promise<void> {
    
    const numericId = this.nodeIdMap.get(nodeId);
    
    await graphWorkerProxy.removeNode(nodeId);
    
    
    if (numericId !== undefined) {
      this.nodeIdMap.delete(nodeId);
      this.reverseNodeIdMap.delete(numericId);
    }
  }

  
  public async removeEdge(edgeId: string): Promise<void> {
    
    const currentData = await graphWorkerProxy.getGraphData();
    currentData.edges = currentData.edges.filter(edge => edge.id !== edgeId);
    await graphWorkerProxy.setGraphData(currentData);
  }

  
  public async updateNodePositions(positionData: ArrayBuffer): Promise<void> {
    
    this.updateCount = (this.updateCount || 0) + 1;
    if (this.updateCount % 100 === 1) { 
      console.log('[GraphDataManager] updateNodePositions called, size:', positionData?.byteLength, 'graph type:', this.graphType, 'count:', this.updateCount);
    }
    
    if (!positionData || positionData.byteLength === 0) {
      if (this.updateCount % 100 === 1) {
        console.log('[GraphDataManager] No position data, returning');
      }
      return;
    }

    
    if (this.graphType !== 'logseq') {
      if (this.updateCount % 100 === 1) {
        console.log('[GraphDataManager] Skipping - not logseq graph, type is:', this.graphType);
      }
      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Skipping binary update for ${this.graphType} graph`);
      }
      return;
    }

    
    const now = Date.now();
    if (now - this.lastBinaryUpdateTime < 16) { 
      if (debugState.isDataDebugEnabled()) {
        logger.debug('Skipping duplicate position update');
      }
      return;
    }
    this.lastBinaryUpdateTime = now;

    try {
      
      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Received binary data: ${positionData.byteLength} bytes`);
        
        
        const remainder = positionData.byteLength % BINARY_NODE_SIZE;
        if (remainder !== 0) {
          logger.warn(`Binary data size (${positionData.byteLength} bytes) is not a multiple of ${BINARY_NODE_SIZE}. Remainder: ${remainder} bytes`);
        }
      }
      
      
      if (this.updateCount % 100 === 1) {
        console.log('[GraphDataManager] Sending to worker proxy for processing');
      }
      await graphWorkerProxy.processBinaryData(positionData);
      if (this.updateCount % 100 === 1) {
        console.log('[GraphDataManager] Worker proxy processing complete');
      }
      
      
      const settings = useSettingsStore.getState().settings;
      const debugEnabled = settings?.system?.debug?.enabled;
      const physicsDebugEnabled = (settings?.system?.debug as any)?.enablePhysicsDebug;
      const nodeDebugEnabled = (settings?.system?.debug as any)?.enableNodeDebug;
      
      if (debugEnabled && (physicsDebugEnabled || nodeDebugEnabled)) {
        const view = new DataView(positionData);
        const nodeCount = Math.min(3, positionData.byteLength / BINARY_NODE_SIZE);
        for (let i = 0; i < nodeCount; i++) {
          const offset = i * BINARY_NODE_SIZE;
          const x = view.getFloat32(offset + 4, true);
          const y = view.getFloat32(offset + 8, true);
          const z = view.getFloat32(offset + 12, true);
          logger.info(`[Physics Debug] Node ${i}: position(${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})`);
        }
      }
      
      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Processed binary data through worker`);
      }
    } catch (error) {
      logger.error('Error processing binary position data:', createErrorMetadata(error));
      
      
      if (debugState.isEnabled()) {
        try {
          
          const view = new DataView(positionData);
          const byteArray = [];
          const maxBytesToShow = Math.min(64, positionData.byteLength);
          
          for (let i = 0; i < maxBytesToShow; i++) {
            byteArray.push(view.getUint8(i).toString(16).padStart(2, '0'));
          }
          
          logger.debug(`First ${maxBytesToShow} bytes of binary data: ${byteArray.join(' ')}${positionData.byteLength > maxBytesToShow ? '...' : ''}`);
        } catch (e) {
          logger.debug('Could not display binary data preview:', e);
        }
      }
    }
  }

  
  
  public async sendNodePositions(): Promise<void> {
    if (!this.binaryUpdatesEnabled || !this.webSocketService || !this.isUserInteracting) {
      return;
    }

    try {
      
      const currentData = await graphWorkerProxy.getGraphData();
      
      
      const binaryNodes: BinaryNodeData[] = currentData.nodes
        .filter(node => node && node.id) 
        .map(node => {
          
          const validatedNode = this.ensureNodeHasValidPosition(node);
          
          
          const numericId = this.nodeIdMap.get(validatedNode.id) || 0;
          if (numericId === 0) {
            logger.warn(`No numeric ID found for node ${validatedNode.id}, skipping`);
            return null;
          }
          
          
          const velocity: Vec3 = (validatedNode.metadata?.velocity as Vec3) || { x: 0, y: 0, z: 0 };
          
          return {
            nodeId: numericId,
            position: {
              x: validatedNode.position.x || 0,
              y: validatedNode.position.y || 0,
              z: validatedNode.position.z || 0
            },
            velocity
          };
        })
        .filter((node): node is BinaryNodeData => node !== null);

      
      const buffer = createBinaryNodeData(binaryNodes);
      
      
      this.webSocketService.send(buffer);
      
      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Sent positions for ${binaryNodes.length} nodes using binary protocol`);
      }
    } catch (error) {
      logger.error('Error sending node positions:', createErrorMetadata(error));
    }
  }

  
  public onGraphDataChange(listener: GraphDataChangeListener): () => void {
    console.log('[GraphDataManager] Adding graph data change listener');
    this.graphDataListeners.push(listener);
    
    
    console.log('[GraphDataManager] Getting current data for new listener');
    graphWorkerProxy.getGraphData().then(data => {
      console.log(`[GraphDataManager] Calling listener with current data: ${data.nodes.length} nodes`);
      listener(data);
    }).catch(error => {
      console.error('[GraphDataManager] Error getting initial graph data for listener:', error);
      logger.error('Error getting initial graph data for listener:', createErrorMetadata(error));
      
      listener({ nodes: [], edges: [] });
    });
    
    
    return () => {
      console.log('[GraphDataManager] Removing graph data change listener');
      this.graphDataListeners = this.graphDataListeners.filter(l => l !== listener);
    };
  }

  
  public onPositionUpdate(listener: PositionUpdateListener): () => void {
    this.positionUpdateListeners.push(listener);
    
    
    return () => {
      this.positionUpdateListeners = this.positionUpdateListeners.filter(l => l !== listener);
    };
  }

  
  private async notifyGraphDataListeners(): Promise<void> {
    try {
      const currentData = await graphWorkerProxy.getGraphData();
      this.graphDataListeners.forEach(listener => {
        try {
          listener(currentData);
        } catch (error) {
          logger.error('Error in graph data listener:', createErrorMetadata(error));
        }
      });
    } catch (error) {
      logger.error('Error getting graph data for listeners:', createErrorMetadata(error));
    }
  }

  
  private notifyPositionUpdateListeners(positions: Float32Array): void {
    this.positionUpdateListeners.forEach(listener => {
      try {
        listener(positions);
      } catch (error) {
        logger.error('Error in position update listener:', createErrorMetadata(error));
      }
    });
  }

  
  public ensureNodeHasValidPosition(node: Node): Node {
    if (!node.position) {
      
      console.warn(`[GraphDataManager] Node ${node.id} missing position - server should provide this!`);
      return {
        ...node,
        position: { x: 0, y: 0, z: 0 }
      };
    } else if (typeof node.position.x !== 'number' ||
               typeof node.position.y !== 'number' ||
               typeof node.position.z !== 'number') {
      
      console.warn(`[GraphDataManager] Node ${node.id} has invalid position coordinates - fixing`);
      node.position.x = typeof node.position.x === 'number' && isFinite(node.position.x) ? node.position.x : 0;
      node.position.y = typeof node.position.y === 'number' && isFinite(node.position.y) ? node.position.y : 0;
      node.position.z = typeof node.position.z === 'number' && isFinite(node.position.z) ? node.position.z : 0;
    }
    return node;
  }

  
  public subscribeToUpdates(listener: GraphDataChangeListener): () => void {
    return this.onGraphDataChange(listener);
  }

  
  public getVisibleNodes(): Node[] {
    
    let nodes: Node[] = [];
    graphWorkerProxy.getGraphData().then(data => {
      nodes = data.nodes;
    }).catch(error => {
      logger.error('Error getting visible nodes:', createErrorMetadata(error));
    });
    return nodes;
  }

  
  public setUserInteracting(isInteracting: boolean): void {
    if (this.isUserInteracting === isInteracting) {
      return; 
    }

    this.isUserInteracting = isInteracting;

    if (isInteracting) {
      
      if (this.interactionTimeoutRef) {
        window.clearTimeout(this.interactionTimeoutRef);
        this.interactionTimeoutRef = null;
      }

      if (debugState.isEnabled()) {
        logger.debug('User interaction started - WebSocket position updates enabled');
      }
    } else {
      
      
      this.interactionTimeoutRef = window.setTimeout(() => {
        this.isUserInteracting = false;
        this.interactionTimeoutRef = null;

        if (debugState.isEnabled()) {
          logger.debug('User interaction ended - WebSocket position updates disabled');
        }
      }, 200); 
    }
  }

  
  public isUserCurrentlyInteracting(): boolean {
    return this.isUserInteracting;
  }

  
  public dispose(): void {
    if (this.retryTimeout) {
      window.clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }

    if (this.interactionTimeoutRef) {
      window.clearTimeout(this.interactionTimeoutRef);
      this.interactionTimeoutRef = null;
    }

    this.graphDataListeners = [];
    this.positionUpdateListeners = [];
    this.webSocketService = null;
    this.nodeIdMap.clear();
    this.reverseNodeIdMap.clear();
    this.isUserInteracting = false;

    if (debugState.isEnabled()) {
      logger.info('GraphDataManager disposed');
    }
  }
}

// Create singleton instance
export const graphDataManager = GraphDataManager.getInstance();

