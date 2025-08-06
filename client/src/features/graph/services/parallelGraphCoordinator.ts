/**
 * Parallel Graph Coordinator
 * Manages both Logseq and VisionFlow graphs running in parallel
 */

import { createLogger } from '../../../utils/logger';
import { graphDataManager } from '../managers/graphDataManager';
import { botsPhysicsWorker } from '../../bots/workers/botsPhysicsWorker';
import { apiService } from '../../../services/apiService';
import type { GraphData } from '../managers/graphWorkerProxy';
import type { BotsAgent, BotsEdge, TokenUsage, BotsVisualConfig } from '../../bots/types/BotsTypes';

const logger = createLogger('ParallelGraphCoordinator');

export interface ParallelGraphState {
  logseq: {
    enabled: boolean;
    data: GraphData | null;
    lastUpdate: number;
  };
  visionflow: {
    enabled: boolean;
    agents: BotsAgent[];
    edges: BotsEdge[];
    tokenUsage: TokenUsage | null;
    lastUpdate: number;
  };
}

class ParallelGraphCoordinator {
  private static instance: ParallelGraphCoordinator;
  private state: ParallelGraphState = {
    logseq: {
      enabled: false,
      data: null,
      lastUpdate: 0
    },
    visionflow: {
      enabled: false,
      agents: [],
      edges: [],
      tokenUsage: null,
      lastUpdate: 0
    }
  };
  private listeners: Set<(state: ParallelGraphState) => void> = new Set();

  private constructor() {}

  public static getInstance(): ParallelGraphCoordinator {
    if (!ParallelGraphCoordinator.instance) {
      ParallelGraphCoordinator.instance = new ParallelGraphCoordinator();
    }
    return ParallelGraphCoordinator.instance;
  }

  /**
   * Initialize both graph systems
   */
  public async initialize(): Promise<void> {
    logger.info('Initializing parallel graph coordinator');

    // Configure graph data manager for Logseq
    graphDataManager.setGraphType('logseq');

    // Configure bots physics worker for VisionFlow
    botsPhysicsWorker.setDataType('visionflow');

    // Set up listeners for both data sources
    this.setupLogseqListeners();
    this.setupVisionFlowListeners();

    logger.info('Parallel graph coordinator initialized');
  }

  /**
   * Enable/disable Logseq graph
   */
  public setLogseqEnabled(enabled: boolean): void {
    this.state.logseq.enabled = enabled;
    logger.info(`Logseq graph ${enabled ? 'enabled' : 'disabled'}`);
    this.notifyListeners();

    if (enabled) {
      // Start fetching Logseq data
      this.fetchLogseqData();
    }
  }

  /**
   * Enable/disable VisionFlow graph
   */
  public setVisionFlowEnabled(enabled: boolean): void {
    this.state.visionflow.enabled = enabled;
    logger.info(`VisionFlow graph ${enabled ? 'enabled' : 'disabled'}`);
    this.notifyListeners();

    if (enabled) {
      // Start polling for VisionFlow data
      this.startVisionFlowPolling();
    } else {
      // Stop polling
      this.stopVisionFlowPolling();
    }
  }

  /**
   * Get current state of both graphs
   */
  public getState(): ParallelGraphState {
    return { ...this.state };
  }

  /**
   * Subscribe to state changes
   */
  public onStateChange(listener: (state: ParallelGraphState) => void): () => void {
    this.listeners.add(listener);
    // Immediately notify with current state
    listener(this.getState());

    // Return unsubscribe function
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Setup Logseq data listeners
   */
  private setupLogseqListeners(): void {
    // Listen for graph data changes from Logseq
    graphDataManager.onGraphDataChange((data) => {
      if (this.state.logseq.enabled) {
        this.state.logseq.data = data;
        this.state.logseq.lastUpdate = Date.now();
        this.notifyListeners();
      }
    });
  }

  /**
   * Setup VisionFlow data listeners
   */
  private setupVisionFlowListeners(): void {
    // VisionFlow data is now fetched via polling the REST API
    // Position updates come through the main WebSocket binary protocol
    logger.info('VisionFlow listeners configured for REST API polling');
  }

  private pollingInterval: NodeJS.Timeout | null = null;

  /**
   * Start polling for VisionFlow data
   */
  private async startVisionFlowPolling(): Promise<void> {
    // Clear any existing interval
    this.stopVisionFlowPolling();

    // Initial fetch
    await this.fetchVisionFlowData();

    // Poll every 5 seconds (matching backend polling rate)
    this.pollingInterval = setInterval(() => {
      this.fetchVisionFlowData();
    }, 5000);
  }

  /**
   * Stop polling for VisionFlow data
   */
  private stopVisionFlowPolling(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
  }

  /**
   * Fetch VisionFlow data from REST API
   */
  private async fetchVisionFlowData(): Promise<void> {
    if (!this.state.visionflow.enabled) {
      return;
    }

    try {
      // Fetch bots data from backend API
      const botsData = await apiService.getBotsData();
      
      if (botsData && botsData.nodes && botsData.edges) {
        // Update physics simulation
        botsPhysicsWorker.updateAgents(botsData.nodes);
        botsPhysicsWorker.updateEdges(botsData.edges);

        // Update state
        this.state.visionflow.agents = botsData.nodes;
        this.state.visionflow.edges = botsData.edges;
        this.state.visionflow.lastUpdate = Date.now();

        this.notifyListeners();
      }
    } catch (error) {
      logger.error('Error fetching VisionFlow data:', error);
    }
  }

  /**
   * Fetch Logseq graph data
   */
  private async fetchLogseqData(): Promise<void> {
    if (!this.state.logseq.enabled) {
      return;
    }

    try {
      const data = await graphDataManager.fetchInitialData();
      this.state.logseq.data = data;
      this.state.logseq.lastUpdate = Date.now();
      this.notifyListeners();
    } catch (error) {
      logger.error('Error fetching Logseq data:', error);
    }
  }


  /**
   * Notify all listeners of state changes
   */
  private notifyListeners(): void {
    const state = this.getState();
    this.listeners.forEach(listener => {
      try {
        listener(state);
      } catch (error) {
        logger.error('Error in state change listener:', error);
      }
    });
  }

  /**
   * Get Logseq node positions
   */
  public async getLogseqPositions(): Promise<Map<string, { x: number; y: number; z: number }>> {
    if (!this.state.logseq.enabled || !this.state.logseq.data) {
      return new Map();
    }

    const positions = new Map();
    this.state.logseq.data.nodes.forEach(node => {
      positions.set(node.id, {
        x: node.position.x,
        y: node.position.y,
        z: node.position.z
      });
    });

    return positions;
  }

  /**
   * Get VisionFlow node positions
   */
  public getVisionFlowPositions(): Map<string, { x: number; y: number; z: number }> {
    if (!this.state.visionflow.enabled) {
      return new Map();
    }

    return botsPhysicsWorker.getPositions();
  }

  /**
   * Update physics configuration
   */
  public updatePhysicsConfig(config: Partial<BotsVisualConfig['physics']>): void {
    botsPhysicsWorker.updateConfig(config);
    logger.debug('Updated physics configuration in parallel graph coordinator:', config);
  }

  /**
   * Cleanup resources
   */
  public dispose(): void {
    this.listeners.clear();
    this.stopVisionFlowPolling();
    logger.info('Parallel graph coordinator disposed');
  }
}

// Export singleton instance
export const parallelGraphCoordinator = ParallelGraphCoordinator.getInstance();