import { createLogger } from '../../../utils/logger';
import { apiService } from '../../../services/apiService';
import type { BotsAgent, BotsEdge } from '../types/BotsTypes';
import { PollingPerformanceMonitor } from '../utils/pollingPerformance';

const logger = createLogger('AgentPollingService');

export interface AgentSwarmData {
  nodes: Array<{
    id: number;
    metadata_id: string;
    label: string;
    node_type: string;
    data?: {
      position: { x: number; y: number; z: number };
      velocity: { x: number; y: number; z: number };
    };
    metadata?: {
      agent_type: string;
      status: string;
      health: string;
      cpu_usage: string;
      memory_usage: string;
      workload: string;
      tokens: string;
      created_at: string;
      age: string;
      swarm_id?: string;
      parent_queen_id?: string;
      capabilities?: string;
    };
  }>;
  edges: Array<{
    id: string;
    source: number;
    target: number;
    weight: number;
  }>;
  metadata?: {
    total_agents: number;
    active_agents: number;
    total_tasks: number;
    completed_tasks: number;
    avg_success_rate: number;
    total_tokens: number;
  };
}

export interface PollingConfig {
  activePollingInterval: number; // Default 1000ms for active tasks
  idlePollingInterval: number;   // Default 5000ms for idle
  enableSmartPolling: boolean;   // Auto-adjust based on activity
  maxRetries: number;
  retryDelay: number;
}

type PollingCallback = (data: AgentSwarmData) => void;
type ErrorCallback = (error: Error) => void;

export class AgentPollingService {
  private static instance: AgentPollingService;
  private pollingTimer: NodeJS.Timeout | null = null;
  private config: PollingConfig;
  private callbacks: Set<PollingCallback> = new Set();
  private errorCallbacks: Set<ErrorCallback> = new Set();
  private isPolling: boolean = false;
  private lastPollTime: number = 0;
  private lastDataHash: string = '';
  private retryCount: number = 0;
  private currentInterval: number;
  private activityLevel: 'active' | 'idle' = 'idle';
  private lastActivityCheck: number = 0;
  private performanceMonitor: PollingPerformanceMonitor;

  private constructor() {
    this.config = {
      activePollingInterval: 1000,  // 1s for active tasks
      idlePollingInterval: 5000,    // 5s for idle
      enableSmartPolling: true,
      maxRetries: 3,
      retryDelay: 2000
    };
    this.currentInterval = this.config.idlePollingInterval;
    this.performanceMonitor = new PollingPerformanceMonitor();
  }

  public static getInstance(): AgentPollingService {
    if (!AgentPollingService.instance) {
      AgentPollingService.instance = new AgentPollingService();
    }
    return AgentPollingService.instance;
  }

  /**
   * Configure polling parameters
   */
  public configure(config: Partial<PollingConfig>): void {
    this.config = { ...this.config, ...config };
    logger.info('Polling configuration updated:', this.config);
    
    // Restart polling with new config if active
    if (this.isPolling) {
      this.stop();
      this.start();
    }
  }

  /**
   * Start polling for agent swarm data
   */
  public start(): void {
    if (this.isPolling) {
      logger.warn('Polling already active');
      return;
    }

    logger.info('Starting agent swarm polling');
    this.isPolling = true;
    this.poll();
  }

  /**
   * Stop polling
   */
  public stop(): void {
    if (this.pollingTimer) {
      clearTimeout(this.pollingTimer);
      this.pollingTimer = null;
    }
    this.isPolling = false;
    logger.info('Agent swarm polling stopped');
  }

  /**
   * Subscribe to polling updates
   */
  public subscribe(callback: PollingCallback, errorCallback?: ErrorCallback): () => void {
    this.callbacks.add(callback);
    if (errorCallback) {
      this.errorCallbacks.add(errorCallback);
    }

    // Return unsubscribe function
    return () => {
      this.callbacks.delete(callback);
      if (errorCallback) {
        this.errorCallbacks.delete(errorCallback);
      }
    };
  }

  /**
   * Force an immediate poll
   */
  public async pollNow(): Promise<void> {
    if (!this.isPolling) {
      logger.warn('Cannot poll now - polling is not active');
      return;
    }
    
    // Cancel existing timer
    if (this.pollingTimer) {
      clearTimeout(this.pollingTimer);
      this.pollingTimer = null;
    }
    
    await this.poll();
  }

  /**
   * Get performance metrics
   */
  public getPerformanceMetrics() {
    return this.performanceMonitor.getMetrics();
  }

  /**
   * Reset performance metrics
   */
  public resetPerformanceMetrics(): void {
    this.performanceMonitor.reset();
  }

  /**
   * Get current polling status
   */
  public getStatus() {
    return {
      isPolling: this.isPolling,
      currentInterval: this.currentInterval,
      activityLevel: this.activityLevel,
      lastPollTime: this.lastPollTime,
      retryCount: this.retryCount,
      subscriberCount: this.callbacks.size,
      performance: this.performanceMonitor.getSummary()
    };
  }

  /**
   * Main polling loop
   */
  private async poll(): Promise<void> {
    if (!this.isPolling) return;

    try {
      const startTime = Date.now();
      
      // Fetch agent swarm data from REST API
      const data = await apiService.get<AgentSwarmData>('/bots/data');
      
      const pollDuration = Date.now() - startTime;
      this.lastPollTime = Date.now();
      this.retryCount = 0; // Reset retry count on success

      // Check if data has changed (simple hash comparison)
      const dataHash = this.hashData(data);
      const hasChanged = dataHash !== this.lastDataHash;
      this.lastDataHash = dataHash;
      
      // Record performance metrics
      this.performanceMonitor.recordPoll(pollDuration, hasChanged);

      // Analyze activity level
      if (this.config.enableSmartPolling) {
        this.updateActivityLevel(data, hasChanged);
      }

      // Log poll metrics
      if (hasChanged || Date.now() - this.lastActivityCheck > 10000) {
        logger.debug('Poll completed', {
          duration: pollDuration,
          hasChanged,
          activityLevel: this.activityLevel,
          nodeCount: data.nodes?.length || 0,
          activeAgents: data.metadata?.active_agents || 0
        });
        this.lastActivityCheck = Date.now();
      }

      // Notify all subscribers if data changed
      if (hasChanged) {
        this.callbacks.forEach(callback => {
          try {
            callback(data);
          } catch (error) {
            logger.error('Error in polling callback:', error);
          }
        });
      }

    } catch (error) {
      this.handlePollingError(error as Error);
    }

    // Schedule next poll
    if (this.isPolling) {
      this.pollingTimer = setTimeout(() => this.poll(), this.currentInterval);
    }
  }

  /**
   * Update activity level based on agent metrics
   */
  private updateActivityLevel(data: AgentSwarmData, hasChanged: boolean): void {
    const activeAgents = data.metadata?.active_agents || 0;
    const totalAgents = data.metadata?.total_agents || 0;
    const activeRatio = totalAgents > 0 ? activeAgents / totalAgents : 0;

    // Determine activity level
    const wasActive = this.activityLevel === 'active';
    
    // Switch to active if:
    // - More than 20% of agents are active
    // - Data is changing frequently
    // - There are ongoing tasks
    if (activeRatio > 0.2 || hasChanged || (data.metadata?.total_tasks || 0) > (data.metadata?.completed_tasks || 0)) {
      this.activityLevel = 'active';
      this.currentInterval = this.config.activePollingInterval;
    } else {
      this.activityLevel = 'idle';
      this.currentInterval = this.config.idlePollingInterval;
    }

    // Log activity level changes
    if (wasActive !== (this.activityLevel === 'active')) {
      logger.info(`Activity level changed to ${this.activityLevel}, interval: ${this.currentInterval}ms`);
    }
  }

  /**
   * Handle polling errors with retry logic
   */
  private handlePollingError(error: Error): void {
    this.retryCount++;
    logger.error('Polling error:', error, { retryCount: this.retryCount });
    
    // Record error in performance monitor
    this.performanceMonitor.recordError();

    // Notify error callbacks
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (err) {
        logger.error('Error in error callback:', err);
      }
    });

    if (this.retryCount >= this.config.maxRetries) {
      logger.error('Max retries reached, stopping polling');
      this.stop();
      return;
    }

    // Exponential backoff for retries
    const retryDelay = this.config.retryDelay * Math.pow(2, this.retryCount - 1);
    logger.info(`Retrying poll in ${retryDelay}ms`);
    
    if (this.isPolling) {
      this.pollingTimer = setTimeout(() => this.poll(), retryDelay);
    }
  }

  /**
   * Simple hash function to detect data changes
   */
  private hashData(data: AgentSwarmData): string {
    const relevant = {
      nodeCount: data.nodes?.length || 0,
      edgeCount: data.edges?.length || 0,
      activeAgents: data.metadata?.active_agents || 0,
      totalTasks: data.metadata?.total_tasks || 0,
      completedTasks: data.metadata?.completed_tasks || 0,
      // Include agent positions for change detection
      positions: data.nodes?.map(n => 
        `${n.id}:${n.data?.position?.x?.toFixed(1)},${n.data?.position?.y?.toFixed(1)},${n.data?.position?.z?.toFixed(1)}`
      ).join('|')
    };
    return JSON.stringify(relevant);
  }
}

// Export singleton instance
export const agentPollingService = AgentPollingService.getInstance();