// Optimization API Client - GPU-accelerated graph optimization and clustering
import { createLogger } from '../utils/loggerConfig';
import { unifiedApiClient, isApiError } from '../services/api/UnifiedApiClient';
import { nostrAuth } from '../services/nostrAuthService';

const API_BASE = '';
const logger = createLogger('OptimizationAPI');

// Optimization task interfaces
export interface OptimizationTask {
  taskId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  algorithm: string;
  startTime: string;
  estimatedCompletion?: string;
  error?: string;
}

export interface OptimizationResult {
  taskId: string;
  algorithm: string;
  confidence: number;
  performanceGain: number;
  clusters: number;
  convergenceRate: number;
  iterations: number;
  gpuUtilization: number;
  recommendations: OptimizationRecommendation[];
  metrics: OptimizationMetrics;
}

export interface OptimizationRecommendation {
  type: 'layout' | 'clustering' | 'performance' | 'gpu' | 'algorithm';
  priority: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  impact: number;
  confidence: number;
  action: string;
}

export interface OptimizationMetrics {
  stressMajorization: {
    finalStress: number;
    stressReduction: number;
    convergenceIterations: number;
  };
  clustering: {
    modularity: number;
    silhouetteScore: number;
    clusters: number;
    coverage: number;
  };
  performance: {
    gpuMemoryUsage: number;
    computeTime: number;
    throughput: number;
    efficiency: number;
  };
}

export interface OptimizationParams {
  algorithm?: 'stress-majorization' | 'force-directed' | 'hierarchical' | 'adaptive';
  optimizationLevel: number; 
  clusteringEnabled: boolean;
  maxIterations?: number;
  convergenceThreshold?: number;
  gpuAcceleration?: boolean;
  performanceMode?: 'battery' | 'balanced' | 'performance' | 'extreme';
}

export interface GraphData {
  nodes: Array<{
    id: string;
    x?: number;
    y?: number;
    z?: number;
    group?: string;
    weight?: number;
    metadata?: Record<string, any>;
  }>;
  edges: Array<{
    source: string;
    target: string;
    weight?: number;
    metadata?: Record<string, any>;
  }>;
}

// API Client Class
class OptimizationApiClient {
  private readonly baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    method: string = 'GET',
    data?: any
  ): Promise<T> {
    try {
      const response = await unifiedApiClient.request<T>(method, `${this.baseUrl}${endpoint}`, data);
      return response.data;
    } catch (error) {
      if (isApiError(error)) {
        logger.error(`API request failed: ${endpoint}`, {
          status: error.status,
          message: error.message,
          data: error.data
        });
      } else {
        logger.error(`API request failed: ${endpoint}`, error);
      }
      throw error;
    }
  }

  
  async triggerStressMajorization(
    graphData: GraphData,
    params: OptimizationParams = {
      algorithm: 'stress-majorization',
      optimizationLevel: 3,
      clusteringEnabled: false,
      gpuAcceleration: true,
    }
  ): Promise<OptimizationTask> {
    logger.info('Triggering stress majorization optimization', { nodeCount: graphData.nodes.length, edgeCount: graphData.edges.length });

    return this.request<OptimizationTask>('/analytics/optimize/stress-majorization', 'POST', {
      graph: graphData,
      parameters: params,
    });
  }

  
  async getOptimizationStatus(taskId: string): Promise<OptimizationTask> {
    return this.request<OptimizationTask>(`/analytics/optimize/status/${taskId}`, 'GET');
  }

  
  async getOptimizationResults(taskId: string): Promise<OptimizationResult> {
    logger.info('Fetching optimization results', { taskId });

    return this.request<OptimizationResult>(`/analytics/optimize/results/${taskId}`, 'GET');
  }

  
  async updateOptimizationParams(
    taskId: string,
    params: Partial<OptimizationParams>
  ): Promise<{ success: boolean; message: string }> {
    logger.info('Updating optimization parameters', { taskId, params });

    return this.request<{ success: boolean; message: string }>(`/analytics/optimize/params/${taskId}`, 'PUT', params);
  }

  
  async cancelOptimization(taskId: string): Promise<{ success: boolean; message: string }> {
    logger.info('Cancelling optimization task', { taskId });

    return this.request<{ success: boolean; message: string }>(`/analytics/optimize/cancel/${taskId}`, 'POST');
  }

  
  async runClusteringAnalysis(
    graphData: GraphData,
    algorithm: 'modularity' | 'louvain' | 'leiden' | 'spectral' = 'louvain'
  ): Promise<OptimizationTask> {
    logger.info('Running clustering analysis', { algorithm, nodeCount: graphData.nodes.length });

    return this.request<OptimizationTask>('/analytics/clustering', 'POST', {
      graph: graphData,
      algorithm,
      gpu_acceleration: true,
    });
  }

  
  async getAvailableAlgorithms(): Promise<string[]> {
    return this.request<string[]>('/analytics/optimize/algorithms', 'GET');
  }

  
  async getGpuStatus(): Promise<{
    available: boolean;
    memory: { total: number; free: number; used: number };
    utilization: number;
    temperature?: number;
    compute_capability?: string;
  }> {
    return this.request('/analytics/gpu/status', 'GET');
  }

  
  async batchOptimization(
    tasks: Array<{
      graphData: GraphData;
      params: OptimizationParams;
      taskName?: string;
    }>
  ): Promise<OptimizationTask[]> {
    logger.info('Starting batch optimization', { taskCount: tasks.length });

    return this.request<OptimizationTask[]>('/analytics/optimize/batch', 'POST', { tasks });
  }

  
  async getOptimizationHistory(
    limit: number = 50
  ): Promise<OptimizationResult[]> {
    return this.request<OptimizationResult[]>(`/analytics/optimize/history?limit=${limit}`, 'GET');
  }

  
  async getPerformanceMetrics(): Promise<{
    gpu: {
      utilization: number;
      memory: number;
      temperature?: number;
    };
    optimization: {
      activeTasksCount: number;
      completedTasksToday: number;
      averageExecutionTime: number;
      successRate: number;
    };
    clustering: {
      averageModularity: number;
      averageClusterCount: number;
      processingSpeed: number;
    };
  }> {
    return this.request('/analytics/optimize/metrics', 'GET');
  }
}

// Export singleton instance
export const optimizationApi = new OptimizationApiClient();

// Export types and utilities
export { OptimizationApiClient };

// WebSocket event types for real-time optimization updates
export interface OptimizationWebSocketEvent {
  type: 'optimization_progress' | 'optimization_complete' | 'optimization_error' | 'gpu_status' | 'clustering_update';
  taskId?: string;
  data: any;
  timestamp: string;
}

// WebSocket connection manager for optimization events
export class OptimizationWebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners = new Map<string, Set<(event: OptimizationWebSocketEvent) => void>>();

  private readonly wsUrl: string;

  constructor(url?: string) {
    if (url) {
      this.wsUrl = url;
    } else {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      this.wsUrl = `${wsProtocol}//${window.location.host}/ws/optimization`;
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.wsUrl);

        const connectionTimeout = setTimeout(() => {
          if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
            this.ws.close();
            reject(new Error('WebSocket connection timeout'));
          }
        }, 5000);

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          // Send auth token as first message instead of URL param
          const token = nostrAuth.getSessionToken();
          if (token) {
            this.ws?.send(JSON.stringify({ type: 'auth', token }));
          }
          logger.info('Optimization WebSocket connected');
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data: OptimizationWebSocketEvent = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            logger.error('Failed to parse WebSocket message', error);
          }
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          logger.warn('Optimization WebSocket closed', { code: event.code, reason: event.reason });
          this.handleReconnect();
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          logger.warn('Optimization WebSocket error (backend may not be available)', error);
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  private handleMessage(event: OptimizationWebSocketEvent): void {
    const typeListeners = this.listeners.get(event.type);
    if (typeListeners) {
      typeListeners.forEach(listener => {
        try {
          listener(event);
        } catch (error) {
          logger.error('WebSocket event listener error', error);
        }
      });
    }

    
    const wildcardListeners = this.listeners.get('*');
    if (wildcardListeners) {
      wildcardListeners.forEach(listener => {
        try {
          listener(event);
        } catch (error) {
          logger.error('WebSocket wildcard listener error', error);
        }
      });
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

      logger.info(`Attempting WebSocket reconnect ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);

      setTimeout(() => {
        this.connect().catch((error) => {
          logger.error('WebSocket reconnect failed', error);
        });
      }, delay);
    } else {
      logger.error('Max WebSocket reconnect attempts reached');
    }
  }

  addEventListener(
    type: string,
    listener: (event: OptimizationWebSocketEvent) => void
  ): void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(listener);
  }

  removeEventListener(
    type: string,
    listener: (event: OptimizationWebSocketEvent) => void
  ): void {
    const typeListeners = this.listeners.get(type);
    if (typeListeners) {
      typeListeners.delete(listener);
      if (typeListeners.size === 0) {
        this.listeners.delete(type);
      }
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Export singleton WebSocket manager
export const optimizationWebSocket = new OptimizationWebSocketManager();