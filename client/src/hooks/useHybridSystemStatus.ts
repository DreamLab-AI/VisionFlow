import { useState, useEffect, useCallback, useRef } from 'react';
import { createLogger } from '../utils/loggerConfig';
import { unifiedApiClient } from '../services/api/UnifiedApiClient';

const logger = createLogger('useHybridSystemStatus');

export interface HybridSystemStatus {
  dockerHealth: 'healthy' | 'degraded' | 'unavailable' | 'unknown';
  mcpHealth: 'connected' | 'reconnecting' | 'disconnected' | 'unknown';
  activeSessions: SessionInfo[];
  telemetryDelay: number;
  networkLatency: number;
  containerHealth?: ContainerHealth;
  lastUpdated: string;
  systemStatus: 'healthy' | 'degraded' | 'critical' | 'unknown';
  failoverActive: boolean;
  performance: PerformanceMetrics;
}

export interface SessionInfo {
  sessionId: string;
  taskDescription: string;
  status: 'spawning' | 'active' | 'paused' | 'completed' | 'failed' | 'unknown';
  createdAt: string;
  lastActivity: string;
  activeWorkers: number;
  method: 'docker' | 'mcp' | 'hybrid';
}

export interface ContainerHealth {
  isRunning: boolean;
  cpuUsage: number;
  memoryUsage: number;
  networkHealthy: boolean;
  diskSpaceGb: number;
  lastResponseMs: number;
  timestamp: string;
}

export interface PerformanceMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTimeMs: number;
  cacheHitRatio: number;
  connectionPoolUtilization: number;
  memoryUsageMb: number;
  activeOptimizations: string[];
}

export interface UseHybridSystemStatusOptions {
  pollingInterval?: number;
  enableWebSocket?: boolean;
  enablePerformanceMetrics?: boolean;
  enableHealthChecks?: bool;
  autoReconnect?: boolean;
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
}

const defaultOptions: UseHybridSystemStatusOptions = {
  pollingInterval: 30000, 
  enableWebSocket: true,
  enablePerformanceMetrics: true,
  enableHealthChecks: true,
  autoReconnect: true,
  reconnectDelay: 5000,
  maxReconnectAttempts: 5,
};

export const useHybridSystemStatus = (options: UseHybridSystemStatusOptions = {}) => {
  const opts = { ...defaultOptions, ...options };

  const [status, setStatus] = useState<HybridSystemStatus>({
    dockerHealth: 'unknown',
    mcpHealth: 'unknown',
    activeSessions: [],
    telemetryDelay: 0,
    networkLatency: 0,
    lastUpdated: new Date().toISOString(),
    systemStatus: 'unknown',
    failoverActive: false,
    performance: {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTimeMs: 0,
      cacheHitRatio: 0,
      connectionPoolUtilization: 0,
      memoryUsageMb: 0,
      activeOptimizations: [],
    },
  });

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  
  const fetchStatus = useCallback(async (): Promise<HybridSystemStatus | null> => {
    try {
      const startTime = Date.now();

      const response = await unifiedApiClient.get('/hybrid/status');
      const networkLatency = Date.now() - startTime;
      const data = response.data;

      
      return {
        ...data,
        networkLatency,
        lastUpdated: new Date().toISOString(),
      };
    } catch (err) {
      console.error('Failed to fetch hybrid system status:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      return null;
    }
  }, []);

  
  const initializeWebSocket = useCallback(() => {
    if (!opts.enableWebSocket) return;

    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        logger.info('WebSocket connected successfully');
        setError(null);
        setReconnectAttempts(0);

        
        ws.send(JSON.stringify({ type: 'request_status' }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'status_update') {
            setStatus(prevStatus => ({
              ...prevStatus,
              ...data.payload,
              lastUpdated: new Date().toISOString(),
            }));
            setIsLoading(false);
          } else if (data.type === 'performance_update') {
            setStatus(prevStatus => ({
              ...prevStatus,
              performance: {
                ...prevStatus.performance,
                ...data.payload,
              },
              lastUpdated: new Date().toISOString(),
            }));
          } else if (data.type === 'session_update') {
            setStatus(prevStatus => ({
              ...prevStatus,
              activeSessions: data.payload.sessions || [],
              lastUpdated: new Date().toISOString(),
            }));
          } else if (data.type === 'health_update') {
            setStatus(prevStatus => ({
              ...prevStatus,
              dockerHealth: data.payload.dockerHealth || prevStatus.dockerHealth,
              mcpHealth: data.payload.mcpHealth || prevStatus.mcpHealth,
              containerHealth: data.payload.containerHealth,
              systemStatus: data.payload.systemStatus || prevStatus.systemStatus,
              lastUpdated: new Date().toISOString(),
            }));
          }
        } catch (parseError) {
          console.error('Failed to parse WebSocket message:', parseError);
        }
      };

      ws.onerror = (error) => {
        console.error('Hybrid system status WebSocket error:', error);
        setError('WebSocket connection error');
      };

      ws.onclose = (event) => {
        logger.warn('WebSocket closed', { code: event.code, reason: event.reason });
        wsRef.current = null;

        if (opts.autoReconnect && reconnectAttempts < (opts.maxReconnectAttempts || 5)) {
          setReconnectAttempts(prev => prev + 1);

          reconnectTimeoutRef.current = setTimeout(() => {
            logger.info(`Attempting WebSocket reconnection`, { attempt: reconnectAttempts + 1, maxAttempts: opts.maxReconnectAttempts });
            initializeWebSocket();
          }, opts.reconnectDelay || 5000);
        } else {
          setError('WebSocket connection lost and max reconnect attempts reached');
        }
      };

    } catch (err) {
      console.error('Failed to initialize WebSocket:', err);
      setError('Failed to initialize WebSocket connection');
    }
  }, [opts.enableWebSocket, opts.autoReconnect, opts.reconnectDelay, opts.maxReconnectAttempts, reconnectAttempts]);

  
  const startPolling = useCallback(() => {
    if (opts.enableWebSocket) return; 

    const poll = async () => {
      const newStatus = await fetchStatus();
      if (newStatus) {
        setStatus(newStatus);
        setIsLoading(false);
      }
    };

    
    poll();

    
    pollingRef.current = setInterval(poll, opts.pollingInterval || 30000);
  }, [opts.enableWebSocket, opts.pollingInterval, fetchStatus]);

  
  const cleanup = useCallback(() => {
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }

    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  
  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    const newStatus = await fetchStatus();
    if (newStatus) {
      setStatus(newStatus);
    }
    setIsLoading(false);
  }, [fetchStatus]);

  
  const reconnect = useCallback(() => {
    cleanup();
    setReconnectAttempts(0);
    setError(null);

    if (opts.enableWebSocket) {
      initializeWebSocket();
    } else {
      startPolling();
    }
  }, [cleanup, opts.enableWebSocket, initializeWebSocket, startPolling]);

  
  const spawnSwarm = useCallback(async (
    taskDescription: string,
    config?: {
      priority?: 'low' | 'medium' | 'high' | 'critical';
      strategy?: 'strategic' | 'tactical' | 'adaptive' | 'hive-mind';
      method?: 'docker' | 'mcp' | 'hybrid';
      maxWorkers?: number;
      autoScale?: boolean;
    }
  ) => {
    try {
      const response = await unifiedApiClient.post('/hybrid/spawn-swarm', {
        task: taskDescription,
        ...config,
      });

      const result = response.data;

      
      setTimeout(refresh, 1000);

      return result;
    } catch (err) {
      console.error('Failed to spawn swarm:', err);
      throw err;
    }
  }, [refresh]);

  
  const stopSwarm = useCallback(async (sessionId: string) => {
    try {
      const response = await unifiedApiClient.post(`/hybrid/swarm/${sessionId}/stop`);

      
      setTimeout(refresh, 1000);

      return response.data;
    } catch (err) {
      console.error('Failed to stop swarm:', err);
      throw err;
    }
  }, [refresh]);

  
  const getPerformanceReport = useCallback(async () => {
    try {
      const response = await unifiedApiClient.get('/hybrid/performance-report');
      return response.data;
    } catch (err) {
      console.error('Failed to get performance report:', err);
      throw err;
    }
  }, []);

  
  useEffect(() => {
    if (opts.enableWebSocket) {
      initializeWebSocket();
    } else {
      startPolling();
    }

    return cleanup;
  }, [opts.enableWebSocket, initializeWebSocket, startPolling, cleanup]);

  
  const isSystemHealthy = status.systemStatus === 'healthy';
  const isSystemDegraded = status.systemStatus === 'degraded';
  const isSystemCritical = status.systemStatus === 'critical';

  
  const isDockerAvailable = status.dockerHealth === 'healthy' || status.dockerHealth === 'degraded';
  const isMcpAvailable = status.mcpHealth === 'connected' || status.mcpHealth === 'reconnecting';
  const isConnected = opts.enableWebSocket ? wsRef.current?.readyState === WebSocket.OPEN : !error;

  return {
    
    status,
    isLoading,
    error,
    reconnectAttempts,

    
    isSystemHealthy,
    isSystemDegraded,
    isSystemCritical,
    isDockerAvailable,
    isMcpAvailable,
    isConnected,

    
    refresh,
    reconnect,
    spawnSwarm,
    stopSwarm,
    getPerformanceReport,

    
    wsRef,
  };
};

export default useHybridSystemStatus;