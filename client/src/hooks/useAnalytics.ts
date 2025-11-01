import { useState, useEffect, useCallback, useRef } from 'react';
import { analyticsAPI, AnalysisTask, GPUPerformanceStats, VisualAnalyticsParams } from '../api/analyticsApi';
import { createLogger } from '../utils/loggerConfig';
import { createErrorMetadata } from '../utils/loggerConfig';
import type { GraphNode, GraphEdge } from '../features/graph/types/graphTypes';

const logger = createLogger('useAnalytics');

export interface AnalyticsState {
  
  params: VisualAnalyticsParams | null;
  paramsLoading: boolean;

  
  performanceStats: GPUPerformanceStats | null;
  statsLoading: boolean;

  
  structuralAnalysis: any | null;
  semanticAnalysis: any | null;
  clusteringResults: any | null;
  anomalies: any[] | null;

  
  activeTasks: Map<string, AnalysisTask>;
  taskHistory: AnalysisTask[];

  
  gpuStatus: any | null;
  isGPUEnabled: boolean;

  
  isAnalyzing: boolean;
  error: string | null;
  lastUpdate: Date | null;
}

export interface UseAnalyticsOptions {
  autoRefreshStats?: boolean;
  refreshInterval?: number;
  enableWebSocket?: boolean;
  cachingEnabled?: boolean;
  retryAttempts?: number;
}

export interface AnalysisRequest {
  type: 'structural' | 'semantic' | 'clustering' | 'anomaly';
  graphData: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  options?: any;
}


export function useAnalytics(options: UseAnalyticsOptions = {}) {
  const {
    autoRefreshStats = true,
    refreshInterval = 5000,
    enableWebSocket = true,
    cachingEnabled = true,
    retryAttempts = 3
  } = options;

  
  const [state, setState] = useState<AnalyticsState>({
    params: null,
    paramsLoading: false,
    performanceStats: null,
    statsLoading: false,
    structuralAnalysis: null,
    semanticAnalysis: null,
    clusteringResults: null,
    anomalies: null,
    activeTasks: new Map(),
    taskHistory: [],
    gpuStatus: null,
    isGPUEnabled: false,
    isAnalyzing: false,
    error: null,
    lastUpdate: null
  });

  
  const refreshInterval_ref = useRef<NodeJS.Timeout | null>(null);
  const taskSubscriptions = useRef<Map<string, () => void>>(new Map());
  const resultCache = useRef<Map<string, { result: any; timestamp: number }>>(new Map());

  
  const updateState = useCallback((updates: Partial<AnalyticsState>) => {
    setState(prev => ({
      ...prev,
      ...updates,
      lastUpdate: new Date()
    }));
  }, []);

  
  const handleError = useCallback((error: any, operation: string, retry?: () => Promise<void>) => {
    logger.error(`Analytics ${operation} failed:`, createErrorMetadata(error));
    updateState({
      error: error.message || `${operation} failed`,
      isAnalyzing: false
    });

    if (retry && retryAttempts > 0) {
      
      setTimeout(retry, 1000);
    }
  }, [retryAttempts, updateState]);

  
  const loadParams = useCallback(async () => {
    try {
      updateState({ paramsLoading: true, error: null });
      const params = await analyticsAPI.getAnalyticsParams();
      updateState({
        params,
        paramsLoading: false,
        isGPUEnabled: params.gpu_acceleration
      });
    } catch (error) {
      handleError(error, 'parameter loading');
      updateState({ paramsLoading: false });
    }
  }, [handleError, updateState]);

  
  const updateParams = useCallback(async (newParams: Partial<VisualAnalyticsParams>) => {
    try {
      updateState({ paramsLoading: true, error: null });
      await analyticsAPI.updateAnalyticsParams(newParams);

      
      const updatedParams = await analyticsAPI.getAnalyticsParams();
      updateState({
        params: updatedParams,
        paramsLoading: false,
        isGPUEnabled: updatedParams.gpu_acceleration
      });
    } catch (error) {
      handleError(error, 'parameter update');
      updateState({ paramsLoading: false });
    }
  }, [handleError, updateState]);

  
  const loadPerformanceStats = useCallback(async () => {
    try {
      updateState({ statsLoading: true });
      const stats = await analyticsAPI.getPerformanceStats();
      updateState({
        performanceStats: stats,
        statsLoading: false,
        isGPUEnabled: stats.gpu_enabled
      });
    } catch (error) {
      logger.warn('Failed to load performance stats:', createErrorMetadata(error));
      updateState({ statsLoading: false });
    }
  }, [updateState]);

  
  const loadGPUStatus = useCallback(async () => {
    try {
      const gpuStatus = await analyticsAPI.getGPUStatus();
      updateState({
        gpuStatus,
        isGPUEnabled: gpuStatus.gpu_available
      });
    } catch (error) {
      logger.warn('Failed to load GPU status:', createErrorMetadata(error));
    }
  }, [updateState]);

  
  const runAnalysis = useCallback(async (request: AnalysisRequest): Promise<string> => {
    try {
      updateState({ isAnalyzing: true, error: null });

      let taskId: string;

      switch (request.type) {
        case 'structural':
          taskId = await analyticsAPI.runStructuralAnalysis({
            graph_data: {
              nodes: request.graphData.nodes.map(n => ({ id: n.id, ...n })),
              edges: request.graphData.edges.map(e => ({
                source: e.source,
                target: e.target,
                ...e
              }))
            },
            analysis_type: 'comprehensive',
            options: request.options
          });
          break;

        case 'semantic':
          taskId = await analyticsAPI.runSemanticAnalysis({
            graph_data: {
              nodes: request.graphData.nodes.map(n => ({
                id: n.id,
                content: n.label || n.id,
                metadata: n
              })),
              edges: request.graphData.edges.map(e => ({
                source: e.source,
                target: e.target,
                weight: e.weight || 1
              }))
            },
            analysis_type: 'similarity',
            options: request.options
          });
          break;

        case 'clustering':
          taskId = await analyticsAPI.runClustering({
            algorithm: 'louvain',
            resolution: request.options?.resolution || 1.0,
            gpu_accelerated: true,
            ...request.options
          });
          break;

        default:
          throw new Error(`Unsupported analysis type: ${request.type}`);
      }

      
      const task: AnalysisTask = {
        task_id: taskId,
        task_type: request.type,
        status: 'pending',
        progress: 0,
        start_time: new Date().toISOString()
      };

      updateState({
        activeTasks: new Map(state.activeTasks.set(taskId, task))
      });

      
      if (enableWebSocket) {
        const unsubscribe = analyticsAPI.subscribeToTask(taskId, (updatedTask) => {
          updateState({
            activeTasks: new Map(state.activeTasks.set(taskId, updatedTask))
          });

          
          if (updatedTask.status === 'completed' || updatedTask.status === 'failed') {
            handleTaskCompletion(updatedTask);
          }
        });

        taskSubscriptions.current.set(taskId, unsubscribe);
      }

      return taskId;
    } catch (error) {
      handleError(error, 'analysis execution');
      updateState({ isAnalyzing: false });
      throw error;
    }
  }, [state.activeTasks, enableWebSocket, updateState, handleError]);

  
  const handleTaskCompletion = useCallback((task: AnalysisTask) => {
    
    setState(prev => {
      const newActiveTasks = new Map(prev.activeTasks);
      newActiveTasks.delete(task.task_id);
      return {
        ...prev,
        activeTasks: newActiveTasks,
        taskHistory: [...prev.taskHistory, task],
        isAnalyzing: newActiveTasks.size > 0
      };
    });

    
    if (task.status === 'completed' && task.result && cachingEnabled) {
      resultCache.current.set(task.task_id, {
        result: task.result,
        timestamp: Date.now()
      });
    }

    
    if (task.status === 'completed' && task.result) {
      const updates: Partial<AnalyticsState> = {};

      switch (task.task_type) {
        case 'structural':
          updates.structuralAnalysis = task.result;
          break;
        case 'semantic':
          updates.semanticAnalysis = task.result;
          break;
        case 'clustering':
          updates.clusteringResults = task.result;
          break;
      }

      updateState(updates);
    }

    
    const unsubscribe = taskSubscriptions.current.get(task.task_id);
    if (unsubscribe) {
      unsubscribe();
      taskSubscriptions.current.delete(task.task_id);
    }
  }, [cachingEnabled, updateState]);

  
  const getCachedResult = useCallback((taskId: string) => {
    if (!cachingEnabled) return null;

    const cached = resultCache.current.get(taskId);
    if (cached && Date.now() - cached.timestamp < 300000) { 
      return cached.result;
    }

    resultCache.current.delete(taskId);
    return null;
  }, [cachingEnabled]);

  
  const cancelTask = useCallback(async (taskId: string) => {
    try {
      await analyticsAPI.cancelTask(taskId);

      setState(prev => {
        const newActiveTasks = new Map(prev.activeTasks);
        newActiveTasks.delete(taskId);
        return {
          ...prev,
          activeTasks: newActiveTasks,
          isAnalyzing: newActiveTasks.size > 0
        };
      });

      
      const unsubscribe = taskSubscriptions.current.get(taskId);
      if (unsubscribe) {
        unsubscribe();
        taskSubscriptions.current.delete(taskId);
      }
    } catch (error) {
      handleError(error, 'task cancellation');
    }
  }, [handleError]);

  
  const loadAnomalies = useCallback(async () => {
    try {
      const anomalies = await analyticsAPI.getCurrentAnomalies();
      updateState({ anomalies });
    } catch (error) {
      logger.warn('Failed to load anomalies:', createErrorMetadata(error));
    }
  }, [updateState]);

  
  const configureAnomalyDetection = useCallback(async (config: any) => {
    try {
      await analyticsAPI.configureAnomalyDetection(config);
      await loadAnomalies(); 
    } catch (error) {
      handleError(error, 'anomaly detection configuration');
    }
  }, [loadAnomalies, handleError]);

  
  const refresh = useCallback(async () => {
    await Promise.allSettled([
      loadParams(),
      loadPerformanceStats(),
      loadGPUStatus(),
      loadAnomalies()
    ]);
  }, [loadParams, loadPerformanceStats, loadGPUStatus, loadAnomalies]);

  
  useEffect(() => {
    if (autoRefreshStats) {
      refreshInterval_ref.current = setInterval(() => {
        loadPerformanceStats();
      }, refreshInterval);
    }

    return () => {
      if (refreshInterval_ref.current) {
        clearInterval(refreshInterval_ref.current);
      }
    };
  }, [autoRefreshStats, refreshInterval, loadPerformanceStats]);

  
  useEffect(() => {
    refresh();
  }, []);

  
  useEffect(() => {
    return () => {
      
      if (refreshInterval_ref.current) {
        clearInterval(refreshInterval_ref.current);
      }

      
      taskSubscriptions.current.forEach(unsubscribe => unsubscribe());
      taskSubscriptions.current.clear();

      
      resultCache.current.clear();

      
      analyticsAPI.cleanup();
    };
  }, []);

  return {
    
    ...state,

    
    loadParams,
    updateParams,
    loadPerformanceStats,
    loadGPUStatus,
    runAnalysis,
    cancelTask,
    loadAnomalies,
    configureAnomalyDetection,
    refresh,
    getCachedResult,

    
    hasActiveTasks: state.activeTasks.size > 0,
    totalTasks: state.activeTasks.size + state.taskHistory.length,
    completedTasks: state.taskHistory.filter(t => t.status === 'completed').length,
    failedTasks: state.taskHistory.filter(t => t.status === 'failed').length,
    successRate: state.taskHistory.length > 0
      ? state.taskHistory.filter(t => t.status === 'completed').length / state.taskHistory.length
      : 0
  };
}