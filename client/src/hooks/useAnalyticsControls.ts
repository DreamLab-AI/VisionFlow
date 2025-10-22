import { useState, useCallback } from 'react';
import { unifiedApiClient } from '../services/api/UnifiedApiClient';
import { useSettingsStore } from '../store/settingsStore';
import { createLogger } from '../utils/loggerConfig';
import { createErrorMetadata } from '../utils/loggerConfig';

const logger = createLogger('useAnalyticsControls');

export interface ClusteringParams {
  method: string;
  numClusters?: number;
  resolution?: number;
  iterations?: number;
}

export interface ClusteringResult {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  clusters?: any;
  error?: string;
}

export interface CommunityDetectionParams {
  algorithm?: string;
  resolution?: number;
}

export interface AnalyticsStats {
  gpu_enabled: boolean;
  total_tasks: number;
  active_tasks: number;
  completed_tasks: number;
  [key: string]: any;
}

/**
 * Hook for controlling analytics operations with backend API integration
 */
export const useAnalyticsControls = () => {
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const settings = useSettingsStore(state => state.settings);

  /**
   * Run clustering algorithm
   * Backend endpoint: POST /api/analytics/clustering/run
   */
  const runClustering = useCallback(async (customParams?: ClusteringParams): Promise<string | null> => {
    setRunning(true);
    setError(null);
    setStatus('starting');

    try {
      const params = customParams || {
        method: settings.analytics?.clustering?.algorithm || 'louvain',
        numClusters: settings.analytics?.clustering?.clusterCount,
        resolution: settings.analytics?.clustering?.resolution || 1.0,
        iterations: settings.analytics?.clustering?.iterations || 50
      };

      logger.info('Starting clustering with params:', params);

      const response = await unifiedApiClient.postData<ClusteringResult>(
        '/api/analytics/clustering/run',
        params
      );

      setStatus('running');
      setResults(response);
      logger.info('Clustering started successfully:', response);

      return response.task_id;
    } catch (err: any) {
      const errorMsg = err.message || 'Clustering failed';
      logger.error('Clustering failed:', createErrorMetadata(err));
      setError(errorMsg);
      setStatus('failed');
      return null;
    } finally {
      setRunning(false);
    }
  }, [settings.analytics]);

  /**
   * Check clustering status
   * Backend endpoint: GET /api/analytics/clustering/status?task_id={taskId}
   */
  const checkClusteringStatus = useCallback(async (taskId: string): Promise<ClusteringResult | null> => {
    try {
      const response = await unifiedApiClient.getData<ClusteringResult>(
        `/api/analytics/clustering/status?task_id=${taskId}`
      );

      setStatus(response.status);
      if (response.status === 'completed') {
        setResults(response);
      } else if (response.status === 'failed') {
        setError(response.error || 'Clustering failed');
      }

      return response;
    } catch (err: any) {
      logger.error('Failed to check clustering status:', createErrorMetadata(err));
      setError(err.message || 'Status check failed');
      return null;
    }
  }, []);

  /**
   * Run community detection
   * Backend endpoint: POST /api/analytics/community/detect
   */
  const runCommunityDetection = useCallback(async (params?: CommunityDetectionParams) => {
    setRunning(true);
    setError(null);

    try {
      const detectionParams = params || {
        algorithm: settings.analytics?.clustering?.algorithm || 'louvain',
        resolution: settings.analytics?.clustering?.resolution || 1.0
      };

      logger.info('Starting community detection:', detectionParams);

      const response = await unifiedApiClient.postData(
        '/api/analytics/community/detect',
        detectionParams
      );

      setResults(response);
      logger.info('Community detection completed:', response);

      return response;
    } catch (err: any) {
      const errorMsg = err.message || 'Community detection failed';
      logger.error('Community detection failed:', createErrorMetadata(err));
      setError(errorMsg);
      throw err;
    } finally {
      setRunning(false);
    }
  }, [settings.analytics]);

  /**
   * Get analytics performance stats
   * Backend endpoint: GET /api/analytics/stats
   */
  const getPerformanceStats = useCallback(async (): Promise<AnalyticsStats | null> => {
    try {
      const response = await unifiedApiClient.getData<AnalyticsStats>('/api/analytics/stats');
      logger.debug('Performance stats retrieved:', response);
      return response;
    } catch (err: any) {
      logger.error('Failed to get performance stats:', createErrorMetadata(err));
      setError(err.message || 'Stats retrieval failed');
      return null;
    }
  }, []);

  /**
   * Cancel running clustering task
   * Backend endpoint: DELETE /api/analytics/clustering/cancel?task_id={taskId}
   */
  const cancelClustering = useCallback(async (taskId: string): Promise<boolean> => {
    try {
      await unifiedApiClient.delete(`/api/analytics/clustering/cancel?task_id=${taskId}`);
      setStatus('cancelled');
      logger.info('Clustering task cancelled:', taskId);
      return true;
    } catch (err: any) {
      logger.error('Failed to cancel clustering:', createErrorMetadata(err));
      setError(err.message || 'Cancellation failed');
      return false;
    }
  }, []);

  /**
   * Clear current results and errors
   */
  const clear = useCallback(() => {
    setResults(null);
    setError(null);
    setStatus(null);
  }, []);

  return {
    // State
    running,
    status,
    results,
    error,

    // Actions
    runClustering,
    checkClusteringStatus,
    runCommunityDetection,
    getPerformanceStats,
    cancelClustering,
    clear,

    // Computed
    hasResults: !!results,
    hasError: !!error
  };
};
