/**
 * Health Service Hook
 *
 * Provides access to health monitoring API endpoints.
 * Integrates with /health/* endpoints for system health monitoring.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { unifiedApiClient } from '@/services/api/UnifiedApiClient';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('useHealthService');

export interface HealthStatus {
  healthy: boolean;
  components: Record<string, boolean>;
  timestamp: string;
  version?: string;
}

export interface PhysicsHealth {
  simulation_id?: string;
  running: boolean;
  statistics?: {
    total_steps: number;
    average_step_time_ms: number;
    average_energy: number;
    gpu_memory_used_mb: number;
  };
}

export interface UseHealthServiceOptions {
  /** Enable automatic health polling */
  pollHealth?: boolean;
  /** Health polling interval in ms. Default: 5000 */
  pollInterval?: number;
}

/**
 * Hook for accessing the Health API (/health/*)
 *
 * Provides:
 * - Overall system health monitoring
 * - Physics simulation health
 * - MCP relay control
 * - Real-time health updates
 */
export function useHealthService(options: UseHealthServiceOptions = {}) {
  const { pollHealth = true, pollInterval = 5000 } = options;

  const [overallHealth, setOverallHealth] = useState<HealthStatus | null>(null);
  const [physicsHealth, setPhysicsHealth] = useState<PhysicsHealth | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef(true);

  // Poll health periodically
  useEffect(() => {
    isMountedRef.current = true;

    const fetchHealth = async () => {
      try {
        const [overallResponse, physicsResponse] = await Promise.all([
          unifiedApiClient.get<HealthStatus>('/health'),
          unifiedApiClient.get<PhysicsHealth>('/health/physics'),
        ]);

        if (isMountedRef.current) {
          setOverallHealth(overallResponse.data);
          setPhysicsHealth(physicsResponse.data);
        }
      } catch (err: any) {
        if (isMountedRef.current) {
          logger.warn('Failed to fetch health status:', err);
        }
      }
    };

    if (pollHealth) {
      // Initial fetch
      fetchHealth();

      // Set up polling
      pollIntervalRef.current = setInterval(fetchHealth, pollInterval);
    }

    return () => {
      isMountedRef.current = false;
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [pollHealth, pollInterval]);

  /**
   * Start MCP relay
   */
  const startMCPRelay = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await unifiedApiClient.post<{ success: boolean; message: string }>(
        '/health/mcp/start',
        {}
      );

      if (isMountedRef.current) {
        setLoading(false);
        logger.info('MCP relay started:', response.data);
      }

      return response.data;
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to start MCP relay';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to start MCP relay:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Get MCP logs
   */
  const getMCPLogs = useCallback(async (): Promise<string> => {
    setLoading(true);
    setError(null);
    try {
      const response = await unifiedApiClient.get<{ logs: string }>('/health/mcp/logs');

      if (isMountedRef.current) {
        setLoading(false);
        logger.info('Retrieved MCP logs');
      }

      return response.data.logs;
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to get MCP logs';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to get MCP logs:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Manually refresh health status
   */
  const refreshHealth = useCallback(async () => {
    try {
      const [overallResponse, physicsResponse] = await Promise.all([
        unifiedApiClient.get<HealthStatus>('/health'),
        unifiedApiClient.get<PhysicsHealth>('/health/physics'),
      ]);

      if (isMountedRef.current) {
        setOverallHealth(overallResponse.data);
        setPhysicsHealth(physicsResponse.data);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        logger.warn('Failed to refresh health:', err);
      }
    }
  }, []);

  return {
    // State
    overallHealth,
    physicsHealth,
    loading,
    error,

    // Actions
    startMCPRelay,
    getMCPLogs,
    refreshHealth,
  };
}
