/**
 * Physics Service Hook
 *
 * Provides access to physics simulation API endpoints from the Phase 5 Physics API.
 * Integrates with /api/physics/* endpoints for complete simulation control.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { unifiedApiClient } from '@/services/api/UnifiedApiClient';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('usePhysicsService');

export interface PhysicsStatus {
  simulation_id?: string;
  running: boolean;
  gpu_status?: {
    device_name: string;
    compute_capability: string;
    total_memory_mb: number;
    free_memory_mb: number;
  };
  statistics?: {
    total_steps: number;
    average_step_time_ms: number;
    average_energy: number;
    gpu_memory_used_mb: number;
  };
}

export interface SimulationParameters {
  time_step?: number;
  damping?: number;
  spring_constant?: number;
  repulsion_strength?: number;
  attraction_strength?: number;
  max_velocity?: number;
  convergence_threshold?: number;
  max_iterations?: number;
  auto_stop_on_convergence?: boolean;
}

export interface NodePosition {
  node_id: number;
  x: number;
  y: number;
  z: number;
}

export interface NodeForce {
  node_id: number;
  force_x: number;
  force_y: number;
  force_z: number;
}

export interface OptimizeLayoutRequest {
  algorithm: string;
  max_iterations?: number;
  target_energy?: number;
}

export interface OptimizeLayoutResponse {
  nodes_updated: number;
  optimization_score: number;
}

export interface UsePhysicsServiceOptions {
  /** Polling interval in ms for status updates. Set to 0 to disable polling. Default: 1000 */
  pollInterval?: number;
  /** Enable automatic error recovery */
  autoRetry?: boolean;
}

/**
 * Hook for accessing the Physics API (/api/physics/*)
 *
 * Provides:
 * - Real-time simulation status
 * - Start/stop controls
 * - Parameter updates
 * - Layout optimization
 * - Node pinning/unpinning
 * - Force application
 */
export function usePhysicsService(options: UsePhysicsServiceOptions = {}) {
  const { pollInterval = 1000, autoRetry = true } = options;

  const [status, setStatus] = useState<PhysicsStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef(true);

  // Poll status periodically
  useEffect(() => {
    isMountedRef.current = true;

    const fetchStatus = async () => {
      try {
        const response = await unifiedApiClient.get<PhysicsStatus>('/api/physics/status');
        if (isMountedRef.current) {
          setStatus(response.data);
          setError(null);
        }
      } catch (err: any) {
        if (isMountedRef.current) {
          logger.warn('Failed to fetch physics status:', err);
          // Don't set error for polling failures to avoid UI noise
        }
      }
    };

    // Initial fetch
    fetchStatus();

    // Set up polling if enabled
    if (pollInterval > 0) {
      pollIntervalRef.current = setInterval(fetchStatus, pollInterval);
    }

    return () => {
      isMountedRef.current = false;
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [pollInterval]);

  /**
   * Start physics simulation
   */
  const startSimulation = useCallback(async (params?: SimulationParameters) => {
    setLoading(true);
    setError(null);
    try {
      const response = await unifiedApiClient.post<{ simulation_id: string; status: string }>(
        '/api/physics/start',
        params || {}
      );

      if (isMountedRef.current) {
        setLoading(false);
        logger.info('Physics simulation started:', response.data);

        // Force immediate status update
        const statusResponse = await unifiedApiClient.get<PhysicsStatus>('/api/physics/status');
        setStatus(statusResponse.data);
      }

      return response.data;
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to start simulation';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to start simulation:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Stop physics simulation
   */
  const stopSimulation = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/stop', {});

      if (isMountedRef.current) {
        setLoading(false);
        logger.info('Physics simulation stopped');

        // Force immediate status update
        const statusResponse = await unifiedApiClient.get<PhysicsStatus>('/api/physics/status');
        setStatus(statusResponse.data);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to stop simulation';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to stop simulation:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Update simulation parameters
   */
  const updateParameters = useCallback(async (params: SimulationParameters) => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/parameters', params);

      if (isMountedRef.current) {
        setLoading(false);
        logger.info('Physics parameters updated:', params);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to update parameters';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to update parameters:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Perform single simulation step
   */
  const performStep = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/step', {});

      if (isMountedRef.current) {
        setLoading(false);
        logger.info('Performed single simulation step');

        // Force immediate status update
        const statusResponse = await unifiedApiClient.get<PhysicsStatus>('/api/physics/status');
        setStatus(statusResponse.data);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to perform step';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to perform step:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Reset simulation state
   */
  const resetSimulation = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/reset', {});

      if (isMountedRef.current) {
        setLoading(false);
        logger.info('Physics simulation reset');

        // Force immediate status update
        const statusResponse = await unifiedApiClient.get<PhysicsStatus>('/api/physics/status');
        setStatus(statusResponse.data);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to reset simulation';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to reset simulation:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Optimize graph layout
   */
  const optimizeLayout = useCallback(async (request: OptimizeLayoutRequest): Promise<OptimizeLayoutResponse> => {
    setLoading(true);
    setError(null);
    try {
      const response = await unifiedApiClient.post<OptimizeLayoutResponse>(
        '/api/physics/optimize',
        request
      );

      if (isMountedRef.current) {
        setLoading(false);
        logger.info('Layout optimized:', response.data);
      }

      return response.data;
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to optimize layout';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to optimize layout:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Pin nodes at specific positions
   */
  const pinNodes = useCallback(async (nodes: NodePosition[]) => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/nodes/pin', { nodes });

      if (isMountedRef.current) {
        setLoading(false);
        logger.info(`Pinned ${nodes.length} nodes`);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to pin nodes';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to pin nodes:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Unpin nodes
   */
  const unpinNodes = useCallback(async (nodeIds: number[]) => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/nodes/unpin', { node_ids: nodeIds });

      if (isMountedRef.current) {
        setLoading(false);
        logger.info(`Unpinned ${nodeIds.length} nodes`);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to unpin nodes';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to unpin nodes:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Apply custom forces to nodes
   */
  const applyForces = useCallback(async (forces: NodeForce[]) => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/forces/apply', { forces });

      if (isMountedRef.current) {
        setLoading(false);
        logger.info(`Applied forces to ${forces.length} nodes`);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        const errorMsg = err.message || 'Failed to apply forces';
        setError(errorMsg);
        setLoading(false);
        logger.error('Failed to apply forces:', err);
      }
      throw err;
    }
  }, []);

  /**
   * Manually refresh status
   */
  const refreshStatus = useCallback(async () => {
    try {
      const response = await unifiedApiClient.get<PhysicsStatus>('/api/physics/status');
      if (isMountedRef.current) {
        setStatus(response.data);
        setError(null);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        logger.warn('Failed to refresh status:', err);
      }
    }
  }, []);

  return {
    // State
    status,
    loading,
    error,

    // Actions
    startSimulation,
    stopSimulation,
    updateParameters,
    performStep,
    resetSimulation,
    optimizeLayout,
    pinNodes,
    unpinNodes,
    applyForces,
    refreshStatus,
  };
}
