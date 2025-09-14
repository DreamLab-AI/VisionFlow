import React from 'react';
import { createLogger } from '../utils/logger';
import { BinaryNodeData } from '../types/binaryProtocol';
import { validateNodePositions } from '../utils/validation';

const logger = createLogger('BatchUpdateApi');

const API_BASE = '/api';

export interface BatchUpdateResult {
  success: boolean;
  processed: number;
  failed: number;
  errors?: string[];
}

export interface NodeUpdate {
  nodeId: number;
  position: { x: number; y: number; z: number };
  velocity?: { x: number; y: number; z: number };
}

export interface SettingUpdate {
  path: string;
  value: any;
}

/**
 * Batch update API for efficient server communication
 */
export const batchUpdateApi = {
  /**
   * Send batch position updates via REST API
   * This is an alternative to WebSocket for less frequent updates
   */
  async updateNodePositions(updates: NodeUpdate[]): Promise<BatchUpdateResult> {
    try {
      // Validate before sending
      const binaryNodes: BinaryNodeData[] = updates.map(u => ({
        nodeId: u.nodeId,
        position: u.position,
        velocity: u.velocity || { x: 0, y: 0, z: 0 }
      }));

      const validation = validateNodePositions(binaryNodes);
      if (!validation.valid) {
        logger.error('Batch validation failed:', validation.errors);
        return {
          success: false,
          processed: 0,
          failed: updates.length,
          errors: validation.errors
        };
      }

      const response = await fetch(`${API_BASE}/graph/nodes/batch-update`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ updates })
      });

      if (!response.ok) {
        throw new Error(`Batch update failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      logger.info(`Batch update successful: ${result.processed} processed, ${result.failed} failed`);
      
      return result;
    } catch (error) {
      logger.error('Batch update error:', error);
      throw error;
    }
  },

  /**
   * Batch update multiple settings
   */
  async updateSettings(updates: SettingUpdate[]): Promise<BatchUpdateResult> {
    try {
      const response = await fetch(`${API_BASE}/settings/batch`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ updates })
      });

      if (!response.ok) {
        throw new Error(`Settings batch update failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      logger.info(`Settings batch update: ${result.processed} processed`);
      
      return result;
    } catch (error) {
      logger.error('Settings batch update error:', error);
      throw error;
    }
  },

  /**
   * Batch create nodes
   */
  async createNodes(nodes: Array<{ id?: number; type: string; position: { x: number; y: number; z: number }; metadata?: any }>): Promise<BatchUpdateResult> {
    try {
      const response = await fetch(`${API_BASE}/graph/nodes/batch-create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ nodes })
      });

      if (!response.ok) {
        throw new Error(`Batch create failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      logger.info(`Batch created ${result.processed} nodes`);
      
      return result;
    } catch (error) {
      logger.error('Batch create error:', error);
      throw error;
    }
  },

  /**
   * Batch delete nodes
   */
  async deleteNodes(nodeIds: number[]): Promise<BatchUpdateResult> {
    try {
      const response = await fetch(`${API_BASE}/graph/nodes/batch-delete`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ nodeIds })
      });

      if (!response.ok) {
        throw new Error(`Batch delete failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      logger.info(`Batch deleted ${result.processed} nodes`);
      
      return result;
    } catch (error) {
      logger.error('Batch delete error:', error);
      throw error;
    }
  },

  /**
   * Batch update edges
   */
  async updateEdges(edges: Array<{ id: number; source?: number; target?: number; weight?: number }>): Promise<BatchUpdateResult> {
    try {
      const response = await fetch(`${API_BASE}/graph/edges/batch-update`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ edges })
      });

      if (!response.ok) {
        throw new Error(`Edge batch update failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      logger.info(`Batch updated ${result.processed} edges`);
      
      return result;
    } catch (error) {
      logger.error('Edge batch update error:', error);
      throw error;
    }
  }
};

/**
 * Hook for using batch updates with React
 */
export function useBatchUpdates() {
  const [isProcessing, setIsProcessing] = React.useState(false);
  const [lastResult, setLastResult] = React.useState<BatchUpdateResult | null>(null);

  const updateNodePositions = React.useCallback(async (updates: NodeUpdate[]) => {
    setIsProcessing(true);
    try {
      const result = await batchUpdateApi.updateNodePositions(updates);
      setLastResult(result);
      return result;
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const updateSettings = React.useCallback(async (updates: SettingUpdate[]) => {
    setIsProcessing(true);
    try {
      const result = await batchUpdateApi.updateSettings(updates);
      setLastResult(result);
      return result;
    } finally {
      setIsProcessing(false);
    }
  }, []);

  return {
    updateNodePositions,
    updateSettings,
    isProcessing,
    lastResult
  };
}