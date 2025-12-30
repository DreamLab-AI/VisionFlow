import React from 'react';
import { createLogger } from '../utils/loggerConfig';
import { BinaryNodeData } from '../types/binaryProtocol';
import { validateNodePositions } from '../utils/validation';
import { unifiedApiClient, isApiError } from '../services/api/UnifiedApiClient';

const logger = createLogger('BatchUpdateApi');

const API_BASE = '';

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



export const batchUpdateApi = {
  
  async updateNodePositions(updates: NodeUpdate[]): Promise<BatchUpdateResult> {
    try {
      
      const binaryNodes: BinaryNodeData[] = updates.map(u => ({
        nodeId: u.nodeId,
        position: u.position,
        velocity: u.velocity || { x: 0, y: 0, z: 0 },
        ssspDistance: Infinity, 
        ssspParent: -1 
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

      const result = await unifiedApiClient.putData(`${API_BASE}/graph/nodes/batch-update`, { updates });
      logger.info(`Batch update successful: ${result.processed} processed, ${result.failed} failed`);
      
      return result;
    } catch (error) {
      logger.error('Batch update error:', error);
      throw error;
    }
  },


  
  async createNodes(nodes: Array<{ id?: number; type: string; position: { x: number; y: number; z: number }; metadata?: any }>): Promise<BatchUpdateResult> {
    try {
      const result = await unifiedApiClient.postData(`${API_BASE}/graph/nodes/batch-create`, { nodes });
      logger.info(`Batch created ${result.processed} nodes`);
      
      return result;
    } catch (error) {
      logger.error('Batch create error:', error);
      throw error;
    }
  },

  
  async deleteNodes(nodeIds: number[]): Promise<BatchUpdateResult> {
    try {
      const result = await unifiedApiClient.request<BatchUpdateResult>('DELETE', `${API_BASE}/graph/nodes/batch-delete`, { nodeIds });
      logger.info(`Batch deleted ${(result as any).processed || 0} nodes`);

      return result as unknown as BatchUpdateResult;
    } catch (error) {
      logger.error('Batch delete error:', error);
      throw error;
    }
  },

  
  async updateEdges(edges: Array<{ id: number; source?: number; target?: number; weight?: number }>): Promise<BatchUpdateResult> {
    try {
      const result = await unifiedApiClient.putData(`${API_BASE}/graph/edges/batch-update`, { edges });
      logger.info(`Batch updated ${result.processed} edges`);
      
      return result;
    } catch (error) {
      logger.error('Edge batch update error:', error);
      throw error;
    }
  }
};


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


  return {
    updateNodePositions,
    isProcessing,
    lastResult
  };
}