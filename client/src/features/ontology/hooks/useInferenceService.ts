/**
 * Inference Service Hook
 *
 * Provides access to ontology inference API endpoints from Phase 7 Inference API.
 * Integrates with /api/inference/* endpoints for ontology reasoning and validation.
 */

import { useState, useCallback, useRef } from 'react';
import { unifiedApiClient } from '@/services/api/UnifiedApiClient';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('useInferenceService');

export interface RunInferenceRequest {
  ontology_id: string;
  force?: boolean;
}

export interface RunInferenceResponse {
  success: boolean;
  ontology_id: string;
  inferred_axioms_count: number;
  inference_time_ms: number;
  reasoner_version: string;
  error?: string;
}

export interface ValidateOntologyRequest {
  ontology_id: string;
}

export interface OntologyClassification {
  classes: number;
  properties: number;
  individuals: number;
  axioms: number;
}

/**
 * Hook for accessing the Inference API (/api/inference/*)
 *
 * Provides:
 * - Run inference on ontologies
 * - Validate ontology consistency
 * - Get inference results
 * - Classification and consistency reports
 * - Cache management
 */
export function useInferenceService() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  /**
   * Run inference on an ontology
   */
  const runInference = useCallback(
    async (request: RunInferenceRequest): Promise<RunInferenceResponse> => {
      setLoading(true);
      setError(null);
      try {
        const response = await unifiedApiClient.post<RunInferenceResponse>(
          '/api/inference/run',
          request
        );

        if (isMountedRef.current) {
          setLoading(false);
          logger.info('Inference completed:', response.data);
        }

        return response.data;
      } catch (err: any) {
        if (isMountedRef.current) {
          const errorMsg = err.message || 'Failed to run inference';
          setError(errorMsg);
          setLoading(false);
          logger.error('Failed to run inference:', err);
        }
        throw err;
      }
    },
    []
  );

  /**
   * Validate ontology consistency
   */
  const validateOntology = useCallback(
    async (request: ValidateOntologyRequest): Promise<{ success: boolean; consistent: boolean; message: string }> => {
      setLoading(true);
      setError(null);
      try {
        const response = await unifiedApiClient.post<{ success: boolean; consistent: boolean; message: string }>(
          '/api/inference/validate',
          request
        );

        if (isMountedRef.current) {
          setLoading(false);
          logger.info('Validation completed:', response.data);
        }

        return response.data;
      } catch (err: any) {
        if (isMountedRef.current) {
          const errorMsg = err.message || 'Failed to validate ontology';
          setError(errorMsg);
          setLoading(false);
          logger.error('Failed to validate ontology:', err);
        }
        throw err;
      }
    },
    []
  );

  /**
   * Get inference results for an ontology
   */
  const getResults = useCallback(
    async (ontologyId: string): Promise<any> => {
      setLoading(true);
      setError(null);
      try {
        const response = await unifiedApiClient.get<any>(
          `/api/inference/results/${ontologyId}`
        );

        if (isMountedRef.current) {
          setLoading(false);
          logger.info('Retrieved inference results:', response.data);
        }

        return response.data;
      } catch (err: any) {
        if (isMountedRef.current) {
          const errorMsg = err.message || 'Failed to get inference results';
          setError(errorMsg);
          setLoading(false);
          logger.error('Failed to get inference results:', err);
        }
        throw err;
      }
    },
    []
  );

  /**
   * Get ontology classification
   */
  const getClassification = useCallback(
    async (ontologyId: string): Promise<OntologyClassification> => {
      setLoading(true);
      setError(null);
      try {
        const response = await unifiedApiClient.get<OntologyClassification>(
          `/api/inference/classify/${ontologyId}`
        );

        if (isMountedRef.current) {
          setLoading(false);
          logger.info('Retrieved classification:', response.data);
        }

        return response.data;
      } catch (err: any) {
        if (isMountedRef.current) {
          const errorMsg = err.message || 'Failed to get classification';
          setError(errorMsg);
          setLoading(false);
          logger.error('Failed to get classification:', err);
        }
        throw err;
      }
    },
    []
  );

  /**
   * Get consistency report
   */
  const getConsistencyReport = useCallback(
    async (ontologyId: string): Promise<any> => {
      setLoading(true);
      setError(null);
      try {
        const response = await unifiedApiClient.get<any>(
          `/api/inference/consistency/${ontologyId}`
        );

        if (isMountedRef.current) {
          setLoading(false);
          logger.info('Retrieved consistency report:', response.data);
        }

        return response.data;
      } catch (err: any) {
        if (isMountedRef.current) {
          const errorMsg = err.message || 'Failed to get consistency report';
          setError(errorMsg);
          setLoading(false);
          logger.error('Failed to get consistency report:', err);
        }
        throw err;
      }
    },
    []
  );

  /**
   * Invalidate inference cache for an ontology
   */
  const invalidateCache = useCallback(
    async (ontologyId: string): Promise<void> => {
      setLoading(true);
      setError(null);
      try {
        await unifiedApiClient.delete(`/api/inference/cache/${ontologyId}`);

        if (isMountedRef.current) {
          setLoading(false);
          logger.info('Cache invalidated for ontology:', ontologyId);
        }
      } catch (err: any) {
        if (isMountedRef.current) {
          const errorMsg = err.message || 'Failed to invalidate cache';
          setError(errorMsg);
          setLoading(false);
          logger.error('Failed to invalidate cache:', err);
        }
        throw err;
      }
    },
    []
  );

  return {
    // State
    loading,
    error,

    // Actions
    runInference,
    validateOntology,
    getResults,
    getClassification,
    getConsistencyReport,
    invalidateCache,
  };
}
