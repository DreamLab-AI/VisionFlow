/**
 * Hook for Solid Pod management
 * Provides pod state, creation, and deletion functionality
 */

import { useState, useEffect, useCallback } from 'react';
import solidPodService, {
  PodInfo,
  PodCreationResult,
} from '../../../services/SolidPodService';

export interface UseSolidPodReturn {
  podInfo: PodInfo | null;
  isLoading: boolean;
  error: string | null;
  checkPod: () => Promise<void>;
  createPod: (name?: string) => Promise<PodCreationResult>;
  deletePod: () => Promise<boolean>;
}

export function useSolidPod(): UseSolidPodReturn {
  const [podInfo, setPodInfo] = useState<PodInfo | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkPod = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const info = await solidPodService.checkPodExists();
      setPodInfo(info);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check pod');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const createPod = useCallback(async (name?: string): Promise<PodCreationResult> => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await solidPodService.createPod(name);
      if (result.success) {
        setPodInfo({
          exists: true,
          podUrl: result.podUrl,
          webId: result.webId,
        });
      } else {
        setError(result.error || 'Failed to create pod');
      }
      return result;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to create pod';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, []);

  const deletePod = useCallback(async (): Promise<boolean> => {
    if (!podInfo?.podUrl) return false;

    setIsLoading(true);
    setError(null);
    try {
      const success = await solidPodService.deleteResource(podInfo.podUrl);
      if (success) {
        setPodInfo({ exists: false });
      }
      return success;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete pod');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [podInfo?.podUrl]);

  useEffect(() => {
    checkPod();
  }, [checkPod]);

  return {
    podInfo,
    isLoading,
    error,
    checkPod,
    createPod,
    deletePod,
  };
}

export default useSolidPod;
