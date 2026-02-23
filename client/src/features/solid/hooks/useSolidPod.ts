/**
 * Hook for Solid Pod management
 * Provides pod state, auto-provisioning, and deletion functionality.
 * Uses initPod() which auto-creates the pod if it doesn't exist.
 */

import { useState, useEffect, useCallback } from 'react';
import solidPodService, {
  PodInfo,
  PodCreationResult,
} from '../../../services/SolidPodService';
import { nostrAuth } from '../../../services/nostrAuthService';

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
      // Use initPod which auto-provisions if pod doesn't exist
      const result = await solidPodService.initPod();
      if (result.success) {
        setPodInfo({
          exists: true,
          podUrl: result.podUrl,
          webId: result.webId,
          structure: result.structure,
        });
      } else {
        // initPod failed (e.g. not authenticated) — fall back to check
        const info = await solidPodService.checkPodExists();
        setPodInfo(info);
      }
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
      // initPod handles create-if-not-exists with full structure
      const result = await solidPodService.initPod();
      if (result.success) {
        setPodInfo({
          exists: true,
          podUrl: result.podUrl,
          webId: result.webId,
          structure: result.structure,
        });
        return {
          success: true,
          podUrl: result.podUrl,
          webId: result.webId,
          created: result.created,
          structure: result.structure,
        };
      } else {
        setError(result.error || 'Failed to create pod');
        return { success: false, error: result.error };
      }
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
    // Only check/init pod if user is authenticated — otherwise we get 401s
    // from the backend and the NIP-07 extension can't sign without a session.
    if (nostrAuth.isAuthenticated()) {
      checkPod();
    }
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
