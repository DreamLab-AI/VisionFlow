/**
 * useSolidPod Hook
 *
 * Main hook for Solid Pod state management:
 * - Checks if user has a pod on auth
 * - Auto-provisions pod if needed
 * - Provides pod URL, WebID, loading/error states
 * - Subscribes to pod notifications via WebSocket
 * - Integrates with existing Nostr auth system
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import solidPodService, {
  PodInfo,
  PodCreationResult,
  SolidNotification,
} from '../services/SolidPodService';
import { useNostrAuth } from './useNostrAuth';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('useSolidPod');

// --- Types ---

export interface SolidPodState {
  /** URL of the user's pod */
  podUrl: string | null;
  /** User's WebID */
  webId: string | null;
  /** Whether pod check/creation is in progress */
  isLoading: boolean;
  /** Error message if any operation failed */
  error: string | null;
  /** Whether the WebSocket is connected */
  isConnected: boolean;
  /** Whether a pod exists for this user */
  hasPod: boolean;
}

export interface UseSolidPodOptions {
  /** Auto-provision pod if user doesn't have one */
  autoProvision?: boolean;
  /** Connect WebSocket for real-time notifications */
  enableWebSocket?: boolean;
  /** Custom pod name for auto-provisioning */
  defaultPodName?: string;
}

export interface UseSolidPodReturn extends SolidPodState {
  /** Manually create a pod */
  createPod: (name?: string) => Promise<PodCreationResult>;
  /** Refresh pod status */
  checkPod: () => Promise<PodInfo>;
  /** Subscribe to resource notifications */
  subscribe: (resourceUrl: string, callback: (notification: SolidNotification) => void) => () => void;
  /** Connect WebSocket manually */
  connectWebSocket: () => void;
  /** Disconnect WebSocket */
  disconnectWebSocket: () => void;
}

// --- Hook Implementation ---

export function useSolidPod(options: UseSolidPodOptions = {}): UseSolidPodReturn {
  const {
    autoProvision = false,
    enableWebSocket = true,
    defaultPodName,
  } = options;

  const { authenticated, user } = useNostrAuth();

  const [state, setState] = useState<SolidPodState>({
    podUrl: null,
    webId: null,
    isLoading: false,
    error: null,
    isConnected: false,
    hasPod: false,
  });

  const wsConnectedRef = useRef(false);
  const checkInProgressRef = useRef(false);

  // --- Pod Check ---

  const checkPod = useCallback(async (): Promise<PodInfo> => {
    if (checkInProgressRef.current) {
      return { exists: state.hasPod, podUrl: state.podUrl ?? undefined, webId: state.webId ?? undefined };
    }

    checkInProgressRef.current = true;
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const podInfo = await solidPodService.checkPodExists();

      setState((prev) => ({
        ...prev,
        podUrl: podInfo.podUrl ?? null,
        webId: podInfo.webId ?? null,
        hasPod: podInfo.exists,
        isLoading: false,
      }));

      logger.debug('Pod check complete', { exists: podInfo.exists, podUrl: podInfo.podUrl });
      return podInfo;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to check pod';
      logger.error('Pod check failed', { error: err });

      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));

      return { exists: false };
    } finally {
      checkInProgressRef.current = false;
    }
  }, [state.hasPod, state.podUrl, state.webId]);

  // --- Pod Creation ---

  const createPod = useCallback(async (name?: string): Promise<PodCreationResult> => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const result = await solidPodService.createPod(name || defaultPodName);

      if (result.success) {
        setState((prev) => ({
          ...prev,
          podUrl: result.podUrl ?? null,
          webId: result.webId ?? null,
          hasPod: true,
          isLoading: false,
        }));

        logger.info('Pod created', { podUrl: result.podUrl });
      } else {
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: result.error ?? 'Pod creation failed',
        }));
      }

      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Pod creation failed';
      logger.error('Pod creation error', { error: err });

      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));

      return { success: false, error: errorMessage };
    }
  }, [defaultPodName]);

  // --- WebSocket Management ---

  const connectWebSocket = useCallback(() => {
    if (wsConnectedRef.current) return;

    solidPodService.connectWebSocket();
    wsConnectedRef.current = true;
    setState((prev) => ({ ...prev, isConnected: true }));
    logger.debug('WebSocket connection initiated');
  }, []);

  const disconnectWebSocket = useCallback(() => {
    solidPodService.disconnect();
    wsConnectedRef.current = false;
    setState((prev) => ({ ...prev, isConnected: false }));
    logger.debug('WebSocket disconnected');
  }, []);

  const subscribe = useCallback(
    (resourceUrl: string, callback: (notification: SolidNotification) => void): (() => void) => {
      // Ensure WebSocket is connected
      if (!wsConnectedRef.current && enableWebSocket) {
        connectWebSocket();
      }

      return solidPodService.subscribe(resourceUrl, callback);
    },
    [connectWebSocket, enableWebSocket]
  );

  // --- Effects ---

  // Check pod on auth change
  useEffect(() => {
    if (!authenticated || !user) {
      // Reset state on logout
      setState({
        podUrl: null,
        webId: null,
        isLoading: false,
        error: null,
        isConnected: false,
        hasPod: false,
      });
      disconnectWebSocket();
      return;
    }

    const initializePod = async () => {
      const podInfo = await checkPod();

      // Auto-provision if enabled and no pod exists
      if (!podInfo.exists && autoProvision) {
        logger.info('Auto-provisioning pod');
        await createPod();
      }

      // Connect WebSocket if enabled and pod exists
      if (enableWebSocket && (podInfo.exists || autoProvision)) {
        connectWebSocket();
      }
    };

    initializePod();
  }, [authenticated, user, autoProvision, enableWebSocket, checkPod, createPod, connectWebSocket, disconnectWebSocket]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnectWebSocket();
    };
  }, [disconnectWebSocket]);

  return {
    ...state,
    createPod,
    checkPod,
    subscribe,
    connectWebSocket,
    disconnectWebSocket,
  };
}

export default useSolidPod;
