/**
 * Hook for Solid Resource operations
 * Provides CRUD operations and real-time notifications for resources
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import solidPodService, {
  JsonLdDocument,
  SolidNotification,
} from '../../../services/SolidPodService';

export interface ResourceMetadata {
  lastModified?: string;
  etag?: string;
  contentType?: string;
}

export interface UseSolidResourceReturn {
  resource: JsonLdDocument | null;
  metadata: ResourceMetadata;
  isLoading: boolean;
  error: string | null;
  fetch: () => Promise<void>;
  save: (data: JsonLdDocument) => Promise<boolean>;
  remove: () => Promise<boolean>;
  subscribe: () => void;
  unsubscribe: () => void;
}

export function useSolidResource(resourcePath: string): UseSolidResourceReturn {
  const [resource, setResource] = useState<JsonLdDocument | null>(null);
  const [metadata, setMetadata] = useState<ResourceMetadata>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  const fetchResource = useCallback(async () => {
    if (!resourcePath) return;

    setIsLoading(true);
    setError(null);
    try {
      const data = await solidPodService.fetchJsonLd(resourcePath);
      setResource(data);
      // Metadata would typically come from response headers
      setMetadata({
        lastModified: new Date().toISOString(),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch resource');
      setResource(null);
    } finally {
      setIsLoading(false);
    }
  }, [resourcePath]);

  const saveResource = useCallback(async (data: JsonLdDocument): Promise<boolean> => {
    if (!resourcePath) return false;

    setIsLoading(true);
    setError(null);
    try {
      const success = await solidPodService.putResource(resourcePath, data);
      if (success) {
        setResource(data);
        setMetadata((prev) => ({
          ...prev,
          lastModified: new Date().toISOString(),
        }));
      }
      return success;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save resource');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [resourcePath]);

  const removeResource = useCallback(async (): Promise<boolean> => {
    if (!resourcePath) return false;

    setIsLoading(true);
    setError(null);
    try {
      const success = await solidPodService.deleteResource(resourcePath);
      if (success) {
        setResource(null);
        setMetadata({});
      }
      return success;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete resource');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [resourcePath]);

  const subscribe = useCallback(() => {
    if (!resourcePath || unsubscribeRef.current) return;

    const handleNotification = (notification: SolidNotification) => {
      if (notification.type === 'pub') {
        // Resource changed, refetch
        fetchResource();
      }
    };

    unsubscribeRef.current = solidPodService.subscribe(resourcePath, handleNotification);
  }, [resourcePath, fetchResource]);

  const unsubscribe = useCallback(() => {
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
      unsubscribeRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      unsubscribe();
    };
  }, [unsubscribe]);

  return {
    resource,
    metadata,
    isLoading,
    error,
    fetch: fetchResource,
    save: saveResource,
    remove: removeResource,
    subscribe,
    unsubscribe,
  };
}

export default useSolidResource;
