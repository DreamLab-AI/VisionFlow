/**
 * Hook for Solid Container (folder) operations
 * Provides listing, navigation, and CRUD for container contents
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import solidPodService, {
  JsonLdDocument,
  SolidNotification,
} from '../../../services/SolidPodService';

export interface ContainerItem {
  name: string;
  url: string;
  type: 'container' | 'resource';
  modified?: string;
}

export interface UseSolidContainerReturn {
  items: ContainerItem[];
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  createResource: (data: JsonLdDocument, slug?: string) => Promise<string | null>;
  createContainer: (slug: string) => Promise<boolean>;
  deleteItem: (itemUrl: string) => Promise<boolean>;
  subscribe: () => void;
  unsubscribe: () => void;
}

export function useSolidContainer(containerPath: string): UseSolidContainerReturn {
  const [items, setItems] = useState<ContainerItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  const parseContainerContents = (data: JsonLdDocument): ContainerItem[] => {
    const result: ContainerItem[] = [];

    // LDP Container typically has ldp:contains predicate
    const contains = data['ldp:contains'] || data['contains'] || [];
    const itemsList = Array.isArray(contains) ? contains : [contains];

    for (const item of itemsList) {
      if (!item) continue;

      const url = typeof item === 'string' ? item : item['@id'] || '';
      if (!url) continue;

      const name = url.split('/').filter(Boolean).pop() || url;
      const types = item['@type'] || [];
      const typeArray = Array.isArray(types) ? types : [types];

      const isContainer = typeArray.some(
        (t: string) => t.includes('Container') || t.includes('BasicContainer')
      ) || url.endsWith('/');

      result.push({
        name: name.replace(/\/$/, ''),
        url,
        type: isContainer ? 'container' : 'resource',
        modified: item['dcterms:modified'] || item['modified'],
      });
    }

    return result.sort((a, b) => {
      // Containers first, then alphabetically
      if (a.type !== b.type) {
        return a.type === 'container' ? -1 : 1;
      }
      return a.name.localeCompare(b.name);
    });
  };

  const refresh = useCallback(async () => {
    if (!containerPath) return;

    setIsLoading(true);
    setError(null);
    try {
      const data = await solidPodService.fetchJsonLd(containerPath);
      const parsedItems = parseContainerContents(data);
      setItems(parsedItems);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load container');
      setItems([]);
    } finally {
      setIsLoading(false);
    }
  }, [containerPath]);

  const createResource = useCallback(
    async (data: JsonLdDocument, slug?: string): Promise<string | null> => {
      if (!containerPath) return null;

      setError(null);
      try {
        const location = await solidPodService.postResource(containerPath, data, slug);
        if (location) {
          await refresh();
        }
        return location;
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to create resource');
        return null;
      }
    },
    [containerPath, refresh]
  );

  const createContainer = useCallback(
    async (slug: string): Promise<boolean> => {
      if (!containerPath) return false;

      setError(null);
      try {
        // Create container by POSTing with Link header for BasicContainer
        const containerData: JsonLdDocument = {
          '@context': 'http://www.w3.org/ns/ldp',
          '@type': 'BasicContainer',
        };
        const location = await solidPodService.postResource(
          containerPath,
          containerData,
          slug.endsWith('/') ? slug : `${slug}/`
        );
        if (location) {
          await refresh();
          return true;
        }
        return false;
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to create container');
        return false;
      }
    },
    [containerPath, refresh]
  );

  const deleteItem = useCallback(
    async (itemUrl: string): Promise<boolean> => {
      setError(null);
      try {
        const success = await solidPodService.deleteResource(itemUrl);
        if (success) {
          await refresh();
        }
        return success;
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to delete item');
        return false;
      }
    },
    [refresh]
  );

  const subscribe = useCallback(() => {
    if (!containerPath || unsubscribeRef.current) return;

    const handleNotification = (notification: SolidNotification) => {
      if (notification.type === 'pub') {
        refresh();
      }
    };

    unsubscribeRef.current = solidPodService.subscribe(containerPath, handleNotification);
  }, [containerPath, refresh]);

  const unsubscribe = useCallback(() => {
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
      unsubscribeRef.current = null;
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    return () => {
      unsubscribe();
    };
  }, [unsubscribe]);

  return {
    items,
    isLoading,
    error,
    refresh,
    createResource,
    createContainer,
    deleteItem,
    subscribe,
    unsubscribe,
  };
}

export default useSolidContainer;
