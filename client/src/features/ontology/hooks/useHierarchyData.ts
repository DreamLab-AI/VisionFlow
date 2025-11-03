/**
 * Ontology Hierarchy Data Hook
 *
 * Fetches and manages class hierarchy data from the backend reasoning service.
 * Provides real-time updates and caching for optimal performance.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('useHierarchyData');

// Backend API types (matching Rust backend schema)
export interface ClassNode {
  iri: string;
  label: string;
  parentIri: string | null;
  childrenIris: string[];
  nodeCount: number;
  depth: number;
}

export interface ClassHierarchy {
  rootClasses: string[];
  hierarchy: Record<string, ClassNode>;
}

export interface HierarchyDataOptions {
  ontologyId?: string;
  maxDepth?: number;
  autoRefresh?: boolean;
  refreshIntervalMs?: number;
}

export interface UseHierarchyDataReturn {
  hierarchy: ClassHierarchy | null;
  loading: boolean;
  error: Error | null;
  maxDepth: number;
  rootCount: number;
  totalClasses: number;

  // Actions
  refetch: () => Promise<void>;
  getClassNode: (iri: string) => ClassNode | undefined;
  getChildren: (iri: string) => ClassNode[];
  getAncestors: (iri: string) => ClassNode[];
  getDescendants: (iri: string) => ClassNode[];
  getRootClasses: () => ClassNode[];
}

/**
 * Hook for fetching and managing ontology class hierarchy
 * @param options Hierarchy query options
 * @returns Hierarchy data and utility functions
 */
export function useHierarchyData(
  options: HierarchyDataOptions = {}
): UseHierarchyDataReturn {
  const {
    ontologyId = 'default',
    maxDepth,
    autoRefresh = false,
    refreshIntervalMs = 30000,
  } = options;

  const [hierarchy, setHierarchy] = useState<ClassHierarchy | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchHierarchy = useCallback(async () => {
    // Cancel previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      setLoading(true);
      setError(null);

      // Build query params
      const params = new URLSearchParams();
      params.set('ontology_id', ontologyId);
      if (maxDepth !== undefined) {
        params.set('max_depth', maxDepth.toString());
      }

      const url = `/api/ontology/hierarchy?${params.toString()}`;
      logger.info('Fetching hierarchy', { ontologyId, maxDepth, url });

      const response = await fetch(url, {
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch hierarchy: ${response.status} ${response.statusText}`);
      }

      const data: ClassHierarchy = await response.json();

      logger.info('Hierarchy loaded', {
        rootClasses: data.rootClasses.length,
        totalClasses: Object.keys(data.hierarchy).length,
      });

      setHierarchy(data);
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          logger.debug('Hierarchy fetch aborted');
          return;
        }
        logger.error('Failed to fetch hierarchy', { error: err.message });
        setError(err);
      } else {
        const unknownError = new Error('Unknown error fetching hierarchy');
        logger.error('Unknown error', { err });
        setError(unknownError);
      }
    } finally {
      setLoading(false);
    }
  }, [ontologyId, maxDepth]);

  // Initial fetch
  useEffect(() => {
    fetchHierarchy();

    // Auto-refresh setup
    if (autoRefresh && refreshIntervalMs > 0) {
      intervalRef.current = setInterval(fetchHierarchy, refreshIntervalMs);
    }

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchHierarchy, autoRefresh, refreshIntervalMs]);

  // Utility functions
  const getClassNode = useCallback(
    (iri: string): ClassNode | undefined => {
      return hierarchy?.hierarchy[iri];
    },
    [hierarchy]
  );

  const getChildren = useCallback(
    (iri: string): ClassNode[] => {
      const node = getClassNode(iri);
      if (!node || !hierarchy) return [];

      return node.childrenIris
        .map((childIri) => hierarchy.hierarchy[childIri])
        .filter(Boolean);
    },
    [hierarchy, getClassNode]
  );

  const getAncestors = useCallback(
    (iri: string): ClassNode[] => {
      if (!hierarchy) return [];

      const ancestors: ClassNode[] = [];
      let currentIri: string | null = iri;

      while (currentIri) {
        const node = hierarchy.hierarchy[currentIri];
        if (!node || !node.parentIri) break;

        const parent = hierarchy.hierarchy[node.parentIri];
        if (parent) {
          ancestors.push(parent);
          currentIri = node.parentIri;
        } else {
          break;
        }
      }

      return ancestors;
    },
    [hierarchy]
  );

  const getDescendants = useCallback(
    (iri: string): ClassNode[] => {
      if (!hierarchy) return [];

      const descendants: ClassNode[] = [];
      const queue: string[] = [iri];
      const visited = new Set<string>();

      while (queue.length > 0) {
        const currentIri = queue.shift()!;
        if (visited.has(currentIri)) continue;
        visited.add(currentIri);

        const node = hierarchy.hierarchy[currentIri];
        if (!node) continue;

        for (const childIri of node.childrenIris) {
          const child = hierarchy.hierarchy[childIri];
          if (child) {
            descendants.push(child);
            queue.push(childIri);
          }
        }
      }

      return descendants;
    },
    [hierarchy]
  );

  const getRootClasses = useCallback((): ClassNode[] => {
    if (!hierarchy) return [];

    return hierarchy.rootClasses
      .map((iri) => hierarchy.hierarchy[iri])
      .filter(Boolean);
  }, [hierarchy]);

  // Computed properties
  const maxDepthValue = hierarchy
    ? Math.max(...Object.values(hierarchy.hierarchy).map((n) => n.depth))
    : 0;

  const rootCount = hierarchy?.rootClasses.length || 0;
  const totalClasses = hierarchy ? Object.keys(hierarchy.hierarchy).length : 0;

  return {
    hierarchy,
    loading,
    error,
    maxDepth: maxDepthValue,
    rootCount,
    totalClasses,

    refetch: fetchHierarchy,
    getClassNode,
    getChildren,
    getAncestors,
    getDescendants,
    getRootClasses,
  };
}
