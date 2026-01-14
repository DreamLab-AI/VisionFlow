/**
 * useVRConnectionsLOD Hook
 *
 * Level of Detail management for VR action connections.
 * Reduces geometry complexity at distance to maintain 72fps on Quest 3.
 *
 * LOD Levels:
 * - high: < 5m - Full detail (24 curve segments, 12 sphere segments)
 * - medium: 5-15m - Reduced detail (16 curve segments, 8 sphere segments)
 * - low: 15-30m - Minimal detail (8 curve segments, 6 sphere segments)
 * - culled: > 30m - Not rendered
 *
 * Performance considerations:
 * - Updates at 30Hz (every other frame at 72fps)
 * - Caches LOD calculations per position
 * - Uses squared distances to avoid sqrt
 */

import { useRef, useCallback, useMemo } from 'react';
import * as THREE from 'three';

export type LODLevel = 'high' | 'medium' | 'low' | 'culled';

/** Distance thresholds (squared for performance) */
const LOD_THRESHOLDS_SQ = {
  high: 25,        // < 5m
  medium: 225,     // 5-15m
  low: 900,        // 15-30m
  // > 30m = culled
};

/** Cache entry for LOD calculation */
interface LODCacheEntry {
  level: LODLevel;
  frameNumber: number;
}

export interface VRConnectionsLODConfig {
  /** Distance for high LOD (meters) */
  highDistance?: number;
  /** Distance for medium LOD (meters) */
  mediumDistance?: number;
  /** Distance for low LOD (meters) */
  lowDistance?: number;
  /** How often to recalculate LOD (frames) */
  updateInterval?: number;
  /** Enable aggressive culling for low-end devices */
  aggressiveCulling?: boolean;
}

const DEFAULT_CONFIG: Required<VRConnectionsLODConfig> = {
  highDistance: 5,
  mediumDistance: 15,
  lowDistance: 30,
  updateInterval: 2, // Every 2 frames (36Hz at 72fps)
  aggressiveCulling: false,
};

/**
 * Hook for managing LOD levels of VR connections
 */
export const useVRConnectionsLOD = (config: VRConnectionsLODConfig = {}) => {
  const settings = useMemo(
    () => ({ ...DEFAULT_CONFIG, ...config }),
    [config]
  );

  // Precompute squared thresholds
  const thresholdsSq = useMemo(
    () => ({
      high: settings.highDistance * settings.highDistance,
      medium: settings.mediumDistance * settings.mediumDistance,
      low: settings.lowDistance * settings.lowDistance,
    }),
    [settings.highDistance, settings.mediumDistance, settings.lowDistance]
  );

  // Camera position reference (updated externally)
  const cameraPosition = useRef(new THREE.Vector3());
  const frameCounter = useRef(0);

  // LOD cache to avoid recalculating every frame
  const lodCache = useRef(new Map<string, LODCacheEntry>());

  /**
   * Update camera position (call from useFrame)
   */
  const updateCameraPosition = useCallback((position: THREE.Vector3) => {
    cameraPosition.current.copy(position);
    frameCounter.current++;

    // Periodically clear stale cache entries
    if (frameCounter.current % 60 === 0) {
      const threshold = frameCounter.current - 30;
      const entries = Array.from(lodCache.current.entries());
      for (const [key, entry] of entries) {
        if (entry.frameNumber < threshold) {
          lodCache.current.delete(key);
        }
      }
    }
  }, []);

  /**
   * Get LOD level for a position in world space
   */
  const getLODLevel = useCallback(
    (position: THREE.Vector3): LODLevel => {
      // Generate cache key from position (rounded to reduce cache size)
      const cacheKey = `${Math.round(position.x)},${Math.round(position.y)},${Math.round(position.z)}`;

      // Check cache (allow update every N frames)
      const cached = lodCache.current.get(cacheKey);
      if (
        cached &&
        frameCounter.current - cached.frameNumber < settings.updateInterval
      ) {
        return cached.level;
      }

      // Calculate squared distance
      const distSq = position.distanceToSquared(cameraPosition.current);

      let level: LODLevel;

      if (distSq < thresholdsSq.high) {
        level = 'high';
      } else if (distSq < thresholdsSq.medium) {
        level = 'medium';
      } else if (distSq < thresholdsSq.low) {
        level = 'low';
      } else {
        level = 'culled';
      }

      // Aggressive culling mode: lower all levels by one tier
      if (settings.aggressiveCulling && level !== 'culled') {
        if (level === 'high') level = 'medium';
        else if (level === 'medium') level = 'low';
        else if (level === 'low') level = 'culled';
      }

      // Update cache
      lodCache.current.set(cacheKey, {
        level,
        frameNumber: frameCounter.current,
      });

      return level;
    },
    [thresholdsSq, settings.updateInterval, settings.aggressiveCulling]
  );

  /**
   * Get LOD levels for multiple positions (batch operation)
   */
  const getLODLevels = useCallback(
    (positions: THREE.Vector3[]): LODLevel[] => {
      return positions.map(getLODLevel);
    },
    [getLODLevel]
  );

  /**
   * Check if position should be rendered at all
   */
  const isVisible = useCallback(
    (position: THREE.Vector3): boolean => {
      return getLODLevel(position) !== 'culled';
    },
    [getLODLevel]
  );

  /**
   * Get segment counts for a position (for geometry generation)
   */
  const getSegmentCounts = useCallback(
    (position: THREE.Vector3): { curve: number; sphere: number } => {
      const level = getLODLevel(position);

      switch (level) {
        case 'high':
          return { curve: 24, sphere: 12 };
        case 'medium':
          return { curve: 16, sphere: 8 };
        case 'low':
          return { curve: 8, sphere: 6 };
        case 'culled':
        default:
          return { curve: 0, sphere: 0 };
      }
    },
    [getLODLevel]
  );

  /**
   * Reset LOD cache (call when camera teleports)
   */
  const resetCache = useCallback(() => {
    lodCache.current.clear();
  }, []);

  /**
   * Get current cache size (for debugging)
   */
  const getCacheStats = useCallback(() => {
    return {
      size: lodCache.current.size,
      frameNumber: frameCounter.current,
    };
  }, []);

  return {
    updateCameraPosition,
    getLODLevel,
    getLODLevels,
    isVisible,
    getSegmentCounts,
    resetCache,
    getCacheStats,
    /** Current config */
    config: settings,
  };
};

/**
 * Calculate optimal LOD thresholds based on device performance
 */
export function calculateOptimalThresholds(
  targetFPS: number,
  connectionCount: number
): VRConnectionsLODConfig {
  // Lower thresholds = more aggressive LOD = better performance
  const performanceFactor = 72 / targetFPS;
  const countFactor = Math.max(1, connectionCount / 10);

  const baseLow = 30 / (performanceFactor * countFactor);
  const baseMedium = baseLow * 0.5;
  const baseHigh = baseMedium * 0.33;

  return {
    highDistance: Math.max(2, baseHigh),
    mediumDistance: Math.max(5, baseMedium),
    lowDistance: Math.max(10, baseLow),
    aggressiveCulling: targetFPS < 72 || connectionCount > 15,
  };
}

/**
 * Get LOD statistics for a set of connections (debugging)
 */
export function getLODDistribution(
  levels: LODLevel[]
): Record<LODLevel, number> {
  const distribution: Record<LODLevel, number> = {
    high: 0,
    medium: 0,
    low: 0,
    culled: 0,
  };

  for (const level of levels) {
    distribution[level]++;
  }

  return distribution;
}

export default useVRConnectionsLOD;
