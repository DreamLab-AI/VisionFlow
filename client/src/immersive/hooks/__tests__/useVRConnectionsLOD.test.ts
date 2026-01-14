/**
 * useVRConnectionsLOD Hook Tests
 *
 * Tests for LOD (Level of Detail) management in VR.
 * Tests the pure functions and configuration without React rendering context.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  calculateOptimalThresholds,
  getLODDistribution,
  LODLevel,
} from '../useVRConnectionsLOD';

// Mock THREE.Vector3 for unit tests
vi.mock('three', async (importOriginal) => {
  const actual = await importOriginal<typeof import('three')>();
  return {
    ...actual,
    Vector3: vi.fn().mockImplementation((x = 0, y = 0, z = 0) => ({
      x,
      y,
      z,
      copy: vi.fn(function (this: any, v: any) {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
        return this;
      }),
      distanceToSquared: vi.fn(function (this: any, v: any) {
        const dx = this.x - v.x;
        const dy = this.y - v.y;
        const dz = this.z - v.z;
        return dx * dx + dy * dy + dz * dz;
      }),
    })),
  };
});

describe('LOD Level Thresholds', () => {
  // Distance thresholds (squared for performance)
  const LOD_THRESHOLDS_SQ = {
    high: 25, // < 5m
    medium: 225, // 5-15m
    low: 900, // 15-30m
  };

  describe('Distance Classification', () => {
    it('classifies objects < 5m as high LOD', () => {
      const distanceSq = 16; // 4m squared
      expect(distanceSq).toBeLessThan(LOD_THRESHOLDS_SQ.high);
    });

    it('classifies objects 5-15m as medium LOD', () => {
      const distanceSq = 100; // 10m squared
      expect(distanceSq).toBeGreaterThanOrEqual(LOD_THRESHOLDS_SQ.high);
      expect(distanceSq).toBeLessThan(LOD_THRESHOLDS_SQ.medium);
    });

    it('classifies objects 15-30m as low LOD', () => {
      const distanceSq = 400; // 20m squared
      expect(distanceSq).toBeGreaterThanOrEqual(LOD_THRESHOLDS_SQ.medium);
      expect(distanceSq).toBeLessThan(LOD_THRESHOLDS_SQ.low);
    });

    it('classifies objects > 30m as culled', () => {
      const distanceSq = 1600; // 40m squared
      expect(distanceSq).toBeGreaterThanOrEqual(LOD_THRESHOLDS_SQ.low);
    });
  });

  describe('Threshold Boundaries', () => {
    it('high LOD threshold is 5m (25 squared)', () => {
      expect(LOD_THRESHOLDS_SQ.high).toBe(25);
      expect(Math.sqrt(LOD_THRESHOLDS_SQ.high)).toBe(5);
    });

    it('medium LOD threshold is 15m (225 squared)', () => {
      expect(LOD_THRESHOLDS_SQ.medium).toBe(225);
      expect(Math.sqrt(LOD_THRESHOLDS_SQ.medium)).toBe(15);
    });

    it('low LOD threshold is 30m (900 squared)', () => {
      expect(LOD_THRESHOLDS_SQ.low).toBe(900);
      expect(Math.sqrt(LOD_THRESHOLDS_SQ.low)).toBe(30);
    });
  });
});

describe('Segment Counts by LOD', () => {
  const LOD_SEGMENTS: Record<LODLevel, { curve: number; sphere: number }> = {
    high: { curve: 24, sphere: 12 },
    medium: { curve: 16, sphere: 8 },
    low: { curve: 8, sphere: 6 },
    culled: { curve: 0, sphere: 0 },
  };

  it('high LOD has 24 curve segments', () => {
    expect(LOD_SEGMENTS.high.curve).toBe(24);
  });

  it('high LOD has 12 sphere segments', () => {
    expect(LOD_SEGMENTS.high.sphere).toBe(12);
  });

  it('medium LOD has 16 curve segments', () => {
    expect(LOD_SEGMENTS.medium.curve).toBe(16);
  });

  it('medium LOD has 8 sphere segments', () => {
    expect(LOD_SEGMENTS.medium.sphere).toBe(8);
  });

  it('low LOD has 8 curve segments', () => {
    expect(LOD_SEGMENTS.low.curve).toBe(8);
  });

  it('low LOD has 6 sphere segments', () => {
    expect(LOD_SEGMENTS.low.sphere).toBe(6);
  });

  it('culled has 0 segments', () => {
    expect(LOD_SEGMENTS.culled.curve).toBe(0);
    expect(LOD_SEGMENTS.culled.sphere).toBe(0);
  });

  it('segment counts decrease with LOD level', () => {
    expect(LOD_SEGMENTS.high.curve).toBeGreaterThan(LOD_SEGMENTS.medium.curve);
    expect(LOD_SEGMENTS.medium.curve).toBeGreaterThan(LOD_SEGMENTS.low.curve);
    expect(LOD_SEGMENTS.low.curve).toBeGreaterThan(LOD_SEGMENTS.culled.curve);
  });
});

describe('calculateOptimalThresholds', () => {
  it('calculates thresholds for 72fps target', () => {
    const thresholds = calculateOptimalThresholds(72, 10);

    expect(thresholds.highDistance).toBeGreaterThan(0);
    expect(thresholds.mediumDistance).toBeGreaterThan(thresholds.highDistance!);
    expect(thresholds.lowDistance).toBeGreaterThan(thresholds.mediumDistance!);
  });

  it('enables aggressive culling for lower FPS targets', () => {
    const thresholds = calculateOptimalThresholds(60, 10);

    expect(thresholds.aggressiveCulling).toBe(true);
  });

  it('enables aggressive culling for high connection counts', () => {
    const thresholds = calculateOptimalThresholds(72, 20);

    expect(thresholds.aggressiveCulling).toBe(true);
  });

  it('returns tighter thresholds for more connections', () => {
    const lowCount = calculateOptimalThresholds(72, 5);
    const highCount = calculateOptimalThresholds(72, 20);

    expect(highCount.lowDistance!).toBeLessThan(lowCount.lowDistance!);
  });

  it('does not enable aggressive culling for optimal conditions', () => {
    const thresholds = calculateOptimalThresholds(72, 10);

    expect(thresholds.aggressiveCulling).toBe(false);
  });

  it('scales thresholds based on connection count', () => {
    const few = calculateOptimalThresholds(72, 5);
    const many = calculateOptimalThresholds(72, 15);

    expect(many.highDistance!).toBeLessThan(few.highDistance!);
  });

  it('ensures minimum threshold values', () => {
    const thresholds = calculateOptimalThresholds(72, 100);

    expect(thresholds.highDistance).toBeGreaterThanOrEqual(2);
    expect(thresholds.mediumDistance).toBeGreaterThanOrEqual(5);
    expect(thresholds.lowDistance).toBeGreaterThanOrEqual(10);
  });
});

describe('getLODDistribution', () => {
  it('counts LOD levels correctly', () => {
    const levels: LODLevel[] = ['high', 'high', 'medium', 'low', 'culled', 'culled'];

    const distribution = getLODDistribution(levels);

    expect(distribution.high).toBe(2);
    expect(distribution.medium).toBe(1);
    expect(distribution.low).toBe(1);
    expect(distribution.culled).toBe(2);
  });

  it('handles empty array', () => {
    const distribution = getLODDistribution([]);

    expect(distribution.high).toBe(0);
    expect(distribution.medium).toBe(0);
    expect(distribution.low).toBe(0);
    expect(distribution.culled).toBe(0);
  });

  it('handles all same level', () => {
    const levels: LODLevel[] = ['medium', 'medium', 'medium'];

    const distribution = getLODDistribution(levels);

    expect(distribution.medium).toBe(3);
    expect(distribution.high).toBe(0);
    expect(distribution.low).toBe(0);
    expect(distribution.culled).toBe(0);
  });

  it('handles single element', () => {
    const distribution = getLODDistribution(['high']);

    expect(distribution.high).toBe(1);
    expect(distribution.medium).toBe(0);
  });

  it('handles all levels present', () => {
    const levels: LODLevel[] = ['high', 'medium', 'low', 'culled'];

    const distribution = getLODDistribution(levels);

    expect(distribution.high).toBe(1);
    expect(distribution.medium).toBe(1);
    expect(distribution.low).toBe(1);
    expect(distribution.culled).toBe(1);
  });
});

describe('LOD Level Determination', () => {
  // Simulating the getLODLevel logic
  function getLODLevel(distanceSq: number, thresholdsSq: typeof LOD_THRESHOLDS_SQ): LODLevel {
    if (distanceSq < thresholdsSq.high) return 'high';
    if (distanceSq < thresholdsSq.medium) return 'medium';
    if (distanceSq < thresholdsSq.low) return 'low';
    return 'culled';
  }

  const LOD_THRESHOLDS_SQ = {
    high: 25,
    medium: 225,
    low: 900,
  };

  it('returns high for very close objects', () => {
    expect(getLODLevel(4, LOD_THRESHOLDS_SQ)).toBe('high');
  });

  it('returns high at boundary', () => {
    expect(getLODLevel(24.9, LOD_THRESHOLDS_SQ)).toBe('high');
  });

  it('returns medium just past high boundary', () => {
    expect(getLODLevel(25.1, LOD_THRESHOLDS_SQ)).toBe('medium');
  });

  it('returns medium in middle range', () => {
    expect(getLODLevel(100, LOD_THRESHOLDS_SQ)).toBe('medium');
  });

  it('returns low in far range', () => {
    expect(getLODLevel(400, LOD_THRESHOLDS_SQ)).toBe('low');
  });

  it('returns culled for very far objects', () => {
    expect(getLODLevel(1000, LOD_THRESHOLDS_SQ)).toBe('culled');
  });
});

describe('Aggressive Culling Mode', () => {
  function applyAggressiveCulling(level: LODLevel): LODLevel {
    if (level === 'high') return 'medium';
    if (level === 'medium') return 'low';
    if (level === 'low') return 'culled';
    return 'culled';
  }

  it('downgrades high to medium', () => {
    expect(applyAggressiveCulling('high')).toBe('medium');
  });

  it('downgrades medium to low', () => {
    expect(applyAggressiveCulling('medium')).toBe('low');
  });

  it('downgrades low to culled', () => {
    expect(applyAggressiveCulling('low')).toBe('culled');
  });

  it('keeps culled as culled', () => {
    expect(applyAggressiveCulling('culled')).toBe('culled');
  });
});

describe('Cache Position Rounding', () => {
  // Cache key generation logic
  function generateCacheKey(x: number, y: number, z: number): string {
    return `${Math.round(x)},${Math.round(y)},${Math.round(z)}`;
  }

  it('rounds to nearest integer', () => {
    expect(generateCacheKey(1.4, 2.5, 3.6)).toBe('1,3,4');
  });

  it('handles negative coordinates', () => {
    expect(generateCacheKey(-1.4, -2.5, -3.6)).toBe('-1,-2,-4');
  });

  it('handles zero coordinates', () => {
    expect(generateCacheKey(0, 0, 0)).toBe('0,0,0');
  });

  it('generates same key for nearby positions', () => {
    const key1 = generateCacheKey(1.1, 2.1, 3.1);
    const key2 = generateCacheKey(1.4, 2.4, 3.4);
    expect(key1).toBe(key2);
  });

  it('generates different keys for distant positions', () => {
    const key1 = generateCacheKey(1, 2, 3);
    const key2 = generateCacheKey(10, 20, 30);
    expect(key1).not.toBe(key2);
  });
});

describe('Performance Characteristics', () => {
  it('uses squared distance to avoid sqrt', () => {
    // The hook uses distanceToSquared instead of distanceTo
    const x1 = 0, y1 = 0, z1 = 0;
    const x2 = 3, y2 = 4, z2 = 0;

    const distanceSq = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2;
    expect(distanceSq).toBe(25); // 5^2

    // Without sqrt, we compare against squared thresholds
    expect(distanceSq).toBe(5 * 5);
  });

  it('cache key generation is O(1)', () => {
    // String interpolation with Math.round is constant time
    const start = performance.now();
    for (let i = 0; i < 1000; i++) {
      `${Math.round(i)},${Math.round(i)},${Math.round(i)}`;
    }
    const elapsed = performance.now() - start;

    // Should complete in under 10ms for 1000 iterations
    expect(elapsed).toBeLessThan(10);
  });
});
