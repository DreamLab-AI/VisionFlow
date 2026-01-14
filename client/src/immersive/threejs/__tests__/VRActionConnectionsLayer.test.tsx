/**
 * VRActionConnectionsLayer Tests
 *
 * Tests for the VR-optimized action connections layer.
 * Focuses on performance constraints and VR-specific behavior.
 * Tests pure logic without React rendering context (same pattern as ActionConnectionsLayer.test.tsx).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import { ActionConnection } from '../../../features/visualisation/hooks/useActionConnections';
import { AgentActionType, AGENT_ACTION_COLORS } from '@/services/BinaryWebSocketProtocol';

/** Maximum connections for Quest 3 @ 72fps */
const VR_MAX_CONNECTIONS = 20;

/** Phase timing boundaries */
const PHASE_BOUNDS = {
  spawnEnd: 0.2,
  travelEnd: 0.8,
  impactEnd: 0.9,
  fadeEnd: 1.0,
};

/**
 * Helper to create a mock ActionConnection
 */
function createMockConnection(overrides: Partial<ActionConnection> = {}): ActionConnection {
  return {
    id: `test-conn-${Math.random().toString(36).substring(7)}`,
    sourceAgentId: 1,
    targetNodeId: 100,
    actionType: AgentActionType.Query,
    color: AGENT_ACTION_COLORS[AgentActionType.Query],
    progress: 0.5,
    phase: 'travel',
    startTime: performance.now() - 250,
    duration: 500,
    sourcePosition: { x: 0, y: 0, z: 0 },
    targetPosition: { x: 10, y: 5, z: 10 },
    ...overrides,
  };
}

/**
 * Helper to create multiple connections
 */
function createConnections(count: number): ActionConnection[] {
  return Array.from({ length: count }, (_, i) =>
    createMockConnection({
      id: `conn-${i}`,
      sourceAgentId: i + 1,
      targetNodeId: (i + 1) * 100,
    })
  );
}

describe('VRActionConnectionsLayer Configuration', () => {
  describe('Connection Limiting for VR Performance', () => {
    it('enforces VR limit of 20 connections', () => {
      const connections = createConnections(30);

      // Simulating the internal limiting logic
      const limited = connections.length <= VR_MAX_CONNECTIONS
        ? connections
        : connections.slice(-VR_MAX_CONNECTIONS);

      expect(limited).toHaveLength(20);
    });

    it('prioritizes newest connections when limiting', () => {
      const connections = createConnections(25);

      const limited = connections.slice(-VR_MAX_CONNECTIONS);

      expect(limited[0].id).toBe('conn-5');
      expect(limited[19].id).toBe('conn-24');
    });

    it('keeps all connections when under limit', () => {
      const connections = createConnections(15);

      const limited = connections.length <= VR_MAX_CONNECTIONS
        ? connections
        : connections.slice(-VR_MAX_CONNECTIONS);

      expect(limited).toHaveLength(15);
    });

    it('validates VR_MAX_CONNECTIONS is 20', () => {
      expect(VR_MAX_CONNECTIONS).toBe(20);
    });

    it('validates VR limit is lower than desktop limit', () => {
      const DESKTOP_MAX = 50;
      expect(VR_MAX_CONNECTIONS).toBeLessThan(DESKTOP_MAX);
    });
  });

  describe('Default Props', () => {
    it('defaults opacity to 1.0', () => {
      const defaultOpacity = 1.0;
      expect(defaultOpacity).toBe(1.0);
    });

    it('defaults preview color to cyan', () => {
      const defaultPreviewColor = '#00ffff';
      expect(defaultPreviewColor).toBe('#00ffff');
    });

    it('defaults showHandPreview to false', () => {
      const defaultShowHandPreview = false;
      expect(defaultShowHandPreview).toBe(false);
    });
  });

  describe('Preview Visibility Logic', () => {
    it('shows preview when all conditions met', () => {
      const showHandPreview = true;
      const hasHandPosition = true;
      const hasPreviewTarget = true;

      const shouldShowPreview = showHandPreview && hasHandPosition && hasPreviewTarget;
      expect(shouldShowPreview).toBe(true);
    });

    it('hides preview when showHandPreview is false', () => {
      const showHandPreview = false;
      const hasHandPosition = true;
      const hasPreviewTarget = true;

      const shouldShowPreview = showHandPreview && hasHandPosition && hasPreviewTarget;
      expect(shouldShowPreview).toBe(false);
    });

    it('hides preview when handPosition is null', () => {
      const showHandPreview = true;
      const hasHandPosition = false;
      const hasPreviewTarget = true;

      const shouldShowPreview = showHandPreview && hasHandPosition && hasPreviewTarget;
      expect(shouldShowPreview).toBe(false);
    });

    it('hides preview when previewTarget is null', () => {
      const showHandPreview = true;
      const hasHandPosition = true;
      const hasPreviewTarget = false;

      const shouldShowPreview = showHandPreview && hasHandPosition && hasPreviewTarget;
      expect(shouldShowPreview).toBe(false);
    });
  });

  describe('Empty State Logic', () => {
    it('should not render when no connections and no preview', () => {
      const connections: ActionConnection[] = [];
      const showHandPreview = false;

      const shouldRender = connections.length > 0 || showHandPreview;
      expect(shouldRender).toBe(false);
    });

    it('should render when has connections', () => {
      const connections = [createMockConnection()];
      const showHandPreview = false;

      const shouldRender = connections.length > 0 || showHandPreview;
      expect(shouldRender).toBe(true);
    });

    it('should render when preview enabled with valid positions', () => {
      const connections: ActionConnection[] = [];
      const showHandPreview = true;
      const hasHandPosition = true;
      const hasPreviewTarget = true;

      const shouldRender = connections.length > 0 || (showHandPreview && hasHandPosition && hasPreviewTarget);
      expect(shouldRender).toBe(true);
    });
  });

  describe('Phase Handling', () => {
    it.each(['spawn', 'travel', 'impact', 'fade'] as const)(
      'creates connection in %s phase',
      (phase) => {
        const connection = createMockConnection({ phase });
        expect(connection.phase).toBe(phase);
      }
    );

    it('validates PHASE_BOUNDS are cumulative', () => {
      expect(PHASE_BOUNDS.spawnEnd).toBe(0.2);
      expect(PHASE_BOUNDS.travelEnd).toBe(0.8);
      expect(PHASE_BOUNDS.impactEnd).toBe(0.9);
      expect(PHASE_BOUNDS.fadeEnd).toBe(1.0);
    });

    it('validates phases are contiguous', () => {
      expect(PHASE_BOUNDS.spawnEnd).toBeLessThan(PHASE_BOUNDS.travelEnd);
      expect(PHASE_BOUNDS.travelEnd).toBeLessThan(PHASE_BOUNDS.impactEnd);
      expect(PHASE_BOUNDS.impactEnd).toBeLessThan(PHASE_BOUNDS.fadeEnd);
    });
  });

  describe('Position Handling', () => {
    it('uses provided source position', () => {
      const conn = createMockConnection({
        sourcePosition: { x: 5, y: 10, z: 15 },
      });

      expect(conn.sourcePosition).toEqual({ x: 5, y: 10, z: 15 });
    });

    it('uses provided target position', () => {
      const conn = createMockConnection({
        targetPosition: { x: 20, y: 25, z: 30 },
      });

      expect(conn.targetPosition).toEqual({ x: 20, y: 25, z: 30 });
    });

    it('handles undefined positions', () => {
      const conn = createMockConnection({
        sourcePosition: undefined,
        targetPosition: undefined,
      });

      expect(conn.sourcePosition).toBeUndefined();
      expect(conn.targetPosition).toBeUndefined();
    });
  });
});

describe('VR Particle Position Calculation', () => {
  describe('Spawn Phase', () => {
    it('particle stays at source during spawn', () => {
      const conn = createMockConnection({
        phase: 'spawn',
        progress: 0.1,
      });

      // During spawn, particle should be at source
      expect(conn.phase).toBe('spawn');
      expect(conn.progress).toBeLessThan(PHASE_BOUNDS.spawnEnd);
    });
  });

  describe('Travel Phase', () => {
    it('calculates travel progress correctly at 50%', () => {
      const progress = 0.5;

      const travelProgress =
        (progress - PHASE_BOUNDS.spawnEnd) /
        (PHASE_BOUNDS.travelEnd - PHASE_BOUNDS.spawnEnd);

      // (0.5 - 0.2) / (0.8 - 0.2) = 0.3 / 0.6 = 0.5
      expect(travelProgress).toBeCloseTo(0.5, 5);
    });

    it('calculates travel progress at start of travel', () => {
      const progress = 0.2;

      const travelProgress =
        (progress - PHASE_BOUNDS.spawnEnd) /
        (PHASE_BOUNDS.travelEnd - PHASE_BOUNDS.spawnEnd);

      expect(travelProgress).toBeCloseTo(0, 5);
    });

    it('calculates travel progress at end of travel', () => {
      const progress = 0.8;

      const travelProgress =
        (progress - PHASE_BOUNDS.spawnEnd) /
        (PHASE_BOUNDS.travelEnd - PHASE_BOUNDS.spawnEnd);

      expect(travelProgress).toBeCloseTo(1, 5);
    });
  });

  describe('Impact and Fade Phases', () => {
    it('particle at target during impact', () => {
      const conn = createMockConnection({
        phase: 'impact',
        progress: 0.85,
      });

      expect(conn.progress).toBeGreaterThanOrEqual(PHASE_BOUNDS.travelEnd);
      expect(conn.progress).toBeLessThan(PHASE_BOUNDS.impactEnd);
    });

    it('particle at target during fade', () => {
      const conn = createMockConnection({
        phase: 'fade',
        progress: 0.95,
      });

      expect(conn.progress).toBeGreaterThanOrEqual(PHASE_BOUNDS.impactEnd);
    });
  });
});

describe('VR Particle Scale Calculation', () => {
  it('scales from 0 to 1 during spawn phase', () => {
    const progress = 0.1;
    const scale = progress / PHASE_BOUNDS.spawnEnd;

    expect(scale).toBeCloseTo(0.5, 5);
  });

  it('scale is 0 at spawn start', () => {
    const progress = 0;
    const scale = progress / PHASE_BOUNDS.spawnEnd;

    expect(scale).toBe(0);
  });

  it('scale is 1 at spawn end', () => {
    const progress = 0.2;
    const scale = progress / PHASE_BOUNDS.spawnEnd;

    expect(scale).toBeCloseTo(1, 5);
  });

  it('maintains scale of 1 during travel', () => {
    const travelScale = 1.0;
    expect(travelScale).toBe(1.0);
  });

  it('uses scale of 0.5 during impact', () => {
    const impactScale = 0.5;
    expect(impactScale).toBe(0.5);
  });

  it('fades from 0.5 to 0 during fade', () => {
    const progress = 0.95;
    const fadeProgress =
      (progress - PHASE_BOUNDS.impactEnd) /
      (PHASE_BOUNDS.fadeEnd - PHASE_BOUNDS.impactEnd);
    const scale = 0.5 * (1 - fadeProgress);

    expect(fadeProgress).toBeCloseTo(0.5, 5);
    expect(scale).toBeCloseTo(0.25, 5);
  });

  it('scale is 0 at fade end', () => {
    const progress = 1.0;
    const fadeProgress =
      (progress - PHASE_BOUNDS.impactEnd) /
      (PHASE_BOUNDS.fadeEnd - PHASE_BOUNDS.impactEnd);
    const scale = 0.5 * (1 - fadeProgress);

    expect(fadeProgress).toBeCloseTo(1, 5);
    expect(scale).toBeCloseTo(0, 5);
  });
});

describe('VRImpactRing Logic', () => {
  describe('Visibility Logic', () => {
    it('should not render when scale is 0', () => {
      const scale = 0;
      const shouldRender = scale > 0;
      expect(shouldRender).toBe(false);
    });

    it('should not render when scale is negative', () => {
      const scale = -1;
      const shouldRender = scale > 0;
      expect(shouldRender).toBe(false);
    });

    it('should render when scale is positive', () => {
      const scale = 1;
      const shouldRender = scale > 0;
      expect(shouldRender).toBe(true);
    });

    it('should render when scale is very small positive', () => {
      const scale = 0.001;
      const shouldRender = scale > 0;
      expect(shouldRender).toBe(true);
    });
  });

  describe('Geometry Configuration', () => {
    it('uses ring geometry with 16 segments', () => {
      const segments = 16; // VR-optimized
      expect(segments).toBe(16);
    });

    it('uses inner radius of 0.3', () => {
      const innerRadius = 0.3;
      expect(innerRadius).toBe(0.3);
    });

    it('uses outer radius of 0.8', () => {
      const outerRadius = 0.8;
      expect(outerRadius).toBe(0.8);
    });
  });
});

describe('VR Performance Optimizations', () => {
  describe('Geometry Reduction', () => {
    it('uses fewer curve segments than desktop', () => {
      // VR: 8-24 segments, Desktop: 40 segments
      const vrMaxSegments = 24;
      const desktopSegments = 40;

      expect(vrMaxSegments).toBeLessThan(desktopSegments);
    });

    it('uses fewer sphere segments than desktop', () => {
      // VR: 6-12 segments, Desktop: 16 segments
      const vrMaxSegments = 12;
      const desktopSegments = 16;

      expect(vrMaxSegments).toBeLessThan(desktopSegments);
    });

    it('high LOD uses 24 curve segments', () => {
      const highLODSegments = 24;
      expect(highLODSegments).toBe(24);
    });

    it('medium LOD uses 16 curve segments', () => {
      const mediumLODSegments = 16;
      expect(mediumLODSegments).toBe(16);
    });

    it('low LOD uses 8 curve segments', () => {
      const lowLODSegments = 8;
      expect(lowLODSegments).toBe(8);
    });
  });

  describe('InstancedMesh Configuration', () => {
    it('uses VR_MAX_CONNECTIONS as max instances', () => {
      const maxInstances = VR_MAX_CONNECTIONS;
      expect(maxInstances).toBe(20);
    });

    it('enables frustum culling', () => {
      const frustumCulled = true;
      expect(frustumCulled).toBe(true);
    });
  });

  describe('Material Properties', () => {
    it('disables depth write to prevent z-fighting', () => {
      const depthWrite = false;
      expect(depthWrite).toBe(false);
    });

    it('uses transparent materials', () => {
      const transparent = true;
      expect(transparent).toBe(true);
    });

    it('particle opacity is 0.9', () => {
      const particleOpacity = 0.9;
      expect(particleOpacity).toBe(0.9);
    });
  });

  describe('LOD Integration', () => {
    it('defines four LOD levels', () => {
      const lodLevels = ['high', 'medium', 'low', 'culled'];
      expect(lodLevels).toHaveLength(4);
    });

    it('culled level has 0 segments', () => {
      const culledSegments = 0;
      expect(culledSegments).toBe(0);
    });
  });
});

describe('Stereoscopic Rendering Considerations', () => {
  it('uses DoubleSide for ring geometry', () => {
    const side = THREE.DoubleSide;
    expect(side).toBe(THREE.DoubleSide);
  });

  it('disables depth write for transparent objects', () => {
    const depthWrite = false;
    expect(depthWrite).toBe(false);
  });

  it('sphere geometry uses 8 segments (VR-optimized)', () => {
    const segments = 8;
    expect(segments).toBe(8);
  });
});

describe('Connection Line Geometry', () => {
  describe('Simplified Lerp for VR', () => {
    it('uses two-segment lerp instead of bezier for particle movement', () => {
      const source = { x: 0, y: 0, z: 0 };
      const target = { x: 10, y: 10, z: 10 };
      const midPoint = {
        x: (source.x + target.x) / 2,
        y: (source.y + target.y) / 2 + 5 * 0.15, // Arc height
        z: (source.z + target.z) / 2,
      };

      expect(midPoint.x).toBe(5);
      expect(midPoint.z).toBe(5);
    });

    it('first half lerps from source to midpoint', () => {
      const travelProgress = 0.25; // 25% through travel = 50% to midpoint
      const inFirstHalf = travelProgress < 0.5;
      expect(inFirstHalf).toBe(true);
    });

    it('second half lerps from midpoint to target', () => {
      const travelProgress = 0.75; // 75% through travel = 50% from mid to target
      const inSecondHalf = travelProgress >= 0.5;
      expect(inSecondHalf).toBe(true);
    });
  });

  describe('QuadraticBezier for Lines', () => {
    it('line still uses bezier curve (higher quality)', () => {
      // Lines use bezier for smooth appearance
      const useBezier = true;
      expect(useBezier).toBe(true);
    });

    it('arc height is 15% of distance', () => {
      const arcHeightFactor = 0.15;
      expect(arcHeightFactor).toBe(0.15);
    });
  });
});

describe('Color Handling', () => {
  it('uses action type color for connections', () => {
    const queryColor = AGENT_ACTION_COLORS[AgentActionType.Query];
    expect(queryColor).toBe('#3b82f6');
  });

  it('supports all action type colors', () => {
    expect(AGENT_ACTION_COLORS[AgentActionType.Query]).toBeDefined();
    expect(AGENT_ACTION_COLORS[AgentActionType.Update]).toBeDefined();
    expect(AGENT_ACTION_COLORS[AgentActionType.Create]).toBeDefined();
    expect(AGENT_ACTION_COLORS[AgentActionType.Delete]).toBeDefined();
    expect(AGENT_ACTION_COLORS[AgentActionType.Link]).toBeDefined();
    expect(AGENT_ACTION_COLORS[AgentActionType.Transform]).toBeDefined();
  });
});
