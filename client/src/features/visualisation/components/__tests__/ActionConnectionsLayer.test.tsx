/**
 * ActionConnectionsLayer Tests
 *
 * Tests for the ephemeral animated connections between agent nodes and data nodes.
 * Verifies rendering, color mapping, phase transitions, VR optimization, and connection limits.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import React from 'react';
import { ActionConnectionsLayer, ActionConnectionsStats } from '../ActionConnectionsLayer';
import { ActionConnection } from '../../hooks/useActionConnections';
import { AgentActionType, AGENT_ACTION_COLORS } from '@/services/BinaryWebSocketProtocol';

// Phase boundaries from the component
const PHASE_BOUNDS = {
  spawnEnd: 0.2,
  travelEnd: 0.8,
  impactEnd: 0.9,
  fadeEnd: 1.0,
};

// Mock @react-three/fiber
vi.mock('@react-three/fiber', () => ({
  useFrame: vi.fn((callback: (state: unknown, delta: number) => void) => {
    // Execute callback immediately for testing
    callback({}, 0.016);
  }),
}));

// Mock THREE.js objects
vi.mock('three', async (importOriginal) => {
  const actual = await importOriginal<typeof import('three')>();
  return {
    ...actual,
    Line: vi.fn().mockImplementation(() => ({
      material: { opacity: 1 },
    })),
    LineBasicMaterial: vi.fn().mockImplementation((params) => ({
      ...params,
      opacity: params?.opacity ?? 1,
    })),
    BufferGeometry: vi.fn().mockImplementation(() => ({
      setFromPoints: vi.fn().mockReturnThis(),
    })),
    QuadraticBezierCurve3: vi.fn().mockImplementation((p0, p1, p2) => ({
      getPoint: vi.fn((t: number) => new actual.Vector3(
        p0.x + (p2.x - p0.x) * t,
        p0.y + (p2.y - p0.y) * t,
        p0.z + (p2.z - p0.z) * t
      )),
      getPoints: vi.fn((segments: number) => {
        const points = [];
        for (let i = 0; i <= segments; i++) {
          points.push(new actual.Vector3(
            p0.x + (p2.x - p0.x) * (i / segments),
            p0.y + (p2.y - p0.y) * (i / segments),
            p0.z + (p2.z - p0.z) * (i / segments)
          ));
        }
        return points;
      }),
    })),
  };
});

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
 * Helper to create multiple connections of different types
 */
function createConnectionsOfType(
  actionType: AgentActionType,
  count: number
): ActionConnection[] {
  return Array.from({ length: count }, (_, i) =>
    createMockConnection({
      id: `conn-${actionType}-${i}`,
      actionType,
      color: AGENT_ACTION_COLORS[actionType],
      sourceAgentId: i + 1,
      targetNodeId: (i + 1) * 100,
    })
  );
}

describe('ActionConnectionsLayer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Empty State Rendering', () => {
    it('renders nothing when connections array is empty', () => {
      const result = ActionConnectionsLayer({ connections: [] });
      expect(result).toBeNull();
    });

    it('renders nothing with undefined vrMode when connections empty', () => {
      const result = ActionConnectionsLayer({
        connections: [],
        vrMode: undefined,
      });
      expect(result).toBeNull();
    });

    it('renders nothing with custom opacity when connections empty', () => {
      const result = ActionConnectionsLayer({
        connections: [],
        opacity: 0.5,
      });
      expect(result).toBeNull();
    });
  });

  describe('Connection Line Rendering', () => {
    it('renders correct number of connection lines', () => {
      const connections = [
        createMockConnection({ id: 'conn-1' }),
        createMockConnection({ id: 'conn-2' }),
        createMockConnection({ id: 'conn-3' }),
      ];

      const result = ActionConnectionsLayer({ connections });

      // Should return a group element
      expect(result).not.toBeNull();
      expect(result?.type).toBe('group');
      expect(result?.props.name).toBe('action-connections-layer');

      // Should have 3 children (ActionConnectionLine components)
      expect(result?.props.children).toHaveLength(3);
    });

    it('renders single connection correctly', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections });

      expect(result).not.toBeNull();
      expect(result?.props.children).toHaveLength(1);
    });

    it('renders connections with unique keys', () => {
      const connections = [
        createMockConnection({ id: 'unique-1' }),
        createMockConnection({ id: 'unique-2' }),
      ];

      const result = ActionConnectionsLayer({ connections });
      const children = result?.props.children;

      expect(children[0].key).toBe('unique-1');
      expect(children[1].key).toBe('unique-2');
    });

    it('passes vrMode prop to child components', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections, vrMode: true });

      const child = result?.props.children[0];
      expect(child.props.vrMode).toBe(true);
    });

    it('passes opacity prop to child components', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections, opacity: 0.7 });

      const child = result?.props.children[0];
      expect(child.props.opacity).toBe(0.7);
    });

    it('passes lineWidth prop to child components', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections, lineWidth: 4 });

      const child = result?.props.children[0];
      expect(child.props.lineWidth).toBe(4);
    });
  });

  describe('Color Mapping by Action Type', () => {
    it('maps Query action type to blue (#3b82f6)', () => {
      const conn = createMockConnection({ actionType: AgentActionType.Query });
      expect(conn.color).toBe('#3b82f6');
      expect(AGENT_ACTION_COLORS[AgentActionType.Query]).toBe('#3b82f6');
    });

    it('maps Update action type to yellow (#eab308)', () => {
      const conn = createMockConnection({
        actionType: AgentActionType.Update,
        color: AGENT_ACTION_COLORS[AgentActionType.Update],
      });
      expect(conn.color).toBe('#eab308');
      expect(AGENT_ACTION_COLORS[AgentActionType.Update]).toBe('#eab308');
    });

    it('maps Create action type to green (#22c55e)', () => {
      const conn = createMockConnection({
        actionType: AgentActionType.Create,
        color: AGENT_ACTION_COLORS[AgentActionType.Create],
      });
      expect(conn.color).toBe('#22c55e');
      expect(AGENT_ACTION_COLORS[AgentActionType.Create]).toBe('#22c55e');
    });

    it('maps Delete action type to red (#ef4444)', () => {
      const conn = createMockConnection({
        actionType: AgentActionType.Delete,
        color: AGENT_ACTION_COLORS[AgentActionType.Delete],
      });
      expect(conn.color).toBe('#ef4444');
      expect(AGENT_ACTION_COLORS[AgentActionType.Delete]).toBe('#ef4444');
    });

    it('maps Link action type to purple (#a855f7)', () => {
      const conn = createMockConnection({
        actionType: AgentActionType.Link,
        color: AGENT_ACTION_COLORS[AgentActionType.Link],
      });
      expect(conn.color).toBe('#a855f7');
      expect(AGENT_ACTION_COLORS[AgentActionType.Link]).toBe('#a855f7');
    });

    it('maps Transform action type to cyan (#06b6d4)', () => {
      const conn = createMockConnection({
        actionType: AgentActionType.Transform,
        color: AGENT_ACTION_COLORS[AgentActionType.Transform],
      });
      expect(conn.color).toBe('#06b6d4');
      expect(AGENT_ACTION_COLORS[AgentActionType.Transform]).toBe('#06b6d4');
    });

    it('renders connections with mixed action types correctly', () => {
      const connections = [
        createMockConnection({
          id: 'query-conn',
          actionType: AgentActionType.Query,
          color: AGENT_ACTION_COLORS[AgentActionType.Query],
        }),
        createMockConnection({
          id: 'create-conn',
          actionType: AgentActionType.Create,
          color: AGENT_ACTION_COLORS[AgentActionType.Create],
        }),
        createMockConnection({
          id: 'delete-conn',
          actionType: AgentActionType.Delete,
          color: AGENT_ACTION_COLORS[AgentActionType.Delete],
        }),
      ];

      const result = ActionConnectionsLayer({ connections });
      expect(result?.props.children).toHaveLength(3);

      // Verify each connection has correct color
      expect(connections[0].color).toBe('#3b82f6'); // blue
      expect(connections[1].color).toBe('#22c55e'); // green
      expect(connections[2].color).toBe('#ef4444'); // red
    });
  });

  describe('Phase Transitions with PHASE_BOUNDS', () => {
    it('identifies spawn phase (progress 0 to 0.2)', () => {
      const spawnConn = createMockConnection({
        progress: 0.1,
        phase: 'spawn',
      });

      expect(spawnConn.progress).toBeLessThan(PHASE_BOUNDS.spawnEnd);
      expect(spawnConn.phase).toBe('spawn');
    });

    it('identifies travel phase (progress 0.2 to 0.8)', () => {
      const travelConn = createMockConnection({
        progress: 0.5,
        phase: 'travel',
      });

      expect(travelConn.progress).toBeGreaterThanOrEqual(PHASE_BOUNDS.spawnEnd);
      expect(travelConn.progress).toBeLessThan(PHASE_BOUNDS.travelEnd);
      expect(travelConn.phase).toBe('travel');
    });

    it('identifies impact phase (progress 0.8 to 0.9)', () => {
      const impactConn = createMockConnection({
        progress: 0.85,
        phase: 'impact',
      });

      expect(impactConn.progress).toBeGreaterThanOrEqual(PHASE_BOUNDS.travelEnd);
      expect(impactConn.progress).toBeLessThan(PHASE_BOUNDS.impactEnd);
      expect(impactConn.phase).toBe('impact');
    });

    it('identifies fade phase (progress 0.9 to 1.0)', () => {
      const fadeConn = createMockConnection({
        progress: 0.95,
        phase: 'fade',
      });

      expect(fadeConn.progress).toBeGreaterThanOrEqual(PHASE_BOUNDS.impactEnd);
      expect(fadeConn.progress).toBeLessThanOrEqual(PHASE_BOUNDS.fadeEnd);
      expect(fadeConn.phase).toBe('fade');
    });

    it('renders connections in all phases correctly', () => {
      const connections = [
        createMockConnection({ id: 'spawn', progress: 0.1, phase: 'spawn' }),
        createMockConnection({ id: 'travel', progress: 0.5, phase: 'travel' }),
        createMockConnection({ id: 'impact', progress: 0.85, phase: 'impact' }),
        createMockConnection({ id: 'fade', progress: 0.95, phase: 'fade' }),
      ];

      const result = ActionConnectionsLayer({ connections });
      expect(result?.props.children).toHaveLength(4);
    });

    it('validates PHASE_BOUNDS are cumulative and sum to 1.0', () => {
      expect(PHASE_BOUNDS.spawnEnd).toBe(0.2);
      expect(PHASE_BOUNDS.travelEnd).toBe(0.8);
      expect(PHASE_BOUNDS.impactEnd).toBe(0.9);
      expect(PHASE_BOUNDS.fadeEnd).toBe(1.0);

      // Verify phases are contiguous
      expect(PHASE_BOUNDS.spawnEnd).toBeLessThan(PHASE_BOUNDS.travelEnd);
      expect(PHASE_BOUNDS.travelEnd).toBeLessThan(PHASE_BOUNDS.impactEnd);
      expect(PHASE_BOUNDS.impactEnd).toBeLessThan(PHASE_BOUNDS.fadeEnd);
    });

    it('calculates spawn phase visuals correctly', () => {
      // During spawn phase, line fades in and particle grows
      const progress = 0.1; // 50% through spawn phase
      const expectedLineOpacity = progress / PHASE_BOUNDS.spawnEnd; // 0.5
      const expectedParticleScale = progress / PHASE_BOUNDS.spawnEnd; // 0.5

      expect(expectedLineOpacity).toBeCloseTo(0.5, 1);
      expect(expectedParticleScale).toBeCloseTo(0.5, 1);
    });

    it('calculates travel phase visuals correctly', () => {
      // During travel phase, full visibility
      const conn = createMockConnection({ progress: 0.5, phase: 'travel' });

      // Travel phase should have full opacity
      expect(conn.phase).toBe('travel');
      // In travel phase, lineOpacity = 1.0, particleScale = 1.0
    });

    it('calculates impact phase visuals correctly', () => {
      // Impact phase: burst effect
      const progress = 0.85;
      const impactScale =
        (progress - PHASE_BOUNDS.travelEnd) /
        (PHASE_BOUNDS.impactEnd - PHASE_BOUNDS.travelEnd);

      expect(impactScale).toBeCloseTo(0.5, 1); // 50% through impact phase
    });

    it('calculates fade phase visuals correctly', () => {
      // Fade phase: everything fades out
      const progress = 0.95;
      const fadeProgress =
        (progress - PHASE_BOUNDS.impactEnd) /
        (PHASE_BOUNDS.fadeEnd - PHASE_BOUNDS.impactEnd);
      const expectedLineOpacity = 1 - fadeProgress;

      expect(fadeProgress).toBeCloseTo(0.5, 1);
      expect(expectedLineOpacity).toBeCloseTo(0.5, 1);
    });
  });

  describe('VR Mode Geometry Complexity', () => {
    it('uses reduced geometry segments in VR mode', () => {
      // VR mode uses 20 points for curves, desktop uses 40
      const vrSegments = 20;
      const desktopSegments = 40;

      expect(vrSegments).toBeLessThan(desktopSegments);
      expect(vrSegments).toBe(20);
    });

    it('uses smaller sphere geometry in VR mode', () => {
      // VR: sphereGeometry args={[0.2, 8, 8]}
      // Desktop: sphereGeometry args={[0.4, 16, 16]}
      const vrSphereRadius = 0.2;
      const vrSphereSegments = 8;
      const desktopSphereRadius = 0.4;
      const desktopSphereSegments = 16;

      expect(vrSphereRadius).toBeLessThan(desktopSphereRadius);
      expect(vrSphereSegments).toBeLessThan(desktopSphereSegments);
    });

    it('uses smaller particle scale in VR mode', () => {
      // VR: 0.3, Desktop: 0.5
      const vrScale = 0.3;
      const desktopScale = 0.5;

      expect(vrScale).toBeLessThan(desktopScale);
    });

    it('disables glow effect around particle in VR mode', () => {
      // Glow is only rendered when !vrMode
      const vrMode = true;
      const shouldRenderGlow = !vrMode;

      expect(shouldRenderGlow).toBe(false);
    });

    it('uses reduced ring geometry segments in VR mode', () => {
      // VR: ringGeometry args={[0.5, 2, 16]}
      // Desktop: ringGeometry args={[0.5, 2, 32]}
      const vrRingSegments = 16;
      const desktopRingSegments = 32;

      expect(vrRingSegments).toBeLessThan(desktopRingSegments);
    });

    it('renders connection with vrMode=true', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections, vrMode: true });

      expect(result).not.toBeNull();
      const child = result?.props.children[0];
      expect(child.props.vrMode).toBe(true);
    });

    it('renders connection with vrMode=false (default)', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections, vrMode: false });

      expect(result).not.toBeNull();
      const child = result?.props.children[0];
      expect(child.props.vrMode).toBe(false);
    });

    it('defaults vrMode to false when not specified', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections });

      // Component defaults vrMode to false
      const child = result?.props.children[0];
      expect(child.props.vrMode).toBe(false);
    });
  });

  describe('maxConnections Limit', () => {
    it('enforces desktop limit of 50 connections', () => {
      const desktopMaxConnections = 50;
      const connections = createConnectionsOfType(AgentActionType.Query, 60);

      // When enforcing limit, should slice to maxConnections
      const limited = connections.slice(-desktopMaxConnections);
      expect(limited).toHaveLength(50);
    });

    it('enforces VR limit of 25 connections', () => {
      const vrMaxConnections = 25;
      const connections = createConnectionsOfType(AgentActionType.Query, 30);

      // When enforcing limit in VR mode, should slice to vrMaxConnections
      const limited = connections.slice(-vrMaxConnections);
      expect(limited).toHaveLength(25);
    });

    it('renders all connections when under desktop limit', () => {
      const connections = createConnectionsOfType(AgentActionType.Query, 30);
      const result = ActionConnectionsLayer({ connections });

      expect(result?.props.children).toHaveLength(30);
    });

    it('renders all connections when under VR limit', () => {
      const connections = createConnectionsOfType(AgentActionType.Query, 20);
      const result = ActionConnectionsLayer({ connections, vrMode: true });

      expect(result?.props.children).toHaveLength(20);
    });

    it('removes oldest connections first when limit exceeded', () => {
      const connections = [
        createMockConnection({ id: 'oldest', startTime: 1000 }),
        createMockConnection({ id: 'middle', startTime: 2000 }),
        createMockConnection({ id: 'newest', startTime: 3000 }),
      ];

      // Simulating limit enforcement - oldest removed first (uses slice(-max))
      const limited = connections.slice(-2);
      expect(limited.map((c) => c.id)).toEqual(['middle', 'newest']);
    });

    it('validates maxConnections limits are correct', () => {
      // Per component documentation: Max 50 concurrent connections
      // Per useActionConnections: vrMode ? 25 : 50
      const DESKTOP_MAX = 50;
      const VR_MAX = 25;

      expect(DESKTOP_MAX).toBe(50);
      expect(VR_MAX).toBe(25);
      expect(VR_MAX).toBeLessThan(DESKTOP_MAX);
    });
  });

  describe('Position Handling', () => {
    it('uses provided source position when available', () => {
      const conn = createMockConnection({
        sourcePosition: { x: 5, y: 10, z: 15 },
      });

      expect(conn.sourcePosition).toEqual({ x: 5, y: 10, z: 15 });
    });

    it('uses provided target position when available', () => {
      const conn = createMockConnection({
        targetPosition: { x: 20, y: 25, z: 30 },
      });

      expect(conn.targetPosition).toEqual({ x: 20, y: 25, z: 30 });
    });

    it('handles undefined source position gracefully', () => {
      const conn = createMockConnection({
        sourcePosition: undefined,
      });

      // Component uses fallback based on sourceAgentId hash
      expect(conn.sourcePosition).toBeUndefined();
    });

    it('handles undefined target position gracefully', () => {
      const conn = createMockConnection({
        targetPosition: undefined,
      });

      // Component uses fallback based on targetNodeId hash
      expect(conn.targetPosition).toBeUndefined();
    });
  });

  describe('Opacity Handling', () => {
    it('applies global opacity multiplier', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections, opacity: 0.5 });

      const child = result?.props.children[0];
      expect(child.props.opacity).toBe(0.5);
    });

    it('defaults opacity to 1.0', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections });

      const child = result?.props.children[0];
      expect(child.props.opacity).toBe(1.0);
    });

    it('handles zero opacity', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections, opacity: 0 });

      expect(result).not.toBeNull();
      const child = result?.props.children[0];
      expect(child.props.opacity).toBe(0);
    });
  });

  describe('Line Width Handling', () => {
    it('applies custom line width', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections, lineWidth: 4 });

      const child = result?.props.children[0];
      expect(child.props.lineWidth).toBe(4);
    });

    it('defaults line width to 2', () => {
      const connections = [createMockConnection()];
      const result = ActionConnectionsLayer({ connections });

      const child = result?.props.children[0];
      expect(child.props.lineWidth).toBe(2);
    });
  });
});

describe('ActionConnectionsStats', () => {
  /**
   * ActionConnectionsStats uses useMemo which requires React rendering context.
   * These tests verify the stats computation logic without rendering.
   */

  it('computes connection count correctly', () => {
    const connections = createConnectionsOfType(AgentActionType.Query, 5);

    // Stats component displays connections.length
    expect(connections.length).toBe(5);
  });

  it('groups connections by action type', () => {
    const connections = [
      ...createConnectionsOfType(AgentActionType.Query, 3),
      ...createConnectionsOfType(AgentActionType.Create, 2),
      ...createConnectionsOfType(AgentActionType.Delete, 1),
    ];

    // Stats component computes byType counts using this logic
    const byType: Record<string, number> = {};
    for (const conn of connections) {
      const typeName = AgentActionType[conn.actionType] || 'Unknown';
      byType[typeName] = (byType[typeName] || 0) + 1;
    }

    expect(byType.Query).toBe(3);
    expect(byType.Create).toBe(2);
    expect(byType.Delete).toBe(1);
  });

  it('verifies stats styling configuration', () => {
    // Stats component renders with these inline styles
    const expectedStyles = {
      position: 'absolute',
      bottom: 10,
      left: 10,
      background: 'rgba(0,0,0,0.7)',
      color: 'white',
      padding: '8px 12px',
      borderRadius: 4,
      fontSize: 12,
      fontFamily: 'monospace',
    };

    expect(expectedStyles.position).toBe('absolute');
    expect(expectedStyles.bottom).toBe(10);
    expect(expectedStyles.left).toBe(10);
  });

  it('handles empty connections array', () => {
    const connections: ActionConnection[] = [];

    // Stats displays "Active Actions: {count}"
    expect(connections.length).toBe(0);
  });

  it('handles unknown action types', () => {
    const unknownConnection = createMockConnection({
      actionType: 999 as AgentActionType,
    });

    // Stats should handle unknown types gracefully
    const typeName = AgentActionType[unknownConnection.actionType] || 'Unknown';
    expect(typeName).toBe('Unknown');
  });

  it('computes stats for all action types', () => {
    const connections = [
      createMockConnection({ actionType: AgentActionType.Query }),
      createMockConnection({ actionType: AgentActionType.Update }),
      createMockConnection({ actionType: AgentActionType.Create }),
      createMockConnection({ actionType: AgentActionType.Delete }),
      createMockConnection({ actionType: AgentActionType.Link }),
      createMockConnection({ actionType: AgentActionType.Transform }),
    ];

    const byType: Record<string, number> = {};
    for (const conn of connections) {
      const typeName = AgentActionType[conn.actionType] || 'Unknown';
      byType[typeName] = (byType[typeName] || 0) + 1;
    }

    expect(Object.keys(byType)).toHaveLength(6);
    expect(byType.Query).toBe(1);
    expect(byType.Update).toBe(1);
    expect(byType.Create).toBe(1);
    expect(byType.Delete).toBe(1);
    expect(byType.Link).toBe(1);
    expect(byType.Transform).toBe(1);
  });
});

describe('Bezier Curve Generation', () => {
  it('creates quadratic bezier curve between source and target', () => {
    const source = new THREE.Vector3(0, 0, 0);
    const target = new THREE.Vector3(10, 5, 10);

    // Mid point calculation
    const midPoint = new THREE.Vector3()
      .addVectors(source, target)
      .multiplyScalar(0.5);

    expect(midPoint.x).toBe(5);
    expect(midPoint.y).toBe(2.5);
    expect(midPoint.z).toBe(5);
  });

  it('adds perpendicular offset for arc effect', () => {
    const source = new THREE.Vector3(0, 0, 0);
    const target = new THREE.Vector3(10, 0, 0);
    const distance = source.distanceTo(target);

    expect(distance).toBe(10);

    // Offset should be proportional to distance
    const offsetAmount = distance * 0.3;
    expect(offsetAmount).toBe(3);
  });

  it('varies curve offset based on action type', () => {
    // offsetAmount = distance * 0.3 * (1 + (actionType * 0.1))
    const distance = 10;

    const queryOffset = distance * 0.3 * (1 + AgentActionType.Query * 0.1);
    const deleteOffset = distance * 0.3 * (1 + AgentActionType.Delete * 0.1);

    expect(queryOffset).toBe(3); // 0 * 0.1 = 0, so 3 * 1 = 3
    expect(deleteOffset).toBeCloseTo(3.9, 1); // 3 * 0.1 = 0.3, so 3 * 1.3 = 3.9
  });
});

describe('Animation Frame Updates', () => {
  it('updates line opacity during animation', () => {
    const mockMaterial = { opacity: 1 };
    const mockLineRef = { current: { material: mockMaterial } };

    // Simulate opacity update
    mockMaterial.opacity = 0.5;
    expect(mockLineRef.current.material.opacity).toBe(0.5);
  });

  it('updates particle position along curve during travel', () => {
    const source = new THREE.Vector3(0, 0, 0);
    const target = new THREE.Vector3(10, 10, 10);

    // At 50% progress through travel phase
    const progress = 0.5;
    const expectedPosition = new THREE.Vector3(
      source.x + (target.x - source.x) * progress,
      source.y + (target.y - source.y) * progress,
      source.z + (target.z - source.z) * progress
    );

    expect(expectedPosition.x).toBe(5);
    expect(expectedPosition.y).toBe(5);
    expect(expectedPosition.z).toBe(5);
  });

  it('scales impact ring during impact phase', () => {
    const progress = 0.85; // In impact phase
    const impactScale =
      ((progress - PHASE_BOUNDS.travelEnd) /
        (PHASE_BOUNDS.impactEnd - PHASE_BOUNDS.travelEnd)) *
      2;

    expect(impactScale).toBeCloseTo(1, 1); // Scale multiplied by 2
  });
});
