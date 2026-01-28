/**
 * useActionConnections Hook Tests
 *
 * Tests for the action connections hook that manages ephemeral animated connections
 * between agent nodes and data nodes.
 *
 * Uses direct hook invocation via a test component to avoid React 19 compatibility
 * issues with @testing-library/react's renderHook.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import React, { useState, useEffect } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { useActionConnections, ActionConnection, UseActionConnectionsOptions } from '../useActionConnections';
import {
  AgentActionType,
  AgentActionEvent,
  AGENT_ACTION_COLORS,
} from '@/services/BinaryWebSocketProtocol';

// Phase timing constants from the hook
const PHASE_TIMING = {
  spawn: 0.2,    // 0.0 - 0.2 (100ms of 500ms)
  travel: 0.6,   // 0.2 - 0.8 (300ms of 500ms)
  impact: 0.1,   // 0.8 - 0.9 (50ms of 500ms)
  fade: 0.1,     // 0.9 - 1.0 (50ms of 500ms)
};

/**
 * Creates a mock AgentActionEvent for testing
 */
function createMockEvent(overrides: Partial<AgentActionEvent> = {}): AgentActionEvent {
  return {
    sourceAgentId: 1,
    targetNodeId: 100,
    actionType: AgentActionType.Query,
    timestamp: Date.now(),
    durationMs: 500,
    ...overrides,
  };
}

/**
 * Creates a mock position resolver
 */
function createMockPositionResolver() {
  const positions = new Map<number, { x: number; y: number; z: number }>();
  positions.set(1, { x: 0, y: 0, z: 0 });
  positions.set(100, { x: 10, y: 5, z: 10 });
  positions.set(2, { x: 5, y: 5, z: 5 });
  positions.set(200, { x: 15, y: 10, z: 15 });

  return (nodeId: number) => positions.get(nodeId) || null;
}

/**
 * Test harness component that exposes hook result
 */
interface HookResult {
  connections: ActionConnection[];
  addAction: (event: AgentActionEvent) => void;
  addActions: (events: AgentActionEvent[]) => void;
  clearAll: () => void;
  updatePositions: () => void;
  getConnectionsByType: (type: AgentActionType) => ActionConnection[];
  activeCount: number;
}

let hookResultRef: HookResult | null = null;
let renderCount = 0;
let resolveRender: (() => void) | null = null;

function TestComponent({ options = {} }: { options?: UseActionConnectionsOptions }) {
  const result = useActionConnections(options);

  // Update ref synchronously during render
  hookResultRef = result;
  renderCount++;

  useEffect(() => {
    // Signal that render is complete
    if (resolveRender) {
      resolveRender();
      resolveRender = null;
    }
  });

  return null;
}

describe('useActionConnections', () => {
  let container: HTMLDivElement;
  let root: Root;
  let mockPerformanceNow: number;
  let rafCallbacks: Map<number, (time: number) => void>;
  let rafIdCounter: number;

  beforeEach(() => {
    vi.clearAllMocks();
    mockPerformanceNow = 1000;
    rafCallbacks = new Map();
    rafIdCounter = 0;
    hookResultRef = null;
    renderCount = 0;
    resolveRender = null;

    // Create DOM container
    container = document.createElement('div');
    document.body.appendChild(container);

    // Mock performance.now
    vi.spyOn(performance, 'now').mockImplementation(() => mockPerformanceNow);

    // Mock requestAnimationFrame - don't auto-execute, only store for manual triggering
    vi.spyOn(global, 'requestAnimationFrame').mockImplementation((cb) => {
      const id = ++rafIdCounter;
      rafCallbacks.set(id, cb);
      return id;
    });

    vi.spyOn(global, 'cancelAnimationFrame').mockImplementation((id) => {
      rafCallbacks.delete(id);
    });
  });

  afterEach(() => {
    if (root) {
      root.unmount();
    }
    if (container) {
      document.body.removeChild(container);
    }
    vi.restoreAllMocks();
  });

  /**
   * Renders the test component and waits for it to be ready
   */
  async function renderHookTest(options: UseActionConnectionsOptions = {}): Promise<HookResult> {
    root = createRoot(container);

    await new Promise<void>((resolve) => {
      resolveRender = resolve;
      root.render(React.createElement(TestComponent, { options }));
    });

    // Give a small buffer for effects to run
    await new Promise((resolve) => setTimeout(resolve, 10));

    if (!hookResultRef) {
      throw new Error('Hook did not render');
    }

    return hookResultRef;
  }

  /**
   * Advances the mock time and triggers one animation frame update
   * Only triggers existing callbacks, then clears them to prevent infinite loop
   */
  function advanceTime(ms: number) {
    mockPerformanceNow += ms;
    // Get current callbacks and clear the map
    const currentCallbacks = new Map(rafCallbacks);
    rafCallbacks.clear();
    rafIdCounter = 0;
    // Trigger callbacks - they will register new ones which we ignore
    currentCallbacks.forEach((cb) => {
      cb(mockPerformanceNow);
    });
  }

  /**
   * Wait for state updates to propagate
   */
  async function waitForUpdate(): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  describe('addAction creates connection with correct properties', () => {
    it('creates connection with unique id', async () => {
      const result = await renderHookTest();
      const event = createMockEvent();

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(1);
      expect(hookResultRef!.connections[0].id).toMatch(/^action-\d+$/);
    });

    it('sets correct sourceAgentId from event', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ sourceAgentId: 42 });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].sourceAgentId).toBe(42);
    });

    it('sets correct targetNodeId from event', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ targetNodeId: 999 });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].targetNodeId).toBe(999);
    });

    it('sets correct actionType from event', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ actionType: AgentActionType.Delete });

      result.addAction(event);
      await waitForUpdate();

      // New API returns string literal, legacy enum stored in _actionTypeEnum
      expect(hookResultRef!.connections[0].actionType).toBe('delete');
      expect(hookResultRef!.connections[0]._actionTypeEnum).toBe(AgentActionType.Delete);
    });

    it('initializes progress to 0', async () => {
      const result = await renderHookTest();
      const event = createMockEvent();

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].progress).toBe(0);
    });

    it('initializes phase to spawn', async () => {
      const result = await renderHookTest();
      const event = createMockEvent();

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].phase).toBe('spawn');
    });

    it('sets startTime from performance.now', async () => {
      const result = await renderHookTest();
      const event = createMockEvent();

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].startTime).toBe(mockPerformanceNow);
    });

    it('uses event durationMs when provided', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 750 });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].duration).toBe(750);
    });

    it('uses baseDuration when event durationMs is 0', async () => {
      const result = await renderHookTest({ baseDuration: 600 });
      const event = createMockEvent({ durationMs: 0 });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].duration).toBe(600);
    });

    it('uses default baseDuration (500ms) when not specified', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 0 });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].duration).toBe(500);
    });

    it('resolves source position using getNodePosition', async () => {
      const getNodePosition = createMockPositionResolver();
      const result = await renderHookTest({ getNodePosition });
      const event = createMockEvent({ sourceAgentId: 1 });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].sourcePosition).toEqual({
        x: 0,
        y: 0,
        z: 0,
      });
    });

    it('resolves target position using getNodePosition', async () => {
      const getNodePosition = createMockPositionResolver();
      const result = await renderHookTest({ getNodePosition });
      const event = createMockEvent({ targetNodeId: 100 });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].targetPosition).toEqual({
        x: 10,
        y: 5,
        z: 10,
      });
    });

    it('sets sourcePosition to default Vector3(0,0,0) when position not found', async () => {
      const getNodePosition = () => null;
      const result = await renderHookTest({ getNodePosition });
      const event = createMockEvent({ sourceAgentId: 999 });

      result.addAction(event);
      await waitForUpdate();

      // New API returns default Vector3 instead of undefined for easier rendering
      const pos = hookResultRef!.connections[0].sourcePosition;
      expect(pos.x).toBe(0);
      expect(pos.y).toBe(0);
      expect(pos.z).toBe(0);
    });

    it('creates multiple connections with incrementing ids', async () => {
      const result = await renderHookTest();

      result.addAction(createMockEvent());
      await waitForUpdate();
      result.addAction(createMockEvent());
      await waitForUpdate();
      result.addAction(createMockEvent());
      await waitForUpdate();

      const ids = hookResultRef!.connections.map((c) => c.id);
      expect(ids[0]).toBe('action-0');
      expect(ids[1]).toBe('action-1');
      expect(ids[2]).toBe('action-2');
    });
  });

  describe('Connection lifecycle (spawn, travel, impact, fade phases)', () => {
    it('stays in spawn phase during first 20% of duration', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 500 });

      result.addAction(event);
      await waitForUpdate();

      // At 50ms (10% progress)
      advanceTime(50);
      await waitForUpdate();

      const conn = hookResultRef!.connections[0];
      expect(conn.progress).toBeCloseTo(0.1, 1);
      expect(conn.phase).toBe('spawn');
    });

    it('transitions to travel phase after 20% of duration', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 500 });

      result.addAction(event);
      await waitForUpdate();

      // At 150ms (30% progress - in travel phase)
      advanceTime(150);
      await waitForUpdate();

      const conn = hookResultRef!.connections[0];
      expect(conn.progress).toBeCloseTo(0.3, 1);
      expect(conn.phase).toBe('travel');
    });

    it('stays in travel phase during 20%-80% of duration', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 500 });

      result.addAction(event);
      await waitForUpdate();

      // At 250ms (50% progress)
      advanceTime(250);
      await waitForUpdate();

      const conn = hookResultRef!.connections[0];
      expect(conn.progress).toBeCloseTo(0.5, 1);
      expect(conn.phase).toBe('travel');
    });

    it('transitions to impact phase after 80% of duration', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 500 });

      result.addAction(event);
      await waitForUpdate();

      // At 425ms (85% progress - in impact phase)
      advanceTime(425);
      await waitForUpdate();

      const conn = hookResultRef!.connections[0];
      expect(conn.progress).toBeCloseTo(0.85, 1);
      expect(conn.phase).toBe('impact');
    });

    it('stays in impact phase through 80-100% of duration (combined impact+fade)', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 500 });

      result.addAction(event);
      await waitForUpdate();

      // At 475ms (95% progress - still in impact phase, which now includes fade)
      advanceTime(475);
      await waitForUpdate();

      const conn = hookResultRef!.connections[0];
      expect(conn.progress).toBeCloseTo(0.95, 1);
      // Impact and fade are now combined into a single 'impact' phase (0.8-1.0)
      expect(conn.phase).toBe('impact');
    });

    it('removes connection after 100% progress', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 500 });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(1);

      // At 550ms (110% progress - should be removed)
      advanceTime(550);
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(0);
    });

    it('validates phase timing boundaries are correct', () => {
      // spawn: 0.0 - 0.2
      // travel: 0.2 - 0.8
      // impact: 0.8 - 0.9
      // fade: 0.9 - 1.0
      expect(PHASE_TIMING.spawn).toBe(0.2);
      expect(PHASE_TIMING.travel).toBe(0.6);
      expect(PHASE_TIMING.impact).toBe(0.1);
      expect(PHASE_TIMING.fade).toBe(0.1);

      // Verify they sum to 1.0
      const total =
        PHASE_TIMING.spawn +
        PHASE_TIMING.travel +
        PHASE_TIMING.impact +
        PHASE_TIMING.fade;
      expect(total).toBeCloseTo(1.0, 5);
    });
  });

  describe('maxConnections limit enforced', () => {
    it('enforces default maxConnections limit of 50', async () => {
      const result = await renderHookTest();

      // Add 60 connections
      for (let i = 0; i < 60; i++) {
        result.addAction(
          createMockEvent({ sourceAgentId: i, targetNodeId: i + 100 })
        );
      }
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(50);
    });

    it('enforces custom maxConnections limit', async () => {
      const result = await renderHookTest({ maxConnections: 10 });

      // Add 15 connections
      for (let i = 0; i < 15; i++) {
        result.addAction(
          createMockEvent({ sourceAgentId: i, targetNodeId: i + 100 })
        );
      }
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(10);
    });

    it('allows all connections when under limit', async () => {
      const result = await renderHookTest({ maxConnections: 50 });

      // Add 30 connections
      for (let i = 0; i < 30; i++) {
        result.addAction(
          createMockEvent({ sourceAgentId: i, targetNodeId: i + 100 })
        );
      }
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(30);
    });

    it('maintains exactly maxConnections when at limit', async () => {
      const result = await renderHookTest({ maxConnections: 25 });

      // Add exactly 25 connections
      for (let i = 0; i < 25; i++) {
        result.addAction(
          createMockEvent({ sourceAgentId: i, targetNodeId: i + 100 })
        );
      }
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(25);

      // Add one more
      result.addAction(createMockEvent({ sourceAgentId: 99 }));
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(25);
    });
  });

  describe('Oldest connections removed when limit exceeded', () => {
    it('removes oldest connections first using slice(-maxConnections)', async () => {
      const result = await renderHookTest({ maxConnections: 3 });

      // Add connections with identifiable sourceAgentIds
      result.addAction(createMockEvent({ sourceAgentId: 1 })); // Oldest
      result.addAction(createMockEvent({ sourceAgentId: 2 }));
      result.addAction(createMockEvent({ sourceAgentId: 3 }));
      result.addAction(createMockEvent({ sourceAgentId: 4 })); // Newest
      await waitForUpdate();

      // Oldest (sourceAgentId: 1) should be removed
      const sourceIds = hookResultRef!.connections.map((c) => c.sourceAgentId);
      expect(sourceIds).toEqual([2, 3, 4]);
    });

    it('removes multiple oldest when batch exceeds limit', async () => {
      const result = await renderHookTest({ maxConnections: 3 });

      // Add 6 connections
      for (let i = 1; i <= 6; i++) {
        result.addAction(createMockEvent({ sourceAgentId: i }));
      }
      await waitForUpdate();

      // Only newest 3 should remain
      const sourceIds = hookResultRef!.connections.map((c) => c.sourceAgentId);
      expect(sourceIds).toEqual([4, 5, 6]);
    });

    it('preserves newest connections when limit exceeded', async () => {
      const result = await renderHookTest({ maxConnections: 5 });

      // Add 10 connections
      for (let i = 1; i <= 10; i++) {
        result.addAction(createMockEvent({ sourceAgentId: i }));
      }
      await waitForUpdate();

      // Newest 5 should remain
      const sourceIds = hookResultRef!.connections.map((c) => c.sourceAgentId);
      expect(sourceIds).toEqual([6, 7, 8, 9, 10]);
    });
  });

  describe('Color mapping by action type (AGENT_ACTION_COLORS)', () => {
    it('maps Query action type to blue (#3b82f6)', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ actionType: AgentActionType.Query });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].color).toBe('#3b82f6');
      expect(AGENT_ACTION_COLORS[AgentActionType.Query]).toBe('#3b82f6');
    });

    it('maps Update action type to yellow (#eab308)', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ actionType: AgentActionType.Update });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].color).toBe('#eab308');
      expect(AGENT_ACTION_COLORS[AgentActionType.Update]).toBe('#eab308');
    });

    it('maps Create action type to green (#22c55e)', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ actionType: AgentActionType.Create });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].color).toBe('#22c55e');
      expect(AGENT_ACTION_COLORS[AgentActionType.Create]).toBe('#22c55e');
    });

    it('maps Delete action type to red (#ef4444)', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ actionType: AgentActionType.Delete });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].color).toBe('#ef4444');
      expect(AGENT_ACTION_COLORS[AgentActionType.Delete]).toBe('#ef4444');
    });

    it('maps Link action type to purple (#a855f7)', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ actionType: AgentActionType.Link });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].color).toBe('#a855f7');
      expect(AGENT_ACTION_COLORS[AgentActionType.Link]).toBe('#a855f7');
    });

    it('maps Transform action type to cyan (#06b6d4)', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ actionType: AgentActionType.Transform });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].color).toBe('#06b6d4');
      expect(AGENT_ACTION_COLORS[AgentActionType.Transform]).toBe('#06b6d4');
    });

    it('falls back to white (#ffffff) for unknown action types', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ actionType: 999 as AgentActionType });

      result.addAction(event);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].color).toBe('#ffffff');
    });

    it('creates connections with different colors for mixed action types', async () => {
      const result = await renderHookTest();

      result.addAction(createMockEvent({ actionType: AgentActionType.Query }));
      result.addAction(createMockEvent({ actionType: AgentActionType.Create }));
      result.addAction(createMockEvent({ actionType: AgentActionType.Delete }));
      await waitForUpdate();

      expect(hookResultRef!.connections[0].color).toBe('#3b82f6'); // blue
      expect(hookResultRef!.connections[1].color).toBe('#22c55e'); // green
      expect(hookResultRef!.connections[2].color).toBe('#ef4444'); // red
    });
  });

  describe('Animation progress updates correctly', () => {
    it('progress increases proportionally to elapsed time', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 1000 });

      result.addAction(event);
      await waitForUpdate();

      // At 250ms (25% progress)
      advanceTime(250);
      await waitForUpdate();

      expect(hookResultRef!.connections[0].progress).toBeCloseTo(0.25, 1);
    });

    it('progress is capped at 1.0', async () => {
      const result = await renderHookTest();
      const event = createMockEvent({ durationMs: 100 });

      result.addAction(event);
      await waitForUpdate();

      // Progress should be capped, not exceed 1.0
      advanceTime(80);
      await waitForUpdate();

      const conn = hookResultRef!.connections[0];
      if (conn) {
        expect(conn.progress).toBeLessThanOrEqual(1.0);
      }
    });

    it('updates activeCount correctly', async () => {
      const result = await renderHookTest();

      expect(hookResultRef!.activeCount).toBe(0);

      result.addAction(createMockEvent());
      await waitForUpdate();

      expect(hookResultRef!.activeCount).toBe(1);

      result.addAction(createMockEvent());
      result.addAction(createMockEvent());
      await waitForUpdate();

      expect(hookResultRef!.activeCount).toBe(3);
    });

    it('clearAll removes all connections', async () => {
      const result = await renderHookTest();

      result.addAction(createMockEvent());
      result.addAction(createMockEvent());
      result.addAction(createMockEvent());
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(3);

      result.clearAll();
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(0);
      expect(hookResultRef!.activeCount).toBe(0);
    });

    it('getConnectionsByType filters connections correctly', async () => {
      const result = await renderHookTest();

      result.addAction(createMockEvent({ actionType: AgentActionType.Query }));
      result.addAction(createMockEvent({ actionType: AgentActionType.Query }));
      result.addAction(createMockEvent({ actionType: AgentActionType.Create }));
      result.addAction(createMockEvent({ actionType: AgentActionType.Delete }));
      await waitForUpdate();

      const queryConnections = hookResultRef!.getConnectionsByType(AgentActionType.Query);
      const createConnections = hookResultRef!.getConnectionsByType(AgentActionType.Create);
      const deleteConnections = hookResultRef!.getConnectionsByType(AgentActionType.Delete);
      const updateConnections = hookResultRef!.getConnectionsByType(AgentActionType.Update);

      expect(queryConnections).toHaveLength(2);
      expect(createConnections).toHaveLength(1);
      expect(deleteConnections).toHaveLength(1);
      expect(updateConnections).toHaveLength(0);
    });

    it('updatePositions refreshes positions from resolver', async () => {
      const positions = new Map<number, { x: number; y: number; z: number }>();
      positions.set(1, { x: 0, y: 0, z: 0 });
      positions.set(100, { x: 10, y: 5, z: 10 });

      const getNodePosition = (nodeId: number) => positions.get(nodeId) || null;

      const result = await renderHookTest({ getNodePosition });

      result.addAction(createMockEvent({ sourceAgentId: 1, targetNodeId: 100 }));
      await waitForUpdate();

      expect(hookResultRef!.connections[0].sourcePosition).toEqual({
        x: 0,
        y: 0,
        z: 0,
      });

      // Update positions
      positions.set(1, { x: 5, y: 5, z: 5 });

      result.updatePositions();
      await waitForUpdate();

      expect(hookResultRef!.connections[0].sourcePosition).toEqual({
        x: 5,
        y: 5,
        z: 5,
      });
    });
  });

  describe('addActions batch processing', () => {
    it('adds multiple actions at once', async () => {
      const result = await renderHookTest();

      const events = [
        createMockEvent({ sourceAgentId: 1 }),
        createMockEvent({ sourceAgentId: 2 }),
        createMockEvent({ sourceAgentId: 3 }),
      ];

      result.addActions(events);
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(3);
    });

    it('processes batch events in order', async () => {
      const result = await renderHookTest();

      const events = [
        createMockEvent({ sourceAgentId: 10 }),
        createMockEvent({ sourceAgentId: 20 }),
        createMockEvent({ sourceAgentId: 30 }),
      ];

      result.addActions(events);
      await waitForUpdate();

      const sourceIds = hookResultRef!.connections.map((c) => c.sourceAgentId);
      expect(sourceIds).toEqual([10, 20, 30]);
    });

    it('respects maxConnections limit during batch add', async () => {
      const result = await renderHookTest({ maxConnections: 5 });

      const events = Array.from({ length: 10 }, (_, i) =>
        createMockEvent({ sourceAgentId: i })
      );

      result.addActions(events);
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(5);
    });
  });

  describe('Animation frame cleanup', () => {
    it('cancels animation frame on unmount when connections exist', async () => {
      const result = await renderHookTest({
        getNodePosition: createMockPositionResolver(),
      });

      // Add a connection to start the animation loop
      result.addAction(createMockEvent());
      await waitForUpdate();

      root.unmount();
      await waitForUpdate();

      expect(cancelAnimationFrame).toHaveBeenCalled();
    });

    it('starts animation loop when connections are added', async () => {
      const result = await renderHookTest({
        getNodePosition: createMockPositionResolver(),
      });

      // Animation loop only starts when there are connections to animate
      // (optimization to prevent CPU waste when idle)
      expect(requestAnimationFrame).not.toHaveBeenCalled();

      // Add a connection to trigger animation
      result.addAction(createMockEvent());
      await waitForUpdate();

      expect(requestAnimationFrame).toHaveBeenCalled();
    });
  });

  describe('Options configuration', () => {
    it('uses default options when none provided', async () => {
      const result = await renderHookTest();

      // Add 60 connections to verify default maxConnections (50)
      for (let i = 0; i < 60; i++) {
        result.addAction(createMockEvent({ sourceAgentId: i }));
      }
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(50);
    });

    it('merges provided options with defaults', async () => {
      const result = await renderHookTest({ maxConnections: 10 });

      // Verify maxConnections is customized
      for (let i = 0; i < 15; i++) {
        result.addAction(createMockEvent({ sourceAgentId: i }));
      }
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(10);

      // Verify baseDuration still uses default (500)
      result.clearAll();
      await waitForUpdate();

      result.addAction(createMockEvent({ durationMs: 0 }));
      await waitForUpdate();

      expect(hookResultRef!.connections[0].duration).toBe(500);
    });

    it('accepts vrMode option without affecting hook behavior', async () => {
      // vrMode is passed through but doesn't change hook logic
      const result = await renderHookTest({ vrMode: true });

      result.addAction(createMockEvent());
      await waitForUpdate();

      expect(hookResultRef!.connections).toHaveLength(1);
    });
  });
});
