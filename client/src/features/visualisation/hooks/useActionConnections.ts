/**
 * useActionConnections Hook
 *
 * Manages ephemeral action connections between agent nodes and data nodes.
 * Connections animate from agent → target with type-specific colors.
 *
 * Animation lifecycle (500ms total):
 * - spawn:  100ms - Line appears, particle grows at source
 * - travel: 300ms - Particle travels along bezier curve
 * - impact: 100ms - Burst effect at target + fade out
 *
 * Color coding by action type:
 * - query:     blue (#3b82f6)
 * - update:    yellow (#eab308)
 * - create:    green (#22c55e)
 * - delete:    red (#ef4444)
 * - link:      purple (#a855f7)
 * - transform: cyan (#06b6d4)
 *
 * Phase 2b: Enhanced with THREE.Vector3 position support and Map-based lookups.
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import {
  AgentActionType,
  AgentActionEvent,
  AGENT_ACTION_COLORS,
} from '@/services/BinaryWebSocketProtocol';
import { useWebSocketStore } from '@/store/websocketStore';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('useActionConnections');

/** Action type string literals for external API compatibility */
export type ActionTypeString = 'query' | 'update' | 'create' | 'delete' | 'link' | 'transform';

/** Map AgentActionType enum to string literal */
const ACTION_TYPE_MAP: Record<AgentActionType, ActionTypeString> = {
  [AgentActionType.Query]: 'query',
  [AgentActionType.Update]: 'update',
  [AgentActionType.Create]: 'create',
  [AgentActionType.Delete]: 'delete',
  [AgentActionType.Link]: 'link',
  [AgentActionType.Transform]: 'transform',
};

/** Single animated action connection with THREE.Vector3 positions */
export interface ActionConnection {
  /** Unique connection identifier */
  id: string;
  /** ID of the source agent node */
  sourceAgentId: number;
  /** ID of the target data node */
  targetNodeId: number;
  /** Action type as string literal */
  actionType: ActionTypeString;
  /** Internal enum action type */
  _actionTypeEnum: AgentActionType;
  /** Action color based on type */
  color: string;
  /** Animation progress 0-1 */
  progress: number;
  /** Current animation phase */
  phase: 'spawn' | 'travel' | 'impact' | 'fade';
  /** When the action started (performance.now()) */
  startTime: number;
  /** Total animation duration (ms) */
  duration: number;
  /** Source agent position in world space */
  sourcePosition: THREE.Vector3;
  /** Target node position in world space */
  targetPosition: THREE.Vector3;
}

/** Legacy ActionConnection interface for backward compatibility */
export interface LegacyActionConnection {
  id: string;
  sourceAgentId: number;
  targetNodeId: number;
  actionType: AgentActionType;
  color: string;
  progress: number;
  phase: 'spawn' | 'travel' | 'impact' | 'fade';
  startTime: number;
  duration: number;
  sourcePosition?: { x: number; y: number; z: number };
  targetPosition?: { x: number; y: number; z: number };
}

export interface UseActionConnectionsOptions {
  /** Maximum concurrent connections to display (default: 50) */
  maxConnections?: number;
  /** Base animation duration in ms (default: 500) */
  baseDuration?: number;
  /** Enable VR-optimized rendering (simplified geometry) */
  vrMode?: boolean;
  /** Position resolver for node IDs (legacy API) */
  getNodePosition?: (nodeId: number) => { x: number; y: number; z: number } | null;
  /** Auto-subscribe to WebSocket events (default: false for backward compat) */
  autoSubscribe?: boolean;
  /** Connection cleanup duration override (default: uses baseDuration) */
  cleanupDuration?: number;
}

const DEFAULT_OPTIONS: Required<UseActionConnectionsOptions> = {
  maxConnections: 50,
  baseDuration: 500,
  vrMode: false,
  getNodePosition: () => null,
  autoSubscribe: false,
  cleanupDuration: 500,
};

/**
 * Animation phase timing (as fraction of total duration)
 * Total duration: 500ms (configurable via baseDuration)
 *
 * Phase breakdown per spec:
 * - spawn:  0.0 - 0.2 (100ms) - Line appears, particle grows at source
 * - travel: 0.2 - 0.8 (300ms) - Particle travels along bezier curve
 * - impact: 0.8 - 1.0 (100ms) - Burst at target + fade out
 */
const PHASE_TIMING = {
  spawn: 0.2,    // 0.0 - 0.2 (100ms of 500ms)
  travel: 0.6,   // Cumulative: 0.2 - 0.8 (300ms of 500ms)
  impact: 0.2,   // Cumulative: 0.8 - 1.0 (100ms combined impact+fade)
  fade: 0.0,     // Combined with impact phase (for backward compat)
};

/** Default position when lookup fails */
const DEFAULT_POSITION = new THREE.Vector3(0, 0, 0);

/**
 * Primary hook interface using Map-based position lookups
 */
export function useActionConnections(
  agentPositions: Map<number, THREE.Vector3>,
  nodePositions: Map<number, THREE.Vector3>
): {
  connections: ActionConnection[];
  updateConnections: () => void;
};

/**
 * Legacy hook interface using options object
 */
export function useActionConnections(
  options?: UseActionConnectionsOptions
): {
  connections: ActionConnection[];
  addAction: (event: AgentActionEvent) => void;
  addActions: (events: AgentActionEvent[]) => void;
  clearAll: () => void;
  updatePositions: () => void;
  getConnectionsByType: (type: AgentActionType) => ActionConnection[];
  activeCount: number;
  updateConnections: () => void;
};

/**
 * Implementation supporting both interfaces
 */
export function useActionConnections(
  agentPositionsOrOptions?: Map<number, THREE.Vector3> | UseActionConnectionsOptions,
  nodePositions?: Map<number, THREE.Vector3>
) {
  // Determine which interface is being used
  const isMapInterface = agentPositionsOrOptions instanceof Map;

  // Extract position maps or create empty ones
  const agentPositionMap = isMapInterface
    ? (agentPositionsOrOptions as Map<number, THREE.Vector3>)
    : useMemo(() => new Map<number, THREE.Vector3>(), []);

  const nodePositionMap = isMapInterface && nodePositions
    ? nodePositions
    : useMemo(() => new Map<number, THREE.Vector3>(), []);

  // Extract options
  const options = isMapInterface
    ? DEFAULT_OPTIONS
    : { ...DEFAULT_OPTIONS, ...(agentPositionsOrOptions as UseActionConnectionsOptions || {}) };

  const config = options;

  // State
  const [connections, setConnections] = useState<ActionConnection[]>([]);
  const connectionIdCounter = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const lastUpdateRef = useRef<number>(performance.now());

  // WebSocket subscription for auto mode
  const wsOn = useWebSocketStore(state => state.on);

  /**
   * Get position from Maps or legacy resolver
   */
  const getPosition = useCallback((
    nodeId: number,
    isAgent: boolean
  ): THREE.Vector3 => {
    // Try Map lookup first
    const map = isAgent ? agentPositionMap : nodePositionMap;
    const mapPosition = map.get(nodeId);
    if (mapPosition) {
      return mapPosition.clone();
    }

    // Fall back to legacy resolver
    if (config.getNodePosition) {
      const legacyPos = config.getNodePosition(nodeId);
      if (legacyPos) {
        return new THREE.Vector3(legacyPos.x, legacyPos.y, legacyPos.z);
      }
    }

    // Return default position
    return DEFAULT_POSITION.clone();
  }, [agentPositionMap, nodePositionMap, config.getNodePosition]);

  /**
   * Add a new action connection from an AgentActionEvent
   */
  const addAction = useCallback((event: AgentActionEvent) => {
    const id = `action-${connectionIdCounter.current++}`;
    const color = AGENT_ACTION_COLORS[event.actionType] || '#ffffff';
    const duration = event.durationMs > 0 ? event.durationMs : config.baseDuration;
    const actionTypeString = ACTION_TYPE_MAP[event.actionType] || 'query';

    const sourcePosition = getPosition(event.sourceAgentId, true);
    const targetPosition = getPosition(event.targetNodeId, false);

    const newConnection: ActionConnection = {
      id,
      sourceAgentId: event.sourceAgentId,
      targetNodeId: event.targetNodeId,
      actionType: actionTypeString,
      _actionTypeEnum: event.actionType,
      color,
      progress: 0,
      phase: 'spawn',
      startTime: performance.now(),
      duration,
      sourcePosition,
      targetPosition,
    };

    setConnections(prev => {
      // Enforce max connections limit (remove oldest first)
      const updated = [...prev, newConnection];
      if (updated.length > config.maxConnections) {
        return updated.slice(-config.maxConnections);
      }
      return updated;
    });

    logger.debug(`Added action connection: ${event.sourceAgentId} → ${event.targetNodeId} (${actionTypeString})`);
  }, [config.baseDuration, config.maxConnections, getPosition]);

  /**
   * Add multiple actions at once (batch from WebSocket)
   */
  const addActions = useCallback((events: AgentActionEvent[]) => {
    events.forEach(addAction);
  }, [addAction]);

  /**
   * Determine animation phase based on progress
   * Phase boundaries: spawn (0-0.2), travel (0.2-0.8), impact (0.8-1.0)
   */
  const getPhase = useCallback((progress: number): ActionConnection['phase'] => {
    if (progress < PHASE_TIMING.spawn) return 'spawn';
    if (progress < PHASE_TIMING.spawn + PHASE_TIMING.travel) return 'travel';
    // Combined impact + fade phase (0.8 - 1.0)
    return 'impact';
  }, []);

  /**
   * Update animation state for all connections.
   * Called per-frame from animation loop or externally.
   */
  const updateConnections = useCallback(() => {
    const now = performance.now();
    const cleanupThreshold = config.cleanupDuration;

    setConnections(prev => {
      const updated: ActionConnection[] = [];

      for (const conn of prev) {
        const elapsed = now - conn.startTime;
        const progress = Math.min(elapsed / conn.duration, 1);

        // Auto-cleanup connections after duration (default 500ms)
        if (elapsed >= cleanupThreshold) {
          continue;
        }

        // Update positions from current Maps
        const sourcePosition = getPosition(conn.sourceAgentId, true);
        const targetPosition = getPosition(conn.targetNodeId, false);

        updated.push({
          ...conn,
          progress,
          phase: getPhase(progress),
          sourcePosition,
          targetPosition,
        });
      }

      return updated;
    });

    lastUpdateRef.current = now;
  }, [config.cleanupDuration, getPosition, getPhase]);

  /**
   * Animation loop - only runs when there are active connections
   * Idle detection prevents CPU waste when no animations are active
   */
  useEffect(() => {
    let running = true;
    let isAnimating = false;

    const startAnimation = () => {
      if (isAnimating || !running) return;
      isAnimating = true;
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    const stopAnimation = () => {
      isAnimating = false;
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };

    const animate = () => {
      if (!running) return;

      updateConnections();

      // Check if we still have connections to animate
      // If not, stop the animation loop to save CPU
      setConnections(prev => {
        if (prev.length === 0) {
          stopAnimation();
        } else {
          animationFrameRef.current = requestAnimationFrame(animate);
        }
        return prev;
      });
    };

    // Only start animation when connections exist
    if (connections.length > 0 && !isAnimating) {
      startAnimation();
    }

    return () => {
      running = false;
      stopAnimation();
    };
  }, [updateConnections, connections.length]);

  /**
   * Auto-subscribe to WebSocket agent-action events
   */
  useEffect(() => {
    if (!config.autoSubscribe && !isMapInterface) return;

    const unsubscribe = wsOn('agent-action', (data: unknown) => {
      const actions = data as AgentActionEvent[];
      if (Array.isArray(actions) && actions.length > 0) {
        addActions(actions);
        logger.debug(`Received ${actions.length} agent actions via WebSocket`);
      }
    });

    return unsubscribe;
  }, [config.autoSubscribe, isMapInterface, wsOn, addActions]);

  /**
   * Clear all active connections
   */
  const clearAll = useCallback(() => {
    setConnections([]);
  }, []);

  /**
   * Update positions for existing connections (legacy API)
   */
  const updatePositions = useCallback(() => {
    setConnections(prev => prev.map(conn => ({
      ...conn,
      sourcePosition: getPosition(conn.sourceAgentId, true),
      targetPosition: getPosition(conn.targetNodeId, false),
    })));
  }, [getPosition]);

  /**
   * Get connections by action type (for filtering)
   */
  const getConnectionsByType = useCallback((type: AgentActionType) => {
    return connections.filter(c => c._actionTypeEnum === type);
  }, [connections]);

  /**
   * Get active connection count
   */
  const activeCount = connections.length;

  // Return appropriate interface
  if (isMapInterface) {
    return {
      connections,
      updateConnections,
    };
  }

  return {
    connections,
    addAction,
    addActions,
    clearAll,
    updatePositions,
    getConnectionsByType,
    activeCount,
    updateConnections,
  };
}

/**
 * Convert ActionConnection to legacy format for backward compatibility
 */
export function toLegacyConnection(conn: ActionConnection): LegacyActionConnection {
  return {
    id: conn.id,
    sourceAgentId: conn.sourceAgentId,
    targetNodeId: conn.targetNodeId,
    actionType: conn._actionTypeEnum,
    color: conn.color,
    progress: conn.progress,
    phase: conn.phase,
    startTime: conn.startTime,
    duration: conn.duration,
    sourcePosition: {
      x: conn.sourcePosition.x,
      y: conn.sourcePosition.y,
      z: conn.sourcePosition.z,
    },
    targetPosition: {
      x: conn.targetPosition.x,
      y: conn.targetPosition.y,
      z: conn.targetPosition.z,
    },
  };
}

export default useActionConnections;
