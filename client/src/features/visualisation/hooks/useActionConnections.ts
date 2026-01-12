/**
 * useActionConnections Hook
 *
 * Manages ephemeral action connections between agent nodes and data nodes.
 * Connections animate from agent → target with type-specific colors.
 *
 * Animation lifecycle: spawn (100ms) → travel (300ms) → impact (50ms) → fade (50ms)
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  AgentActionType,
  AgentActionEvent,
  AGENT_ACTION_COLORS,
} from '@/services/BinaryWebSocketProtocol';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('useActionConnections');

/** Single animated action connection */
export interface ActionConnection {
  id: string;
  sourceAgentId: number;
  targetNodeId: number;
  actionType: AgentActionType;
  color: string;
  /** Animation progress 0-1 */
  progress: number;
  /** Current animation phase */
  phase: 'spawn' | 'travel' | 'impact' | 'fade';
  /** When the action started (ms) */
  startTime: number;
  /** Total duration (ms) */
  duration: number;
  /** Source position in world space */
  sourcePosition?: { x: number; y: number; z: number };
  /** Target position in world space */
  targetPosition?: { x: number; y: number; z: number };
}

export interface UseActionConnectionsOptions {
  /** Maximum concurrent connections to display */
  maxConnections?: number;
  /** Base animation duration in ms */
  baseDuration?: number;
  /** Enable VR-optimized rendering (simplified geometry) */
  vrMode?: boolean;
  /** Position resolver for node IDs */
  getNodePosition?: (nodeId: number) => { x: number; y: number; z: number } | null;
}

const DEFAULT_OPTIONS: Required<UseActionConnectionsOptions> = {
  maxConnections: 50,
  baseDuration: 500,
  vrMode: false,
  getNodePosition: () => null,
};

/** Animation phase timing (as fraction of total duration) */
const PHASE_TIMING = {
  spawn: 0.2,    // 0.0 - 0.2 (100ms of 500ms)
  travel: 0.6,   // 0.2 - 0.8 (300ms of 500ms)
  impact: 0.1,   // 0.8 - 0.9 (50ms of 500ms)
  fade: 0.1,     // 0.9 - 1.0 (50ms of 500ms)
};

export const useActionConnections = (options: UseActionConnectionsOptions = {}) => {
  const config = { ...DEFAULT_OPTIONS, ...options };
  const [connections, setConnections] = useState<ActionConnection[]>([]);
  const connectionIdCounter = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const lastUpdateRef = useRef<number>(performance.now());

  /**
   * Add a new action connection from an AgentActionEvent
   */
  const addAction = useCallback((event: AgentActionEvent) => {
    const id = `action-${connectionIdCounter.current++}`;
    const color = AGENT_ACTION_COLORS[event.actionType] || '#ffffff';
    const duration = event.durationMs > 0 ? event.durationMs : config.baseDuration;

    const sourcePosition = config.getNodePosition(event.sourceAgentId);
    const targetPosition = config.getNodePosition(event.targetNodeId);

    const newConnection: ActionConnection = {
      id,
      sourceAgentId: event.sourceAgentId,
      targetNodeId: event.targetNodeId,
      actionType: event.actionType,
      color,
      progress: 0,
      phase: 'spawn',
      startTime: performance.now(),
      duration,
      sourcePosition: sourcePosition || undefined,
      targetPosition: targetPosition || undefined,
    };

    setConnections(prev => {
      // Enforce max connections limit (remove oldest first)
      const updated = [...prev, newConnection];
      if (updated.length > config.maxConnections) {
        return updated.slice(-config.maxConnections);
      }
      return updated;
    });

    logger.debug(`Added action connection: ${event.sourceAgentId} → ${event.targetNodeId} (${AgentActionType[event.actionType]})`);
  }, [config.baseDuration, config.maxConnections, config.getNodePosition]);

  /**
   * Add multiple actions at once (batch from WebSocket)
   */
  const addActions = useCallback((events: AgentActionEvent[]) => {
    events.forEach(addAction);
  }, [addAction]);

  /**
   * Determine animation phase based on progress
   */
  const getPhase = (progress: number): ActionConnection['phase'] => {
    if (progress < PHASE_TIMING.spawn) return 'spawn';
    if (progress < PHASE_TIMING.spawn + PHASE_TIMING.travel) return 'travel';
    if (progress < PHASE_TIMING.spawn + PHASE_TIMING.travel + PHASE_TIMING.impact) return 'impact';
    return 'fade';
  };

  /**
   * Update animation state for all connections
   */
  const updateAnimations = useCallback(() => {
    const now = performance.now();

    setConnections(prev => {
      const updated: ActionConnection[] = [];

      for (const conn of prev) {
        const elapsed = now - conn.startTime;
        const progress = Math.min(elapsed / conn.duration, 1);

        if (progress >= 1) {
          // Animation complete, remove connection
          continue;
        }

        updated.push({
          ...conn,
          progress,
          phase: getPhase(progress),
        });
      }

      return updated;
    });

    lastUpdateRef.current = now;
  }, []);

  /**
   * Animation loop
   */
  useEffect(() => {
    let running = true;

    const animate = () => {
      if (!running) return;

      updateAnimations();
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      running = false;
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [updateAnimations]);

  /**
   * Clear all active connections
   */
  const clearAll = useCallback(() => {
    setConnections([]);
  }, []);

  /**
   * Update positions for existing connections (when nodes move)
   */
  const updatePositions = useCallback(() => {
    setConnections(prev => prev.map(conn => ({
      ...conn,
      sourcePosition: config.getNodePosition(conn.sourceAgentId) || conn.sourcePosition,
      targetPosition: config.getNodePosition(conn.targetNodeId) || conn.targetPosition,
    })));
  }, [config.getNodePosition]);

  /**
   * Get connections by action type (for filtering)
   */
  const getConnectionsByType = useCallback((type: AgentActionType) => {
    return connections.filter(c => c.actionType === type);
  }, [connections]);

  /**
   * Get active connection count
   */
  const activeCount = connections.length;

  return {
    connections,
    addAction,
    addActions,
    clearAll,
    updatePositions,
    getConnectionsByType,
    activeCount,
  };
};

export default useActionConnections;
