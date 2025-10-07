import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { agentPollingService, AgentSwarmData, PollingConfig } from '../services/AgentPollingService';
import type { BotsAgent, BotsEdge } from '../types/BotsTypes';
import { createLogger } from '../../../utils/loggerConfig';
import { agentTelemetry } from '../../../telemetry/AgentTelemetry';

const logger = createLogger('useAgentPolling');

export interface UseAgentPollingOptions {
  enabled?: boolean;
  config?: Partial<PollingConfig>;
  onError?: (error: Error) => void;
}

export interface AgentPollingState {
  agents: BotsAgent[];
  edges: BotsEdge[];
  metadata?: {
    totalAgents: number;
    activeAgents: number;
    totalTasks: number;
    completedTasks: number;
    avgSuccessRate: number;
    totalTokens: number;
  };
  isPolling: boolean;
  activityLevel: 'active' | 'idle';
  lastUpdate: number;
  error: Error | null;
}

/**
 * React hook for efficient agent swarm polling with state management
 */
export function useAgentPolling(options: UseAgentPollingOptions = {}) {
  const { enabled = true, config, onError } = options;
  
  // Use refs to prevent unnecessary re-renders
  const agentsMapRef = useRef<Map<string, BotsAgent>>(new Map());
  const edgesMapRef = useRef<Map<string, BotsEdge>>(new Map());
  const lastUpdateRef = useRef<number>(0);
  
  const [state, setState] = useState<AgentPollingState>({
    agents: [],
    edges: [],
    metadata: undefined,
    isPolling: false,
    activityLevel: 'idle',
    lastUpdate: 0,
    error: null
  });

  // Transform raw agent data to frontend format
  const transformAgentData = useCallback((data: AgentSwarmData): {
    agents: BotsAgent[],
    edges: BotsEdge[]
  } => {
    // Create node ID to agent ID mapping
    const nodeIdToAgentId = new Map<number, string>();
    data.nodes?.forEach(node => {
      nodeIdToAgentId.set(node.id, node.metadataId || String(node.id));
    });

    // Transform nodes to agents
    const agents = data.nodes?.map(node => {
      const agentType = node.metadata?.agent_type || node.type || 'specialist';

      // Handle both nested position object and flat x/y/z coordinates
      const position = node.data?.position || {
        x: node.data?.x || 0,
        y: node.data?.y || 0,
        z: node.data?.z || 0
      };

      const velocity = node.data?.velocity || {
        x: node.data?.vx || 0,
        y: node.data?.vy || 0,
        z: node.data?.vz || 0
      };

      return {
        id: node.metadataId || String(node.id),
        name: node.label || `Agent-${node.id}`,
        type: agentType as BotsAgent['type'],
        status: (node.metadata?.status || 'active') as BotsAgent['status'],
        position,
        velocity,
        cpuUsage: parseFloat(node.metadata?.cpu_usage || '0'),
        memoryUsage: parseFloat(node.metadata?.memory_usage || '0'),
        health: parseFloat(node.metadata?.health || '100'),
        workload: parseFloat(node.metadata?.workload || '0'),
        tokens: parseInt(node.metadata?.tokens || '0'),
        createdAt: node.metadata?.created_at || new Date().toISOString(),
        age: parseInt(node.metadata?.age || '0'),
        swarmId: node.metadata?.swarm_id,
        parentQueenId: node.metadata?.parent_queen_id,
        capabilities: node.metadata?.capabilities ?
          node.metadata.capabilities.split(',').map(cap => cap.trim()).filter(cap => cap) :
          undefined,
      } as BotsAgent;
    }) || [];

    // Transform edges
    const edges = data.edges?.map(edge => ({
      id: edge.id,
      source: nodeIdToAgentId.get(edge.source) || String(edge.source),
      target: nodeIdToAgentId.get(edge.target) || String(edge.target),
      dataVolume: edge.weight * 1000,
      messageCount: Math.floor(edge.weight * 10),
      lastMessageTime: Date.now()
    } as BotsEdge)) || [];

    return { agents, edges };
  }, []);

  // Store the update function in a ref to avoid re-renders
  const updateStateRef = useRef<(data: AgentSwarmData) => void>();
  updateStateRef.current = (data: AgentSwarmData) => {
    const { agents, edges } = transformAgentData(data);
    const now = Date.now();

    // Update agent positions efficiently
    let hasAgentChanges = false;
    agents.forEach(agent => {
      const existing = agentsMapRef.current.get(agent.id);
      if (!existing ||
          existing.position.x !== agent.position.x ||
          existing.position.y !== agent.position.y ||
          existing.position.z !== agent.position.z ||
          existing.status !== agent.status ||
          existing.health !== agent.health) {
        hasAgentChanges = true;
        agentsMapRef.current.set(agent.id, agent);
      }
    });

    // Update edges efficiently
    let hasEdgeChanges = false;
    const newEdgeIds = new Set<string>();
    edges.forEach(edge => {
      newEdgeIds.add(edge.id);
      const existing = edgesMapRef.current.get(edge.id);
      if (!existing ||
          existing.dataVolume !== edge.dataVolume ||
          existing.messageCount !== edge.messageCount) {
        hasEdgeChanges = true;
        edgesMapRef.current.set(edge.id, edge);
      }
    });

    // Remove edges that no longer exist
    edgesMapRef.current.forEach((edge, id) => {
      if (!newEdgeIds.has(id)) {
        edgesMapRef.current.delete(id);
        hasEdgeChanges = true;
      }
    });

    // Only update state if there are actual changes
    if (hasAgentChanges || hasEdgeChanges || now - lastUpdateRef.current > 5000) {
      lastUpdateRef.current = now;

      setState(prev => ({
        ...prev,
        agents: Array.from(agentsMapRef.current.values()),
        edges: Array.from(edgesMapRef.current.values()),
        metadata: data.metadata ? {
          totalAgents: data.metadata.total_agents,
          activeAgents: data.metadata.active_agents,
          totalTasks: data.metadata.total_tasks,
          completedTasks: data.metadata.completed_tasks,
          avgSuccessRate: data.metadata.avg_success_rate,
          totalTokens: data.metadata.total_tokens
        } : prev.metadata,
        lastUpdate: now,
        error: null
      }));

      // Log telemetry for significant updates
      if (hasAgentChanges) {
        agentTelemetry.logAgentAction('polling', 'update', 'agents_changed', {
          agentCount: agents.length,
          activeCount: agents.filter(a => a.status === 'active').length
        });
      }
    }
  };

  // Efficient state update with change detection
  const updateState = useCallback((data: AgentSwarmData) => {
    updateStateRef.current?.(data);
  }, []);

  // Handle polling errors - no dependencies to prevent re-renders
  const handleErrorRef = useRef<(error: Error) => void>();
  handleErrorRef.current = (error: Error) => {
    logger.error('Polling error:', error);
    setState(prev => ({ ...prev, error }));
    onError?.(error);
  };

  const handleError = useCallback((error: Error) => {
    handleErrorRef.current?.(error);
  }, []);

  // Configure polling service
  useEffect(() => {
    if (config) {
      agentPollingService.configure(config);
    }
  }, [config]);

  // Start/stop polling based on enabled state
  useEffect(() => {
    if (!enabled) {
      agentPollingService.stop();
      setState(prev => ({ ...prev, isPolling: false }));
      return;
    }

    // Subscribe to polling updates
    const unsubscribe = agentPollingService.subscribe(updateState, handleError);

    // Start polling
    agentPollingService.start();
    setState(prev => ({ ...prev, isPolling: true }));

    // Update polling status periodically
    const statusInterval = setInterval(() => {
      const status = agentPollingService.getStatus();
      setState(prev => {
        // Only update if values have changed
        if (prev.isPolling === status.isPolling && prev.activityLevel === status.activityLevel) {
          return prev;
        }
        return {
          ...prev,
          isPolling: status.isPolling,
          activityLevel: status.activityLevel
        };
      });
    }, 2000);

    return () => {
      unsubscribe();
      agentPollingService.stop();
      clearInterval(statusInterval);
    };
  }, [enabled]); // Remove updateState and handleError from dependencies

  // Memoized return value to prevent unnecessary re-renders
  const result = useMemo(() => ({
    agents: state.agents,
    edges: state.edges,
    metadata: state.metadata,
    isPolling: state.isPolling,
    activityLevel: state.activityLevel,
    lastUpdate: state.lastUpdate,
    error: state.error,
    // Utility methods
    pollNow: () => agentPollingService.pollNow(),
    getStatus: () => agentPollingService.getStatus(),
    configure: (newConfig: Partial<PollingConfig>) => agentPollingService.configure(newConfig)
  }), [state]);

  return result;
}