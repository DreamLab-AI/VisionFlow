import React, { createContext, useContext, useState, useEffect } from 'react';
import type { BotsAgent, BotsEdge, BotsFullUpdateMessage } from '../types/BotsTypes';
import { botsWebSocketIntegration } from '../services/BotsWebSocketIntegration';

interface BotsData {
  nodeCount: number;
  edgeCount: number;
  tokenCount: number;
  mcpConnected: boolean;
  dataSource: string;
  // Enhanced fields for full agent data
  agents: BotsAgent[];
  edges: BotsEdge[];  // Added edges array
  multiAgentMetrics?: {
    totalAgents: number;
    activeAgents: number;
    totalTasks: number;
    completedTasks: number;
    avgSuccessRate: number;
    totalTokens: number;
  };
  lastUpdate?: string;
}

interface BotsDataContextType {
  botsData: BotsData | null;
  updateBotsData: (data: BotsData) => void;
  updateFromFullUpdate: (update: BotsFullUpdateMessage) => void;
}

const BotsDataContext = createContext<BotsDataContextType | undefined>(undefined);

export const BotsDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [botsData, setBotsData] = useState<BotsData | null>({
    nodeCount: 0,
    edgeCount: 0,
    tokenCount: 0,
    mcpConnected: false,
    dataSource: 'live',
    agents: [],
    edges: []  // Initialize edges array
  });

  const updateBotsData = (data: BotsData) => {
    setBotsData(data);
  };

  const updateFromFullUpdate = (update: BotsFullUpdateMessage) => {
    setBotsData(prev => ({
      ...prev!,
      agents: update.agents || [],
      nodeCount: update.agents?.length || 0,
      edgeCount: 0, // Will be calculated from communication patterns
      tokenCount: update.multiAgentMetrics?.totalTokens || 0,
      mcpConnected: true,
      dataSource: 'live',
      multiAgentMetrics: update.multiAgentMetrics || {
        totalAgents: 0,
        activeAgents: 0,
        totalTasks: 0,
        completedTasks: 0,
        avgSuccessRate: 0,
        totalTokens: 0
      },
      lastUpdate: update.timestamp
    }));
  };

  // Handler for graph data with type conversion
  const updateFromGraphData = (data: any) => {
    // Transform backend nodes to BotsAgent format
    const transformedAgents = (data.nodes || []).map((node: any) => ({
      // Use metadata_id as the agent ID (original string ID)
      id: node.metadata_id || String(node.id),
      name: node.label || node.metadata?.name || `Agent-${node.id}`,
      type: node.metadata?.agent_type || 'coordinator',  // 'type' not 'agent_type'
      status: node.metadata?.status || 'active',
      position: node.data?.position || { x: 0, y: 0, z: 0 },
      velocity: node.data?.velocity || { x: 0, y: 0, z: 0 },
      force: { x: 0, y: 0, z: 0 },
      // Parse numeric values from metadata (use camelCase for frontend)
      cpuUsage: parseFloat(node.metadata?.cpu_usage || '0'),
      memoryUsage: parseFloat(node.metadata?.memory_usage || '0'),
      health: parseFloat(node.metadata?.health || '100'),
      workload: parseFloat(node.metadata?.workload || '0'),
      tokens: parseInt(node.metadata?.tokens || '0'),
      createdAt: node.metadata?.created_at || new Date().toISOString(),
      age: parseInt(node.metadata?.age || '0'),
      // Other fields from metadata
      swarmId: node.metadata?.swarm_id,
      parentQueenId: node.metadata?.parent_queen_id,
      capabilities: node.metadata?.capabilities ? JSON.parse(node.metadata.capabilities) : undefined,
      connections: [],
    }));

    // Transform backend edges (u32 IDs) to frontend format (string IDs)
    // Need to map numeric node IDs to agent string IDs
    const nodeIdToAgentId = new Map();
    data.nodes?.forEach((node: any) => {
      nodeIdToAgentId.set(node.id, node.metadata_id || String(node.id));
    });

    const transformedEdges = (data.edges || []).map((edge: any) => ({
      id: edge.id,
      source: nodeIdToAgentId.get(edge.source) || String(edge.source),
      target: nodeIdToAgentId.get(edge.target) || String(edge.target),
      dataVolume: edge.weight * 1000,  // Use weight as proxy for data volume
      messageCount: Math.floor(edge.weight * 10),  // Derive from weight
    }));
    
    setBotsData(prev => ({
      ...prev!,
      agents: transformedAgents,
      edges: transformedEdges,
      nodeCount: transformedAgents.length,
      edgeCount: transformedEdges.length,
      tokenCount: transformedAgents.reduce((sum: number, agent: any) => sum + (agent.tokens || 0), 0),
      mcpConnected: true,
      dataSource: 'live',
      lastUpdate: new Date().toISOString()
    }));
  };

  // Subscribe to WebSocket updates
  useEffect(() => {
    const unsubscribe1 = botsWebSocketIntegration.on('bots-full-update', (update: BotsFullUpdateMessage) => {
      updateFromFullUpdate(update);
    });
    
    // Subscribe to new graph update event
    const unsubscribe2 = botsWebSocketIntegration.on('bots-graph-update', updateFromGraphData);

    return () => {
      unsubscribe1();
      unsubscribe2();
    };
  }, []);

  return (
    <BotsDataContext.Provider value={{ botsData, updateBotsData, updateFromFullUpdate }}>
      {children}
    </BotsDataContext.Provider>
  );
};

export const useBotsData = () => {
  const context = useContext(BotsDataContext);
  if (!context) {
    throw new Error('useBotsData must be used within a BotsDataProvider');
  }
  return context;
};