import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import type { BotsAgent, BotsEdge, BotsFullUpdateMessage } from '../types/BotsTypes';
import { botsWebSocketIntegration } from '../services/BotsWebSocketIntegration';
import { parseBinaryNodeData, isAgentNode, getActualNodeId } from '../../../types/binaryProtocol';
import { useAgentPolling } from '../hooks/useAgentPolling';
import { agentPollingService } from '../services/AgentPollingService';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('BotsDataContext');

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
  pollingStatus?: {
    isPolling: boolean;
    activityLevel: 'active' | 'idle';
    lastUpdate: number;
    error: Error | null;
  };
  pollNow?: () => Promise<void>;
  configurePolling?: (config: any) => void;
}

const BotsDataContext = createContext<BotsDataContextType | undefined>(undefined);

export const BotsDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Use the new polling hook for REST-based updates
  const pollingData = useAgentPolling({
    enabled: true,
    config: {
      activePollingInterval: 1000,  // 1s for active
      idlePollingInterval: 5000,    // 5s for idle
      enableSmartPolling: true
    },
    onError: (error) => {
      logger.error('Polling error:', error);
    }
  });

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
    // Safety check for undefined data
    if (!data) {
      console.warn('updateFromGraphData received undefined data');
      return;
    }
    
    // Transform backend nodes to BotsAgent format
    const transformedAgents = (data.nodes || []).map((node: any) => {
      // Read agent type from correct field - check multiple possible locations
      const agentType = node.metadata?.agent_type || node.node_type || node.nodeType;
      
      if (!agentType) {
        console.error('Missing agent type for node:', {
          nodeId: node.id,
          metadataId: node.metadata_id,
          metadata: node.metadata,
          node_type: node.node_type,
          nodeType: node.nodeType
        });
      }
      
      return {
        // Use metadata_id as the agent ID (original string ID)
        id: node.metadata_id || String(node.id),
        name: node.label || node.metadata?.name || `Agent-${node.id}`,
        type: agentType,
        status: node.metadata?.status || 'active', // 'active' is the default status in backend
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
        capabilities: node.metadata?.capabilities ? 
          node.metadata.capabilities.split(',').map(cap => cap.trim()).filter(cap => cap) : 
          undefined,
        connections: [],
      };
    });

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

  // Handler for binary position updates
  const updateFromBinaryPositions = (binaryData: ArrayBuffer) => {
    try {
      // Parse binary data to get agent positions
      const nodeUpdates = parseBinaryNodeData(binaryData);

      // Filter for agent nodes only
      const agentUpdates = nodeUpdates.filter(node => isAgentNode(node.nodeId));

      if (agentUpdates.length === 0) {
        return; // No agent data to process
      }

      logger.debug(`Processing ${agentUpdates.length} agent position updates from binary data`);

      setBotsData(prev => {
        if (!prev) return prev;

        // Create updated agents array with new positions
        const updatedAgents = prev.agents.map(agent => {
          // Find matching position update by node ID
          const positionUpdate = agentUpdates.find(update => {
            const actualNodeId = getActualNodeId(update.nodeId);
            // Try to match by ID conversion (numeric to string)
            return String(actualNodeId) === agent.id || actualNodeId.toString() === agent.id;
          });

          if (positionUpdate) {
            // Merge binary position/velocity data with existing agent metadata
            return {
              ...agent,
              position: positionUpdate.position,
              velocity: positionUpdate.velocity,
              // Store additional SSSP data for path visualization
              ssspDistance: positionUpdate.ssspDistance,
              ssspParent: positionUpdate.ssspParent,
              // Update timestamp to indicate fresh data
              lastPositionUpdate: Date.now()
            };
          }

          return agent;
        });

        return {
          ...prev,
          agents: updatedAgents,
          lastUpdate: new Date().toISOString()
        };
      });
    } catch (error) {
      logger.error('Error processing binary position updates:', error);
    }
  };

  // Update botsData when polling data changes
  useEffect(() => {
    if (pollingData.agents.length > 0 || pollingData.edges.length > 0) {
      setBotsData({
        nodeCount: pollingData.agents.length,
        edgeCount: pollingData.edges.length,
        tokenCount: pollingData.metadata?.totalTokens || 0,
        mcpConnected: pollingData.isPolling,
        dataSource: 'live',
        agents: pollingData.agents,
        edges: pollingData.edges,
        multiAgentMetrics: pollingData.metadata,
        lastUpdate: new Date(pollingData.lastUpdate).toISOString()
      });
    }
  }, [pollingData]);

  // Subscribe to WebSocket updates for real-time position data
  useEffect(() => {
    // Subscribe to binary position updates for real-time agent movement
    const unsubscribe = botsWebSocketIntegration.on('bots-binary-position-update', (binaryData: ArrayBuffer) => {
      updateFromBinaryPositions(binaryData);
    });

    // Also subscribe to the REST polling service for metadata updates
    const unsubscribePolling = agentPollingService.subscribe((data) => {
      updateFromGraphData(data);
    });

    return () => {
      unsubscribe();
      unsubscribePolling();
    };
  }, []);

  // Provide both polling data and traditional botsData
  const contextValue = useMemo(() => ({
    botsData,
    updateBotsData,
    updateFromFullUpdate,
    // Additional polling controls
    pollingStatus: {
      isPolling: pollingData.isPolling,
      activityLevel: pollingData.activityLevel,
      lastUpdate: pollingData.lastUpdate,
      error: pollingData.error
    },
    pollNow: pollingData.pollNow,
    configurePolling: pollingData.configure
  }), [botsData, pollingData]);

  return (
    <BotsDataContext.Provider value={contextValue}>
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