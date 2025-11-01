
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
  
  agents: BotsAgent[];
  edges: BotsEdge[];  
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
  
  
  const pollingData = useAgentPolling({
    enabled: true,
    config: {
      activePollingInterval: 3000,  
      idlePollingInterval: 15000,   
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
    edges: []  
  });

  const updateBotsData = (data: BotsData) => {
    setBotsData(data);
  };

  const updateFromFullUpdate = (update: BotsFullUpdateMessage) => {
    setBotsData(prev => ({
      ...prev!,
      agents: update.agents || [],
      nodeCount: update.agents?.length || 0,
      edgeCount: 0, 
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

  
  const updateFromGraphData = (data: any) => {
    
    if (!data) {
      logger.warn('updateFromGraphData received undefined data');
      return;
    }
    
    
    const transformedAgents = (data.nodes || []).map((node: any) => {
      
      const agentType = node.metadata?.agent_type || node.type || node.node_type || node.nodeType;

      if (!agentType) {
        console.error('Missing agent type for node:', {
          nodeId: node.id,
          metadataId: node.metadataId || node.metadata_id,
          metadata: node.metadata,
          type: node.type,
          node_type: node.node_type,
          nodeType: node.nodeType
        });
      }

      
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
        
        id: node.metadataId || node.metadata_id || String(node.id),
        name: node.label || node.metadata?.name || `Agent-${node.id}`,
        type: agentType,
        status: node.metadata?.status || 'active', 
        position,
        velocity,
        force: { x: 0, y: 0, z: 0 },
        
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
        connections: [],
      };
    });

    
    
    const nodeIdToAgentId = new Map();
    data.nodes?.forEach((node: any) => {
      nodeIdToAgentId.set(node.id, node.metadataId || node.metadata_id || String(node.id));
    });

    const transformedEdges = (data.edges || []).map((edge: any) => ({
      id: edge.id,
      source: nodeIdToAgentId.get(edge.source) || String(edge.source),
      target: nodeIdToAgentId.get(edge.target) || String(edge.target),
      dataVolume: edge.weight * 1000,  
      messageCount: Math.floor(edge.weight * 10),  
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

  
  const updateFromBinaryPositions = (binaryData: ArrayBuffer) => {
    try {
      
      const nodeUpdates = parseBinaryNodeData(binaryData);

      
      const agentUpdates = nodeUpdates.filter(node => isAgentNode(node.nodeId));

      if (agentUpdates.length === 0) {
        return; 
      }

      logger.debug(`Processing ${agentUpdates.length} agent position updates from binary data`);

      setBotsData(prev => {
        if (!prev) return prev;

        
        const updatedAgents = prev.agents.map(agent => {
          
          const positionUpdate = agentUpdates.find(update => {
            const actualNodeId = getActualNodeId(update.nodeId);
            
            return String(actualNodeId) === agent.id || actualNodeId.toString() === agent.id;
          });

          if (positionUpdate) {
            
            return {
              ...agent,
              position: positionUpdate.position,
              velocity: positionUpdate.velocity,
              
              ssspDistance: positionUpdate.ssspDistance,
              ssspParent: positionUpdate.ssspParent,
              
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

  
  
  useEffect(() => {
    
    const unsubscribe = botsWebSocketIntegration.on('bots-binary-position-update', (binaryData: ArrayBuffer) => {
      updateFromBinaryPositions(binaryData);
    });

    
    

    return () => {
      unsubscribe();
    };
  }, []);

  
  const contextValue = useMemo(() => ({
    botsData,
    updateBotsData,
    updateFromFullUpdate,
    
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