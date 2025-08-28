import React, { createContext, useContext, useState, useEffect } from 'react';
import type { BotsAgent, BotsFullUpdateMessage } from '../types/BotsTypes';
import { botsWebSocketIntegration } from '../services/BotsWebSocketIntegration';

interface BotsData {
  nodeCount: number;
  edgeCount: number;
  tokenCount: number;
  mcpConnected: boolean;
  dataSource: string;
  // Enhanced fields for full agent data
  agents: BotsAgent[];
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
    dataSource: 'mock',
    agents: []
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

  // Subscribe to WebSocket updates
  useEffect(() => {
    const unsubscribe = botsWebSocketIntegration.on('bots-full-update', (update: BotsFullUpdateMessage) => {
      updateFromFullUpdate(update);
    });

    return unsubscribe;
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