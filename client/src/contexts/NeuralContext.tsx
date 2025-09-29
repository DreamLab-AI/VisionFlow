/**
 * Neural Context - Global state management for neural capabilities
 */

import React, { createContext, useContext, useReducer, useCallback, useEffect } from 'react';
import { NeuralAgent, SwarmTopology, NeuralMemory, ConsensusState, ResourceMetrics, NeuralDashboardState } from '../types/neural';

// Extended types for the unified interface
interface WorkflowStep {
  id: string;
  name: string;
  description?: string;
  agentType: string;
  dependencies: string[];
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'pending' | 'running' | 'completed' | 'failed';
  estimatedDuration: number;
}

interface Workflow {
  id: string;
  name: string;
  description?: string;
  steps: WorkflowStep[];
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  createdAt: Date;
  assignedAgents: string[];
}

interface ChatMessage {
  id: string;
  type: 'user' | 'agent' | 'system';
  content: string;
  timestamp: Date;
  agentId?: string;
  agentType?: string;
  cognitivePattern?: string;
  metadata?: Record<string, any>;
}

interface CognitiveInsight {
  id: string;
  type: 'pattern' | 'optimization' | 'anomaly' | 'prediction';
  title: string;
  description: string;
  confidence: number;
  impact: 'low' | 'medium' | 'high';
  actionable: boolean;
  timestamp: Date;
  source: string;
  data?: any;
}

// Extended neural context state
interface ExtendedNeuralState extends NeuralDashboardState {
  workflows: Workflow[];
  chatMessages: ChatMessage[];
  insights: CognitiveInsight[];
  selectedAgent: string | null;
  selectedWorkflow: string | null;
  commandHistory: string[];
  notifications: Notification[];
  settings: {
    autoRefresh: boolean;
    refreshInterval: number;
    soundEnabled: boolean;
    darkMode: boolean;
  };
}

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  actions?: Array<{ label: string; action: () => void }>;
}

// Action types
type NeuralAction =
  | { type: 'SET_AGENTS'; payload: NeuralAgent[] }
  | { type: 'UPDATE_AGENT'; payload: NeuralAgent }
  | { type: 'ADD_AGENT'; payload: NeuralAgent }
  | { type: 'REMOVE_AGENT'; payload: string }
  | { type: 'SET_TOPOLOGY'; payload: SwarmTopology }
  | { type: 'UPDATE_TOPOLOGY'; payload: Partial<SwarmTopology> }
  | { type: 'SET_MEMORY'; payload: NeuralMemory[] }
  | { type: 'ADD_MEMORY'; payload: NeuralMemory }
  | { type: 'UPDATE_CONSENSUS'; payload: ConsensusState }
  | { type: 'ADD_METRICS'; payload: ResourceMetrics }
  | { type: 'SET_CONNECTION_STATUS'; payload: boolean }
  | { type: 'SET_WORKFLOWS'; payload: Workflow[] }
  | { type: 'ADD_WORKFLOW'; payload: Workflow }
  | { type: 'UPDATE_WORKFLOW'; payload: { id: string; updates: Partial<Workflow> } }
  | { type: 'ADD_CHAT_MESSAGE'; payload: ChatMessage }
  | { type: 'CLEAR_CHAT'; payload: void }
  | { type: 'SET_INSIGHTS'; payload: CognitiveInsight[] }
  | { type: 'ADD_INSIGHT'; payload: CognitiveInsight }
  | { type: 'SELECT_AGENT'; payload: string | null }
  | { type: 'SELECT_WORKFLOW'; payload: string | null }
  | { type: 'ADD_COMMAND'; payload: string }
  | { type: 'ADD_NOTIFICATION'; payload: Notification }
  | { type: 'MARK_NOTIFICATION_READ'; payload: string }
  | { type: 'CLEAR_NOTIFICATIONS'; payload: void }
  | { type: 'UPDATE_SETTINGS'; payload: Partial<ExtendedNeuralState['settings']> };

// Context interface
interface NeuralContextValue extends ExtendedNeuralState {
  // Agent management
  spawnAgent: (config: Partial<NeuralAgent>) => Promise<void>;
  updateAgent: (id: string, updates: Partial<NeuralAgent>) => Promise<void>;
  removeAgent: (id: string) => Promise<void>;
  selectAgent: (id: string | null) => void;

  // Swarm management
  initializeSwarm: (topology: SwarmTopology['type'], maxAgents?: number) => Promise<void>;
  scaleSwarm: (targetAgents: number) => Promise<void>;
  destroySwarm: () => Promise<void>;

  // Workflow management
  createWorkflow: (workflow: Omit<Workflow, 'id' | 'createdAt'>) => Promise<void>;
  executeWorkflow: (workflowId: string) => Promise<void>;
  updateWorkflow: (id: string, updates: Partial<Workflow>) => Promise<void>;
  selectWorkflow: (id: string | null) => void;

  // Communication
  sendMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => Promise<void>;
  clearChat: () => void;

  // Insights and analytics
  refreshInsights: () => Promise<void>;
  dismissInsight: (id: string) => Promise<void>;

  // Utilities
  executeCommand: (command: string, args?: any[]) => Promise<any>;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  updateSettings: (settings: Partial<ExtendedNeuralState['settings']>) => void;
}

// Initial state
const initialState: ExtendedNeuralState = {
  agents: [],
  topology: {
    type: 'mesh',
    nodes: [],
    connections: [],
    consensus: 'proof-of-learning'
  },
  memory: [],
  consensus: {
    mechanism: 'proof-of-learning',
    round: 0,
    participants: [],
    proposals: [],
    decisions: [],
    health: 100,
    latency: 0
  },
  metrics: [],
  workflows: [],
  chatMessages: [],
  insights: [],
  isConnected: false,
  lastUpdate: new Date(),
  selectedAgent: null,
  selectedWorkflow: null,
  commandHistory: [],
  notifications: [],
  settings: {
    autoRefresh: true,
    refreshInterval: 5000,
    soundEnabled: true,
    darkMode: true
  }
};

// Reducer
const neuralReducer = (state: ExtendedNeuralState, action: NeuralAction): ExtendedNeuralState => {
  switch (action.type) {
    case 'SET_AGENTS':
      return { ...state, agents: action.payload, lastUpdate: new Date() };

    case 'UPDATE_AGENT':
      return {
        ...state,
        agents: state.agents.map(agent =>
          agent.id === action.payload.id ? action.payload : agent
        ),
        lastUpdate: new Date()
      };

    case 'ADD_AGENT':
      return {
        ...state,
        agents: [...state.agents, action.payload],
        lastUpdate: new Date()
      };

    case 'REMOVE_AGENT':
      return {
        ...state,
        agents: state.agents.filter(agent => agent.id !== action.payload),
        selectedAgent: state.selectedAgent === action.payload ? null : state.selectedAgent,
        lastUpdate: new Date()
      };

    case 'SET_TOPOLOGY':
      return { ...state, topology: action.payload, lastUpdate: new Date() };

    case 'UPDATE_TOPOLOGY':
      return {
        ...state,
        topology: { ...state.topology, ...action.payload },
        lastUpdate: new Date()
      };

    case 'SET_MEMORY':
      return { ...state, memory: action.payload, lastUpdate: new Date() };

    case 'ADD_MEMORY':
      return {
        ...state,
        memory: [...state.memory, action.payload],
        lastUpdate: new Date()
      };

    case 'UPDATE_CONSENSUS':
      return { ...state, consensus: action.payload, lastUpdate: new Date() };

    case 'ADD_METRICS':
      return {
        ...state,
        metrics: [...state.metrics.slice(-99), action.payload], // Keep last 100 metrics
        lastUpdate: new Date()
      };

    case 'SET_CONNECTION_STATUS':
      return { ...state, isConnected: action.payload };

    case 'SET_WORKFLOWS':
      return { ...state, workflows: action.payload };

    case 'ADD_WORKFLOW':
      return { ...state, workflows: [...state.workflows, action.payload] };

    case 'UPDATE_WORKFLOW':
      return {
        ...state,
        workflows: state.workflows.map(workflow =>
          workflow.id === action.payload.id
            ? { ...workflow, ...action.payload.updates }
            : workflow
        )
      };

    case 'ADD_CHAT_MESSAGE':
      return {
        ...state,
        chatMessages: [...state.chatMessages, action.payload]
      };

    case 'CLEAR_CHAT':
      return { ...state, chatMessages: [] };

    case 'SET_INSIGHTS':
      return { ...state, insights: action.payload };

    case 'ADD_INSIGHT':
      return { ...state, insights: [...state.insights, action.payload] };

    case 'SELECT_AGENT':
      return { ...state, selectedAgent: action.payload };

    case 'SELECT_WORKFLOW':
      return { ...state, selectedWorkflow: action.payload };

    case 'ADD_COMMAND':
      return {
        ...state,
        commandHistory: [...state.commandHistory.slice(-49), action.payload] // Keep last 50 commands
      };

    case 'ADD_NOTIFICATION':
      return {
        ...state,
        notifications: [...state.notifications, {
          ...action.payload,
          id: `notif-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          timestamp: new Date(),
          read: false
        }]
      };

    case 'MARK_NOTIFICATION_READ':
      return {
        ...state,
        notifications: state.notifications.map(notif =>
          notif.id === action.payload ? { ...notif, read: true } : notif
        )
      };

    case 'CLEAR_NOTIFICATIONS':
      return { ...state, notifications: [] };

    case 'UPDATE_SETTINGS':
      return {
        ...state,
        settings: { ...state.settings, ...action.payload }
      };

    default:
      return state;
  }
};

// Context creation
const NeuralContext = createContext<NeuralContextValue | null>(null);

// Provider component
export const NeuralProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(neuralReducer, initialState);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(process.env.REACT_APP_WS_URL || 'ws://localhost:9500');

        ws.onopen = () => {
          dispatch({ type: 'SET_CONNECTION_STATUS', payload: true });
        };

        ws.onclose = () => {
          dispatch({ type: 'SET_CONNECTION_STATUS', payload: false });
          // Attempt reconnection after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        return ws;
      } catch (error) {
        console.error('WebSocket connection failed:', error);
        dispatch({ type: 'SET_CONNECTION_STATUS', payload: false });
        return null;
      }
    };

    const ws = connectWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'agent_update':
        dispatch({ type: 'UPDATE_AGENT', payload: message.payload });
        break;
      case 'swarm_topology':
        dispatch({ type: 'SET_TOPOLOGY', payload: message.payload });
        break;
      case 'memory_sync':
        dispatch({ type: 'ADD_MEMORY', payload: message.payload });
        break;
      case 'consensus_update':
        dispatch({ type: 'UPDATE_CONSENSUS', payload: message.payload });
        break;
      case 'metrics_update':
        dispatch({ type: 'ADD_METRICS', payload: message.payload });
        break;
      default:
        console.log('Unhandled WebSocket message:', message);
    }
  };

  // Agent management
  const spawnAgent = useCallback(async (config: Partial<NeuralAgent>) => {
    try {
      const response = await fetch('/api/neural/agents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (!response.ok) throw new Error('Failed to spawn agent');

      const agent = await response.json();
      dispatch({ type: 'ADD_AGENT', payload: agent });
    } catch (error) {
      console.error('Failed to spawn agent:', error);
      throw error;
    }
  }, []);

  const updateAgent = useCallback(async (id: string, updates: Partial<NeuralAgent>) => {
    try {
      const response = await fetch(`/api/neural/agents/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });

      if (!response.ok) throw new Error('Failed to update agent');

      const agent = await response.json();
      dispatch({ type: 'UPDATE_AGENT', payload: agent });
    } catch (error) {
      console.error('Failed to update agent:', error);
      throw error;
    }
  }, []);

  const removeAgent = useCallback(async (id: string) => {
    try {
      const response = await fetch(`/api/neural/agents/${id}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to remove agent');

      dispatch({ type: 'REMOVE_AGENT', payload: id });
    } catch (error) {
      console.error('Failed to remove agent:', error);
      throw error;
    }
  }, []);

  // Swarm management
  const initializeSwarm = useCallback(async (topology: SwarmTopology['type'], maxAgents = 8) => {
    try {
      const response = await fetch('/api/neural/swarm/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topology, maxAgents })
      });

      if (!response.ok) throw new Error('Failed to initialize swarm');

      const swarmData = await response.json();
      dispatch({ type: 'SET_TOPOLOGY', payload: swarmData.topology });
      dispatch({ type: 'SET_AGENTS', payload: swarmData.agents });
    } catch (error) {
      console.error('Failed to initialize swarm:', error);
      throw error;
    }
  }, []);

  // Workflow management
  const createWorkflow = useCallback(async (workflow: Omit<Workflow, 'id' | 'createdAt'>) => {
    try {
      const response = await fetch('/api/neural/workflows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...workflow,
          createdAt: new Date().toISOString()
        })
      });

      if (!response.ok) throw new Error('Failed to create workflow');

      const newWorkflow = await response.json();
      dispatch({ type: 'ADD_WORKFLOW', payload: newWorkflow });
    } catch (error) {
      console.error('Failed to create workflow:', error);
      throw error;
    }
  }, []);

  const executeWorkflow = useCallback(async (workflowId: string) => {
    try {
      const response = await fetch(`/api/neural/workflows/${workflowId}/execute`, {
        method: 'POST'
      });

      if (!response.ok) throw new Error('Failed to execute workflow');

      dispatch({ type: 'UPDATE_WORKFLOW', payload: {
        id: workflowId,
        updates: { status: 'running' }
      }});
    } catch (error) {
      console.error('Failed to execute workflow:', error);
      throw error;
    }
  }, []);

  // Communication
  const sendMessage = useCallback(async (message: Omit<ChatMessage, 'id' | 'timestamp'>) => {
    const chatMessage: ChatMessage = {
      ...message,
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date()
    };

    dispatch({ type: 'ADD_CHAT_MESSAGE', payload: chatMessage });

    try {
      await fetch('/api/neural/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chatMessage)
      });
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  }, []);

  // Command execution
  const executeCommand = useCallback(async (command: string, args?: any[]) => {
    dispatch({ type: 'ADD_COMMAND', payload: command });

    try {
      const response = await fetch('/api/neural/commands', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, args })
      });

      if (!response.ok) throw new Error('Command execution failed');

      return await response.json();
    } catch (error) {
      console.error('Command execution failed:', error);
      throw error;
    }
  }, []);

  // Utilities
  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => {
    dispatch({ type: 'ADD_NOTIFICATION', payload: notification });
  }, []);

  const contextValue: NeuralContextValue = {
    ...state,
    spawnAgent,
    updateAgent,
    removeAgent,
    selectAgent: (id) => dispatch({ type: 'SELECT_AGENT', payload: id }),
    initializeSwarm,
    scaleSwarm: async () => {}, // Placeholder
    destroySwarm: async () => {}, // Placeholder
    createWorkflow,
    executeWorkflow,
    updateWorkflow: (id, updates) => dispatch({ type: 'UPDATE_WORKFLOW', payload: { id, updates } }),
    selectWorkflow: (id) => dispatch({ type: 'SELECT_WORKFLOW', payload: id }),
    sendMessage,
    clearChat: () => dispatch({ type: 'CLEAR_CHAT', payload: undefined }),
    refreshInsights: async () => {}, // Placeholder
    dismissInsight: async () => {}, // Placeholder
    executeCommand,
    addNotification,
    updateSettings: (settings) => dispatch({ type: 'UPDATE_SETTINGS', payload: settings })
  };

  return (
    <NeuralContext.Provider value={contextValue}>
      {children}
    </NeuralContext.Provider>
  );
};

// Hook to use neural context
export const useNeural = (): NeuralContextValue => {
  const context = useContext(NeuralContext);
  if (!context) {
    throw new Error('useNeural must be used within a NeuralProvider');
  }
  return context;
};

export default NeuralContext;