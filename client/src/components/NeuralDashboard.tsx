import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Grid, Box, Paper, Typography, Tab, Tabs, Alert, Chip, IconButton } from '@mui/material';
import {
  Psychology,
  Settings,
  Visibility,
  Memory,
  HowToVote,
  MonitorHeart,
  Refresh,
  PlayArrow,
  Stop,
  AutoAwesome
} from '@mui/icons-material';
import { CognitiveAgentPanel } from './CognitiveAgentPanel';
import { SwarmVisualization } from './SwarmVisualization';
import { NeuralMemoryExplorer } from './NeuralMemoryExplorer';
import { ConsensusMonitor } from './ConsensusMonitor';
import { ResourceMetrics } from './ResourceMetrics';
import { useNeuralWebSocket } from '../hooks/useNeuralWebSocket';
import { neuralAPI } from '../services/neuralAPI';
import { NeuralDashboardState, NeuralAgent, SwarmTopology } from '../types/neural';
import toast from 'react-hot-toast';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <div hidden={value !== index} style={{ height: '100%' }}>
      {value === index && (
        <Box sx={{ p: 0, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export const NeuralDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [dashboardState, setDashboardState] = useState<NeuralDashboardState>({
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
      health: 1.0,
      latency: 0
    },
    metrics: [],
    isConnected: false,
    lastUpdate: new Date()
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const {
    isConnected,
    isConnecting,
    error: wsError,
    connect,
    disconnect,
    subscribe
  } = useNeuralWebSocket(undefined, {
    onConnect: () => {
      toast.success('Neural network connected');
      setDashboardState(prev => ({ ...prev, isConnected: true }));
    },
    onDisconnect: () => {
      toast.error('Neural network disconnected');
      setDashboardState(prev => ({ ...prev, isConnected: false }));
    },
    onError: (err) => {
      toast.error(`Connection error: ${err.message}`);
      setError(err.message);
    }
  });

  const loadInitialData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [agents, topology, memory, consensus, metrics] = await Promise.all([
        neuralAPI.getAgents().catch(() => []),
        neuralAPI.getSwarmStatus().catch(() => dashboardState.topology),
        neuralAPI.getMemories().catch(() => []),
        neuralAPI.getConsensusState().catch(() => dashboardState.consensus),
        neuralAPI.getResourceMetrics().catch(() => [])
      ]);

      setDashboardState(prev => ({
        ...prev,
        agents,
        topology,
        memory,
        consensus,
        metrics,
        lastUpdate: new Date()
      }));
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleRefreshData = () => {
    loadInitialData();
    toast.success('Dashboard data refreshed');
  };

  const handleToggleConnection = () => {
    if (isConnected) {
      disconnect();
    } else {
      connect();
    }
  };

  const initializeSwarm = async (topology: string) => {
    try {
      const swarmTopology = await neuralAPI.initializeSwarm(topology);
      setDashboardState(prev => ({ ...prev, topology: swarmTopology }));
      toast.success(`Swarm initialized with ${topology} topology`);
    } catch (err) {
      toast.error('Failed to initialize swarm');
      console.error(err);
    }
  };

  // Subscribe to real-time updates
  useEffect(() => {
    const unsubscribeAgent = subscribe('agent_update', (data: NeuralAgent) => {
      setDashboardState(prev => ({
        ...prev,
        agents: prev.agents.map(agent =>
          agent.id === data.id ? data : agent
        ),
        lastUpdate: new Date()
      }));
    });

    const unsubscribeTopology = subscribe('swarm_topology', (data: SwarmTopology) => {
      setDashboardState(prev => ({
        ...prev,
        topology: data,
        lastUpdate: new Date()
      }));
    });

    const unsubscribeMetrics = subscribe('metrics_update', (data) => {
      setDashboardState(prev => ({
        ...prev,
        metrics: [data, ...prev.metrics.slice(0, 99)], // Keep last 100 entries
        lastUpdate: new Date()
      }));
    });

    return () => {
      unsubscribeAgent();
      unsubscribeTopology();
      unsubscribeMetrics();
    };
  }, [subscribe]);

  useEffect(() => {
    loadInitialData();
  }, []);

  const tabs = [
    { label: 'Agents', icon: <Psychology />, component: CognitiveAgentPanel },
    { label: 'Swarm', icon: <Visibility />, component: SwarmVisualization },
    { label: 'Memory', icon: <Memory />, component: NeuralMemoryExplorer },
    { label: 'Consensus', icon: <HowToVote />, component: ConsensusMonitor },
    { label: 'Metrics', icon: <MonitorHeart />, component: ResourceMetrics }
  ];

  if (loading && dashboardState.agents.length === 0) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
        }}
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        >
          <AutoAwesome sx={{ fontSize: 48, color: 'white' }} />
        </motion.div>
        <Typography variant="h5" sx={{ ml: 2, color: 'white' }}>
          Initializing Neural Network...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: '#0a0a0a' }}>
      {/* Header */}
      <Paper
        elevation={2}
        sx={{
          p: 2,
          background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
          borderRadius: 0
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <motion.div
              animate={{ rotate: isConnected ? 360 : 0 }}
              transition={{ duration: 2, repeat: isConnected ? Infinity : 0, ease: 'linear' }}
            >
              <Psychology sx={{ fontSize: 32, color: '#64ffda' }} />
            </motion.div>
            <Typography variant="h4" sx={{ color: 'white', fontWeight: 'bold' }}>
              Neural Command Center
            </Typography>
            <Chip
              label={isConnected ? 'Connected' : isConnecting ? 'Connecting...' : 'Disconnected'}
              color={isConnected ? 'success' : isConnecting ? 'warning' : 'error'}
              variant="outlined"
            />
          </Box>

          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label={`${dashboardState.agents.length} Agents`}
              color="primary"
              variant="outlined"
            />
            <Chip
              label={`${dashboardState.topology.nodes.length} Nodes`}
              color="secondary"
              variant="outlined"
            />
            <IconButton
              onClick={handleRefreshData}
              sx={{ color: 'white' }}
              title="Refresh Data"
            >
              <Refresh />
            </IconButton>
            <IconButton
              onClick={handleToggleConnection}
              sx={{ color: 'white' }}
              title={isConnected ? 'Disconnect' : 'Connect'}
            >
              {isConnected ? <Stop /> : <PlayArrow />}
            </IconButton>
          </Box>
        </Box>

        {/* Status Alerts */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              style={{ marginTop: 8 }}
            >
              <Alert severity="error" onClose={() => setError(null)}>
                {error}
              </Alert>
            </motion.div>
          )}
          {wsError && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              style={{ marginTop: 8 }}
            >
              <Alert severity="warning">
                WebSocket Error: {wsError.message}
              </Alert>
            </motion.div>
          )}
        </AnimatePresence>
      </Paper>

      {/* Navigation Tabs */}
      <Paper
        elevation={1}
        sx={{
          background: 'linear-gradient(90deg, #0f3460 0%, #0e4b99 100%)',
          borderRadius: 0
        }}
      >
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          variant="fullWidth"
          sx={{
            '& .MuiTab-root': {
              color: 'rgba(255, 255, 255, 0.7)',
              minHeight: 64,
              '&.Mui-selected': {
                color: '#64ffda'
              }
            },
            '& .MuiTabs-indicator': {
              backgroundColor: '#64ffda',
              height: 3
            }
          }}
        >
          {tabs.map((tab, index) => (
            <Tab
              key={index}
              icon={tab.icon}
              label={tab.label}
              iconPosition="start"
              sx={{ fontSize: '0.9rem', fontWeight: 'medium' }}
            />
          ))}
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
        <TabPanel value={activeTab} index={0}>
          <CognitiveAgentPanel
            agents={dashboardState.agents}
            onRefresh={loadInitialData}
            isConnected={isConnected}
          />
        </TabPanel>
        <TabPanel value={activeTab} index={1}>
          <SwarmVisualization
            topology={dashboardState.topology}
            agents={dashboardState.agents}
            onInitializeSwarm={initializeSwarm}
            isConnected={isConnected}
          />
        </TabPanel>
        <TabPanel value={activeTab} index={2}>
          <NeuralMemoryExplorer
            memories={dashboardState.memory}
            onRefresh={loadInitialData}
            isConnected={isConnected}
          />
        </TabPanel>
        <TabPanel value={activeTab} index={3}>
          <ConsensusMonitor
            consensus={dashboardState.consensus}
            agents={dashboardState.agents}
            onRefresh={loadInitialData}
            isConnected={isConnected}
          />
        </TabPanel>
        <TabPanel value={activeTab} index={4}>
          <ResourceMetrics
            metrics={dashboardState.metrics}
            isConnected={isConnected}
          />
        </TabPanel>
      </Box>
    </Box>
  );
};

export default NeuralDashboard;