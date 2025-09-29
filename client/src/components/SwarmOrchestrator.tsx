/**
 * Swarm Orchestrator - Advanced swarm management and visualization
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNeural } from '../contexts/NeuralContext';
import '../styles/neural-theme.css';

interface NodePosition {
  x: number;
  y: number;
}

interface SwarmVisualization {
  nodes: Array<{
    id: string;
    agentId: string;
    position: NodePosition;
    status: 'online' | 'offline' | 'syncing' | 'error';
    load: number;
    connections: string[];
    role: 'worker' | 'coordinator' | 'validator' | 'optimizer';
  }>;
  connections: Array<{
    id: string;
    sourceId: string;
    targetId: string;
    weight: number;
    latency: number;
    bandwidth: number;
    status: 'active' | 'inactive' | 'congested';
  }>;
}

const topologyLayouts = {
  mesh: (nodeCount: number) => {
    const radius = Math.min(200, 50 + nodeCount * 15);
    return Array.from({ length: nodeCount }, (_, i) => ({
      x: 300 + radius * Math.cos((i * 2 * Math.PI) / nodeCount),
      y: 300 + radius * Math.sin((i * 2 * Math.PI) / nodeCount)
    }));
  },
  hierarchical: (nodeCount: number) => {
    const levels = Math.ceil(Math.log2(nodeCount + 1));
    const positions: NodePosition[] = [];
    let nodeIndex = 0;

    for (let level = 0; level < levels && nodeIndex < nodeCount; level++) {
      const nodesAtLevel = Math.min(Math.pow(2, level), nodeCount - nodeIndex);
      const startX = 300 - (nodesAtLevel - 1) * 60;

      for (let i = 0; i < nodesAtLevel && nodeIndex < nodeCount; i++) {
        positions.push({
          x: startX + i * 120,
          y: 100 + level * 100
        });
        nodeIndex++;
      }
    }
    return positions;
  },
  ring: (nodeCount: number) => {
    const radius = Math.min(250, 80 + nodeCount * 20);
    return Array.from({ length: nodeCount }, (_, i) => ({
      x: 300 + radius * Math.cos((i * 2 * Math.PI) / nodeCount),
      y: 300 + radius * Math.sin((i * 2 * Math.PI) / nodeCount)
    }));
  },
  star: (nodeCount: number) => {
    const positions: NodePosition[] = [{ x: 300, y: 300 }]; // Center node
    const radius = Math.min(200, 100 + nodeCount * 10);

    for (let i = 1; i < nodeCount; i++) {
      positions.push({
        x: 300 + radius * Math.cos(((i - 1) * 2 * Math.PI) / (nodeCount - 1)),
        y: 300 + radius * Math.sin(((i - 1) * 2 * Math.PI) / (nodeCount - 1))
      });
    }
    return positions;
  }
};

const SwarmOrchestrator: React.FC = () => {
  const neural = useNeural();
  const [selectedTopology, setSelectedTopology] = useState<'mesh' | 'hierarchical' | 'ring' | 'star'>('mesh');
  const [maxAgents, setMaxAgents] = useState(8);
  const [strategy, setStrategy] = useState<'balanced' | 'specialized' | 'adaptive'>('balanced');
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);
  const [visualization, setVisualization] = useState<SwarmVisualization>({ nodes: [], connections: [] });
  const canvasRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // Generate visualization data from neural state
  const generateVisualization = useCallback((): SwarmVisualization => {
    const positions = topologyLayouts[neural.topology.type || selectedTopology](neural.agents.length || 3);

    const nodes = neural.topology.nodes.length > 0
      ? neural.topology.nodes.map((node, index) => ({
          ...node,
          position: positions[index] || { x: 300, y: 300 }
        }))
      : neural.agents.slice(0, maxAgents).map((agent, index) => ({
          id: `node-${agent.id}`,
          agentId: agent.id,
          position: positions[index] || { x: 300, y: 300 },
          status: agent.status === 'active' ? 'online' as const : 'offline' as const,
          load: Math.random() * 100, // Mock load
          connections: [],
          role: agent.type === 'coordinator' ? 'coordinator' as const : 'worker' as const
        }));

    const connections = neural.topology.connections.length > 0
      ? neural.topology.connections
      : generateMockConnections(nodes, neural.topology.type || selectedTopology);

    return { nodes, connections };
  }, [neural, selectedTopology, maxAgents]);

  // Generate mock connections based on topology
  const generateMockConnections = (nodes: any[], topology: string) => {
    const connections: any[] = [];

    switch (topology) {
      case 'mesh':
        // Connect every node to every other node
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            connections.push({
              id: `conn-${i}-${j}`,
              sourceId: nodes[i].id,
              targetId: nodes[j].id,
              weight: Math.random(),
              latency: Math.random() * 100,
              bandwidth: Math.random() * 1000,
              status: 'active'
            });
          }
        }
        break;

      case 'ring':
        // Connect each node to its neighbors in a circle
        for (let i = 0; i < nodes.length; i++) {
          const nextIndex = (i + 1) % nodes.length;
          connections.push({
            id: `conn-${i}-${nextIndex}`,
            sourceId: nodes[i].id,
            targetId: nodes[nextIndex].id,
            weight: Math.random(),
            latency: Math.random() * 50,
            bandwidth: Math.random() * 1000,
            status: 'active'
          });
        }
        break;

      case 'star':
        // Connect all nodes to the center node
        if (nodes.length > 0) {
          for (let i = 1; i < nodes.length; i++) {
            connections.push({
              id: `conn-0-${i}`,
              sourceId: nodes[0].id,
              targetId: nodes[i].id,
              weight: Math.random(),
              latency: Math.random() * 30,
              bandwidth: Math.random() * 1000,
              status: 'active'
            });
          }
        }
        break;

      case 'hierarchical':
        // Connect in a tree structure
        for (let i = 1; i < nodes.length; i++) {
          const parentIndex = Math.floor((i - 1) / 2);
          connections.push({
            id: `conn-${parentIndex}-${i}`,
            sourceId: nodes[parentIndex].id,
            targetId: nodes[i].id,
            weight: Math.random(),
            latency: Math.random() * 40,
            bandwidth: Math.random() * 1000,
            status: 'active'
          });
        }
        break;
    }

    return connections;
  };

  useEffect(() => {
    setVisualization(generateVisualization());
  }, [generateVisualization]);

  const initializeSwarm = async () => {
    setIsInitializing(true);
    try {
      await neural.initializeSwarm({
        topology: selectedTopology,
        maxAgents,
        strategy
      });
    } catch (error) {
      console.error('Failed to initialize swarm:', error);
      neural.addNotification({
        type: 'error',
        title: 'Swarm Initialization Failed',
        message: error instanceof Error ? error.message : 'Unknown error occurred'
      });
    } finally {
      setIsInitializing(false);
    }
  };

  const spawnAgent = async (agentType: string) => {
    try {
      await neural.spawnAgent({
        type: agentType as any,
        cognitivePattern: 'adaptive',
        capabilities: []
      });
    } catch (error) {
      console.error('Failed to spawn agent:', error);
    }
  };

  const getNodeColor = (node: any) => {
    switch (node.status) {
      case 'online': return 'neural-glow-success';
      case 'syncing': return 'neural-glow-accent';
      case 'error': return 'neural-glow-error';
      default: return 'neural-glow-muted';
    }
  };

  const getNodeIcon = (node: any) => {
    switch (node.role) {
      case 'coordinator': return '🎯';
      case 'validator': return '✅';
      case 'optimizer': return '⚡';
      default: return '🤖';
    }
  };

  const getConnectionColor = (connection: any) => {
    switch (connection.status) {
      case 'active': return 'rgba(16, 185, 129, 0.6)';
      case 'congested': return 'rgba(245, 158, 11, 0.6)';
      default: return 'rgba(99, 102, 241, 0.3)';
    }
  };

  const handleNodeClick = (nodeId: string) => {
    setSelectedNode(selectedNode === nodeId ? null : nodeId);
  };

  const selectedNodeData = selectedNode
    ? visualization.nodes.find(n => n.id === selectedNode)
    : null;

  const agentData = selectedNodeData
    ? neural.agents.find(a => a.id === selectedNodeData.agentId)
    : null;

  return (
    <div className="neural-theme h-full neural-flex">
      {/* Control Panel */}
      <div className="w-80 neural-card neural-flex neural-flex-col">
        <div className="neural-card-header">
          <h2 className="neural-heading neural-heading-md">Swarm Orchestrator</h2>
          <p className="neural-text-muted">Manage and visualize AI swarms</p>
        </div>

        <div className="neural-card-body neural-flex-1 neural-space-y-6">
          {/* Swarm Configuration */}
          <div>
            <h3 className="neural-heading neural-heading-sm mb-3">Configuration</h3>
            <div className="neural-space-y-3">
              <div>
                <label className="block neural-text-secondary text-sm mb-1">Topology</label>
                <select
                  value={selectedTopology}
                  onChange={(e) => setSelectedTopology(e.target.value as any)}
                  className="neural-input neural-select"
                >
                  <option value="mesh">Mesh Network</option>
                  <option value="hierarchical">Hierarchical</option>
                  <option value="ring">Ring Network</option>
                  <option value="star">Star Network</option>
                </select>
              </div>

              <div>
                <label className="block neural-text-secondary text-sm mb-1">Max Agents</label>
                <input
                  type="number"
                  value={maxAgents}
                  onChange={(e) => setMaxAgents(Math.max(1, parseInt(e.target.value) || 1))}
                  min="1"
                  max="20"
                  className="neural-input"
                />
              </div>

              <div>
                <label className="block neural-text-secondary text-sm mb-1">Strategy</label>
                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value as any)}
                  className="neural-input neural-select"
                >
                  <option value="balanced">Balanced</option>
                  <option value="specialized">Specialized</option>
                  <option value="adaptive">Adaptive</option>
                </select>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div>
            <h3 className="neural-heading neural-heading-sm mb-3">Actions</h3>
            <div className="neural-space-y-2">
              <button
                onClick={initializeSwarm}
                disabled={isInitializing}
                className="w-full neural-btn neural-btn-primary"
              >
                {isInitializing ? 'Initializing...' : 'Initialize Swarm'}
              </button>

              <div className="neural-grid neural-grid-2 gap-2">
                <button
                  onClick={() => spawnAgent('researcher')}
                  className="neural-btn neural-btn-outline neural-btn-sm"
                >
                  + Researcher
                </button>
                <button
                  onClick={() => spawnAgent('coder')}
                  className="neural-btn neural-btn-outline neural-btn-sm"
                >
                  + Coder
                </button>
                <button
                  onClick={() => spawnAgent('analyst')}
                  className="neural-btn neural-btn-outline neural-btn-sm"
                >
                  + Analyst
                </button>
                <button
                  onClick={() => spawnAgent('optimizer')}
                  className="neural-btn neural-btn-outline neural-btn-sm"
                >
                  + Optimizer
                </button>
              </div>
            </div>
          </div>

          {/* Swarm Status */}
          <div>
            <h3 className="neural-heading neural-heading-sm mb-3">Status</h3>
            <div className="neural-space-y-3">
              <div className="neural-flex neural-flex-between">
                <span className="neural-text-secondary">Total Agents</span>
                <span className="neural-text-primary">{neural.agents.length}</span>
              </div>
              <div className="neural-flex neural-flex-between">
                <span className="neural-text-secondary">Active Nodes</span>
                <span className="neural-text-primary">
                  {visualization.nodes.filter(n => n.status === 'online').length}
                </span>
              </div>
              <div className="neural-flex neural-flex-between">
                <span className="neural-text-secondary">Connections</span>
                <span className="neural-text-primary">{visualization.connections.length}</span>
              </div>
              <div className="neural-flex neural-flex-between">
                <span className="neural-text-secondary">Topology</span>
                <span className="neural-text-primary capitalize">
                  {neural.topology.type || selectedTopology}
                </span>
              </div>
            </div>
          </div>

          {/* Node Details */}
          {selectedNodeData && (
            <div>
              <h3 className="neural-heading neural-heading-sm mb-3">
                Node Details
                <button
                  onClick={() => setSelectedNode(null)}
                  className="neural-btn neural-btn-ghost neural-btn-sm ml-2"
                >
                  ×
                </button>
              </h3>
              <div className="neural-card neural-card-body neural-bg-tertiary">
                <div className="neural-space-y-2">
                  <div className="neural-flex neural-flex-between">
                    <span className="neural-text-secondary text-sm">Role</span>
                    <span className="neural-text-primary text-sm capitalize">
                      {selectedNodeData.role}
                    </span>
                  </div>
                  <div className="neural-flex neural-flex-between">
                    <span className="neural-text-secondary text-sm">Status</span>
                    <span className={`neural-status neural-status-${
                      selectedNodeData.status === 'online' ? 'active' : 'error'
                    }`}>
                      <div className="neural-status-dot"></div>
                      <span className="text-sm capitalize">{selectedNodeData.status}</span>
                    </span>
                  </div>
                  <div className="neural-flex neural-flex-between">
                    <span className="neural-text-secondary text-sm">Load</span>
                    <span className="neural-text-primary text-sm">
                      {Math.round(selectedNodeData.load)}%
                    </span>
                  </div>
                  <div>
                    <span className="neural-text-secondary text-sm">Load Progress</span>
                    <div className="neural-progress mt-1">
                      <div
                        className="neural-progress-bar"
                        style={{ width: `${selectedNodeData.load}%` }}
                      />
                    </div>
                  </div>
                  {agentData && (
                    <>
                      <div className="neural-flex neural-flex-between">
                        <span className="neural-text-secondary text-sm">Success Rate</span>
                        <span className="neural-text-primary text-sm">
                          {Math.round(agentData.performance.successRate * 100)}%
                        </span>
                      </div>
                      <div className="neural-flex neural-flex-between">
                        <span className="neural-text-secondary text-sm">Tasks Completed</span>
                        <span className="neural-text-primary text-sm">
                          {agentData.performance.tasksCompleted}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Topology Info */}
          <div>
            <h3 className="neural-heading neural-heading-sm mb-3">Topology Guide</h3>
            <div className="neural-space-y-2 text-sm">
              <div>
                <strong className="neural-text-primary">Mesh:</strong>
                <span className="neural-text-muted"> Every node connects to every other node</span>
              </div>
              <div>
                <strong className="neural-text-primary">Ring:</strong>
                <span className="neural-text-muted"> Nodes form a circular chain</span>
              </div>
              <div>
                <strong className="neural-text-primary">Star:</strong>
                <span className="neural-text-muted"> All nodes connect to central hub</span>
              </div>
              <div>
                <strong className="neural-text-primary">Hierarchical:</strong>
                <span className="neural-text-muted"> Tree-like parent-child structure</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Visualization Canvas */}
      <div className="flex-1 relative overflow-hidden neural-bg-primary">
        <div
          ref={canvasRef}
          className="w-full h-full relative"
          style={{
            transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`
          }}
        >
          {/* Grid Background */}
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: `
                linear-gradient(rgba(99, 102, 241, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(99, 102, 241, 0.1) 1px, transparent 1px)
              `,
              backgroundSize: '30px 30px'
            }}
          />

          {/* Connections */}
          <svg className="absolute inset-0 w-full h-full">
            {visualization.connections.map(connection => {
              const sourceNode = visualization.nodes.find(n => n.id === connection.sourceId);
              const targetNode = visualization.nodes.find(n => n.id === connection.targetId);

              if (!sourceNode || !targetNode) return null;

              return (
                <line
                  key={connection.id}
                  x1={sourceNode.position.x + 30}
                  y1={sourceNode.position.y + 30}
                  x2={targetNode.position.x + 30}
                  y2={targetNode.position.y + 30}
                  stroke={getConnectionColor(connection)}
                  strokeWidth={Math.max(1, connection.weight * 4)}
                  strokeDasharray={connection.status === 'active' ? 'none' : '5,5'}
                />
              );
            })}
          </svg>

          {/* Nodes */}
          {visualization.nodes.map(node => (
            <div
              key={node.id}
              className={`absolute neural-card cursor-pointer transition-all duration-300 ${
                selectedNode === node.id ? 'neural-glow-primary' : getNodeColor(node)
              }`}
              style={{
                left: node.position.x,
                top: node.position.y,
                width: 60,
                height: 60
              }}
              onClick={() => handleNodeClick(node.id)}
            >
              <div className="neural-card-body p-2 h-full neural-flex neural-flex-col neural-flex-center">
                <div className="text-lg mb-1">{getNodeIcon(node)}</div>
                <div className="text-xs neural-text-primary font-medium text-center">
                  {node.role}
                </div>
              </div>

              {/* Load indicator */}
              <div className="absolute bottom-0 left-0 right-0 h-1 neural-bg-tertiary">
                <div
                  className="h-full neural-bg-accent transition-all duration-500"
                  style={{ width: `${node.load}%` }}
                />
              </div>
            </div>
          ))}

          {/* Canvas Controls */}
          <div className="absolute top-4 right-4 neural-flex neural-flex-col gap-2">
            <button
              onClick={() => setZoom(prev => Math.min(prev + 0.1, 2))}
              className="neural-btn neural-btn-ghost neural-btn-sm"
            >
              +
            </button>
            <button
              onClick={() => setZoom(prev => Math.max(prev - 0.1, 0.5))}
              className="neural-btn neural-btn-ghost neural-btn-sm"
            >
              -
            </button>
            <button
              onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
              className="neural-btn neural-btn-ghost neural-btn-sm"
            >
              Reset
            </button>
          </div>

          {/* Legend */}
          <div className="absolute bottom-4 left-4 neural-card neural-card-body p-3">
            <h4 className="neural-text-secondary font-semibold mb-2 text-sm">Legend</h4>
            <div className="neural-space-y-1 text-xs">
              <div className="neural-flex items-center gap-2">
                <div className="w-3 h-3 rounded neural-bg-success"></div>
                <span className="neural-text-muted">Online</span>
              </div>
              <div className="neural-flex items-center gap-2">
                <div className="w-3 h-3 rounded neural-bg-accent"></div>
                <span className="neural-text-muted">Syncing</span>
              </div>
              <div className="neural-flex items-center gap-2">
                <div className="w-3 h-3 rounded neural-bg-error"></div>
                <span className="neural-text-muted">Error</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SwarmOrchestrator;