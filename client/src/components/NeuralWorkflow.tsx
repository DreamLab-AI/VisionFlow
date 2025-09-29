/**
 * Neural Workflow Builder - Visual workflow creation and management
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useNeural } from '../contexts/NeuralContext';
import '../styles/neural-theme.css';

interface WorkflowNode {
  id: string;
  type: 'start' | 'agent' | 'decision' | 'end';
  agentType?: string;
  name: string;
  description?: string;
  position: { x: number; y: number };
  connections: string[];
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedDuration: number;
  config?: Record<string, any>;
}

interface WorkflowConnection {
  id: string;
  sourceId: string;
  targetId: string;
  condition?: string;
  label?: string;
}

const agentTypes = [
  { type: 'researcher', name: 'Researcher', icon: '🔍', color: 'neural-badge-primary' },
  { type: 'coder', name: 'Coder', icon: '⚡', color: 'neural-badge-success' },
  { type: 'analyst', name: 'Analyst', icon: '📊', color: 'neural-badge-accent' },
  { type: 'optimizer', name: 'Optimizer', icon: '🚀', color: 'neural-badge-warning' },
  { type: 'coordinator', name: 'Coordinator', icon: '🎯', color: 'neural-badge-secondary' }
];

const NeuralWorkflow: React.FC = () => {
  const neural = useNeural();
  const [nodes, setNodes] = useState<WorkflowNode[]>([]);
  const [connections, setConnections] = useState<WorkflowConnection[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [dragNode, setDragNode] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [isConnecting, setIsConnecting] = useState<string | null>(null);
  const [workflowName, setWorkflowName] = useState('');
  const [workflowDescription, setWorkflowDescription] = useState('');
  const canvasRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  useEffect(() => {
    // Initialize with start and end nodes
    if (nodes.length === 0) {
      setNodes([
        {
          id: 'start',
          type: 'start',
          name: 'Start',
          position: { x: 100, y: 200 },
          connections: [],
          priority: 'medium',
          estimatedDuration: 0
        },
        {
          id: 'end',
          type: 'end',
          name: 'End',
          position: { x: 700, y: 200 },
          connections: [],
          priority: 'medium',
          estimatedDuration: 0
        }
      ]);
    }
  }, [nodes.length]);

  const addNode = useCallback((type: 'agent' | 'decision', agentType?: string) => {
    const newNode: WorkflowNode = {
      id: `node-${Date.now()}`,
      type,
      agentType,
      name: type === 'agent' ? `${agentType || 'Agent'} Task` : 'Decision Point',
      position: { x: 400, y: 300 },
      connections: [],
      priority: 'medium',
      estimatedDuration: 30
    };

    setNodes(prev => [...prev, newNode]);
    setSelectedNode(newNode.id);
  }, []);

  const updateNode = useCallback((id: string, updates: Partial<WorkflowNode>) => {
    setNodes(prev => prev.map(node =>
      node.id === id ? { ...node, ...updates } : node
    ));
  }, []);

  const deleteNode = useCallback((id: string) => {
    if (id === 'start' || id === 'end') return;

    setNodes(prev => prev.filter(node => node.id !== id));
    setConnections(prev => prev.filter(conn =>
      conn.sourceId !== id && conn.targetId !== id
    ));
    if (selectedNode === id) {
      setSelectedNode(null);
    }
  }, [selectedNode]);

  const addConnection = useCallback((sourceId: string, targetId: string) => {
    if (sourceId === targetId) return;

    const existingConnection = connections.find(conn =>
      conn.sourceId === sourceId && conn.targetId === targetId
    );

    if (existingConnection) return;

    const newConnection: WorkflowConnection = {
      id: `conn-${Date.now()}`,
      sourceId,
      targetId
    };

    setConnections(prev => [...prev, newConnection]);

    // Update source node connections
    setNodes(prev => prev.map(node =>
      node.id === sourceId
        ? { ...node, connections: [...node.connections, targetId] }
        : node
    ));
  }, [connections]);

  const deleteConnection = useCallback((connectionId: string) => {
    const connection = connections.find(conn => conn.id === connectionId);
    if (!connection) return;

    setConnections(prev => prev.filter(conn => conn.id !== connectionId));

    // Update source node connections
    setNodes(prev => prev.map(node =>
      node.id === connection.sourceId
        ? { ...node, connections: node.connections.filter(id => id !== connection.targetId) }
        : node
    ));
  }, [connections]);

  const handleMouseDown = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.preventDefault();

    if (isConnecting) {
      addConnection(isConnecting, nodeId);
      setIsConnecting(null);
      return;
    }

    setSelectedNode(nodeId);
    setDragNode(nodeId);

    const node = nodes.find(n => n.id === nodeId);
    if (node) {
      const rect = e.currentTarget.getBoundingClientRect();
      setDragOffset({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
    }
  }, [isConnecting, nodes, addConnection]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragNode || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left - dragOffset.x) / zoom - pan.x;
    const y = (e.clientY - rect.top - dragOffset.y) / zoom - pan.y;

    updateNode(dragNode, { position: { x: Math.max(0, x), y: Math.max(0, y) } });
  }, [dragNode, dragOffset, zoom, pan, updateNode]);

  const handleMouseUp = useCallback(() => {
    setDragNode(null);
  }, []);

  const saveWorkflow = useCallback(async () => {
    if (!workflowName.trim()) {
      alert('Please enter a workflow name');
      return;
    }

    const workflowSteps = nodes
      .filter(node => node.type === 'agent')
      .map(node => ({
        id: node.id,
        name: node.name,
        description: node.description,
        agentType: node.agentType || 'coordinator',
        dependencies: connections
          .filter(conn => conn.targetId === node.id)
          .map(conn => conn.sourceId),
        priority: node.priority,
        status: 'pending' as const,
        estimatedDuration: node.estimatedDuration
      }));

    try {
      await neural.createWorkflow({
        name: workflowName,
        description: workflowDescription,
        steps: workflowSteps,
        status: 'pending',
        progress: 0,
        assignedAgents: []
      });

      alert('Workflow saved successfully!');
    } catch (error) {
      alert('Failed to save workflow: ' + (error instanceof Error ? error.message : 'Unknown error'));
    }
  }, [workflowName, workflowDescription, nodes, connections, neural]);

  const executeWorkflow = useCallback(async () => {
    if (!workflowName.trim()) {
      alert('Please save the workflow first');
      return;
    }

    try {
      await saveWorkflow();
      // Find the saved workflow and execute it
      const savedWorkflow = neural.workflows.find(w => w.name === workflowName);
      if (savedWorkflow) {
        await neural.executeWorkflow(savedWorkflow.id);
        alert('Workflow execution started!');
      }
    } catch (error) {
      alert('Failed to execute workflow: ' + (error instanceof Error ? error.message : 'Unknown error'));
    }
  }, [workflowName, saveWorkflow, neural]);

  const getNodeIcon = (node: WorkflowNode) => {
    if (node.type === 'start') return '▶️';
    if (node.type === 'end') return '🏁';
    if (node.type === 'decision') return '❓';
    if (node.type === 'agent' && node.agentType) {
      const agentConfig = agentTypes.find(a => a.type === node.agentType);
      return agentConfig?.icon || '🤖';
    }
    return '🤖';
  };

  const getNodeColor = (node: WorkflowNode) => {
    if (node.type === 'start') return 'neural-glow-success';
    if (node.type === 'end') return 'neural-glow-error';
    if (node.type === 'decision') return 'neural-glow-accent';
    if (node.type === 'agent' && node.agentType) {
      const agentConfig = agentTypes.find(a => a.type === node.agentType);
      return agentConfig?.color || 'neural-badge-primary';
    }
    return 'neural-badge-primary';
  };

  const selectedNodeData = selectedNode ? nodes.find(n => n.id === selectedNode) : null;

  return (
    <div className="neural-theme h-screen flex">
      {/* Toolbar */}
      <div className="w-80 neural-card neural-flex neural-flex-col">
        <div className="neural-card-header">
          <h2 className="neural-heading neural-heading-md">Workflow Builder</h2>
          <p className="neural-text-muted">Design AI agent workflows</p>
        </div>

        <div className="neural-card-body neural-flex-1 neural-space-y-6">
          {/* Workflow Info */}
          <div>
            <h3 className="neural-heading neural-heading-sm mb-3">Workflow Info</h3>
            <div className="neural-space-y-3">
              <div>
                <label className="block neural-text-secondary text-sm mb-1">Name</label>
                <input
                  type="text"
                  value={workflowName}
                  onChange={(e) => setWorkflowName(e.target.value)}
                  placeholder="Enter workflow name..."
                  className="neural-input"
                />
              </div>
              <div>
                <label className="block neural-text-secondary text-sm mb-1">Description</label>
                <textarea
                  value={workflowDescription}
                  onChange={(e) => setWorkflowDescription(e.target.value)}
                  placeholder="Describe the workflow..."
                  className="neural-input neural-textarea"
                  rows={3}
                />
              </div>
            </div>
          </div>

          {/* Add Nodes */}
          <div>
            <h3 className="neural-heading neural-heading-sm mb-3">Add Agents</h3>
            <div className="neural-space-y-2">
              {agentTypes.map(agent => (
                <button
                  key={agent.type}
                  onClick={() => addNode('agent', agent.type)}
                  className="w-full neural-btn neural-btn-outline neural-flex items-center gap-2 justify-start"
                >
                  <span>{agent.icon}</span>
                  <span>{agent.name}</span>
                </button>
              ))}
              <button
                onClick={() => addNode('decision')}
                className="w-full neural-btn neural-btn-outline neural-flex items-center gap-2 justify-start"
              >
                <span>❓</span>
                <span>Decision Point</span>
              </button>
            </div>
          </div>

          {/* Node Properties */}
          {selectedNodeData && selectedNodeData.type !== 'start' && selectedNodeData.type !== 'end' && (
            <div>
              <h3 className="neural-heading neural-heading-sm mb-3">
                Node Properties
                <button
                  onClick={() => deleteNode(selectedNodeData.id)}
                  className="neural-btn neural-btn-ghost neural-btn-sm ml-2 neural-text-error"
                >
                  Delete
                </button>
              </h3>
              <div className="neural-space-y-3">
                <div>
                  <label className="block neural-text-secondary text-sm mb-1">Name</label>
                  <input
                    type="text"
                    value={selectedNodeData.name}
                    onChange={(e) => updateNode(selectedNodeData.id, { name: e.target.value })}
                    className="neural-input"
                  />
                </div>
                <div>
                  <label className="block neural-text-secondary text-sm mb-1">Description</label>
                  <textarea
                    value={selectedNodeData.description || ''}
                    onChange={(e) => updateNode(selectedNodeData.id, { description: e.target.value })}
                    className="neural-input"
                    rows={2}
                  />
                </div>
                <div>
                  <label className="block neural-text-secondary text-sm mb-1">Priority</label>
                  <select
                    value={selectedNodeData.priority}
                    onChange={(e) => updateNode(selectedNodeData.id, { priority: e.target.value as any })}
                    className="neural-input neural-select"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
                <div>
                  <label className="block neural-text-secondary text-sm mb-1">Duration (minutes)</label>
                  <input
                    type="number"
                    value={selectedNodeData.estimatedDuration}
                    onChange={(e) => updateNode(selectedNodeData.id, { estimatedDuration: parseInt(e.target.value) || 0 })}
                    className="neural-input"
                    min="0"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="neural-space-y-2">
            <button
              onClick={saveWorkflow}
              disabled={!workflowName.trim()}
              className="w-full neural-btn neural-btn-primary"
            >
              Save Workflow
            </button>
            <button
              onClick={executeWorkflow}
              disabled={!workflowName.trim()}
              className="w-full neural-btn neural-btn-secondary"
            >
              Save & Execute
            </button>
          </div>

          {/* Connection Mode */}
          <div>
            <button
              onClick={() => setIsConnecting(isConnecting ? null : 'start')}
              className={`w-full neural-btn ${
                isConnecting ? 'neural-btn-accent' : 'neural-btn-outline'
              }`}
            >
              {isConnecting ? 'Cancel Connection' : 'Connect Nodes'}
            </button>
            {isConnecting && (
              <p className="neural-text-muted text-sm mt-2">
                Click on nodes to connect them. Starting from: {isConnecting}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 relative overflow-hidden neural-bg-primary">
        <div
          ref={canvasRef}
          className="w-full h-full relative cursor-grab"
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
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
              backgroundSize: '20px 20px'
            }}
          />

          {/* Connections */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {connections.map(connection => {
              const sourceNode = nodes.find(n => n.id === connection.sourceId);
              const targetNode = nodes.find(n => n.id === connection.targetId);

              if (!sourceNode || !targetNode) return null;

              const startX = sourceNode.position.x + 60;
              const startY = sourceNode.position.y + 30;
              const endX = targetNode.position.x;
              const endY = targetNode.position.y + 30;

              const midX = startX + (endX - startX) / 2;

              return (
                <g key={connection.id}>
                  <path
                    d={`M ${startX} ${startY} C ${midX} ${startY} ${midX} ${endY} ${endX} ${endY}`}
                    stroke="rgba(99, 102, 241, 0.6)"
                    strokeWidth="2"
                    fill="none"
                    markerEnd="url(#arrowhead)"
                  />
                  <circle
                    cx={midX}
                    cy={(startY + endY) / 2}
                    r="8"
                    fill="rgba(239, 68, 68, 0.8)"
                    className="cursor-pointer"
                    onClick={() => deleteConnection(connection.id)}
                    style={{ pointerEvents: 'all' }}
                  />
                  <text
                    x={midX}
                    y={(startY + endY) / 2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize="10"
                    className="cursor-pointer"
                    style={{ pointerEvents: 'all' }}
                    onClick={() => deleteConnection(connection.id)}
                  >
                    ×
                  </text>
                </g>
              );
            })}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon
                  points="0 0, 10 3.5, 0 7"
                  fill="rgba(99, 102, 241, 0.6)"
                />
              </marker>
            </defs>
          </svg>

          {/* Nodes */}
          {nodes.map(node => (
            <div
              key={node.id}
              className={`absolute neural-card cursor-pointer transition-all duration-200 ${
                selectedNode === node.id ? 'neural-glow-primary' : ''
              } ${getNodeColor(node)}`}
              style={{
                left: node.position.x,
                top: node.position.y,
                width: 120,
                height: 60,
                userSelect: 'none'
              }}
              onMouseDown={(e) => handleMouseDown(e, node.id)}
            >
              <div className="neural-card-body p-2 h-full neural-flex neural-flex-col justify-center items-center text-center">
                <div className="text-lg mb-1">{getNodeIcon(node)}</div>
                <div className="text-xs neural-text-primary font-medium">
                  {node.name}
                </div>
                {node.type === 'agent' && (
                  <div className={`neural-badge neural-badge-xs mt-1 ${getNodeColor(node)}`}>
                    {node.agentType}
                  </div>
                )}
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
        </div>
      </div>
    </div>
  );
};

export default NeuralWorkflow;