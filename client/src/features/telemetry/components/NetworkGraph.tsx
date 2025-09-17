import React, { useMemo, useRef, useEffect, useState } from 'react';
import { Card } from '../../design-system/components/Card';
import { Switch } from '../../design-system/components/Switch';
import { Label } from '../../design-system/components/Label';
import { createLogger } from '../../../utils/logger';
import { AgentCommunication, AgentPosition, NetworkGraphNode, NetworkGraphEdge } from '../types';

const logger = createLogger('NetworkGraph');

interface NetworkGraphProps {
  communications: AgentCommunication[];
  agentPositions: Record<string, AgentPosition>;
  className?: string;
}

export const NetworkGraph: React.FC<NetworkGraphProps> = ({
  communications,
  agentPositions,
  className = ''
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showWeights, setShowWeights] = useState(false);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // Build network graph data
  const graphData = useMemo(() => {
    const nodes: NetworkGraphNode[] = [];
    const edges: NetworkGraphEdge[] = [];
    const nodeMap = new Map<string, NetworkGraphNode>();
    const edgeMap = new Map<string, NetworkGraphEdge>();

    // Create nodes from agent positions and communications
    const agentIds = new Set<string>();
    Object.keys(agentPositions).forEach(id => agentIds.add(id));
    communications.forEach(comm => {
      agentIds.add(comm.fromAgentId);
      agentIds.add(comm.toAgentId);
    });

    agentIds.forEach(agentId => {
      const position = agentPositions[agentId];
      const agentComms = communications.filter(c => c.fromAgentId === agentId || c.toAgentId === agentId);

      const errorRate = agentComms.length > 0
        ? agentComms.filter(c => !c.success).length / agentComms.length
        : 0;

      const avgLatency = agentComms.length > 0
        ? agentComms.reduce((sum, c) => sum + (c.latency || 0), 0) / agentComms.length
        : 0;

      const node: NetworkGraphNode = {
        id: agentId,
        label: agentId.slice(0, 8),
        agentType: 'agent', // Could be derived from agent type if available
        status: errorRate > 0.2 ? 'error' : agentComms.length > 0 ? 'active' : 'idle',
        position: position ? {
          x: (position.position.x + 10) * 20, // Scale and center
          y: (position.position.y + 10) * 15
        } : undefined,
        metrics: {
          messageCount: agentComms.length,
          errorRate,
          performance: Math.max(0, 100 - errorRate * 100 - avgLatency)
        }
      };

      nodes.push(node);
      nodeMap.set(agentId, node);
    });

    // Create edges from communications
    communications.forEach(comm => {
      const edgeId = `${comm.fromAgentId}-${comm.toAgentId}`;
      const reverseEdgeId = `${comm.toAgentId}-${comm.fromAgentId}`;

      if (edgeMap.has(edgeId) || edgeMap.has(reverseEdgeId)) {
        // Update existing edge
        const existing = edgeMap.get(edgeId) || edgeMap.get(reverseEdgeId);
        if (existing) {
          existing.messageCount++;
          existing.weight++;
          existing.latency = (existing.latency + (comm.latency || 0)) / 2;
          if (comm.timestamp > existing.lastActivity) {
            existing.lastActivity = comm.timestamp;
          }
        }
      } else {
        // Create new edge
        const edge: NetworkGraphEdge = {
          id: edgeId,
          source: comm.fromAgentId,
          target: comm.toAgentId,
          weight: 1,
          latency: comm.latency || 0,
          messageCount: 1,
          lastActivity: comm.timestamp
        };
        edges.push(edge);
        edgeMap.set(edgeId, edge);
      }
    });

    // Auto-position nodes if they don't have positions
    const unpositionedNodes = nodes.filter(n => !n.position);
    if (unpositionedNodes.length > 0) {
      const radius = Math.min(150, 50 + unpositionedNodes.length * 10);
      unpositionedNodes.forEach((node, index) => {
        const angle = (index / unpositionedNodes.length) * 2 * Math.PI;
        node.position = {
          x: 200 + Math.cos(angle) * radius,
          y: 150 + Math.sin(angle) * radius
        };
      });
    }

    return { nodes, edges };
  }, [communications, agentPositions]);

  const getNodeColor = (node: NetworkGraphNode) => {
    switch (node.status) {
      case 'active': return '#10b981';
      case 'error': return '#ef4444';
      case 'idle': return '#6b7280';
      default: return '#3b82f6';
    }
  };

  const getEdgeColor = (edge: NetworkGraphEdge) => {
    const age = Date.now() - edge.lastActivity.getTime();
    if (age < 5000) return '#3b82f6'; // Recent activity - blue
    if (age < 30000) return '#10b981'; // Moderately recent - green
    return '#9ca3af'; // Old - gray
  };

  const getEdgeWidth = (edge: NetworkGraphEdge) => {
    return Math.max(1, Math.min(5, edge.weight));
  };

  return (
    <Card className={className}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Agent Communication Network</h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Label htmlFor="show-labels" className="text-sm">Labels</Label>
              <Switch
                id="show-labels"
                checked={showLabels}
                onCheckedChange={setShowLabels}
                size="sm"
              />
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="show-weights" className="text-sm">Weights</Label>
              <Switch
                id="show-weights"
                checked={showWeights}
                onCheckedChange={setShowWeights}
                size="sm"
              />
            </div>
          </div>
        </div>

        {/* Network Graph */}
        <div className="relative">
          <svg
            ref={svgRef}
            width="100%"
            height="400"
            viewBox="0 0 400 300"
            className="border border-gray-200 rounded-lg bg-gray-50"
          >
            {/* Edges */}
            {graphData.edges.map(edge => {
              const sourceNode = graphData.nodes.find(n => n.id === edge.source);
              const targetNode = graphData.nodes.find(n => n.id === edge.target);

              if (!sourceNode?.position || !targetNode?.position) return null;

              return (
                <g key={edge.id}>
                  <line
                    x1={sourceNode.position.x}
                    y1={sourceNode.position.y}
                    x2={targetNode.position.x}
                    y2={targetNode.position.y}
                    stroke={getEdgeColor(edge)}
                    strokeWidth={getEdgeWidth(edge)}
                    opacity="0.7"
                  />

                  {showWeights && (
                    <text
                      x={(sourceNode.position.x + targetNode.position.x) / 2}
                      y={(sourceNode.position.y + targetNode.position.y) / 2}
                      textAnchor="middle"
                      className="text-xs fill-gray-600"
                      dy="-2"
                    >
                      {edge.messageCount}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Nodes */}
            {graphData.nodes.map(node => {
              if (!node.position) return null;

              const isSelected = selectedNode === node.id;

              return (
                <g key={node.id}>
                  {/* Node circle */}
                  <circle
                    cx={node.position.x}
                    cy={node.position.y}
                    r={isSelected ? 12 : 8}
                    fill={getNodeColor(node)}
                    stroke="#ffffff"
                    strokeWidth="2"
                    className="cursor-pointer transition-all hover:r-10"
                    onClick={() => setSelectedNode(selectedNode === node.id ? null : node.id)}
                  />

                  {/* Performance indicator */}
                  <circle
                    cx={node.position.x}
                    cy={node.position.y}
                    r={isSelected ? 16 : 12}
                    fill="none"
                    stroke={getNodeColor(node)}
                    strokeWidth="2"
                    strokeDasharray={`${(node.metrics.performance / 100) * 2 * Math.PI * 12} ${2 * Math.PI * 12}`}
                    opacity="0.5"
                    transform={`rotate(-90 ${node.position.x} ${node.position.y})`}
                  />

                  {/* Node label */}
                  {showLabels && (
                    <text
                      x={node.position.x}
                      y={node.position.y + (isSelected ? 25 : 20)}
                      textAnchor="middle"
                      className="text-xs fill-gray-700 font-medium pointer-events-none"
                    >
                      {node.label}
                    </text>
                  )}

                  {/* Error indicator */}
                  {node.metrics.errorRate > 0.1 && (
                    <text
                      x={node.position.x + (isSelected ? 15 : 10)}
                      y={node.position.y - (isSelected ? 10 : 8)}
                      className="text-xs fill-red-600"
                    >
                      ⚠️
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          {/* Legend */}
          <div className="absolute top-2 right-2 bg-white bg-opacity-90 p-3 rounded-lg text-xs space-y-1">
            <div className="font-medium">Status</div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>Active</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span>Error</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
              <span>Idle</span>
            </div>
            <div className="mt-2 font-medium">Connections</div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-blue-500"></div>
              <span>Recent</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-green-500"></div>
              <span>Active</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-gray-400"></div>
              <span>Stale</span>
            </div>
          </div>
        </div>

        {/* Node Details */}
        {selectedNode && (() => {
          const node = graphData.nodes.find(n => n.id === selectedNode);
          if (!node) return null;

          const nodeEdges = graphData.edges.filter(e =>
            e.source === selectedNode || e.target === selectedNode
          );

          return (
            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
              <h4 className="font-medium mb-2">{node.label} Details</h4>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-600">Status:</span>
                  <span className="ml-2 capitalize font-medium">{node.status}</span>
                </div>
                <div>
                  <span className="text-gray-600">Messages:</span>
                  <span className="ml-2 font-mono">{node.metrics.messageCount}</span>
                </div>
                <div>
                  <span className="text-gray-600">Error Rate:</span>
                  <span className="ml-2 font-mono">{(node.metrics.errorRate * 100).toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-gray-600">Performance:</span>
                  <span className="ml-2 font-mono">{node.metrics.performance.toFixed(1)}%</span>
                </div>
                <div className="col-span-2">
                  <span className="text-gray-600">Connections:</span>
                  <span className="ml-2 font-mono">{nodeEdges.length}</span>
                </div>
              </div>

              {nodeEdges.length > 0 && (
                <div className="mt-3">
                  <div className="text-sm font-medium mb-1">Active Connections:</div>
                  <div className="space-y-1 max-h-20 overflow-y-auto">
                    {nodeEdges.slice(0, 5).map(edge => (
                      <div key={edge.id} className="text-xs flex justify-between">
                        <span>
                          {edge.source === selectedNode ? '→ ' : '← '}
                          {edge.source === selectedNode ? edge.target : edge.source}
                        </span>
                        <span className="text-gray-500">
                          {edge.messageCount} msgs, {edge.latency.toFixed(0)}ms
                        </span>
                      </div>
                    ))}
                    {nodeEdges.length > 5 && (
                      <div className="text-xs text-gray-500">
                        +{nodeEdges.length - 5} more connections
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })()}

        {/* Graph Statistics */}
        <div className="mt-4 pt-4 border-t border-gray-200 text-sm text-gray-600">
          <div className="flex justify-between">
            <span>{graphData.nodes.length} agents, {graphData.edges.length} connections</span>
            <span>{communications.length} total messages</span>
          </div>
        </div>
      </div>
    </Card>
  );
};