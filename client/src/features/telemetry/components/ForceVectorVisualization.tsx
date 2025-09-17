import React, { useMemo, useState } from 'react';
import { Card } from '../../design-system/components/Card';
import { Switch } from '../../design-system/components/Switch';
import { Label } from '../../design-system/components/Label';
import { Select } from '../../design-system/components/Select';
import { createLogger } from '../../../utils/logger';
import { AgentPosition } from '../types';

const logger = createLogger('ForceVectorVisualization');

interface ForceVectorVisualizationProps {
  positions: Record<string, AgentPosition>;
  className?: string;
}

export const ForceVectorVisualization: React.FC<ForceVectorVisualizationProps> = ({
  positions,
  className = ''
}) => {
  const [showVelocity, setShowVelocity] = useState(true);
  const [showForces, setShowForces] = useState(true);
  const [showTrails, setShowTrails] = useState(false);
  const [vectorScale, setVectorScale] = useState<'1' | '2' | '5' | '10'>('5');

  const visualizationData = useMemo(() => {
    const agentData = Object.entries(positions).map(([agentId, position]) => {
      // Calculate derived forces if not provided
      let forceVector = position.forceVector;

      if (!forceVector && position.velocity) {
        // Estimate force from velocity change (simplified physics)
        const velocityMagnitude = Math.sqrt(
          Math.pow(position.velocity.x, 2) + Math.pow(position.velocity.y, 2)
        );

        if (velocityMagnitude > 0) {
          forceVector = {
            x: position.velocity.x / velocityMagnitude * 0.5,
            y: position.velocity.y / velocityMagnitude * 0.5,
            magnitude: velocityMagnitude * 0.5
          };
        }
      }

      // Generate artificial forces for demonstration if none exist
      if (!forceVector && !position.velocity) {
        const time = Date.now() / 1000;
        const agentHash = agentId.split('').reduce((a, b) => a + b.charCodeAt(0), 0);

        forceVector = {
          x: Math.sin(time * 0.5 + agentHash) * 0.3,
          y: Math.cos(time * 0.7 + agentHash) * 0.3,
          magnitude: 0.3
        };
      }

      return {
        agentId,
        position: position.position,
        velocity: position.velocity,
        forceVector,
        timestamp: position.timestamp
      };
    });

    // Calculate bounds
    const xs = agentData.map(d => d.position.x);
    const ys = agentData.map(d => d.position.y);

    const bounds = {
      minX: Math.min(...xs) - 5,
      maxX: Math.max(...xs) + 5,
      minY: Math.min(...ys) - 5,
      maxY: Math.max(...ys) + 5
    };

    const width = bounds.maxX - bounds.minX;
    const height = bounds.maxY - bounds.minY;

    return { agentData, bounds, width, height };
  }, [positions]);

  const scalePosition = (pos: { x: number; y: number }) => ({
    x: ((pos.x - visualizationData.bounds.minX) / visualizationData.width) * 400,
    y: ((pos.y - visualizationData.bounds.minY) / visualizationData.height) * 300
  });

  const scale = parseFloat(vectorScale);

  return (
    <Card className={className}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Force Vector Visualization</h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Label htmlFor="show-velocity" className="text-sm">Velocity</Label>
              <Switch
                id="show-velocity"
                checked={showVelocity}
                onCheckedChange={setShowVelocity}
                size="sm"
              />
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="show-forces" className="text-sm">Forces</Label>
              <Switch
                id="show-forces"
                checked={showForces}
                onCheckedChange={setShowForces}
                size="sm"
              />
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="show-trails" className="text-sm">Trails</Label>
              <Switch
                id="show-trails"
                checked={showTrails}
                onCheckedChange={setShowTrails}
                size="sm"
              />
            </div>
            <Select value={vectorScale} onValueChange={(value: any) => setVectorScale(value)}>
              <Select.Trigger className="w-20">
                <Select.Value />
              </Select.Trigger>
              <Select.Content>
                <Select.Item value="1">1x</Select.Item>
                <Select.Item value="2">2x</Select.Item>
                <Select.Item value="5">5x</Select.Item>
                <Select.Item value="10">10x</Select.Item>
              </Select.Content>
            </Select>
          </div>
        </div>

        {/* Visualization */}
        <div className="relative">
          <svg
            width="100%"
            height="300"
            viewBox="0 0 400 300"
            className="border border-gray-200 rounded-lg bg-gray-50"
          >
            {/* Grid */}
            <defs>
              <pattern id="force-grid" width="40" height="30" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 30" fill="none" stroke="#e5e7eb" strokeWidth="1" />
              </pattern>

              {/* Arrow markers for different vector types */}
              <marker
                id="force-arrow"
                markerWidth="8"
                markerHeight="6"
                refX="7"
                refY="3"
                orient="auto"
                markerUnits="strokeWidth"
              >
                <path d="M0,0 L0,6 L8,3 z" fill="#ef4444" />
              </marker>

              <marker
                id="velocity-arrow"
                markerWidth="8"
                markerHeight="6"
                refX="7"
                refY="3"
                orient="auto"
                markerUnits="strokeWidth"
              >
                <path d="M0,0 L0,6 L8,3 z" fill="#10b981" />
              </marker>
            </defs>

            <rect width="100%" height="100%" fill="url(#force-grid)" />

            {/* Agent trails */}
            {showTrails && visualizationData.agentData.map(agent => {
              const scaledPos = scalePosition(agent.position);

              return (
                <g key={`trail-${agent.agentId}`}>
                  {/* Radial trail effect */}
                  <circle
                    cx={scaledPos.x}
                    cy={scaledPos.y}
                    r="15"
                    fill="rgba(59, 130, 246, 0.1)"
                    className="animate-pulse"
                  />
                  <circle
                    cx={scaledPos.x}
                    cy={scaledPos.y}
                    r="10"
                    fill="rgba(59, 130, 246, 0.15)"
                    className="animate-ping"
                  />
                </g>
              );
            })}

            {/* Force vectors */}
            {showForces && visualizationData.agentData.map(agent => {
              if (!agent.forceVector) return null;

              const scaledPos = scalePosition(agent.position);
              const forceLength = Math.min(agent.forceVector.magnitude * scale * 30, 60);
              const angle = Math.atan2(agent.forceVector.y, agent.forceVector.x);

              const endX = scaledPos.x + Math.cos(angle) * forceLength;
              const endY = scaledPos.y + Math.sin(angle) * forceLength;

              return (
                <g key={`force-${agent.agentId}`}>
                  <line
                    x1={scaledPos.x}
                    y1={scaledPos.y}
                    x2={endX}
                    y2={endY}
                    stroke="#ef4444"
                    strokeWidth="2"
                    markerEnd="url(#force-arrow)"
                    opacity="0.8"
                  />

                  {/* Force magnitude label */}
                  <text
                    x={endX + 5}
                    y={endY - 5}
                    className="text-xs fill-red-600 font-medium"
                    textAnchor="start"
                  >
                    {agent.forceVector.magnitude.toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* Velocity vectors */}
            {showVelocity && visualizationData.agentData.map(agent => {
              if (!agent.velocity) return null;

              const scaledPos = scalePosition(agent.position);
              const velocityMagnitude = Math.sqrt(
                Math.pow(agent.velocity.x, 2) + Math.pow(agent.velocity.y, 2)
              );
              const velocityLength = Math.min(velocityMagnitude * scale * 40, 80);

              if (velocityLength < 1) return null;

              const angle = Math.atan2(agent.velocity.y, agent.velocity.x);
              const endX = scaledPos.x + Math.cos(angle) * velocityLength;
              const endY = scaledPos.y + Math.sin(angle) * velocityLength;

              return (
                <g key={`velocity-${agent.agentId}`}>
                  <line
                    x1={scaledPos.x}
                    y1={scaledPos.y}
                    x2={endX}
                    y2={endY}
                    stroke="#10b981"
                    strokeWidth="2"
                    markerEnd="url(#velocity-arrow)"
                    opacity="0.7"
                  />

                  {/* Velocity magnitude label */}
                  <text
                    x={endX + 5}
                    y={endY + 10}
                    className="text-xs fill-green-600 font-medium"
                    textAnchor="start"
                  >
                    {velocityMagnitude.toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* Agent nodes */}
            {visualizationData.agentData.map(agent => {
              const scaledPos = scalePosition(agent.position);
              const hasForce = agent.forceVector && agent.forceVector.magnitude > 0.01;
              const hasVelocity = agent.velocity &&
                Math.sqrt(Math.pow(agent.velocity.x, 2) + Math.pow(agent.velocity.y, 2)) > 0.01;

              return (
                <g key={`agent-${agent.agentId}`}>
                  {/* Agent circle */}
                  <circle
                    cx={scaledPos.x}
                    cy={scaledPos.y}
                    r="6"
                    fill={hasForce || hasVelocity ? "#3b82f6" : "#6b7280"}
                    stroke="#ffffff"
                    strokeWidth="2"
                  />

                  {/* Agent label */}
                  <text
                    x={scaledPos.x}
                    y={scaledPos.y - 12}
                    textAnchor="middle"
                    className="text-xs fill-gray-700 font-medium"
                  >
                    {agent.agentId.slice(0, 6)}
                  </text>

                  {/* Force/velocity indicators */}
                  {(hasForce || hasVelocity) && (
                    <circle
                      cx={scaledPos.x}
                      cy={scaledPos.y}
                      r="10"
                      fill="none"
                      stroke="#3b82f6"
                      strokeWidth="1"
                      strokeDasharray="2,2"
                      opacity="0.5"
                    />
                  )}
                </g>
              );
            })}
          </svg>

          {/* Legend */}
          <div className="absolute top-2 right-2 bg-white bg-opacity-90 p-3 rounded-lg text-xs space-y-1">
            <div className="font-medium">Vectors</div>
            {showForces && (
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-red-500"></div>
                <span>Force</span>
              </div>
            )}
            {showVelocity && (
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-green-500"></div>
                <span>Velocity</span>
              </div>
            )}
            <div className="mt-2 font-medium">Agents</div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>Active</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
              <span>Static</span>
            </div>
          </div>
        </div>

        {/* Vector Statistics */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-lg font-bold text-blue-600">
              {visualizationData.agentData.length}
            </div>
            <div className="text-sm text-blue-700">Total Agents</div>
          </div>

          <div className="text-center p-3 bg-red-50 rounded-lg">
            <div className="text-lg font-bold text-red-600">
              {visualizationData.agentData.filter(a =>
                a.forceVector && a.forceVector.magnitude > 0.01
              ).length}
            </div>
            <div className="text-sm text-red-700">With Forces</div>
          </div>

          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-lg font-bold text-green-600">
              {visualizationData.agentData.filter(a =>
                a.velocity && Math.sqrt(Math.pow(a.velocity.x, 2) + Math.pow(a.velocity.y, 2)) > 0.01
              ).length}
            </div>
            <div className="text-sm text-green-700">Moving</div>
          </div>

          <div className="text-center p-3 bg-purple-50 rounded-lg">
            <div className="text-lg font-bold text-purple-600">
              {(() => {
                const totalForce = visualizationData.agentData.reduce((sum, agent) =>
                  sum + (agent.forceVector?.magnitude || 0), 0
                );
                return (totalForce / Math.max(visualizationData.agentData.length, 1)).toFixed(2);
              })()}
            </div>
            <div className="text-sm text-purple-700">Avg Force</div>
          </div>
        </div>

        {/* Detailed Agent List */}
        <div className="mt-6">
          <h4 className="font-medium mb-3">Agent Details</h4>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {visualizationData.agentData.slice(0, 10).map(agent => (
              <div key={agent.agentId} className="flex items-center justify-between p-2 bg-gray-50 rounded text-sm">
                <div>
                  <span className="font-medium">{agent.agentId}</span>
                  <span className="ml-2 text-gray-600">
                    ({agent.position.x.toFixed(2)}, {agent.position.y.toFixed(2)})
                  </span>
                </div>
                <div className="flex gap-3 text-xs">
                  {agent.forceVector && (
                    <span className="text-red-600">
                      F: {agent.forceVector.magnitude.toFixed(2)}
                    </span>
                  )}
                  {agent.velocity && (
                    <span className="text-green-600">
                      V: {Math.sqrt(
                        Math.pow(agent.velocity.x, 2) + Math.pow(agent.velocity.y, 2)
                      ).toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            ))}
            {visualizationData.agentData.length > 10 && (
              <div className="text-xs text-gray-500 text-center">
                +{visualizationData.agentData.length - 10} more agents
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};