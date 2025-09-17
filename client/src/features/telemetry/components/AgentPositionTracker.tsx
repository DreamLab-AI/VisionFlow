import React, { useMemo, useState } from 'react';
import { Card } from '../../design-system/components/Card';
import { Select } from '../../design-system/components/Select';
import { Switch } from '../../design-system/components/Switch';
import { Label } from '../../design-system/components/Label';
import { createLogger } from '../../../utils/logger';
import { AgentPosition } from '../types';

const logger = createLogger('AgentPositionTracker');

interface AgentPositionTrackerProps {
  positions: Record<string, AgentPosition>;
  selectedAgentId?: string | null;
  onAgentSelect?: (agentId: string | null) => void;
  className?: string;
}

export const AgentPositionTracker: React.FC<AgentPositionTrackerProps> = ({
  positions,
  selectedAgentId,
  onAgentSelect,
  className = ''
}) => {
  const [showTrails, setShowTrails] = useState(true);
  const [showVectors, setShowVectors] = useState(false);

  // Calculate visualization bounds
  const bounds = useMemo(() => {
    const agentPositions = Object.values(positions);
    if (agentPositions.length === 0) {
      return { minX: -10, maxX: 10, minY: -10, maxY: 10 };
    }

    const xs = agentPositions.map(p => p.position.x);
    const ys = agentPositions.map(p => p.position.y);

    const padding = 2;
    return {
      minX: Math.min(...xs) - padding,
      maxX: Math.max(...xs) + padding,
      minY: Math.min(...ys) - padding,
      maxY: Math.max(...ys) + padding
    };
  }, [positions]);

  const viewBoxWidth = bounds.maxX - bounds.minX;
  const viewBoxHeight = bounds.maxY - bounds.minY;

  // Scale positions to SVG coordinates
  const scalePosition = (pos: { x: number; y: number }) => ({
    x: ((pos.x - bounds.minX) / viewBoxWidth) * 400,
    y: ((pos.y - bounds.minY) / viewBoxHeight) * 300
  });

  const agentIds = Object.keys(positions);

  return (
    <Card className={className}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Agent Position Tracker</h3>
          <div className="flex items-center gap-2">
            <Label htmlFor="show-trails" className="text-sm">Trails</Label>
            <Switch
              id="show-trails"
              checked={showTrails}
              onCheckedChange={setShowTrails}
              size="sm"
            />
            <Label htmlFor="show-vectors" className="text-sm">Vectors</Label>
            <Switch
              id="show-vectors"
              checked={showVectors}
              onCheckedChange={setShowVectors}
              size="sm"
            />
          </div>
        </div>

        {/* Agent Selection */}
        <div className="mb-4">
          <Select
            value={selectedAgentId || 'all'}
            onValueChange={(value) => onAgentSelect?.(value === 'all' ? null : value)}
          >
            <Select.Trigger className="w-full">
              <Select.Value placeholder="Select agent to track" />
            </Select.Trigger>
            <Select.Content>
              <Select.Item value="all">All Agents</Select.Item>
              {agentIds.map(agentId => (
                <Select.Item key={agentId} value={agentId}>
                  {positions[agentId]?.agentId || agentId}
                </Select.Item>
              ))}
            </Select.Content>
          </Select>
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
              <pattern id="grid" width="40" height="30" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 30" fill="none" stroke="#e5e7eb" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />

            {/* Trails */}
            {showTrails && Object.entries(positions).map(([agentId, position]) => {
              if (selectedAgentId && selectedAgentId !== agentId) return null;

              const scaledPos = scalePosition(position.position);

              // Generate a simple trail effect (could be enhanced with position history)
              return (
                <g key={`trail-${agentId}`}>
                  <circle
                    cx={scaledPos.x}
                    cy={scaledPos.y}
                    r="8"
                    fill="rgba(59, 130, 246, 0.2)"
                    className="animate-pulse"
                  />
                  <circle
                    cx={scaledPos.x}
                    cy={scaledPos.y}
                    r="12"
                    fill="rgba(59, 130, 246, 0.1)"
                    className="animate-ping"
                  />
                </g>
              );
            })}

            {/* Force Vectors */}
            {showVectors && Object.entries(positions).map(([agentId, position]) => {
              if (selectedAgentId && selectedAgentId !== agentId) return null;
              if (!position.forceVector) return null;

              const scaledPos = scalePosition(position.position);
              const vectorLength = Math.min(position.forceVector.magnitude * 20, 50);
              const angle = Math.atan2(position.forceVector.y, position.forceVector.x);

              const endX = scaledPos.x + Math.cos(angle) * vectorLength;
              const endY = scaledPos.y + Math.sin(angle) * vectorLength;

              return (
                <g key={`vector-${agentId}`}>
                  <line
                    x1={scaledPos.x}
                    y1={scaledPos.y}
                    x2={endX}
                    y2={endY}
                    stroke="#ef4444"
                    strokeWidth="2"
                    markerEnd="url(#arrowhead)"
                  />
                </g>
              );
            })}

            {/* Arrow marker for vectors */}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
              </marker>
            </defs>

            {/* Agent Positions */}
            {Object.entries(positions).map(([agentId, position]) => {
              if (selectedAgentId && selectedAgentId !== agentId) return null;

              const scaledPos = scalePosition(position.position);
              const isSelected = selectedAgentId === agentId;

              return (
                <g key={agentId}>
                  <circle
                    cx={scaledPos.x}
                    cy={scaledPos.y}
                    r={isSelected ? "6" : "4"}
                    fill={isSelected ? "#3b82f6" : "#6b7280"}
                    stroke={isSelected ? "#1d4ed8" : "#374151"}
                    strokeWidth="2"
                    className="cursor-pointer transition-all hover:r-6"
                    onClick={() => onAgentSelect?.(agentId)}
                  />

                  {/* Agent Label */}
                  <text
                    x={scaledPos.x}
                    y={scaledPos.y - 10}
                    textAnchor="middle"
                    className="text-xs fill-gray-600 font-medium pointer-events-none"
                  >
                    {agentId.slice(0, 8)}
                  </text>

                  {/* Velocity indicator */}
                  {position.velocity && (
                    <g>
                      <line
                        x1={scaledPos.x}
                        y1={scaledPos.y}
                        x2={scaledPos.x + position.velocity.x * 10}
                        y2={scaledPos.y + position.velocity.y * 10}
                        stroke="#10b981"
                        strokeWidth="1"
                        opacity="0.6"
                      />
                    </g>
                  )}
                </g>
              );
            })}
          </svg>

          {/* Legend */}
          <div className="absolute top-2 right-2 bg-white bg-opacity-90 p-2 rounded-lg text-xs">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-gray-600 rounded-full"></div>
                <span>Agent</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-600 rounded-full"></div>
                <span>Selected</span>
              </div>
              {showVectors && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-red-500"></div>
                  <span>Force</span>
                </div>
              )}
              {position.velocity && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-green-500"></div>
                  <span>Velocity</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Position Details */}
        {selectedAgentId && positions[selectedAgentId] && (
          <div className="mt-4 p-3 bg-gray-50 rounded-lg">
            <h4 className="font-medium mb-2">Position Details</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-600">X:</span>
                <span className="ml-2 font-mono">{positions[selectedAgentId].position.x.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-gray-600">Y:</span>
                <span className="ml-2 font-mono">{positions[selectedAgentId].position.y.toFixed(3)}</span>
              </div>
              {positions[selectedAgentId].position.z !== undefined && (
                <div>
                  <span className="text-gray-600">Z:</span>
                  <span className="ml-2 font-mono">{positions[selectedAgentId].position.z.toFixed(3)}</span>
                </div>
              )}
              {positions[selectedAgentId].velocity && (
                <>
                  <div>
                    <span className="text-gray-600">Velocity:</span>
                    <span className="ml-2 font-mono">
                      {Math.sqrt(
                        Math.pow(positions[selectedAgentId].velocity.x, 2) +
                        Math.pow(positions[selectedAgentId].velocity.y, 2)
                      ).toFixed(3)}
                    </span>
                  </div>
                </>
              )}
              <div className="col-span-2">
                <span className="text-gray-600">Last Update:</span>
                <span className="ml-2">{positions[selectedAgentId].timestamp.toLocaleString()}</span>
              </div>
            </div>
          </div>
        )}

        {/* Status */}
        <div className="mt-4 text-sm text-gray-500 text-center">
          Tracking {agentIds.length} agent{agentIds.length !== 1 ? 's' : ''}
        </div>
      </div>
    </Card>
  );
};