import React, { useState, useEffect } from 'react';
import { useBotsData } from '../contexts/BotsDataContext';
import { Card } from '../../design-system/components/Card';
import { Select } from '../../design-system/components/Select';
import type { BotsAgent } from '../types/BotsTypes';
import { createLogger } from '../../../utils/logger';

const logger = createLogger('AgentDetailPanel');

interface AgentDetailPanelProps {
  className?: string;
  selectedAgentId?: string;
  onAgentSelect?: (agentId: string) => void;
}

export const AgentDetailPanel: React.FC<AgentDetailPanelProps> = ({ 
  className,
  selectedAgentId,
  onAgentSelect
}) => {
  const { botsData } = useBotsData();
  const [selectedAgent, setSelectedAgent] = useState<BotsAgent | null>(null);

  // Update selected agent when ID changes or data updates
  useEffect(() => {
    if (!botsData || !botsData.agents) {
      setSelectedAgent(null);
      return;
    }

    if (selectedAgentId) {
      const agent = botsData.agents.find(a => a.id === selectedAgentId);
      setSelectedAgent(agent || null);
    } else if (botsData.agents.length > 0 && !selectedAgent) {
      // Auto-select first agent if none selected
      setSelectedAgent(botsData.agents[0]);
    }
  }, [botsData, selectedAgentId, selectedAgent]);

  const handleAgentChange = (value: string) => {
    const agent = botsData?.agents.find(a => a.id === value);
    setSelectedAgent(agent || null);
    if (onAgentSelect && agent) {
      onAgentSelect(agent.id);
    }
  };

  if (!botsData || !botsData.agents || botsData.agents.length === 0) {
    return (
      <Card className={className}>
        <div className="p-4">
          <h3 className="text-lg font-semibold mb-3">Agent Details</h3>
          <p className="text-sm text-gray-500">No agents available</p>
        </div>
      </Card>
    );
  }

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      active: 'bg-green-500',
      busy: 'bg-yellow-500',
      idle: 'bg-gray-500',
      error: 'bg-red-500',
      initializing: 'bg-blue-500',
      terminating: 'bg-purple-500',
      offline: 'bg-gray-800'
    };
    return colors[status] || 'bg-gray-500';
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  return (
    <Card className={className}>
      <div className="p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Agent Details</h3>
          <Select
            value={selectedAgent?.id || ''}
            onChange={handleAgentChange}
            className="w-48"
          >
            <option value="">Select Agent</option>
            {botsData.agents.map(agent => (
              <option key={agent.id} value={agent.id}>
                {agent.name || agent.id} ({agent.type})
              </option>
            ))}
          </Select>
        </div>

        {selectedAgent ? (
          <div className="space-y-4">
            {/* Agent Header */}
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${getStatusColor(selectedAgent.status)}`} />
              <div>
                <h4 className="font-semibold">{selectedAgent.name || selectedAgent.id}</h4>
                <p className="text-sm text-gray-600">
                  {selectedAgent.type} • {selectedAgent.status}
                </p>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 rounded p-3">
                <div className="text-xs text-gray-600">Health</div>
                <div className="text-lg font-semibold">{selectedAgent.health.toFixed(1)}%</div>
              </div>
              <div className="bg-gray-50 rounded p-3">
                <div className="text-xs text-gray-600">Success Rate</div>
                <div className="text-lg font-semibold">
                  {(selectedAgent.successRate || 0).toFixed(1)}%
                </div>
              </div>
              <div className="bg-gray-50 rounded p-3">
                <div className="text-xs text-gray-600">CPU Usage</div>
                <div className="text-lg font-semibold">{selectedAgent.cpuUsage.toFixed(1)}%</div>
              </div>
              <div className="bg-gray-50 rounded p-3">
                <div className="text-xs text-gray-600">Memory Usage</div>
                <div className="text-lg font-semibold">{selectedAgent.memoryUsage.toFixed(1)}%</div>
              </div>
            </div>

            {/* Task Information */}
            <div>
              <h5 className="font-semibold text-sm mb-2">Task Activity</h5>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Active Tasks:</span>
                  <span className="font-medium">{selectedAgent.tasksActive || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Completed Tasks:</span>
                  <span className="font-medium">{selectedAgent.tasksCompleted || 0}</span>
                </div>
                {selectedAgent.currentTask && (
                  <div className="mt-2 p-2 bg-blue-50 rounded">
                    <div className="text-xs text-blue-600 font-semibold">Current Task</div>
                    <div className="text-sm mt-1">{selectedAgent.currentTask}</div>
                  </div>
                )}
              </div>
            </div>

            {/* Capabilities */}
            {selectedAgent.capabilities && selectedAgent.capabilities.length > 0 && (
              <div>
                <h5 className="font-semibold text-sm mb-2">Capabilities</h5>
                <div className="flex flex-wrap gap-1">
                  {selectedAgent.capabilities.map((cap, index) => (
                    <span
                      key={index}
                      className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded"
                    >
                      {cap}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Token Usage */}
            {selectedAgent.tokens !== undefined && (
              <div>
                <h5 className="font-semibold text-sm mb-2">Token Usage</h5>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Tokens:</span>
                    <span className="font-medium">{selectedAgent.tokens.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Token Rate:</span>
                    <span className="font-medium">
                      {(selectedAgent.tokenRate || 0).toFixed(1)} tokens/min
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Swarm Information */}
            {selectedAgent.swarmId && (
              <div>
                <h5 className="font-semibold text-sm mb-2">Swarm Information</h5>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Swarm ID:</span>
                    <span className="font-mono text-xs">{selectedAgent.swarmId}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Mode:</span>
                    <span className="font-medium">{selectedAgent.agentMode || 'unknown'}</span>
                  </div>
                  {selectedAgent.parentQueenId && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Parent:</span>
                      <span className="font-mono text-xs">{selectedAgent.parentQueenId}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Agent Age */}
            <div className="pt-3 border-t border-gray-200 text-xs text-gray-500">
              Agent Age: {formatDuration(selectedAgent.age)}
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-500">Select an agent to view details</p>
        )}
      </div>
    </Card>
  );
};