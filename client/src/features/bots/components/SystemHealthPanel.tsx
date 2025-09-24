import React from 'react';
import { useBotsData } from '../contexts/BotsDataContext';
import { Card } from '../../design-system/components/Card';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('SystemHealthPanel');

interface SystemHealthPanelProps {
  className?: string;
}

export const SystemHealthPanel: React.FC<SystemHealthPanelProps> = ({ className }) => {
  const { botsData } = useBotsData();

  if (!botsData || !botsData.agents || botsData.agents.length === 0) {
    return (
      <Card className={className}>
        <div className="p-4">
          <h3 className="text-lg font-semibold mb-3">System Health</h3>
          <p className="text-sm text-gray-500">No agents connected</p>
        </div>
      </Card>
    );
  }

  // Calculate overall health metrics
  const totalAgents = botsData.agents.length;
  const activeAgents = botsData.agents.filter(a => a.status === 'active' || a.status === 'busy').length;
  const errorAgents = botsData.agents.filter(a => a.status === 'error').length;
  const avgHealth = botsData.agents.reduce((sum, a) => sum + (a.health || 0), 0) / totalAgents;
  const avgCpuUsage = botsData.agents.reduce((sum, a) => sum + (a.cpuUsage || 0), 0) / totalAgents;
  const avgMemoryUsage = botsData.agents.reduce((sum, a) => sum + (a.memoryUsage || 0), 0) / totalAgents;

  // Get multi-agent metrics if available
  const multiAgentMetrics = botsData.multiAgentMetrics;

  const getHealthColor = (value: number) => {
    if (value >= 80) return 'text-green-600';
    if (value >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getUsageColor = (value: number) => {
    if (value <= 50) return 'text-green-600';
    if (value <= 75) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <Card className={className}>
      <div className="p-4">
        <h3 className="text-lg font-semibold mb-4">System Health</h3>

        {/* Agent Status Summary */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{totalAgents}</div>
            <div className="text-xs text-gray-600">Total Agents</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{activeAgents}</div>
            <div className="text-xs text-gray-600">Active</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{errorAgents}</div>
            <div className="text-xs text-gray-600">Errors</div>
          </div>
        </div>

        {/* Health Metrics */}
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Overall Health</span>
              <span className={getHealthColor(avgHealth)}>{avgHealth.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-green-400 to-green-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${avgHealth}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>CPU Usage</span>
              <span className={getUsageColor(avgCpuUsage)}>{avgCpuUsage.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-blue-400 to-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${avgCpuUsage}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Memory Usage</span>
              <span className={getUsageColor(avgMemoryUsage)}>{avgMemoryUsage.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-purple-400 to-purple-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${avgMemoryUsage}%` }}
              />
            </div>
          </div>
        </div>

        {/* multi-agent Metrics */}
        {multiAgentMetrics && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <h4 className="text-sm font-semibold mb-2">multi-agent Performance</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-600">Success Rate:</span>
                <span className="ml-2 font-medium">{multiAgentMetrics.avgSuccessRate.toFixed(1)}%</span>
              </div>
              <div>
                <span className="text-gray-600">Active Tasks:</span>
                <span className="ml-2 font-medium">{multiAgentMetrics.totalTasks}</span>
              </div>
              <div>
                <span className="text-gray-600">Completed:</span>
                <span className="ml-2 font-medium">{multiAgentMetrics.completedTasks}</span>
              </div>
              <div>
                <span className="text-gray-600">Tokens Used:</span>
                <span className="ml-2 font-medium">{multiAgentMetrics.totalTokens.toLocaleString()}</span>
              </div>
            </div>
          </div>
        )}

        {/* Connection Status */}
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">MCP Connection</span>
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${botsData.mcpConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm font-medium">{botsData.mcpConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};