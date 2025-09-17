import React, { useMemo, useState } from 'react';
import { Card } from '../../design-system/components/Card';
import { Select } from '../../design-system/components/Select';
import { Switch } from '../../design-system/components/Switch';
import { Label } from '../../design-system/components/Label';
import { createLogger } from '../../../utils/logger';
import { AgentCommunication } from '../types';

const logger = createLogger('MessageFlowDebugger');

interface MessageFlowDebuggerProps {
  communications: AgentCommunication[];
  selectedAgentId?: string | null;
  onAgentSelect?: (agentId: string | null) => void;
  className?: string;
}

export const MessageFlowDebugger: React.FC<MessageFlowDebuggerProps> = ({
  communications,
  selectedAgentId,
  onAgentSelect,
  className = ''
}) => {
  const [timeWindow, setTimeWindow] = useState<'1m' | '5m' | '15m' | 'all'>('5m');
  const [showOnlyErrors, setShowOnlyErrors] = useState(false);
  const [messageTypeFilter, setMessageTypeFilter] = useState<string>('all');

  const { filteredCommunications, messageTypes, agents, flowStats } = useMemo(() => {
    let filtered = [...communications];

    // Apply time window filter
    if (timeWindow !== 'all') {
      const windowMs = timeWindow === '1m' ? 60000 : timeWindow === '5m' ? 300000 : 900000;
      const cutoffTime = new Date(Date.now() - windowMs);
      filtered = filtered.filter(comm => comm.timestamp >= cutoffTime);
    }

    // Apply error filter
    if (showOnlyErrors) {
      filtered = filtered.filter(comm => !comm.success);
    }

    // Apply agent filter
    if (selectedAgentId) {
      filtered = filtered.filter(comm =>
        comm.fromAgentId === selectedAgentId || comm.toAgentId === selectedAgentId
      );
    }

    // Apply message type filter
    if (messageTypeFilter !== 'all') {
      filtered = filtered.filter(comm => comm.messageType === messageTypeFilter);
    }

    // Get unique message types and agents
    const messageTypeSet = new Set<string>();
    const agentSet = new Set<string>();
    communications.forEach(comm => {
      messageTypeSet.add(comm.messageType);
      agentSet.add(comm.fromAgentId);
      agentSet.add(comm.toAgentId);
    });

    // Calculate flow statistics
    const totalMessages = filtered.length;
    const errorMessages = filtered.filter(c => !c.success).length;
    const avgLatency = filtered.length > 0
      ? filtered.reduce((sum, c) => sum + (c.latency || 0), 0) / filtered.length
      : 0;

    const messagesByType = filtered.reduce((acc, comm) => {
      acc[comm.messageType] = (acc[comm.messageType] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      filteredCommunications: filtered.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()),
      messageTypes: Array.from(messageTypeSet).sort(),
      agents: Array.from(agentSet).sort(),
      flowStats: {
        totalMessages,
        errorMessages,
        avgLatency,
        messagesByType
      }
    };
  }, [communications, timeWindow, showOnlyErrors, selectedAgentId, messageTypeFilter]);

  const getMessageTypeColor = (messageType: string) => {
    const colors = [
      '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
      '#ec4899', '#14b8a6', '#f97316', '#84cc16', '#6366f1'
    ];
    const index = messageType.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return colors[index % colors.length];
  };

  const formatLatency = (latency?: number) => {
    if (!latency) return 'N/A';
    return `${latency.toFixed(0)}ms`;
  };

  return (
    <Card className={className}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Message Flow Debugger</h3>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2">
              <Label htmlFor="errors-only" className="text-sm">Errors Only</Label>
              <Switch
                id="errors-only"
                checked={showOnlyErrors}
                onCheckedChange={setShowOnlyErrors}
                size="sm"
              />
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
          <Select value={timeWindow} onValueChange={(value: any) => setTimeWindow(value)}>
            <Select.Trigger>
              <Select.Value />
            </Select.Trigger>
            <Select.Content>
              <Select.Item value="1m">Last 1 minute</Select.Item>
              <Select.Item value="5m">Last 5 minutes</Select.Item>
              <Select.Item value="15m">Last 15 minutes</Select.Item>
              <Select.Item value="all">All time</Select.Item>
            </Select.Content>
          </Select>

          <Select value={messageTypeFilter} onValueChange={setMessageTypeFilter}>
            <Select.Trigger>
              <Select.Value placeholder="Message type" />
            </Select.Trigger>
            <Select.Content>
              <Select.Item value="all">All Types</Select.Item>
              {messageTypes.map(type => (
                <Select.Item key={type} value={type}>
                  {type}
                </Select.Item>
              ))}
            </Select.Content>
          </Select>

          <Select
            value={selectedAgentId || 'all'}
            onValueChange={(value) => onAgentSelect?.(value === 'all' ? null : value)}
          >
            <Select.Trigger>
              <Select.Value placeholder="Agent filter" />
            </Select.Trigger>
            <Select.Content>
              <Select.Item value="all">All Agents</Select.Item>
              {agents.map(agent => (
                <Select.Item key={agent} value={agent}>
                  {agent}
                </Select.Item>
              ))}
            </Select.Content>
          </Select>
        </div>

        {/* Flow Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {flowStats.totalMessages}
            </div>
            <div className="text-sm text-blue-700">Total Messages</div>
          </div>
          <div className="text-center p-3 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">
              {flowStats.errorMessages}
            </div>
            <div className="text-sm text-red-700">Errors</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {(((flowStats.totalMessages - flowStats.errorMessages) / Math.max(flowStats.totalMessages, 1)) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-green-700">Success Rate</div>
          </div>
          <div className="text-center p-3 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">
              {flowStats.avgLatency.toFixed(0)}ms
            </div>
            <div className="text-sm text-purple-700">Avg Latency</div>
          </div>
        </div>

        {/* Message Type Distribution */}
        <div className="mb-6">
          <h4 className="font-medium mb-3">Message Type Distribution</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(flowStats.messagesByType)
              .sort(([,a], [,b]) => b - a)
              .slice(0, 8)
              .map(([type, count]) => (
                <div
                  key={type}
                  className="flex items-center gap-2 px-3 py-1 rounded-full text-sm"
                  style={{
                    backgroundColor: `${getMessageTypeColor(type)}20`,
                    color: getMessageTypeColor(type)
                  }}
                >
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: getMessageTypeColor(type) }}
                  />
                  <span className="font-medium">{type}</span>
                  <span className="bg-white bg-opacity-60 px-1.5 py-0.5 rounded-full text-xs">
                    {count}
                  </span>
                </div>
              ))}
          </div>
        </div>

        {/* Message Flow Visualization */}
        <div className="mb-6">
          <h4 className="font-medium mb-3">Recent Message Flow</h4>
          <div className="relative">
            <svg width="100%" height="200" viewBox="0 0 600 160" className="border rounded-lg bg-gray-50">
              {/* Timeline */}
              <line x1="50" y1="140" x2="550" y2="140" stroke="#d1d5db" strokeWidth="1" />

              {/* Messages */}
              {filteredCommunications.slice(0, 20).map((comm, index) => {
                const x = 50 + (index / 19) * 500;
                const y = 20 + (comm.fromAgentId.charCodeAt(0) % 100);
                const isError = !comm.success;

                return (
                  <g key={comm.id}>
                    {/* Message dot */}
                    <circle
                      cx={x}
                      cy={y}
                      r={isError ? "4" : "3"}
                      fill={isError ? '#ef4444' : getMessageTypeColor(comm.messageType)}
                      stroke="#ffffff"
                      strokeWidth="1"
                    />

                    {/* Connection line */}
                    <line
                      x1={x}
                      y1={y}
                      x2={x}
                      y2="140"
                      stroke={isError ? '#ef4444' : '#9ca3af'}
                      strokeWidth="1"
                      strokeDasharray={isError ? "3,3" : "none"}
                      opacity="0.5"
                    />

                    {/* Timestamp marker */}
                    <text
                      x={x}
                      y="155"
                      textAnchor="middle"
                      className="text-xs fill-gray-500"
                    >
                      {comm.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit'
                      })}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Message List */}
        <div>
          <h4 className="font-medium mb-3">Message Details</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {filteredCommunications.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                No messages match the current filters
              </div>
            ) : (
              filteredCommunications.slice(0, 50).map(comm => (
                <div
                  key={comm.id}
                  className={`p-3 rounded-lg border ${
                    comm.success
                      ? 'bg-white border-gray-200'
                      : 'bg-red-50 border-red-200'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getMessageTypeColor(comm.messageType) }}
                      />
                      <div>
                        <div className="font-medium text-sm">
                          {comm.fromAgentId} → {comm.toAgentId}
                        </div>
                        <div className="text-xs text-gray-600">
                          {comm.messageType} • {comm.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className={`text-sm font-medium ${
                        comm.success ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {comm.success ? '✓' : '✗'}
                      </div>
                      <div className="text-xs text-gray-500">
                        {formatLatency(comm.latency)}
                      </div>
                    </div>
                  </div>

                  {/* Error details */}
                  {!comm.success && comm.metadata && (
                    <div className="mt-2 p-2 bg-red-100 rounded text-xs">
                      <details>
                        <summary className="cursor-pointer text-red-700">
                          Error Details
                        </summary>
                        <pre className="mt-1 overflow-x-auto text-red-600">
                          {JSON.stringify(comm.metadata, null, 2)}
                        </pre>
                      </details>
                    </div>
                  )}

                  {/* Latency warning */}
                  {comm.latency && comm.latency > 1000 && (
                    <div className="mt-2 text-xs text-orange-600 bg-orange-50 p-1 rounded">
                      ⚠️ High latency: {formatLatency(comm.latency)}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Summary */}
        <div className="mt-6 pt-4 border-t border-gray-200 text-sm text-gray-600">
          Showing {Math.min(filteredCommunications.length, 50)} of {filteredCommunications.length} messages
        </div>
      </div>
    </Card>
  );
};