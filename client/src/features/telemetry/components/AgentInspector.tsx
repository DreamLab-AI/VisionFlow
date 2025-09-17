import React, { useMemo } from 'react';
import { Card } from '../../design-system/components/Card';
import { Select } from '../../design-system/components/Select';
import { Tabs } from '../../design-system/components/Tabs';
import { createLogger } from '../../../utils/logger';
import {
  AgentLifecycleEvent,
  AgentPosition,
  LogEntry,
  AgentCommunication,
  PerformanceMetrics
} from '../types';

const logger = createLogger('AgentInspector');

interface TelemetryData {
  lifecycleEvents: AgentLifecycleEvent[];
  agentPositions: Record<string, AgentPosition>;
  logEntries: LogEntry[];
  communications: AgentCommunication[];
  performanceMetrics: PerformanceMetrics[];
}

interface AgentInspectorProps {
  agentIds: string[];
  selectedAgentId: string | null;
  onAgentSelect: (agentId: string | null) => void;
  telemetryData: TelemetryData;
  className?: string;
}

export const AgentInspector: React.FC<AgentInspectorProps> = ({
  agentIds,
  selectedAgentId,
  onAgentSelect,
  telemetryData,
  className = ''
}) => {
  const agentDetails = useMemo(() => {
    if (!selectedAgentId) return null;

    const position = telemetryData.agentPositions[selectedAgentId];
    const events = telemetryData.lifecycleEvents
      .filter(e => e.agentId === selectedAgentId)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 10);

    const logs = telemetryData.logEntries
      .filter(l => l.agentId === selectedAgentId)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 20);

    const communications = telemetryData.communications
      .filter(c => c.fromAgentId === selectedAgentId || c.toAgentId === selectedAgentId)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 15);

    // Find agent performance metrics
    const latestMetrics = telemetryData.performanceMetrics[telemetryData.performanceMetrics.length - 1];
    const agentPerformance = latestMetrics?.agents.find(a => a.agentId === selectedAgentId);

    const currentStatus = events[0]?.eventType || 'unknown';

    return {
      agentId: selectedAgentId,
      status: currentStatus,
      position,
      events,
      logs,
      communications,
      performance: agentPerformance,
      errorCount: logs.filter(l => l.level === 'error' || l.level === 'critical').length,
      messagesSent: communications.filter(c => c.fromAgentId === selectedAgentId).length,
      messagesReceived: communications.filter(c => c.toAgentId === selectedAgentId).length
    };
  }, [selectedAgentId, telemetryData]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'spawn':
      case 'activate': return 'text-green-600 bg-green-50';
      case 'error': return 'text-red-600 bg-red-50';
      case 'idle': return 'text-gray-600 bg-gray-50';
      case 'complete': return 'text-blue-600 bg-blue-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <Card className={className}>
      <div className="p-4">
        <h3 className="text-lg font-semibold mb-4">Agent Inspector</h3>

        {/* Agent Selection */}
        <div className="mb-6">
          <Select
            value={selectedAgentId || ''}
            onValueChange={(value) => onAgentSelect(value || null)}
          >
            <Select.Trigger className="w-full">
              <Select.Value placeholder="Select an agent to inspect" />
            </Select.Trigger>
            <Select.Content>
              {agentIds.map(agentId => (
                <Select.Item key={agentId} value={agentId}>
                  {agentId}
                </Select.Item>
              ))}
            </Select.Content>
          </Select>
        </div>

        {!agentDetails ? (
          <div className="text-center text-gray-500 py-8">
            Select an agent to view detailed information
          </div>
        ) : (
          <div className="space-y-6">
            {/* Agent Status Overview */}
            <div className="grid grid-cols-2 gap-4">
              <div className={`p-3 rounded-lg ${getStatusColor(agentDetails.status)}`}>
                <div className="font-medium">Status</div>
                <div className="text-sm capitalize">{agentDetails.status}</div>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg">
                <div className="font-medium text-blue-800">Health</div>
                <div className="text-sm text-blue-700">
                  {agentDetails.performance?.health?.toFixed(1) || 'N/A'}%
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            {agentDetails.performance && (
              <div>
                <h4 className="font-medium mb-3">Performance Metrics</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-gray-600">CPU Usage:</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div
                          className="h-2 bg-red-500 rounded-full"
                          style={{ width: `${Math.min(agentDetails.performance.cpu, 100)}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs">
                        {agentDetails.performance.cpu.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600">Memory Usage:</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div
                          className="h-2 bg-blue-500 rounded-full"
                          style={{ width: `${Math.min(agentDetails.performance.memory, 100)}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs">
                        {agentDetails.performance.memory.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600">Active Tasks:</div>
                    <div className="font-mono">{agentDetails.performance.taskCount}</div>
                  </div>
                  <div>
                    <div className="text-gray-600">Health Score:</div>
                    <div className="font-mono">{agentDetails.performance.health.toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            )}

            {/* Position Information */}
            {agentDetails.position && (
              <div>
                <h4 className="font-medium mb-3">Position & Movement</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-gray-600">Position:</div>
                    <div className="font-mono">
                      ({agentDetails.position.position.x.toFixed(3)}, {agentDetails.position.position.y.toFixed(3)}
                      {agentDetails.position.position.z !== undefined && `, ${agentDetails.position.position.z.toFixed(3)}`})
                    </div>
                  </div>
                  {agentDetails.position.velocity && (
                    <div>
                      <div className="text-gray-600">Velocity:</div>
                      <div className="font-mono">
                        {Math.sqrt(
                          Math.pow(agentDetails.position.velocity.x, 2) +
                          Math.pow(agentDetails.position.velocity.y, 2)
                        ).toFixed(3)}
                      </div>
                    </div>
                  )}
                  <div className="col-span-2">
                    <div className="text-gray-600">Last Update:</div>
                    <div className="text-xs">{agentDetails.position.timestamp.toLocaleString()}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Communication Stats */}
            <div>
              <h4 className="font-medium mb-3">Communication Stats</h4>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <div className="text-xl font-bold text-green-600">
                    {agentDetails.messagesSent}
                  </div>
                  <div className="text-sm text-green-700">Sent</div>
                </div>
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <div className="text-xl font-bold text-blue-600">
                    {agentDetails.messagesReceived}
                  </div>
                  <div className="text-sm text-blue-700">Received</div>
                </div>
                <div className="text-center p-3 bg-red-50 rounded-lg">
                  <div className="text-xl font-bold text-red-600">
                    {agentDetails.errorCount}
                  </div>
                  <div className="text-sm text-red-700">Errors</div>
                </div>
              </div>
            </div>

            {/* Detailed Information Tabs */}
            <Tabs defaultValue="events" className="w-full">
              <Tabs.List className="grid w-full grid-cols-3">
                <Tabs.Trigger value="events">Events</Tabs.Trigger>
                <Tabs.Trigger value="communications">Messages</Tabs.Trigger>
                <Tabs.Trigger value="logs">Logs</Tabs.Trigger>
              </Tabs.List>

              <Tabs.Content value="events" className="mt-4">
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {agentDetails.events.length === 0 ? (
                    <div className="text-center text-gray-500 py-4">
                      No events recorded
                    </div>
                  ) : (
                    agentDetails.events.map(event => (
                      <div
                        key={event.id}
                        className={`p-2 rounded border ${getStatusColor(event.eventType)}`}
                      >
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="font-medium text-sm capitalize">
                              {event.eventType}
                            </div>
                            <div className="text-xs opacity-75">
                              {event.timestamp.toLocaleString()}
                            </div>
                          </div>
                          {event.metadata?.performance && (
                            <div className="text-xs">
                              CPU: {event.metadata.performance.cpu.toFixed(1)}%
                            </div>
                          )}
                        </div>
                        {Object.keys(event.details).length > 0 && (
                          <details className="mt-1">
                            <summary className="text-xs cursor-pointer">Details</summary>
                            <pre className="text-xs mt-1 overflow-x-auto">
                              {JSON.stringify(event.details, null, 2)}
                            </pre>
                          </details>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </Tabs.Content>

              <Tabs.Content value="communications" className="mt-4">
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {agentDetails.communications.length === 0 ? (
                    <div className="text-center text-gray-500 py-4">
                      No communications recorded
                    </div>
                  ) : (
                    agentDetails.communications.map(comm => (
                      <div
                        key={comm.id}
                        className="p-2 bg-gray-50 rounded border"
                      >
                        <div className="flex justify-between items-start">
                          <div className="text-sm">
                            <div className="font-medium">
                              {comm.fromAgentId === selectedAgentId ? '→' : '←'} {comm.messageType}
                            </div>
                            <div className="text-xs text-gray-600">
                              {comm.fromAgentId === selectedAgentId ? `To: ${comm.toAgentId}` : `From: ${comm.fromAgentId}`}
                            </div>
                            <div className="text-xs text-gray-500">
                              {comm.timestamp.toLocaleString()}
                            </div>
                          </div>
                          <div className="text-right text-xs">
                            <div className={comm.success ? 'text-green-600' : 'text-red-600'}>
                              {comm.success ? '✓' : '✗'}
                            </div>
                            {comm.latency && (
                              <div className="text-gray-500">
                                {comm.latency.toFixed(0)}ms
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </Tabs.Content>

              <Tabs.Content value="logs" className="mt-4">
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {agentDetails.logs.length === 0 ? (
                    <div className="text-center text-gray-500 py-4">
                      No logs recorded
                    </div>
                  ) : (
                    agentDetails.logs.map(log => (
                      <div
                        key={log.id}
                        className={`p-2 rounded border text-sm ${
                          log.level === 'error' || log.level === 'critical'
                            ? 'bg-red-50 border-red-200'
                            : log.level === 'warn'
                            ? 'bg-yellow-50 border-yellow-200'
                            : 'bg-blue-50 border-blue-200'
                        }`}
                      >
                        <div className="flex justify-between items-start mb-1">
                          <span className={`text-xs px-1.5 py-0.5 rounded uppercase font-medium ${
                            log.level === 'error' || log.level === 'critical'
                              ? 'bg-red-200 text-red-800'
                              : log.level === 'warn'
                              ? 'bg-yellow-200 text-yellow-800'
                              : 'bg-blue-200 text-blue-800'
                          }`}>
                            {log.level}
                          </span>
                          <span className="text-xs text-gray-500">
                            {log.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <div className="text-sm break-words">{log.message}</div>
                        {log.tags && log.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {log.tags.map(tag => (
                              <span key={tag} className="text-xs bg-white px-1 py-0.5 rounded">
                                #{tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </Tabs.Content>
            </Tabs>
          </div>
        )}
      </div>
    </Card>
  );
};