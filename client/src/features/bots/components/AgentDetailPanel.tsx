import React, { useState, useEffect } from 'react';
import { useBotsData } from '../contexts/BotsDataContext';
import { Card } from '../../design-system/components/Card';
import { Select } from '../../design-system/components/Select';
import { Button } from '../../design-system/components/Button';
import { Input } from '../../design-system/components/Input';
import type { BotsAgent } from '../types/BotsTypes';
import { createLogger } from '../../../utils/loggerConfig';
import { apiService } from '../../../services/apiService';

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
  const [taskDescription, setTaskDescription] = useState('');
  const [taskStatus, setTaskStatus] = useState<{
    id?: string;
    status: 'idle' | 'submitting' | 'running' | 'completed' | 'error';
    message?: string;
    progress?: number;
  }>({ status: 'idle' });
  const [taskPriority, setTaskPriority] = useState<'low' | 'medium' | 'high' | 'critical'>('medium');

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
            onValueChange={handleAgentChange}
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

            {/* multi-agent Information */}
            {selectedAgent.multi-agentId && (
              <div>
                <h5 className="font-semibold text-sm mb-2">multi-agent Information</h5>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">multi-agent ID:</span>
                    <span className="font-mono text-xs">{selectedAgent.multi-agentId}</span>
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

            {/* Task Submission Section */}
            <div className="pt-3 border-t border-gray-200">
              <h5 className="font-semibold text-sm mb-2">Submit Task to Swarm</h5>
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-gray-600">Task Description</label>
                  <textarea
                    className="w-full mt-1 p-2 text-sm border rounded resize-none"
                    rows={3}
                    placeholder="Enter task description..."
                    value={taskDescription}
                    onChange={(e) => setTaskDescription(e.target.value)}
                    disabled={taskStatus.status === 'submitting' || taskStatus.status === 'running'}
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-600">Priority</label>
                  <select
                    className="w-full mt-1 p-2 text-sm border rounded"
                    value={taskPriority}
                    onChange={(e) => setTaskPriority(e.target.value as 'low' | 'medium' | 'high' | 'critical')}
                    disabled={taskStatus.status === 'submitting' || taskStatus.status === 'running'}
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
                <Button
                  className="w-full"
                  variant={taskStatus.status === 'error' ? 'destructive' : 'default'}
                  disabled={!taskDescription.trim() || taskStatus.status === 'submitting' || taskStatus.status === 'running'}
                  onClick={async () => {
                    try {
                      setTaskStatus({ status: 'submitting', message: 'Submitting task to swarm...' });

                      const response = await apiService.post('/bots/submit-task', {
                        task: taskDescription,
                        priority: taskPriority,
                        strategy: 'adaptive',
                        swarmId: selectedAgent.swarmId || 'default'
                      });

                      if (response.taskId) {
                        setTaskStatus({
                          id: response.taskId,
                          status: 'running',
                          message: `Task ${response.taskId} submitted successfully`,
                          progress: 0
                        });

                        // Start polling for task status
                        const pollInterval = setInterval(async () => {
                          try {
                            const statusRes = await apiService.get(`/bots/task-status/${response.taskId}`);

                            if (statusRes.status === 'completed') {
                              setTaskStatus({
                                id: response.taskId,
                                status: 'completed',
                                message: 'Task completed successfully',
                                progress: 100
                              });
                              clearInterval(pollInterval);
                              // Clear task input after completion
                              setTimeout(() => {
                                setTaskDescription('');
                                setTaskStatus({ status: 'idle' });
                              }, 3000);
                            } else if (statusRes.status === 'error' || statusRes.status === 'failed') {
                              setTaskStatus({
                                id: response.taskId,
                                status: 'error',
                                message: statusRes.error || 'Task failed',
                                progress: statusRes.progress || 0
                              });
                              clearInterval(pollInterval);
                            } else {
                              setTaskStatus({
                                id: response.taskId,
                                status: 'running',
                                message: statusRes.message || 'Task in progress...',
                                progress: statusRes.progress || 0
                              });
                            }
                          } catch (error) {
                            logger.error('Failed to poll task status:', error);
                            clearInterval(pollInterval);
                          }
                        }, 2000); // Poll every 2 seconds

                        // Stop polling after 5 minutes
                        setTimeout(() => clearInterval(pollInterval), 300000);
                      } else {
                        setTaskStatus({
                          status: 'error',
                          message: response.error || 'Failed to submit task'
                        });
                      }
                    } catch (error) {
                      logger.error('Failed to submit task:', error);
                      setTaskStatus({
                        status: 'error',
                        message: 'Failed to submit task to swarm'
                      });
                    }
                  }}
                >
                  {taskStatus.status === 'submitting' ? 'Submitting...' :
                   taskStatus.status === 'running' ? `Running... ${taskStatus.progress || 0}%` :
                   taskStatus.status === 'completed' ? 'Completed!' :
                   taskStatus.status === 'error' ? 'Retry Task' :
                   'Submit Task'}
                </Button>

                {/* Task Status Display */}
                {taskStatus.message && (
                  <div className={`p-2 rounded text-xs ${
                    taskStatus.status === 'error' ? 'bg-red-50 text-red-600' :
                    taskStatus.status === 'completed' ? 'bg-green-50 text-green-600' :
                    taskStatus.status === 'running' || taskStatus.status === 'submitting' ? 'bg-blue-50 text-blue-600' :
                    'bg-gray-50 text-gray-600'
                  }`}>
                    {taskStatus.message}
                    {taskStatus.id && (
                      <div className="font-mono text-xs opacity-75 mt-1">
                        Task ID: {taskStatus.id}
                      </div>
                    )}
                    {taskStatus.status === 'running' && taskStatus.progress !== undefined && (
                      <div className="mt-2">
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div
                            className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${taskStatus.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

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