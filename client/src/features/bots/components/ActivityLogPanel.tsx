import React, { useState, useEffect, useRef } from 'react';
import { useBotsData } from '../contexts/BotsDataContext';
import { Card } from '../../design-system/components/Card';
import { createLogger } from '../../../utils/logger';

const logger = createLogger('ActivityLogPanel');

interface ActivityLogEntry {
  id: string;
  timestamp: Date;
  agentId: string;
  agentName: string;
  agentType: string;
  message: string;
  level: 'info' | 'warning' | 'error' | 'success';
}

interface ActivityLogPanelProps {
  className?: string;
  maxEntries?: number;
}

export const ActivityLogPanel: React.FC<ActivityLogPanelProps> = ({ 
  className,
  maxEntries = 100 
}) => {
  const { botsData } = useBotsData();
  const [logEntries, setLogEntries] = useState<ActivityLogEntry[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const lastUpdateRef = useRef<string | undefined>();

  // Generate activity log entries from agent data
  useEffect(() => {
    if (!botsData || !botsData.agents) return;

    // Check if this is a new update
    if (botsData.lastUpdate === lastUpdateRef.current) return;
    lastUpdateRef.current = botsData.lastUpdate;

    const newEntries: ActivityLogEntry[] = [];

    // Process each agent's status and logs
    botsData.agents.forEach(agent => {
      // Add status change entries
      if (agent.status === 'active' || agent.status === 'busy') {
        newEntries.push({
          id: `${agent.id}-status-${Date.now()}`,
          timestamp: new Date(),
          agentId: agent.id,
          agentName: agent.name || agent.id,
          agentType: agent.type,
          message: `Agent is ${agent.status}${agent.currentTask ? `: ${agent.currentTask}` : ''}`,
          level: 'info'
        });
      } else if (agent.status === 'error') {
        newEntries.push({
          id: `${agent.id}-error-${Date.now()}`,
          timestamp: new Date(),
          agentId: agent.id,
          agentName: agent.name || agent.id,
          agentType: agent.type,
          message: 'Agent encountered an error',
          level: 'error'
        });
      }

      // Add processing logs if available
      if (agent.processingLogs && agent.processingLogs.length > 0) {
        agent.processingLogs.slice(-3).forEach((log, index) => {
          newEntries.push({
            id: `${agent.id}-log-${Date.now()}-${index}`,
            timestamp: new Date(),
            agentId: agent.id,
            agentName: agent.name || agent.id,
            agentType: agent.type,
            message: log,
            level: 'info'
          });
        });
      }

      // Add performance alerts
      if (agent.health < 50) {
        newEntries.push({
          id: `${agent.id}-health-${Date.now()}`,
          timestamp: new Date(),
          agentId: agent.id,
          agentName: agent.name || agent.id,
          agentType: agent.type,
          message: `Low health: ${agent.health.toFixed(1)}%`,
          level: 'warning'
        });
      }

      if (agent.cpuUsage > 80) {
        newEntries.push({
          id: `${agent.id}-cpu-${Date.now()}`,
          timestamp: new Date(),
          agentId: agent.id,
          agentName: agent.name || agent.id,
          agentType: agent.type,
          message: `High CPU usage: ${agent.cpuUsage.toFixed(1)}%`,
          level: 'warning'
        });
      }
    });

    // Add new entries and maintain max entries limit
    setLogEntries(prev => {
      const combined = [...prev, ...newEntries];
      return combined.slice(-maxEntries);
    });
  }, [botsData, maxEntries]);

  // Auto-scroll to bottom when new entries are added
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logEntries, autoScroll]);

  const getLevelColor = (level: ActivityLogEntry['level']) => {
    switch (level) {
      case 'error': return 'text-red-600 bg-red-50';
      case 'warning': return 'text-yellow-600 bg-yellow-50';
      case 'success': return 'text-green-600 bg-green-50';
      default: return 'text-blue-600 bg-blue-50';
    }
  };

  const getLevelIcon = (level: ActivityLogEntry['level']) => {
    switch (level) {
      case 'error': return '❌';
      case 'warning': return '⚠️';
      case 'success': return '✅';
      default: return 'ℹ️';
    }
  };

  const clearLog = () => {
    setLogEntries([]);
  };

  return (
    <Card className={className}>
      <div className="p-4 h-full flex flex-col">
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-lg font-semibold">Activity Log</h3>
          <div className="flex items-center gap-2">
            <label className="flex items-center text-sm">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="mr-1"
              />
              Auto-scroll
            </label>
            <button
              onClick={clearLog}
              className="text-xs px-2 py-1 bg-gray-200 hover:bg-gray-300 rounded transition-colors"
            >
              Clear
            </button>
          </div>
        </div>

        <div
          ref={logContainerRef}
          className="flex-1 overflow-y-auto space-y-1 text-xs font-mono"
          style={{ maxHeight: '400px' }}
        >
          {logEntries.length === 0 ? (
            <div className="text-gray-500 text-center py-4">No activity yet</div>
          ) : (
            logEntries.map(entry => (
              <div
                key={entry.id}
                className={`p-2 rounded flex items-start gap-2 ${getLevelColor(entry.level)}`}
              >
                <span className="flex-shrink-0">{getLevelIcon(entry.level)}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline gap-2">
                    <span className="text-gray-600">
                      {entry.timestamp.toLocaleTimeString()}
                    </span>
                    <span className="font-semibold truncate">
                      [{entry.agentType}] {entry.agentName}
                    </span>
                  </div>
                  <div className="mt-1 break-words">{entry.message}</div>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
          Showing {logEntries.length} entries
        </div>
      </div>
    </Card>
  );
};