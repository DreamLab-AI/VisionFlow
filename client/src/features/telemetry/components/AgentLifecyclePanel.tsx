import React, { useMemo } from 'react';
import { Card } from '../../design-system/components/Card';
import { createLogger } from '../../../utils/logger';
import { AgentLifecycleEvent } from '../types';

const logger = createLogger('AgentLifecyclePanel');

interface AgentLifecyclePanelProps {
  events: AgentLifecycleEvent[];
  className?: string;
  compact?: boolean;
}

export const AgentLifecyclePanel: React.FC<AgentLifecyclePanelProps> = ({
  events,
  className = '',
  compact = false
}) => {
  const eventStats = useMemo(() => {
    const stats = {
      total: events.length,
      spawned: 0,
      active: 0,
      errors: 0,
      completed: 0,
      recent: events.slice(-10)
    };

    events.forEach(event => {
      switch (event.eventType) {
        case 'spawn':
          stats.spawned++;
          break;
        case 'activate':
          stats.active++;
          break;
        case 'error':
          stats.errors++;
          break;
        case 'complete':
          stats.completed++;
          break;
      }
    });

    return stats;
  }, [events]);

  const getEventIcon = (eventType: AgentLifecycleEvent['eventType']) => {
    switch (eventType) {
      case 'spawn': return '🚀';
      case 'activate': return '⚡';
      case 'deactivate': return '⏸️';
      case 'error': return '❌';
      case 'complete': return '✅';
      case 'idle': return '😴';
      default: return '📋';
    }
  };

  const getEventColor = (eventType: AgentLifecycleEvent['eventType']) => {
    switch (eventType) {
      case 'spawn': return 'bg-blue-50 border-blue-200';
      case 'activate': return 'bg-green-50 border-green-200';
      case 'deactivate': return 'bg-yellow-50 border-yellow-200';
      case 'error': return 'bg-red-50 border-red-200';
      case 'complete': return 'bg-emerald-50 border-emerald-200';
      case 'idle': return 'bg-gray-50 border-gray-200';
      default: return 'bg-slate-50 border-slate-200';
    }
  };

  return (
    <Card className={className}>
      <div className="p-4">
        <h3 className="text-lg font-semibold mb-4">Agent Lifecycle Events</h3>

        {/* Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{eventStats.spawned}</div>
            <div className="text-sm text-blue-700">Spawned</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{eventStats.active}</div>
            <div className="text-sm text-green-700">Active</div>
          </div>
          <div className="text-center p-3 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">{eventStats.errors}</div>
            <div className="text-sm text-red-700">Errors</div>
          </div>
          <div className="text-center p-3 bg-emerald-50 rounded-lg">
            <div className="text-2xl font-bold text-emerald-600">{eventStats.completed}</div>
            <div className="text-sm text-emerald-700">Completed</div>
          </div>
        </div>

        {/* Recent Events */}
        <div>
          <h4 className="font-medium mb-3">Recent Events</h4>
          <div className={`space-y-2 overflow-y-auto ${compact ? 'max-h-48' : 'max-h-64'}`}>
            {eventStats.recent.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                No lifecycle events recorded yet
              </div>
            ) : (
              eventStats.recent.reverse().map(event => (
                <div
                  key={event.id}
                  className={`p-3 rounded-lg border ${getEventColor(event.eventType)} transition-colors`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{getEventIcon(event.eventType)}</span>
                      <div>
                        <div className="font-medium text-sm">
                          {event.agentName} ({event.eventType})
                        </div>
                        <div className="text-xs text-gray-600">
                          {event.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>

                    {/* Performance metrics if available */}
                    {event.metadata?.performance && (
                      <div className="flex gap-2 text-xs">
                        <span className="px-1 py-0.5 bg-white rounded">
                          CPU: {event.metadata.performance.cpu.toFixed(1)}%
                        </span>
                        {event.metadata.performance.gpu && (
                          <span className="px-1 py-0.5 bg-white rounded">
                            GPU: {event.metadata.performance.gpu.toFixed(1)}%
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Event details */}
                  {Object.keys(event.details).length > 0 && (
                    <div className="mt-2 text-xs">
                      <details className="cursor-pointer">
                        <summary className="text-gray-600 hover:text-gray-800">
                          Details
                        </summary>
                        <pre className="mt-1 p-2 bg-white rounded text-xs overflow-x-auto">
                          {JSON.stringify(event.details, null, 2)}
                        </pre>
                      </details>
                    </div>
                  )}

                  {/* Position info if available */}
                  {event.metadata?.position && (
                    <div className="mt-2 text-xs text-gray-600">
                      Position: ({event.metadata.position.x.toFixed(2)}, {event.metadata.position.y.toFixed(2)})
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};