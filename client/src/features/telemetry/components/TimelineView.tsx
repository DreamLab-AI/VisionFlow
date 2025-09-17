import React, { useMemo, useState } from 'react';
import { Card } from '../../design-system/components/Card';
import { Select } from '../../design-system/components/Select';
import { Switch } from '../../design-system/components/Switch';
import { Label } from '../../design-system/components/Label';
import { createLogger } from '../../../utils/logger';
import { AgentLifecycleEvent, TelemetryFilters } from '../types';

const logger = createLogger('TimelineView');

interface TimelineViewProps {
  events: AgentLifecycleEvent[];
  filters: TelemetryFilters;
  onFiltersChange: (filters: Partial<TelemetryFilters>) => void;
  className?: string;
}

export const TimelineView: React.FC<TimelineViewProps> = ({
  events,
  filters,
  onFiltersChange,
  className = ''
}) => {
  const [timeScale, setTimeScale] = useState<'minute' | 'hour' | 'day'>('hour');
  const [showDetails, setShowDetails] = useState(true);

  // Filter and sort events
  const filteredEvents = useMemo(() => {
    let filtered = [...events];

    // Apply date range filter
    if (filters.dateRange) {
      filtered = filtered.filter(event =>
        event.timestamp >= filters.dateRange!.start &&
        event.timestamp <= filters.dateRange!.end
      );
    }

    // Apply event type filter
    if (filters.eventTypes && filters.eventTypes.length > 0) {
      filtered = filtered.filter(event =>
        filters.eventTypes!.includes(event.eventType)
      );
    }

    // Apply agent filter
    if (filters.agentIds && filters.agentIds.length > 0) {
      filtered = filtered.filter(event =>
        filters.agentIds!.includes(event.agentId)
      );
    }

    return filtered.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }, [events, filters]);

  // Group events by time scale
  const timelineData = useMemo(() => {
    if (filteredEvents.length === 0) return [];

    const firstEvent = filteredEvents[0];
    const lastEvent = filteredEvents[filteredEvents.length - 1];
    const timeSpan = lastEvent.timestamp.getTime() - firstEvent.timestamp.getTime();

    let bucketSize: number;
    switch (timeScale) {
      case 'minute':
        bucketSize = 60 * 1000;
        break;
      case 'hour':
        bucketSize = 60 * 60 * 1000;
        break;
      case 'day':
        bucketSize = 24 * 60 * 60 * 1000;
        break;
    }

    const buckets: Array<{
      timestamp: Date;
      events: AgentLifecycleEvent[];
      count: number;
    }> = [];

    let currentTime = new Date(Math.floor(firstEvent.timestamp.getTime() / bucketSize) * bucketSize);
    const endTime = lastEvent.timestamp.getTime();

    while (currentTime.getTime() <= endTime) {
      const bucketEvents = filteredEvents.filter(event => {
        const eventTime = event.timestamp.getTime();
        return eventTime >= currentTime.getTime() && eventTime < currentTime.getTime() + bucketSize;
      });

      buckets.push({
        timestamp: new Date(currentTime),
        events: bucketEvents,
        count: bucketEvents.length
      });

      currentTime = new Date(currentTime.getTime() + bucketSize);
    }

    return buckets.slice(0, 50); // Limit to 50 buckets for performance
  }, [filteredEvents, timeScale]);

  const maxEventsInBucket = Math.max(...timelineData.map(bucket => bucket.count), 1);

  const getEventTypeColor = (eventType: AgentLifecycleEvent['eventType']) => {
    switch (eventType) {
      case 'spawn': return '#3b82f6';
      case 'activate': return '#10b981';
      case 'deactivate': return '#f59e0b';
      case 'error': return '#ef4444';
      case 'complete': return '#06d6a0';
      case 'idle': return '#6b7280';
      default: return '#8b5cf6';
    }
  };

  const getEventTypeIcon = (eventType: AgentLifecycleEvent['eventType']) => {
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

  const formatTimestamp = (timestamp: Date) => {
    switch (timeScale) {
      case 'minute':
        return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      case 'hour':
        return timestamp.toLocaleString([], {
          month: 'short',
          day: 'numeric',
          hour: '2-digit'
        });
      case 'day':
        return timestamp.toLocaleDateString([], {
          month: 'short',
          day: 'numeric'
        });
    }
  };

  return (
    <Card className={className}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Timeline View</h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Label htmlFor="show-details">Details</Label>
              <Switch
                id="show-details"
                checked={showDetails}
                onCheckedChange={setShowDetails}
                size="sm"
              />
            </div>
            <Select value={timeScale} onValueChange={(value: any) => setTimeScale(value)}>
              <Select.Trigger className="w-32">
                <Select.Value />
              </Select.Trigger>
              <Select.Content>
                <Select.Item value="minute">Per Minute</Select.Item>
                <Select.Item value="hour">Per Hour</Select.Item>
                <Select.Item value="day">Per Day</Select.Item>
              </Select.Content>
            </Select>
          </div>
        </div>

        {/* Timeline Visualization */}
        <div className="relative">
          {timelineData.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              No events in selected time range
            </div>
          ) : (
            <div className="space-y-1">
              {timelineData.map((bucket, index) => (
                <div key={bucket.timestamp.toISOString()} className="flex items-center gap-3">
                  {/* Timestamp */}
                  <div className="w-24 text-xs text-gray-600 font-mono flex-shrink-0">
                    {formatTimestamp(bucket.timestamp)}
                  </div>

                  {/* Event bar */}
                  <div className="flex-1 relative">
                    <div className="h-8 bg-gray-100 rounded-md relative overflow-hidden">
                      {bucket.count > 0 && (
                        <div
                          className="h-full bg-blue-200 rounded-md transition-all"
                          style={{
                            width: `${(bucket.count / maxEventsInBucket) * 100}%`
                          }}
                        />
                      )}

                      {/* Event type indicators */}
                      <div className="absolute inset-0 flex items-center px-1">
                        {bucket.events.slice(0, 10).map((event, eventIndex) => (
                          <div
                            key={event.id}
                            className="w-1.5 h-4 rounded-sm mr-0.5"
                            style={{
                              backgroundColor: getEventTypeColor(event.eventType)
                            }}
                            title={`${event.agentName}: ${event.eventType}`}
                          />
                        ))}
                        {bucket.events.length > 10 && (
                          <div className="text-xs text-gray-600 ml-1">
                            +{bucket.events.length - 10}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Event count */}
                    <div className="absolute -right-8 top-1 text-xs text-gray-500">
                      {bucket.count}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Event Details */}
        {showDetails && filteredEvents.length > 0 && (
          <div className="mt-6">
            <h4 className="font-medium mb-3">Event Details</h4>
            <div className="max-h-64 overflow-y-auto space-y-2">
              {filteredEvents.slice(-20).map(event => (
                <div
                  key={event.id}
                  className="flex items-center gap-3 p-2 bg-gray-50 rounded-md text-sm"
                >
                  <div className="flex-shrink-0 text-lg">
                    {getEventTypeIcon(event.eventType)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium truncate">
                        {event.agentName}
                      </span>
                      <span
                        className="px-2 py-0.5 rounded-full text-xs text-white"
                        style={{ backgroundColor: getEventTypeColor(event.eventType) }}
                      >
                        {event.eventType}
                      </span>
                    </div>
                    <div className="text-gray-600 text-xs">
                      {event.timestamp.toLocaleString()}
                    </div>
                  </div>
                  {event.metadata?.performance && (
                    <div className="flex-shrink-0 text-xs">
                      <div>CPU: {event.metadata.performance.cpu.toFixed(1)}%</div>
                      {event.metadata.performance.gpu && (
                        <div>GPU: {event.metadata.performance.gpu.toFixed(1)}%</div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="mt-6 pt-4 border-t border-gray-200">
          <h4 className="font-medium mb-2">Event Types</h4>
          <div className="flex flex-wrap gap-3 text-xs">
            {['spawn', 'activate', 'deactivate', 'error', 'complete', 'idle'].map(eventType => (
              <div key={eventType} className="flex items-center gap-1">
                <div
                  className="w-3 h-3 rounded-sm"
                  style={{ backgroundColor: getEventTypeColor(eventType as any) }}
                />
                <span className="capitalize">{eventType}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Statistics */}
        <div className="mt-4 text-sm text-gray-600">
          Showing {filteredEvents.length} events across {timelineData.length} time buckets
        </div>
      </div>
    </Card>
  );
};