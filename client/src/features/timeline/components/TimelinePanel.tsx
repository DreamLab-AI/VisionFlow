import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Clock, Play, Pause, RotateCcw, Calendar } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('TimelinePanel');

interface TimelinePanelProps {
  className?: string;
}

interface TimelineEvent {
  id: string;
  title: string;
  type: 'user' | 'system' | 'data' | 'process';
  timestamp: Date;
  duration?: number;
  status: 'completed' | 'active' | 'pending' | 'failed';
  description: string;
  metadata?: Record<string, any>;
}

export const TimelinePanel: React.FC<TimelinePanelProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [selectedType, setSelectedType] = useState('all');
  const [isPlaying, setIsPlaying] = useState(false);
  
  // Subscribe only to timeline settings
  const timelineSettings = useSelectiveSettings({
    enabled: 'timeline.enabled',
    autoScroll: 'timeline.autoScroll',
    showDuration: 'timeline.showDuration',
    groupEvents: 'timeline.groupEvents',
    updateInterval: 'timeline.updateIntervalSeconds',
    maxEvents: 'timeline.maxEvents'
  });
  
  // Mock timeline events
  const events: TimelineEvent[] = useMemo(() => [
    {
      id: '1',
      title: 'User Login',
      type: 'user',
      timestamp: new Date(Date.now() - 5 * 60 * 1000),
      status: 'completed',
      description: 'Alice Johnson logged in successfully',
      metadata: { userId: 'alice', method: 'oauth' }
    },
    {
      id: '2',
      title: 'Data Processing',
      type: 'process',
      timestamp: new Date(Date.now() - 15 * 60 * 1000),
      duration: 45000,
      status: 'completed',
      description: 'Processed 1,247 records from analytics dataset',
      metadata: { records: 1247, source: 'analytics' }
    },
    {
      id: '3',
      title: 'System Update',
      type: 'system',
      timestamp: new Date(Date.now() - 30 * 60 * 1000),
      duration: 120000,
      status: 'completed',
      description: 'Updated application to version 2.1.3',
      metadata: { version: '2.1.3', type: 'patch' }
    },
    {
      id: '4',
      title: 'Data Import',
      type: 'data',
      timestamp: new Date(Date.now() - 45 * 60 * 1000),
      duration: 180000,
      status: 'completed',
      description: 'Imported user behavior data from CSV file',
      metadata: { fileName: 'user-behavior.csv', size: '2.4MB' }
    },
    {
      id: '5',
      title: 'Bot Monitoring',
      type: 'process',
      timestamp: new Date(Date.now() - 60 * 60 * 1000),
      status: 'active',
      description: 'Performance monitoring bot is actively scanning system metrics',
      metadata: { botId: 'perf-monitor-1', checks: 156 }
    }
  ], []);
  
  const timeRanges = [
    { value: '1h', label: 'Last Hour' },
    { value: '6h', label: 'Last 6 Hours' },
    { value: '24h', label: 'Last 24 Hours' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' }
  ];
  
  const typeFilters = [
    { value: 'all', label: 'All Events' },
    { value: 'user', label: 'User Events' },
    { value: 'system', label: 'System Events' },
    { value: 'data', label: 'Data Events' },
    { value: 'process', label: 'Process Events' }
  ];
  
  const filteredEvents = useMemo(() => {
    let filtered = events;
    
    // Filter by type
    if (selectedType !== 'all') {
      filtered = filtered.filter(event => event.type === selectedType);
    }
    
    // Filter by time range
    const now = Date.now();
    const timeLimit = (() => {
      switch (selectedTimeRange) {
        case '1h': return 60 * 60 * 1000;
        case '6h': return 6 * 60 * 60 * 1000;
        case '24h': return 24 * 60 * 60 * 1000;
        case '7d': return 7 * 24 * 60 * 60 * 1000;
        case '30d': return 30 * 24 * 60 * 60 * 1000;
        default: return 24 * 60 * 60 * 1000;
      }
    })();
    
    filtered = filtered.filter(event => now - event.timestamp.getTime() <= timeLimit);
    
    return filtered.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }, [events, selectedType, selectedTimeRange]);
  
  const getStatusColor = (status: TimelineEvent['status']) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'active': return 'bg-blue-100 text-blue-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'failed': return 'bg-red-100 text-red-800';
    }
  };
  
  const getTypeColor = (type: TimelineEvent['type']) => {
    switch (type) {
      case 'user': return 'bg-purple-100 text-purple-800';
      case 'system': return 'bg-orange-100 text-orange-800';
      case 'data': return 'bg-green-100 text-green-800';
      case 'process': return 'bg-blue-100 text-blue-800';
    }
  };
  
  const formatDuration = (durationMs: number) => {
    const seconds = Math.floor(durationMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };
  
  const formatTimeAgo = (timestamp: Date) => {
    const diffMs = Date.now() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };
  
  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
    logger.info('Timeline playback toggled', { isPlaying: !isPlaying });
  };
  
  if (!timelineSettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock size={20} />
            Timeline
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Clock size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">Timeline is disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable timeline to track events and activities over time
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock size={20} />
            Timeline
            <Badge variant="outline">
              {filteredEvents.length} events
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={togglePlayback}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            </Button>
            <Button size="sm" variant="outline">
              <RotateCcw size={16} />
            </Button>
          </div>
        </CardTitle>
        
        {/* Filters */}
        <div className="grid grid-cols-2 gap-4 mt-4">
          <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {timeRanges.map(range => (
                <SelectItem key={range.value} value={range.value}>
                  {range.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Select value={selectedType} onValueChange={setSelectedType}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {typeFilters.map(filter => (
                <SelectItem key={filter.value} value={filter.value}>
                  {filter.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      
      <CardContent>
        <ScrollArea className="h-[500px]">
          <div className="relative">
            {/* Timeline Line */}
            <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-200" />
            
            <div className="space-y-6">
              {filteredEvents.map((event, index) => (
                <div key={event.id} className="relative flex items-start gap-4">
                  {/* Timeline Dot */}
                  <div className={`relative z-10 w-4 h-4 rounded-full border-4 ${
                    event.status === 'active' 
                      ? 'bg-blue-500 border-blue-200 animate-pulse'
                      : event.status === 'completed'
                      ? 'bg-green-500 border-green-200'
                      : event.status === 'failed'
                      ? 'bg-red-500 border-red-200'
                      : 'bg-yellow-500 border-yellow-200'
                  }`} />
                  
                  {/* Event Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <h3 className="font-medium">{event.title}</h3>
                        <Badge className={getTypeColor(event.type)}>
                          {event.type}
                        </Badge>
                        <Badge className={getStatusColor(event.status)}>
                          {event.status}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Calendar size={12} />
                        {formatTimeAgo(event.timestamp)}
                      </div>
                    </div>
                    
                    <p className="text-sm text-muted-foreground mb-2">
                      {event.description}
                    </p>
                    
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>{event.timestamp.toLocaleString()}</span>
                      {timelineSettings.showDuration && event.duration && (
                        <span>Duration: {formatDuration(event.duration)}</span>
                      )}
                    </div>
                    
                    {event.metadata && Object.keys(event.metadata).length > 0 && (
                      <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                        {Object.entries(event.metadata).map(([key, value]) => (
                          <span key={key} className="mr-3">
                            <span className="font-medium">{key}:</span> {String(value)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </ScrollArea>
        
        {filteredEvents.length === 0 && (
          <div className="text-center py-8">
            <Clock size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">No events found</p>
            <p className="text-sm text-muted-foreground mt-2">
              Try adjusting your filters or time range
            </p>
          </div>
        )}
        
        {/* Timeline Settings */}
        <div className="mt-6 pt-4 border-t">
          <h3 className="font-medium mb-3">Timeline Settings</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Auto Scroll</span>
              <Button
                variant={timelineSettings.autoScroll ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('timeline.autoScroll', !timelineSettings.autoScroll)}
              >
                {timelineSettings.autoScroll ? 'ON' : 'OFF'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Show Duration</span>
              <Button
                variant={timelineSettings.showDuration ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('timeline.showDuration', !timelineSettings.showDuration)}
              >
                {timelineSettings.showDuration ? 'ON' : 'OFF'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Group Events</span>
              <Button
                variant={timelineSettings.groupEvents ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('timeline.groupEvents', !timelineSettings.groupEvents)}
              >
                {timelineSettings.groupEvents ? 'ON' : 'OFF'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Max Events</span>
              <Badge variant="outline">{timelineSettings.maxEvents}</Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TimelinePanel;